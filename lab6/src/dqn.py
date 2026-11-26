"""DQN agents for single- and two-robot warehouse tasks (PyTorch)."""
from collections import deque
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from warehouse import SingleRobotEnv, TwoRobotEnv, START, GOAL, GAMMA, MAX_STEPS, one_hot, joint_one_hot

DEVICE = torch.device('cpu')

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,64), nn.ReLU(),
                                 nn.Linear(64,64), nn.ReLU(),
                                 nn.Linear(64,output_dim))
    def forward(self,x): return self.net(x)

class ReplayBuffer:
    def __init__(self,capacity=20000): self.data=deque(maxlen=capacity)
    def add(self,*transition): self.data.append(transition)
    def sample(self,batch_size):
        batch=random.sample(self.data,batch_size)
        s,a,r,ns,d=zip(*batch)
        return (torch.tensor(np.asarray(s),dtype=torch.float32),
                torch.tensor(a,dtype=torch.int64),
                torch.tensor(r,dtype=torch.float32),
                torch.tensor(np.asarray(ns),dtype=torch.float32),
                torch.tensor(d,dtype=torch.float32))
    def __len__(self): return len(self.data)


def dqn_targets(rewards, next_max, terminated, gamma=GAMMA):
    """True termination suppresses bootstrap; truncation is stored as not terminated."""
    return rewards + gamma * (1.0-terminated) * next_max


def validate_dqn_targets():
    # Required synthetic checks: non-terminal and truncation bootstrap; terminal does not.
    assert abs(float(dqn_targets(torch.tensor([1.]),torch.tensor([2.]),torch.tensor([0.]))[0])-2.9)<1e-6
    assert abs(float(dqn_targets(torch.tensor([5.]),torch.tensor([99.]),torch.tensor([1.]))[0])-5.0)<1e-6
    assert abs(float(dqn_targets(torch.tensor([1.]),torch.tensor([2.]),torch.tensor([0.]))[0])-2.9)<1e-6


def train_dqn(two_robot=False, total_steps=None, seed=7):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    torch.set_num_threads(1)
    total_steps = total_steps or (60000 if two_robot else 12000)
    env = TwoRobotEnv() if two_robot else SingleRobotEnv()
    input_dim, output_dim = (50,25) if two_robot else (25,5)
    online=QNetwork(input_dim,output_dim).to(DEVICE)
    target=QNetwork(input_dim,output_dim).to(DEVICE); target.load_state_dict(online.state_dict())
    optimizer=torch.optim.Adam(online.parameters(),lr=0.001)
    replay=ReplayBuffer(20000); batch_size=64; warmup=500
    eps_start,eps_end=1.0,0.05; decay_steps=int(total_steps*0.8)
    state=env.reset(); episode_return=0.0; episode_steps=0
    returns=[]; successes=[]; collisions=[]; timeouts=[]; invalids=[]; losses=[]
    ep_invalid=0; ep_collision=False; ep_success=False; optimizer_updates=0
    for step in range(total_steps):
        eps=max(eps_end,eps_start-(eps_start-eps_end)*step/max(1,decay_steps))
        s=joint_one_hot(state) if two_robot else one_hot(state)
        if random.random()<eps: action=random.randrange(output_dim)
        else:
            with torch.no_grad(): action=int(torch.argmax(online(torch.tensor(s,dtype=torch.float32).unsqueeze(0))).item())
        next_state,reward,terminated,truncated,info=env.step(action)
        ns=joint_one_hot(next_state) if two_robot else one_hot(next_state)
        # Only true task termination is marked done. Time-limit truncation bootstraps.
        replay.add(s,action,reward,ns,float(terminated))
        episode_return += reward; episode_steps += 1; ep_invalid += info['invalid_moves'] if two_robot else info['invalid']
        ep_collision = ep_collision or (info['collision'] if two_robot else False)
        state=next_state
        if len(replay)>=warmup and (step+1)%4==0:
            bs,ba,br,bns,bd=replay.sample(batch_size)
            q_sa=online(bs).gather(1,ba.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_max=target(bns).max(dim=1).values
                y=dqn_targets(br,next_max,bd)
            loss=F.mse_loss(q_sa,y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            optimizer_updates += 1; losses.append(float(loss.item()))
            if optimizer_updates%250==0: target.load_state_dict(online.state_dict())
        if terminated or truncated:
            ep_success = (env.arrived == (True,True)) if two_robot else bool(terminated)
            returns.append(episode_return); successes.append(int(ep_success)); collisions.append(int(ep_collision)); timeouts.append(int(truncated)); invalids.append(ep_invalid)
            state=env.reset(); episode_return=0.; episode_steps=0; ep_invalid=0; ep_collision=False; ep_success=False
    model_path=None
    return online, {"returns":returns,"successes":successes,"collisions":collisions,"timeouts":timeouts,
                    "invalids":invalids,"losses":losses,"optimizer_updates":optimizer_updates,
                    "environment_steps":total_steps,"seed":seed,"two_robot":two_robot}


def evaluate_dqn(model, two_robot=False, starts=None):
    env=TwoRobotEnv() if two_robot else SingleRobotEnv()
    state=env.reset(starts) if two_robot and starts is not None else env.reset(starts) if (not two_robot and starts is not None) else env.reset()
    path=[state]; invalids=0; total=0.0; collision=False
    for _ in range(MAX_STEPS):
        s=joint_one_hot(state) if two_robot else one_hot(state)
        with torch.no_grad(): action=int(torch.argmax(model(torch.tensor(s,dtype=torch.float32).unsqueeze(0))).item())
        state,reward,terminated,truncated,info=env.step(action)
        path.append(state); total+=reward
        invalids += info['invalid_moves'] if two_robot else info['invalid']
        collision = collision or (info['collision'] if two_robot else False)
        if terminated or truncated: break
    success=(env.arrived==(True,True)) if two_robot else bool(terminated)
    return {"path":path,"success":bool(success),"collision":bool(collision),"timeout":bool(truncated),
            "steps":len(path)-1,"invalid_moves":invalids,"return":total}


def plot_dqn(history,out_path,title):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4.5)); w=20
    x=np.asarray(history['successes'],dtype=float)
    ma=np.convolve(x,np.ones(min(w,len(x)))/min(w,len(x)),mode='valid') if len(x) else x
    ax.plot(ma); ax.set(xlabel='Completed episode',ylabel='Success fraction',title=title); ax.set_ylim(-.05,1.05); fig.tight_layout(); fig.savefig(out_path,dpi=160); plt.close(fig)


def plot_single_robot(path,out_path,title='Single-robot greedy DQN trajectory'):
    import matplotlib.pyplot as plt
    import os
    from warehouse import OBSTACLES
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    fig,ax=plt.subplots(figsize=(5,5))
    for r,c in OBSTACLES: ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,fc='gray'))
    ax.plot([p[1] for p in path],[p[0] for p in path],'-o')
    for t,p in enumerate(path): ax.annotate(str(t),(p[1],p[0]),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.set_xlim(-.5,4.5); ax.set_ylim(4.5,-.5); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid(); ax.set_title(title); fig.tight_layout(); fig.savefig(out_path,dpi=160); plt.close(fig)


def plot_two_robot(path,out_path):
    import matplotlib.pyplot as plt
    import os
    os.makedirs(os.path.dirname(out_path),exist_ok=True)
    fig,ax=plt.subplots(figsize=(6,5))
    from warehouse import OBSTACLES
    for r,c in OBSTACLES: ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,fc='gray'))
    if path:
        ax.plot([p[0][1] for p in path],[p[0][0] for p in path],'-o',label='Robot A')
        ax.plot([p[1][1] for p in path],[p[1][0] for p in path],'-s',label='Robot B')
        for t,p in enumerate(path):
            ax.annotate(f'{t}',(p[0][1],p[0][0]),xytext=(-7,-8),textcoords='offset points',fontsize=7)
            ax.annotate(f'{t}',(p[1][1],p[1][0]),xytext=(5,6),textcoords='offset points',fontsize=7)
    ax.set_xlim(-.5,4.5); ax.set_ylim(4.5,-.5); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid(); ax.legend(); ax.set_title('Two-robot greedy DQN trajectory (time labels on A)'); fig.tight_layout(); fig.savefig(out_path,dpi=160); plt.close(fig)
