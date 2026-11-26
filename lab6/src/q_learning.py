"""Tabular Q-learning implementation for the single-robot warehouse."""
import numpy as np
import matplotlib.pyplot as plt
from warehouse import SingleRobotEnv, START, GOAL, OBSTACLES, GAMMA, MAX_STEPS

ALPHA = 0.2
SEED = 7
EPISODES = 1500


def epsilon_at(ep, total=EPISODES):
    return max(0.05, 1.0 - 0.95 * min(ep, 1200) / 1200)


def choose_action(q, state, epsilon, rng, greedy_tie_first=False):
    if rng.random() < epsilon:
        return int(rng.integers(5))
    best = np.flatnonzero(np.isclose(q[state], np.max(q[state])))
    return int(best[0] if greedy_tie_first else rng.choice(best))


def q_update(q, state, action, reward, next_state, terminated, alpha=ALPHA, gamma=GAMMA):
    target = reward if terminated else reward + gamma * np.max(q[next_state])
    q[state, action] += alpha * (target - q[state, action])
    return float(target), float(q[state, action])


def train(episodes=EPISODES, seed=SEED):
    rng = np.random.default_rng(seed)
    env, q = SingleRobotEnv(), np.zeros((25, 5), dtype=np.float64)
    returns, successes, epsilons, invalids, timeouts = [], [], [], [], []
    for ep in range(episodes):
        pos = env.reset()
        total, invalid_count, success = 0.0, 0, False
        eps = epsilon_at(ep)
        for _ in range(MAX_STEPS):
            s = 5 * pos[0] + pos[1]
            a = choose_action(q, s, eps, rng)
            nxt, reward, terminated, truncated, info = env.step(a)
            ns = 5 * nxt[0] + nxt[1]
            # A 40-step cutoff is truncation, not true termination: bootstrap.
            q_update(q, s, a, reward, ns, terminated)
            total += reward
            invalid_count += info['invalid']
            pos = nxt
            if terminated or truncated:
                success = terminated
                break
        returns.append(total); successes.append(int(success)); epsilons.append(eps)
        invalids.append(invalid_count); timeouts.append(int(not success and env.steps >= MAX_STEPS))
    return q, {"returns": returns, "successes": successes, "epsilons": epsilons,
               "invalids": invalids, "timeouts": timeouts}


def evaluate(q, start=START):
    env = SingleRobotEnv(); pos = env.reset(start)
    path, invalids, total = [pos], 0, 0.0
    for _ in range(MAX_STEPS):
        s = 5 * pos[0] + pos[1]
        # Deterministic tie-breaking: lowest-index maximizing action.
        a = int(np.argmax(q[s]))
        pos, reward, terminated, truncated, info = env.step(a)
        path.append(pos); total += reward; invalids += info['invalid']
        if terminated or truncated:
            return {"path": path, "success": terminated, "timeout": truncated,
                    "steps": len(path)-1, "invalid_moves": invalids, "return": total}
    raise RuntimeError("Evaluation exceeded the specified cutoff")


def random_baseline(n=100, seed=7):
    rng = np.random.default_rng(seed); results=[]
    for _ in range(n):
        env=SingleRobotEnv(); pos=env.reset(); invalids=0
        for t in range(MAX_STEPS):
            pos, _, term, trunc, info=env.step(int(rng.integers(5)))
            invalids += info['invalid']
            if term or trunc: break
        results.append({"success": bool(term), "timeout": bool(trunc),
                        "steps": t+1 if term else None, "invalid_moves": invalids})
    return results


def moving_average(x, window=50):
    x=np.asarray(x,dtype=float)
    return np.convolve(x,np.ones(window)/window,mode='valid') if len(x)>=window else x


def make_plots(history, q_eval, out_dir):
    import os
    os.makedirs(out_dir,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(moving_average(history['returns']),label='Return (50-episode MA)')
    ax.set(xlabel='Episode',ylabel='Return',title='Tabular Q-learning: training return'); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir,'q_learning_return.png'),dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.plot(moving_average(history['successes']),label='Success (50-episode MA)')
    ax.set(xlabel='Episode',ylabel='Success fraction',title='Tabular Q-learning: training success'); ax.set_ylim(-.05,1.05); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir,'q_learning_success.png'),dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5));
    for r in range(5):
        for c in range(5):
            if (r,c) in OBSTACLES: ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,fc='gray'))
            ax.text(c,r,'X' if (r,c) in OBSTACLES else '',ha='center',va='center')
    p=q_eval['path']; ax.plot([c for r,c in p],[r for r,c in p],'-o'); ax.scatter([0,4],[2,2],s=100,marker='s'); ax.set_xlim(-.5,4.5); ax.set_ylim(4.5,-.5); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid(); ax.set_title('Greedy Q-policy trajectory'); fig.tight_layout(); fig.savefig(os.path.join(out_dir,'q_learning_trajectory.png'),dpi=160); plt.close(fig)

if __name__=='__main__':
    import json, os
    here=os.path.dirname(os.path.dirname(__file__)); results=os.path.join(here,'results'); plots=os.path.join(here,'plots'); os.makedirs(results,exist_ok=True)
    q,h=train(); ev=evaluate(q); rb=random_baseline(); make_plots(h,ev,plots)
    np.save(os.path.join(results,'q_table.npy'),q)
    with open(os.path.join(results,'q_learning.json'),'w') as f: json.dump({"evaluation":ev,"random_baseline":rb,"history":h},f,indent=2)
    print('Q evaluation:',ev); print('Random success fraction:',np.mean([x['success'] for x in rb]))
