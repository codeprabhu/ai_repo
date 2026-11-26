import os,sys,json
import numpy as np, torch
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'src'))
from q_learning import train as qtrain,evaluate as qeval,random_baseline,make_plots
from dqn import train_dqn,evaluate_dqn,plot_dqn,plot_two_robot,plot_single_robot
from warehouse import TwoRobotEnv
ROOT=os.path.dirname(__file__); R=os.path.join(ROOT,'results'); P=os.path.join(ROOT,'plots'); os.makedirs(R,exist_ok=True); os.makedirs(P,exist_ok=True)
# Q table
q,h=qtrain(1500,7); qe=qeval(q); rb=random_baseline(100,7); make_plots(h,qe,P)
np.save(os.path.join(R,'q_table.npy'),q)
qres={'evaluation':qe,'random_baseline':{'runs':rb,'success_fraction':float(np.mean([x['success'] for x in rb])),'timeout_fraction':float(np.mean([x['timeout'] for x in rb])),'mean_success_steps':float(np.mean([x['steps'] for x in rb if x['success']])) if any(x['success'] for x in rb) else None,'mean_invalid_moves':float(np.mean([x['invalid_moves'] for x in rb]))},'history':h}
json.dump(qres,open(os.path.join(R,'q_learning.json'),'w'),indent=2)
print('Q RESULT',qe,'random',qres['random_baseline'])
# single DQN
m1,h1=train_dqn(False,12000,7); e1=evaluate_dqn(m1,False); e1_alt=evaluate_dqn(m1,False,(0,0))
torch.save(m1.state_dict(),os.path.join(R,'single_dqn.pt')); json.dump({'evaluation':e1,'additional_start_0_0':e1_alt,'history':h1},open(os.path.join(R,'single_dqn.json'),'w'),indent=2)
plot_dqn(h1,os.path.join(P,'single_dqn_success.png'),'Single-robot DQN: success (20-episode MA)'); plot_single_robot(e1['path'],os.path.join(P,'single_dqn_trajectory.png'))
print('SINGLE DQN',e1,'alt',e1_alt,'episodes',len(h1['successes']),'updates',h1['optimizer_updates'])
# two robot DQN
m2,h2=train_dqn(True,60000,7); e2=evaluate_dqn(m2,True); e2_alt=evaluate_dqn(m2,True,((0,0),(4,4)))
torch.save(m2.state_dict(),os.path.join(R,'two_robot_dqn.pt')); plot_dqn(h2,os.path.join(P,'two_robot_dqn_success.png'),'Two-robot DQN: success (20-episode MA)')
plot_two_robot(e2['path'],os.path.join(P,'two_robot_trajectory.png'))
json.dump({'evaluation':e2,'additional_start':e2_alt,'history':h2},open(os.path.join(R,'two_robot_dqn.json'),'w'),indent=2)
print('TWO DQN',e2,'alt',e2_alt,'episodes',len(h2['successes']),'updates',h2['optimizer_updates'])
