import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
from warehouse import SingleRobotEnv, TwoRobotEnv
from q_learning import q_update
from dqn import validate_dqn_targets


def test_single_environment():
    e=SingleRobotEnv()
    e.reset((0,0)); assert e.step(0)[:4] == ((0,0),-1.0,False,False)
    e.reset((1,1)); assert e.step(3)[:4] == ((1,1),-1.0,False,False)
    e.reset((2,1)); assert e.step(3)[:2] == ((2,2),-0.1)
    e.reset((2,1)); assert e.step(4)[:2] == ((2,1),-0.1)
    e.reset((2,3)); assert e.step(3)[:4] == ((2,4),10.0,True,False)
    e.reset((2,0))
    for _ in range(40): out=e.step(4)
    assert out[3] is True and out[2] is False


def test_q_updates():
    q=np.zeros((25,5)); q[0,3]=0.4; q[1,:]=[1,2,0,0,0]
    y,v=q_update(q,0,3,1,1,False)
    assert abs(y-2.9)<1e-9 and abs(v-0.9)<1e-9
    q=np.zeros((25,5)); q[0,3]=0.4
    y,v=q_update(q,0,3,5,1,True)
    assert abs(y-5.0)<1e-9 and abs(v-1.32)<1e-9
    # Truncation is not true termination, so the same non-terminal bootstrap applies.
    q=np.zeros((25,5)); q[0,3]=0.4; q[1,0]=2.0
    y,v=q_update(q,0,3,1,1,False)
    assert abs(y-2.9)<1e-9 and abs(v-0.9)<1e-9


def test_joint_collisions_and_movement():
    e=TwoRobotEnv(); e.reset(((2,1),(2,3)))
    p,r,t,tr,i=e.step(5*3+2) # A right, B left
    assert i['collision'] and t and p==((2,1),(2,3)) and r==-10
    e.reset(((2,1),(2,2)))
    p,r,t,tr,i=e.step(5*3+2) # swap
    assert i['collision'] and t and p==((2,1),(2,2))
    e.reset(((2,1),(2,2)))
    p,r,t,tr,i=e.step(5*3+3) # A right, B right; follow into vacated cell
    assert not i['collision'] and p==((2,2),(2,3))
    e.reset(((2,1),(2,2)))
    p,r,t,tr,i=e.step(5*0+4) # A up, B wait
    assert not i['collision'] and p==((1,1),(2,2))


def test_goal_robot_stays_without_repeated_arrival():
    e=TwoRobotEnv(); e.reset(((2,4),(2,3))) # A is already at its own goal
    p,r,t,tr,i=e.step(24) # both wait
    assert p[0]==(2,4) and i['newly_arrived']==0
    p,r,t,tr,i=e.step(24)
    assert p[0]==(2,4) and i['newly_arrived']==0


def test_dqn_targets(): validate_dqn_targets()

if __name__=='__main__':
    for f in [test_single_environment,test_q_updates,test_joint_collisions_and_movement,test_goal_robot_stays_without_repeated_arrival,test_dqn_targets]:
        f(); print('PASS',f.__name__)
