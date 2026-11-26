# LLM prompts and validation record

These prompts record the engineering specifications used to implement the laboratory. They are included so the implementation can be audited and reproduced.

## Prompt 1 — Tabular Q-learning

Implement tabular Q-learning in Python using NumPy for the warehouse environment. Use a 5x5 grid, obstacles {(0,2),(1,2),(3,2),(4,2)}, start (2,0), goal (2,4), and actions up/down/left/right/wait with codes 0–4. Obstacle or boundary moves stay in place. Rewards are mutually exclusive: +10 on reaching the goal, -1 for an invalid move, and -0.1 for another valid action. Use alpha=0.2, gamma=0.95, a zero-initialized 25x5 Q-table, seed 7, and 1,500 episodes. Decrease epsilon linearly from 1.0 to 0.05 over the first 1,200 episodes. Cut off unfinished episodes after 40 steps. Return separate termination and truncation flags; bootstrap at truncation but not at goal termination. Include separate environment, action selection, update, training, and greedy evaluation functions. Do not use path search or a ready-made RL library.

## Prompt 2 — Single-robot DQN

Extend the validated single-robot environment to DQN in PyTorch without changing its map, rewards, start, goal, or discount. Encode the position as a 25-element one-hot vector. Use a 25-64-64-5 network with ReLU hidden layers and linear Q-value outputs. Use epsilon-greedy selection, replay capacity 20,000, batch size 64, 500 environment steps of warm-up, one optimizer update every four environment steps, and target-network synchronization every 250 optimizer updates. Use Adam at 0.001 and mean squared TD error. Train for 12,000 environment steps with seed 7 and decay epsilon from 1.0 to 0.05 over the first 80% of training. Use reward-only targets for true termination and bootstrap at truncation. Keep greedy evaluation separate.

## Prompt 3 — Two-robot DQN

Extend the DQN to two robots. State is the ordered pair of positions encoded by concatenating two 25-element one-hot vectors (50 inputs). Joint action j=5*a_A+a_B gives 25 outputs. Both moves are computed from the same old state. Invalid moves stay in place; robots already at their own goals stay. Detect same-cell and swap collisions. Entering a waiting robot's cell is a collision; following a robot into a cell it vacates is allowed if final cells differ and there is no swap. On collision retain old positions and terminate with reward -10. Otherwise use r=-0.1-b+5k+5I, where b counts invalid moves by robots not already at goals, k counts newly reached goals, and I is 1 when both robots are now at their goals. Terminate when both arrive; truncate after 40 joint steps. Train for 60,000 environment steps with seed 7 and the same DQN settings. Do not use a planner or action filter.

## Validation and corrections recorded

- Verified all five specified single-robot movement/reward cases and the 40-step truncation.
- Verified synthetic Q-learning targets: non-terminal target 2.9 and updated value 0.9; terminal target 5 and updated value 1.32. Truncation uses the non-terminal bootstrap target.
- Verified same-cell collision, swap collision, following into a vacated cell, and moving while the other robot waits.
- Verified a robot already at its goal remains in place and does not receive repeated arrival reward.
- Verified DQN target values for non-terminal, terminal, and truncated transitions. The recorded action is selected using `gather` from the online network output; target computation is under `torch.no_grad()`.
- A first test assertion incorrectly expected an already-arrived robot to receive a new-arrival event. The test was corrected: arrival is initialized by `reset`, so subsequent steps correctly count zero new arrivals for that robot.
- The two-robot learned policy succeeds on the default start but fails to generalize to the specified alternate start within 40 steps. This result is retained rather than replaced with a planned path.
