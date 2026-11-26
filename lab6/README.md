# CS F407 Reinforcement Learning Laboratory

This project implements the warehouse RL laboratory: tabular Q-learning, single-robot DQN, and joint-action two-robot DQN.

## Environment
- Python 3.10+
- Install: `python -m pip install -r requirements.txt`
- Tests: `PYTHONPATH=src python -m pytest -q`
- Re-run all experiments: `PYTHONPATH=src python run_experiments.py`

The experiment runner writes metrics and model weights to `results/` and figures to `plots/`.

## Files
- `src/warehouse.py`: deterministic environment and state encodings.
- `src/q_learning.py`: tabular Q-learning, random baseline, and greedy evaluation.
- `src/dqn.py`: replay buffer, online/target networks, DQN training and evaluation.
- `tests/test_lab.py`: environment, collision, Q-update, and DQN-target tests.
- `results/`: recorded outputs and trained model weights from seed 7.
- `plots/`: training curves and learned trajectories.
- `report.pdf` and `report.docx`: laboratory report.
- `prompts.md`: LLM prompts and implementation/validation notes.

## Reproducibility
All reported runs use seed 7. Q-learning uses 1,500 episodes, alpha 0.2, gamma 0.95, and epsilon linearly decayed from 1.0 to 0.05 over 1,200 episodes. Single-robot DQN uses 12,000 environment steps; two-robot DQN uses 60,000 environment steps. DQN uses a 20,000-transition replay buffer, batch size 64, 500-step warm-up, updates every four environment steps, target synchronization every 250 optimizer updates, and Adam at 0.001.

The evaluation results are single greedy rollouts from the specified starts, not independent statistical trials. The additional-start tests use the trained policy without retraining.
