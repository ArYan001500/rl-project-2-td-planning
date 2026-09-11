# RL project 2: Temporal-difference learning and planning

This project studies Cliff Walking with temporal-difference and model-based reinforcement-learning methods.

## Contents

- `question_1_n_step_sarsa`: custom Cliff Walking environment, environment test, and n-step SARSA.
- `question_2_dyna_q`: Dyna-Q on the same Cliff Walking setting.
- `question_3_dyna_q_plus`: Dyna-Q, Dyna-Q with an environment change, and Dyna-Q+.

## Results from the project notes

The n-step SARSA experiment used a 5x20 Cliff Walking grid and 500 training episodes. Dyna-Q learned faster when planning steps were used. In the changing-environment experiment, Dyna-Q+ adapted faster after the change because its exploration bonus encouraged the agent to revisit older actions.
