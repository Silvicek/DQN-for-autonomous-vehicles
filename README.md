# DQN-for-autonomous-vehicles

This project uses Deep Q-networks for solving an 'Autonomous vehicles' task.
The main algorithm: [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)
It's also possible to run with [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

The autonomous vehicle task is present at my [gym fork](https://github.com/Silvicek/gym/tree/target), although the algorithm can be used in any gym environment.

### Usage:
Run `duel.py` with an array of different arguments, see the script for more info.

An example run:
    `python duel.py ACar-v0 --memory_steps 0 --save_path models/tests/ --episodes 2000 --advantage max --hidden_size [50,50]`

If you want to perform an evolution parameter search, run `evolution.py`.

### Requirements
* numpy
* keras
* OpenAI gym
* DEAP (for evolution parameter search)