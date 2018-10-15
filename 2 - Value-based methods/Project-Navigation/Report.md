# Project 1: Navigation

Author: [Aman Dalmia](http://github.com/dalmia) 

The project demonstrates the ability of value-based methods, specifically, [Deep Q-learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and its variants, to learn a suitable policy in a model-free Reinforcement Learning setting using a Unity environment, which consists of a continuous state space of 37 dimensions, with the goal to navigate around and collect yellow bananas (reward: +1) while avoiding blue bananas (reward: -1). There are 4 actions to choose from: move left, move right, move forward and move backward. A agent choosing actions randomly can be seen in motion below:

![random agent](results/random_agent.gif) 



The following report is written in four parts:

- **Implementation**
- **Results**
- **Ideas for improvement** 



## Implementation

The 