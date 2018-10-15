[//]: # (Image References)

[video_random]: https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/2%20-%20Value-based%20methods/Project-Navigation/results/random_agent.gif "Random Agent"

[video_trained]: https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/2%20-%20Value-based%20methods/Project-Navigation/results/trained_agent.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.  



| Random agent             |  Trained agent |
:-------------------------:|:-------------------------:
![Random Agent][video_random]  |  ![Trained Agent][video_trained]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in this folder, unzip (or decompress) the file and then write the correct path in the argument for creating the environment under the notebook `Navigation_solution.ipynb`:

```python
env = env = UnityEnvironment(file_name="Banana.app")

```

### Description

- `dqn_agent.py`: code for the agent used in the environment
- `model.py`: code containing the Q-Network used as the function approximator by the agent
- `dqn.pth`: saved model weights for the original DQN model
- `ddqn.pth`: saved model weights for the Double DQN model
- `ddqn.pth`: saved model weights for the Dueling Double DQN model
- `Navigation_exploration.ipynb`: explore the unity environment
- `Navigation_solution.ipynb`: notebook containing the solution
- `Navigation_Pixels.ipynb`: notebook containing the code for the pixel-action problem (see below)

### Instructions

Follow the instructions in `Navigation_solution.ipynb` to get started with training your own agent! 
To watch a trained smart agent, follow the instructions below:

- **DQN**: If you want to run the original DQN algorithm, use the checkpoint `dqn.pth` for loading the trained model. Also, choose the parameter `qnetwork` as `QNetwork` while defining the agent and the parameter `update_type` as `dqn`.
- **Double DQN**: If you want to run the Double DQN algorithm, use the checkpoint `ddqn.pth` for loading the trained model. Also, choose the parameter `qnetwork` as `QNetwork` while defining the agent and the parameter `update_type` as `double_dqn`.
- **Dueling Double DQN**: If you want to run the Dueling Double DQN algorithm, use the checkpoint `dddqn.pth` for loading the trained model. Also, choose the parameter `qnetwork` as `DuelingQNetwork` while defining the agent and the parameter `update_type` as `double_dqn`.

### Enhancements

Several enhancements to the original DQN algorithm have also been incorporated:

- Double DQN [[Paper](https://arxiv.org/abs/1509.06461)] [[Code](https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/2%20-%20Value-based%20methods/Project-Navigation/dqn_agent.py#L94)]
- Prioritized Experience Replay [[Paper](https://arxiv.org/abs/1511.05952)] [[Code]()] (WIP)
- Dueling DQN [[Paper](https://arxiv.org/abs/1511.06581)] [[Code](https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/2%20-%20Value-based%20methods/Project-Navigation/model.py)]

### Results

Plot showing the score per episode over all the episodes. The environment was solved in **377** episodes (currently).

| Double DQN | DQN | Dueling DQN |
:-------------------------:|:-------------------------:|:-------------------------:
![double-dqn-scores](https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/2%20-%20Value-based%20methods/Project-Navigation/results/ddqn_new_scores.png) |  ![dqn-scores](https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/2%20-%20Value-based%20methods/Project-Navigation/results/dqn_new_scores.png) | ![dueling-double-dqn-scores](https://github.com/dalmia/udacity-deep-reinforcement-learning/blob/master/2%20-%20Value-based%20methods/Project-Navigation/results/dddqn_new_scores.png) 


### Challenge: Learning from Pixels

In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in this folder, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

(_For AWS_) If you'd like to train the agent on AWS, you must follow the instructions to [set up X Server](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above.

### Dependencies

Use the `requirements.txt` file (in the [main](https://github.com/dalmia/udacity-deep-reinforcement-learning) folder) to install the required dependencies via `pip`.

```
pip install -r requirements.txt

```
