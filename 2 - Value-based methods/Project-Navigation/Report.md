# Project 1: Navigation

Author: [Aman Dalmia](http://github.com/dalmia) 

The project demonstrates the ability of value-based methods, specifically, [Deep Q-learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and its variants, to learn a suitable policy in a model-free Reinforcement Learning setting using a Unity environment, which consists of a continuous state space of 37 dimensions, with the goal to navigate around and collect yellow bananas (reward: +1) while avoiding blue bananas (reward: -1). There are 4 actions to choose from: move left, move right, move forward and move backward. A agent choosing actions randomly can be seen in motion below:

![random agent](results/random_agent.gif) 



The following report is written in four parts:

- **Implementation**
- **Results**
- **Ideas for improvement** 



## Implementation

At the heart of the learning algorithm is the , [Deep Q-learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf), which surpassed human-level performance in Atari games. It is an off-policy learning algorithm where the policy being evaluated is different from the policy being learned.



To step back for a bit, the idea of Q-learning is to learn the action-value function, often denoted as          `Q(s, a)` , where `s` represents the current state and `a` represents the action being evaluated. Q-learning is a form of Temporal-Difference learning (TD-learning), where unlike Monte-Carlo methods, we can learn from each step rather than waiting for an episode to complete. The idea is that once we take an action and are thrust into a new state, we use the current Q-value of that state as the estimate for future rewards. 



![q-learning-update](images/q-learning.png)  



There's one specific problem here. Since our space is continuous, we can't use a tabular representation. Hence, we use a `Function Approximator`. The idea behind a function approximator is to introduce a new parameter $\theta$ that helps us to obtain an approximation of the `Q(s, a)`, $\hat{Q} (s, a; \theta)$. So, this becomes a supervised learning problem where the approximation $\hat{Q}$ represents the expected value and $R + \gamma * max (Q(s', a))$ becomes the target. We then use mean-square error as the loss function and update the weights accordingly using gradient descent. Now, the choice remains to choose the function approximator. Enter **Deep Learning**! We use a neural network as function approximator here. More specifically, we choose a 2-hidden layer network with both the layers having 64 hidden units with `relu` activation applied after each fully-connected layer. `Adam` was used as the optimizer for finding the optimal weights:

![fa equation](images/fa_equation.png) 

However, the algorithm described above in its raw form is highly unstable. Two techniques contributed significantly towards stabilizing the training:

- **Fixed Q-targets**: As can be seen from the equation above, the target during training itself is dependent on `w`, the parameter being updated. This leads to constantly moving targets and hurts training. The idea behind fixed q-targets is to fix the parameter `w` used in the calculation of the target, $\hat{Q}(s, a; w)$. This is achieved by having two separate networks, one is the online network being learned and the other being the target network. The weights of the target network are taken from the online network itself by freezing the model parameters for a few iterations and updating it periodically after a few steps. By freezing the parameters this way, it ensures that the target network parameters are significantly different from the online network parameters.
- **Experience Replay**: This is the other important technique used for stabilizing training. If we keep learning from experiences as they come, then we are basically observed a sequence of observations each of which are linked to each other. This destroys the assumption of the samples being independent. In ER, we maintain a Replay Buffer of fixed size (say N). We run a few episodes and store each of the experiences in the buffer. After a fixed number of iterations, we sample a few experiences from this replay buffer and use that to calculate the loss and eventually update the parameters. Sampling randomly this way breaks the sequential nature of experiences and stabilizes learning. It also helps us use an experience more than once.

Both of the above mentioned techniques were incorporated. The entire implementation was done in PyTorch.  Apart from th