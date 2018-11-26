"""
DDPG agent.
"""

import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, model, action_size=4, seed=0, load_file=None,
                 n_agents=20,
                 buffer_size=int(1e5),
                 batch_size=128,
                 gamma=0.99,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 weight_decay=0.0001,
                 update_every=2,
                 use_prioritized_experience_replay=False,
                 alpha_start=0.5,
                 alpha_decay=0.9992):
        """
        Params
        ======
            model: model object
            action_size (int): dimension of each action
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            n_agents (int): number of agents to train simultaneously
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
            update_every (int): how often to update the network
            use_prioritized_experience_replay (bool): wheter to use PER algorithm
            alpha_start (float): initial value for alpha, used in PER
            alpha_decay (float): decay rate for alpha, used in PER
        """
        random.seed(seed)

        self.action_size = action_size
        self.n_agents = n_agents
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.update_every = update_every
        self.use_prioritized_experience_replay = use_prioritized_experience_replay

        self.loss_list = []       # track loss across steps

        # Actor Network
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        if load_file:
            self.actor_local.load_state_dict(torch.load(load_file + '.actor.pth'))
            self.actor_target.load_state_dict(torch.load(load_file + '.actor.pth'))
            self.critic_local.load_state_dict(torch.load(load_file + '.critic.pth'))
            self.critic_target.load_state_dict(torch.load(load_file + '.critic.pth'))
            print('Loaded: {}'.format(load_file))

        # Noise process
        self.noise = OUNoise((n_agents, action_size), seed)

        # Replay memory
        if use_prioritized_experience_replay:
            self.memory = PrioritizedReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        else:
            self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        # initalize alpha (used in prioritized experience sampling probability)
        self.alpha_start = alpha_start
        self.alpha_decay = alpha_decay
        self.alpha = self.alpha_start


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.use_prioritized_experience_replay:
            priority = 100.0   # set initial priority to max value
            for i in range(self.n_agents):
                self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i], priority[i, :])
        else:
            for i in range(self.n_agents):
                self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i])

        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # if prioritized experience replay is enabled
                if self.use_prioritized_experience_replay:
                    self.memory.sort()
                    indexes, experiences = self.memory.sample(self.alpha)
                    self.learn(experiences, self.gamma, indexes)
                    self.alpha = self.alpha_decay*self.alpha
                else:
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma)


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # calculate action values
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()


    def learn(self, experiences, gamma, indexes=None):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self.use_prioritized_experience_replay:
            states, actions, rewards, next_states, dones, priorities = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, actions_next)
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        # compute critic loss
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        critic_loss = F.mse_loss(q_expected, q_targets)
        # minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # ----------- optionally update prioritized experience replay ---------- #
        if self.use_prioritized_experience_replay:
            with torch.no_grad():
                new_priorities = torch.abs(q_targets - q_expected).to(device)
                self.memory.batch_update(indexes, (states, actions, rewards, next_states, dones, new_priorities))

        # ---------------------------- update stats ---------------------------- #
        with torch.no_grad():
            self.loss_list.append(critic_loss.item())


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, action_size, buffer_size, batch_size, seed):
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])

    def batch_update(self, indexes, experiences):
        """ Batch update existing elements in memory. """
        states, actions, rewards, next_states, dones, new_priorities = experiences
        for i in range(self.batch_size):
            e = self.experience(states[i], int(actions[i]), float(rewards[i]), next_states[i], bool(dones[i]), float(new_priorities[i]))
            self.memory[indexes[i]] = e

    def sort(self):
        """ Sort memory based on priority (TD error) """
        # sort memory based on priority (sixth item in experience tuple)
        items = [self.memory.pop() for i in range(len(self.memory))]
        items.sort(key=lambda x: x[5], reverse=True)
        self.memory.extend(items)

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self, alpha):
        """ Sample a batch of experiences from memory using Prioritized Experience. """
        # get the number of items in the experience replay memory
        n_items = len(self.memory)
        # calculate the sum of all the probabilities of all the items
        sum_of_probs = sum((1/i) ** alpha for i in range(1, n_items + 1))
        # build a probability list for all the items
        probs = [(1/i) ** alpha / sum_of_probs for i in range(1, n_items + 1)]
        # sample from the replay memory using the probability list
        indexes = np.random.choice(n_items, self.batch_size, p=probs)
        # use the indexes to generate a list of experience tuples
        experiences = [self.memory[i] for i in indexes]

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)

        return indexes, (states, actions, rewards, next_states, dones, priorities)
