"""
UnityML Environment.
"""


from unityagents import UnityEnvironment

class UnityMLVectorMultiAgent():
    """Multi-agent UnityML environment with vector observations."""

    def __init__(self, name, seed=0):
        self.seed = seed
        print('SEED: {}'.format(self.seed))
        self.env = UnityEnvironment(file_name=name, seed=seed)
        self.brain_name = self.env.brain_names[0]

    def reset(self):
        """Reset the environment."""
        info = self.env.reset(train_mode=True)[self.brain_name]
        state = info.vector_observations
        return state

    def step(self, action):
        """Take a step in the environment."""
        info = self.env.step(action)[self.brain_name]
        state = info.vector_observations
        reward = info.rewards
        done = info.local_done
        return state, reward, done
