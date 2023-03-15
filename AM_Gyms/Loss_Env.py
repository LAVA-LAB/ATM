import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class Measure_Loss_Env(gym.Env):
    """
    Measure Loss environment
    This simple environment describes an MDP with 3 states: one initial state s0,
    one 'positive' state s+ and one 'negative' state s-.
    From every state, taking action 1 (backward) returns to the initial state.
    From s0, taking action 0 has a chance p to change the state to s+,
    and a chance (1-p) to change to s-
    From s+, taking action 0 ends the run and gives reward r.
    From s-, taking action 0 also end the run but gives no reward.
    This environment is described in the report Merlijn Krale (link here!), 
    and is used to test Active-Measuring algorithms.
    """
    def __init__(self, p=0.8, r=1):

        self.p = p
        self.r = r

        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(4)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        reward = 0
        next_state = 0

        if action:  # 'backwards': go back to the beginning
            next_state = 0
        else:
            match self.state:
                case 0:
                    if np.random.rand() < self.p:   # s+
                        next_state = 1
                    else:
                        next_state = 2
                case 1:
                    next_state = 3
                    done = True
                    reward = self.r
                case 2:
                    next_state = 3
                    done = True
                case 3:
                    raise Exception("Invalid state: agent tried moving when done!")
        self.state = next_state
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

    def getname(self):
        return "Loss"