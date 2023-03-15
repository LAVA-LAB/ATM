import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class GenericGym(gym.Env):
    
    def __init__(self, P, R, s_init, has_terminal_state=True, terminal_prob=0):
        
        self.P      = P
        self.R      = R
        self.s_init = s_init
        self.state  = s_init
        self.terminal_prob = terminal_prob
        
        dimension = np.shape(P)
        self.StateSize = dimension[0]
        self.ActionSize = dimension[1]
        
        self.action_space = spaces.Discrete(self.ActionSize)
        self.observation_space = spaces.Discrete(self.StateSize)
        
        assert (np.shape(R) == (self.StateSize, self.ActionSize))
        
        # We assume the last state is the terminal state
        self.has_terminal_state = has_terminal_state
        
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        assert self.action_space.contains(action)
        
        reward = self.R[self.state, action]
        self.state = np.random.choice(self.StateSize, p=self.P[self.state, action])
        
        done = (self.has_terminal_state and self.state == self.StateSize-1)
        done = done or self.terminal_prob > np.random.random()

        return self.state, reward, done, {}
        
    def reset(self):
        self.state = 0
        return self.state