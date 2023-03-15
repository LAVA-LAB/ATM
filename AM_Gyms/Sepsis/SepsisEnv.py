import numpy as np, random
from AM_Gyms.Sepsis.MDP import MDP
from AM_Gyms.Sepsis.State import State
from AM_Gyms.Sepsis.Action import Action
from AM_Gyms.Sepsis.sepsis_tabular import SepsisEnv

from gym import spaces
import gym



class SepsisEnv_Or(gym.Env):
    def __init__(self,
                obs_cost=0, # turn this off (since obs cost added from examples/sepsis)
                noise=False,
                per_step_reward=False,
                counter=False,
                locf=False,
                no_missingness=False,
                action_aug=False,
                ):
        self.timestep = 0
        self.max_t = 5
        self.env = None
        self.locf = locf
        self.cost = obs_cost
        self.counter = counter
        self.viewer = None
        actions_n = 8
        if not no_missingness:
            actions_n *= 2
        self.action_space = spaces.Discrete(actions_n)
        high = np.array([3, 3, 2, 5, 1, 1, 1])
        low = np.array([0, 0, 0, 0, 0, 0, 0])
        self.observation_space = spaces.Box(low, high)
        
        self.state = None
        self.obs = None

    def separate_action(self, action):
        obs = False
        if action < 8:
            obs = True
        return (action % 8), obs

    def step(self, action):
        a, obs = self.separate_action(action)
        self.timestep += 1
        reward = self.env.transition(Action(action_idx=a))
        state = self.env.state.get_state_vector()
        # Add +1 to every vital value since 0 is used for NULL
        # Is this required? AMRL just needs a state int, so I'll just ignore this and ask for that instead...
        state[:4] += 1
        done = bool(self.timestep == self.max_t or reward != 0)
        if self.counter:
            self.obs[-1] += 1
        else:
            self.obs = np.concatenate((np.zeros(4,), state[-3:]))
        state_idx = self.env.state.get_state_idx()
        return state_idx, reward, done, {}

    def reset(self, init_idx = 256):
        self.timestep = 0
        self.env = MDP(init_state_idx=init_idx, 
                        init_state_idx_type='obs', 
                        p_diabetes=0.)
        state = self.env.state.get_state_vector()
        state_idx = self.env.state.get_state_idx()
        #print(state_idx)
        # Add +1 to every vital value since 0 is used for NULL
        state[:4] += 1
        self.state = state.copy()
        self.obs = state.copy()
        if self.counter:
            self.obs = np.concatenate((self.obs, [0]))
        return np.array(self.obs)

        def render(self, mode='human'):
            pass

    def __obs_to_state__(self):
        return 