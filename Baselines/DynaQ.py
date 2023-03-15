"""
File containing a number of basic RL-algorithms for MDPs, reworked for the ACNO-MDP setting.
Currently unused and untested, so use at own risk!
"""

from AM_Gyms.ModelLearner import ModelLearner
from AM_Gyms.AM_Env_wrapper import AM_ENV
import numpy as np


class QBasic:
    """Class for standard Q-learning of AM-environments"""
    
    def __init__(self, ENV:AM_ENV):
        self.env = ENV
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = self.env.get_vars()
        
        self.T_counter = np.zeros((self.StateSize, self.ActionSize, self.StateSize))
        self.T = np.zeros((self.StateSize, self.ActionSize, self.StateSize)) + 1/self.StateSize
        self.Q = np.zeros((self.StateSize, self.ActionSize)) + 0.8
        
        self.lr = 0.3
        self.df = 0.95
        self.selfLoopPenalty = self.MeasureCost
        self.includeCost = False
        
        
    def update_Q(self,s,action,reward,obs):
        Psi = 0
        Psi += np.sum(self.T[s,action] * np.max(self.Q, axis=1))
        self.Q[s,action] = (1-self.lr) * self.Q[s,action] + self.lr * ( reward + self.df * Psi   )
    
    def update_T(self,s,action,obs):
        self.T_counter[s,action,obs] += 1
        self.T[s,action] = self.T_counter[s,action] / np.sum(self.T_counter[s,action])
        
    def pick_action(self,s):
        return np.argmax(self.Q[s])
    
    def run_step(self,s):
        action = self.pick_action(s)
        #print(s,action)

        (reward, done) = self.env.step(action, s)
        (obs, cost) = self.env.measure()
        
        if self.includeCost:
            reward -= cost
        
        self.update_Q(s, action, reward, obs)
        self.update_T(s, action, obs)
        return obs, reward, done
    
    def run_episode(self): 
        s = self.s_init
        done = False
        totalReward, steps = 0, 0
        self.env.reset()
        
        while not done:
            
            obs, reward, done = self.run_step(s)
            totalReward += reward
            steps += 1
            s = obs
        #print(self.Q)
        return totalReward, steps
        
    def run(self, episodes, logging = False):
        rewards, steps = np.zeros(episodes), np.zeros(episodes)
        for i in range(episodes):
            rewards[i], steps[i] = self.run_episode()
            if logging and i%100 == 0:
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {})".format( 
                        i, episodes, np.average(rewards[(i-100):i]), np.average(steps[(i-100):i]) ) )
        return np.sum(rewards), rewards, steps, np.ones(episodes)
    
class QOptimistic(QBasic):
    
    def __init__(self, ENV):
        super().__init__(ENV)
        self.Q_unbiased = np.zeros((self.StateSize, self.ActionSize)) + 0.8
        self.N_since_last_tried = np.ones((self.StateSize, self.ActionSize))
        self.optBias = 10**-600
        
    
    def update_Q(self, s, action, reward, obs):
        Psi = 0
        Psi += np.sum(self.T[s,action] * np.max(self.Q, axis=1))
        self.Q_unbiased[s,action] = (1-self.lr) * self.Q[s,action] + self.lr * ( reward + self.df * Psi )
        
        self.Q[s,action] = self.Q_unbiased[s,action] #+ self.optBias*np.sqrt(self.N_since_last_tried)
        self.N_since_last_tried += 1
        self.N_since_last_tried[s,action] = 1 
        self.Q = self.Q + self.optBias * ( np.sqrt(self.N_since_last_tried-1) + np.sqrt(self.N_since_last_tried)  )
    
class QDyna(QBasic):
    
    def __init__(self, ENV: AM_ENV):
        super().__init__(ENV)
        self.R_counter = np.zeros((self.StateSize, self.ActionSize))
        self.trainingSteps = 10
        
    def update_R(self,s,action,reward):
        self.R_counter[s,action] += reward
        
    def update_Q(self,s,action,reward,obs, isReal=True):
        super().update_Q(s,action,reward,obs)
        if isReal:
            self.update_R(s,action,reward)
    
    def run_step(self, s):
        obs,reward, done = super().run_step(s)
        for i in range(self.trainingSteps):
            s = np.random.randint(self.StateSize)
            action = self.pick_action(s)
            if np.sum(self.T_counter[s,action]) > 1:
                self.simulate_step(s,action)
        return obs, reward, done
    
    def simulate_step(self,s,action):
        snext = np.random.choice(self.StateSize, p=self.T[s,action])
        r = self.R_counter[s,action]/np.sum(self.T_counter[s,action])
        #r -= self.MeasureCost
        self.update_Q(s,action,r,snext, isReal=False) #NO COST!!!