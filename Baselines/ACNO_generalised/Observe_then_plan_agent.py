"""
File containing the Observe-then-Plan agent described by Nam et al (2021).
Code is based on the original code used in their paper (https://github.com/nam630/acno_mdp),
But adepted to work for general ACNO-MDP settings, and to conform to the naming conventions
used in other baseline code.
"""

import numpy as np
import math as m
import time
from pomdpy.solvers.pomcp import POMCP
from pomdpy.pomdp.history import Histories, HistoryEntry

from ACNO_generalised.ACNO_ENV import ACNO_ENV


class ACNO_Agent_OTP:
    """Agent running the Observe-then-Plan algorithm as described by Nam et al (2021)"""
    
    def __init__(self, env=ACNO_ENV):
        self.model = env
        
        self.explore_episodes = 2000
        self.max_steps = 1000
        
        # Variables needed for POMCP
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        
        # The POMCP-solver
        self.solver = POMCP(self)
        self.solver_factory = POMCP
         
        
    
    def run(self, nmbr_episodes, get_full_results=True):
        
        # Declare variables
        rewards, steps, measurements = np.zeros(nmbr_episodes), np.zeros(nmbr_episodes), np.zeros(nmbr_episodes)
        
        # Run exploration phase
        self.explore_episodes = nmbr_episodes-50
        exp_eps = self.explore_episodes #readibilty re-define
        rewards[:exp_eps], steps[:exp_eps] = self.model.sample_model(self.explore_episodes)
        measurements[:exp_eps] = steps[:exp_eps]
        
        # Run episodes
        for i in range(self.explore_episodes, nmbr_episodes):
            rewards[i], steps[i], measurements[i] =  self.run_episode(i)
        return np.sum(rewards), rewards, steps, measurements
    
    
    def run_episode(self, epoch):
        
        # Remake solver & reset env
        self.histories = Histories() 
        self.model.reset_for_epoch()
        solver = self.solver_factory(self)
        
        # Initialise episode variables
        steps, measurements, totalreward =0, 0, 0

        done = False
        while not done and steps<self.max_steps:
            start_time = time.time()
            # Get optimal action from solver
            action = solver.select_eps_greedy_action(0, start_time, greedy_select=False).bin_number
            
            # Perform action
            (reward, obs, done) = self.model.take_real_step(action, False)
            print(action, obs, reward)
            
            # Update solver & history
            stepResult = self.model.to_StepResult(action, obs, obs, reward, done)
            if not done:
                solver.update(stepResult)
            
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, reward, action, obs, obs)
            
            # Update logging variables
            
            steps += 1
            totalreward += reward
            if action < self.model.CActionSize:
                measurements += 1
        
        # Message after each episode
        print ("{} episodes complete (current reward = {}, nmbr steps = {}, nmbr measurements = {})"
               .format( epoch,  totalreward, steps+1, measurements ))
        
        return totalreward, steps, measurements