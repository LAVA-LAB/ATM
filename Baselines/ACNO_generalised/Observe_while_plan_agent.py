"""
Implementation of the Observe-while-Planning algorithm described by Nam et al (2021).
I gave up on using this method, so use at your own risk!
"""



from __future__ import print_function, division
import time
import logging
import os
from pomdpy.pomdp import Statistic
from pomdpy.pomdp.history import Histories, HistoryEntry
from pomdpy.util import console, print_divider
from pomdpy.solvers.pomcp import POMCP
# from experiments.scripts.pickle_wrapper import save_pkl
import numpy as np

module = "agent"

DIR = './exp/'
SAVE_K = '0'
if not os.path.exists(DIR + SAVE_K):
    os.makedirs(DIR + SAVE_K)

_eval = False

class ACNO_Agent_OWP:
    """
    WARNING: NOT FULLY TESTED
    Agent for running the ACNO-POMCP algorithm. 
    Edited from https://github.com/nam630/acno_mdp to work for any openAI environment.

    """

    def __init__(self, model, is_mdp=False):
        """
        Initialize the POMDPY agent
        :param model:
        :param solver:
        :return:
        """
        if is_mdp == 1:
            self.solver_type = 'MDP'
        else:
            self.solver_type = 'POMDP'
        self.logger = logging.getLogger('POMDPy.Solver')
        self.model = model
        self.need_init = True
        self.cost = model.cost
        self.results = Results()
        self.experiment_results = Results()
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        self.solver = POMCP(self)
        self.solver_factory = self.solver.reset  # Factory method for generating instances of the solver

    def multi_epoch(self):
        '''Runs the ACNO-MDP algorithm for a set number of epochs'''
        eps = self.model.epsilon_start
        epochs = self.model.n_epochs
        rewards, steps, measurements = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)
        temp = None
        self.model.reset_for_epoch()
        self.model.sample_model_empty()
        for i in range(epochs):
            # Reset the epoch stats3
            start = time.time()
            self.results = Results()
            if self.model.solver == 'POMCP' or self.model.solver == 'MCP':
                rewards[i], steps[i], measurements[i], eps, reward, temp = self.run_pomcp(i + 1, eps, temp)
                self.model.reset_for_epoch()
            #print('Runtime: ', time.time() - start)
            if self.experiment_results.time.running_total > self.model.timeout:
                #print("TIMEOUT")
                break
            if i % 100 == 0:
                np.save(open(DIR+'{}/epoch_{}_rewards.npy'.format(SAVE_K, i), 'wb'), np.array(rewards))
        rewards = np.array(rewards)
        np.save(DIR+'pomcp_{}.npy'.format(SAVE_K), rewards)
        return rewards, steps, measurements 
    
    def run(self, nmbr_eps, get_full_results=False):
        '''Renaming of multi_epoch to be consistent with other algorithms.'''
        self.model.n_epochs = nmbr_eps
        rs, ss, ms = self.multi_epoch()
        totalReward = np.sum(rs)
        if get_full_results:
            return (totalReward, rs, ss, ms)
        return totalReward
        

    """
    Observe then plan uses the pre-learned transition and reward estimates (set observe_then_plan to True).
    ACNO-POMCP (observe while planning) updates the model estimates after every consecutive observed pair // otherwise both use same MC rollouts.
    """
    def run_pomcp(self, epoch, eps, temp=None, observe_then_plan=False):
        '''Runs the ACNO-POMCP algorithm untill the goal or the maximum number of steps is reached'''
        epoch_start = time.time()
        
        # Periodic logging:
        if not observe_then_plan and (epoch % 100 == 0):
            np.save(open(DIR+'{}/epoch{}_T.npy'.format(SAVE_K, epoch), 'wb'), old_t)
            np.save(open(DIR+'{}/epoch{}_N.npy'.format(SAVE_K, epoch), 'wb'), self.model.n_counts)
            np.save(open(DIR+'{}/epoch{}_R.npy'.format(SAVE_K, epoch), 'wb'), old_r)
        
        # Initialise a new solver
        if epoch == 1:
            solver = self.solver_factory(self)
            temp = solver.fast_UCB.copy()
            self.need_init = False
        else:
            assert(temp is not None)
            solver = self.solver_factory(self)
            solver.fast_UCB = temp
        
        
        # Monte-Carlo start state
        self.model.reset_for_epoch()
        state = solver.belief_tree_index.sample_particle()
        
        #initialise epoch variables
        nmbr_measurements = 0
        discounted_reward = 0
        reward = 0
        discount = 1.0 # starts with 1, drops by 0.7
        past_obs = state.position # always gives true state
        for i in range(self.model.max_steps):
            start_time = time.time()

            # Decide greedy step, according to solver
            action = solver.select_eps_greedy_action(eps, start_time, greedy_select=(not _eval))
            
            if action.bin_number > self.model.CActionSize and np.random.rand() < eps:
                action.bin_number -= self.model.CActionSize
            
            
            # Take step & update local variables
            # Note: step results only used internally!
            thisReward, obs, done = self.model.take_real_step(action.bin_number)
            start_time = time.time()
            
            # If previous state known and measuring:
            if action.bin_number < self.model.CActionSize:
                nmbr_measurements += 1
                if past_obs != self.model.states_n : 
                    # Update model counters
                    #print("Learning!")
                   # print((past_obs, action.bin_number % self.model.c_actions, step_result.observation.position))
                    self.model.update_model(past_obs,action.bin_number, obs, thisReward)
            
            # Update variables
            discounted_reward += discount * (thisReward)
            reward += thisReward
            past_obs = obs
            start_time = time.time()
            discount *= self.model.discount # model discount = 0.7
            
            # Update epsilon every episode
            if eps > self.model.epsilon_minimum:
                eps *= self.model.epsilon_decay
            
            # Update solver & history
            # if not step_result.is_terminal or not is_legal:
            #     solver.update(step_result)
            print(action.bin_number, obs, thisReward)
            
            step_result = self.model.to_StepResult(action.bin_number, obs, obs, thisReward, done)
            if not done:
                solver.update(step_result)
            
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, thisReward, action, obs, obs)
            
            # Printing & Logging:            
            # print("true state in epoch:", state.position)
            # self.display_step_result(i, step_result)
            # print(step_result.is_terminal)
            if done:
            #     console(3, module, 'Terminated after episode step ' + str(i + 1))
                break
        
        # Log variables
        self.results.time.add(time.time() - epoch_start)
        self.results.update_reward_results(reward, discounted_reward)
        
        self.experiment_results.time.add(self.results.time.running_total)
        self.experiment_results.undiscounted_return.count += (self.results.undiscounted_return.count - 1)
        self.experiment_results.undiscounted_return.add(self.results.undiscounted_return.running_total)
        self.experiment_results.discounted_return.count += (self.results.discounted_return.count - 1)
        self.experiment_results.discounted_return.add(self.results.discounted_return.running_total)

        # Print results
        # Pretty Print results
        # print_divider('large')
        # solver.history.show()
        # self.results.show(epoch)
        print ("{} / {} runs complete (current reward = {}, nmbr steps = {}, nmbr measurements = {})".format( epoch, self.model.n_epochs, reward, i+1, nmbr_measurements ))
        # console(3, module, 'Total possible undiscounted return: ' + str(self.model.get_max_undiscounted_return()))
        # print_divider('medium')
        # print(discounted_reward)
        # print(i)
        #print(self.model.r_counts, self.model.n_counts, self.model.r_estimates)

        return reward, i, nmbr_measurements, eps, discounted_reward, temp

    @staticmethod
    def display_step_result(step_num, step_result):
        """
        Pretty prints step result information
        :param step_num:
        :param step_result:
        :return:
        """
        console(3, module, 'Step Number = ' + str(step_num))
        console(3, module, 'Step Result.Action = ' + step_result.action.to_string())
        console(3, module, 'Step Result.Observation = ' + step_result.observation.to_string())
        # console(3, module, 'Step Result.Next_State = ' + step_result.next_state.to_string())
        console(3, module, 'Step Result.Reward = ' + str(step_result.reward))


class Results(object):
    """
    Maintain the statistics for each run
    """
    def __init__(self):
        self.time = Statistic('Time')
        self.discounted_return = Statistic('discounted return')
        self.undiscounted_return = Statistic('undiscounted return')

    def update_reward_results(self, r, dr):
        self.undiscounted_return.add(r)
        self.discounted_return.add(dr)

    def reset_running_totals(self):
        self.time.running_total = 0.0
        self.discounted_return.running_total = 0.0
        self.undiscounted_return.running_total = 0.0

    def show(self, epoch):
        print_divider('large')
        print('\tEpoch #' + str(epoch) + ' RESULTS')
        print_divider('large')
        console(2, module, 'discounted return statistics')
        print_divider('medium')
        self.discounted_return.show()
        print_divider('medium')
        console(2, module, 'undiscounted return statistics')
        print_divider('medium')
        self.undiscounted_return.show()
        print_divider('medium')
        console(2, module, 'Time')
        print_divider('medium')
        self.time.show()
        print_divider('medium')