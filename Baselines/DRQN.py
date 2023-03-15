"""
File containing code for DRQN, a generic RL POMDP algorithm. Currently unused and untested, so use at own risk!
"""



from collections import deque
from multiprocessing.forkserver import MAXFDS_TO_SEND
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable
import sys
import pandas as pd
# from env_Tmaze import EnvTMaze
import numpy as np
import math
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AM_Gyms.AM_Env_wrapper import AM_ENV

"""
Generic DRQN with LSTM:
Start from a fixed patient state represented by 256.
One hot encode observations, including the missingness observation
and learn the LSTM policy mapping from observations.
"""

class ReplayMemory(object):
    def __init__(self, max_epi_num=50, max_epi_len=300):
        # capacity is the maximum number of episodes
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.memory = deque(maxlen=self.max_epi_num)
        self.is_av = False
        self.current_epi = 0
        self.memory.append([])

    def reset(self):
        self.current_epi = 0
        self.memory.clear()
        self.memory.append([])

    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def remember(self, state, action, reward):
        if len(self.memory[self.current_epi]) < self.max_epi_len:
            self.memory[self.current_epi].append([state, action, reward])

    # samples a trajectory of length 5
    def sample(self):
        epi_index = random.randint(0, len(self.memory)-2)
        if self.is_available():
            return self.memory[epi_index]
        else:
            return []

    def size(self):
        return len(self.memory)

    def is_available(self):
        self.is_av = True
        if len(self.memory) <= 1:
            self.is_av = False
        return self.is_av

    def print_info(self):
        for i in range(len(self.memory)):
            print('epi', i, 'length', len(self.memory[i]))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DRQN(nn.Module):
    def __init__(self, env:AM_ENV):
        super(DRQN, self).__init__()
        
        # Import environment variables
        self.env = env
        self.StateSize, self.CActionSize, self.MeasureCost, self.s_init = self.env.get_vars()
        self.ActionSize = 2*self.CActionSize
        
        self.lstm_i_dim = 10    # input dimension of LSTM
        self.lstm_h_dim = 10     # output dimension of LSTM
        self.lstm_N_layer = 10   # number of layers of LSTM
        self.convSize = int(self.StateSize/2)
        self.flat1 = nn.Linear(self.StateSize, self.lstm_h_dim)
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc1 = nn.Linear(self.lstm_h_dim, self.convSize)
        self.fc2 = nn.Linear(self.convSize, self.ActionSize)

    def forward(self, x, hidden):
        h2 = self.flat1(x)
        h2 = h2.unsqueeze(0)
        h3, new_hidden = self.lstm(h2, hidden)
        h4 = F.relu(self.fc1(h3))
        h5 = self.fc2(h4)
        return h5, new_hidden

class DRQN_Agent(object):
    def __init__(self, env:AM_ENV):
        self.env = env
        self.StateSize, self.CActionSize, self.MeasureCost, self.s_init = self.env.get_vars()
        self.ActionSize = self.CActionSize * 2
        
        self.EmptyObservation = -1
        
        self.drqn = DRQN(self.env)
        self.buffer = ReplayMemory() #Should be correcly set later!
        
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.drqn.parameters(), lr=1e-3)
        
        
        self.gamma = 1
        self.maxsteps = 500

    def remember(self, state, action, reward):
        self.buffer.remember(state, action, reward)
        
    def get_empty_obs(self):
        return np.zeros(self.StateSize)
    
    def get_hidden_empty(self):
        return (Variable(torch.zeros(self.drqn.lstm_N_layer, 1, self.drqn.lstm_h_dim).float()), Variable(torch.zeros(self.drqn.lstm_N_layer, 1, self.drqn.lstm_h_dim).float()))
    
    def stateNmbr_to_state(self, stateNmbr):
        state = np.zeros(self.StateSize)
        state[stateNmbr] += 1
        return state


    def train(self):
        if self.buffer.is_available():
            memo = self.buffer.sample()
            obs_list = []
            action_list = []
            reward_list = []
            Q_est = []
            Qs = []
            hidden = self.get_hidden_empty()
            for i in range(len(memo)):
                action_list.append(memo[i][1])
                reward_list.append(memo[i][2])
                obs = torch.Tensor(memo[i][0]).unsqueeze(0)
                Q, hidden = self.drqn.forward(obs, hidden)
                Qs.append(Q)
                Q_est.append(Q.clone())

            losses = []
            for t in range(len(memo) - 1):
                max_next_q = torch.max(Q_est[t+1]).clone().detach()
                q_target = Qs[t].clone().detach()
                q_target[0, 0, action_list[t]] = reward_list[t] + self.gamma * max_next_q
                losses.append(self.loss_fn(Qs[t], q_target))

            T = len(memo) - 1
            q_target = Qs[T].clone().detach()
            q_target[0, 0, action_list[T]] = reward_list[T]
            losses.append(self.loss_fn(Qs[T], q_target))
            loss = torch.stack(losses).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def get_action(self, obs, hidden, epsilon):
        obs = torch.Tensor([obs])
        if len(obs.shape) == 1: # batch sz: 1
            obs = obs.unsqueeze(0)
        if random.random() > epsilon:
            q, new_hidden = self.drqn.forward(obs, hidden)
            action = q[0].max(1)[1].data[0].item()
        else:
            q, new_hidden = self.drqn.forward(obs, hidden)
            action = random.randint(0, self.CActionSize-1)
        return action, new_hidden
    
    def run(self, nmbr_episodes, logging = True):
        rewards, steps, measurements = np.zeros(nmbr_episodes), np.zeros(nmbr_episodes), np.zeros(nmbr_episodes)
        self.buffer = ReplayMemory(nmbr_episodes, self.maxsteps)
        
        # Run episodes
        for i in range(nmbr_episodes):
            rewards[i], steps[i], measurements[i] =  self.run_episode(i)
        return rewards, steps, measurements
    
    
    def run_episode(self, episode):
        hidden = self.get_hidden_empty()
        returns = 0
        obs = self.stateNmbr_to_state(self.s_init)
        nmbr_measurements = 0
        self.env.reset()
        actions = []
        for step in range(self.maxsteps):
            # env.render()
            action, hidden = self.get_action(obs, hidden, get_decay(episode))
            actions.append(action)
            ac = action % self.CActionSize
            reward, done= self.env.step(ac)
            
            if action < self.CActionSize:
                nmbr_measurements += 1
                new_obs, cost = self.env.measure()
                new_obs = self.stateNmbr_to_state(new_obs)
                reward -= cost
            else:
                new_obs = self.get_empty_obs()
            
            
            _obs, _cost = self.env.measure()
            _obs = self.stateNmbr_to_state(_obs)
            if done or np.all(obs == _obs):
                reward -= 0.1
                pass
            obs = new_obs
            returns += reward #* self.gamma ** (step)
            self.remember(obs, action, reward)
            # if reward != 0 or MC_iter == max_MC_iter-1:
            if done: #or step >= self.maxsteps-1:
                self.buffer.create_new_epi()
                break
        print('Episode', episode, 'returns', returns, 'measurements: ', nmbr_measurements, "steps: ", step) #  'where', env.if_up)
        if self.buffer.is_available():
            self.train()
        return returns, step, nmbr_measurements
            
    def evaluate(self):
        """To Be Written!"""
        
        
        # STOP TRAINING and only use the last model to evaluate on 100 new trials
        return_list = []
        for eval_iter in range(50):
            self.env.reset()
            hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
            returns = 0
            for MC_iter in range(self.maxsteps):
                # env.render()
                action, hidden = self.get_action(obs, hidden, 0)
                obs, reward, done, info = self.env.step(action)
                observed = bool(action < 8)
                obs = one_hot_encoding(obs)
                print('Observed:', observed)
                print('Action: ', action)
                returns += reward * self.gamma ** (MC_iter)
                self.remember(obs, action, reward)
                # if reward != 0 or MC_iter == max_MC_iter-1:
                if done or MC_iter == self.maxsteps-1:
                    self.buffer.create_new_epi()
                    break
            print('Returns: ', returns)

            return_list.append(returns)
        print('Obs cost: ', self.MeasureCost)
        print('Eval mean return: ', np.mean(return_list))
        print('Eval ste return: ', np.std(return_list))
        print('Eval ste return: ', np.std(return_list)/ np.sqrt(50))

        # new is same as w/out "new" but just want to make sure the stats are consistent
        # np.save("exp_0523/drqn/new_2k_hidden_DRQN_sepsis_obs_cost_0.05_test.npy", np.array(train_curve))
        print("drqn runtime: NOT IMPLEMENTED! ", time.time() - 0) 










def get_decay(epi_iter):
    min_decay = 0.01
    return 0.2*max(min_decay, math.pow(0.99, epi_iter))


