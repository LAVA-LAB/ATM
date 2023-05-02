### Implementation of AMRL-Algorithm as described in https://arxiv.org/abs/2005.12697

import numpy as np

class AMRL_Agent:
    '''Creates a AMRL-Agent, as described in https://arxiv.org/abs/2005.12697'''

    def __init__(self,env, eta=0.1, m_bias = 0.1, turn_greedy=True, greedy_perc = 0.9):
        #load all environment-specific variables
        self.env = env
        self.StateSize, self.ActionSize, self.measureCost, self.s_init = env.get_vars()

        #load all algo-specific vars (if provided)
        self.eta, self.m_bias = eta, m_bias
        self.greedy_perc, self.turn_greedy = greedy_perc, turn_greedy
        self.be_greedy = False
        self.MeasureSize = 2
        self.init_Q = 0
        self.lr = 0.3
        self.df = 0.95

        # Create all episode and run-specific variables
        self.reset_Run_Variables()

    def reset_Run_Variables(self):
        # Variables for one run
        self.QTable = np.zeros( (self.StateSize,self.ActionSize,self.MeasureSize) )
        self.QTable[:,:,1] = self.m_bias
        self.QTable += self.init_Q
        self.QTriesTable = np.zeros( (self.StateSize, self.ActionSize, self.MeasureSize) )
        self.TransTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) ) + 1/self.StateSize
        self.TriesTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) ) 
        self.totalReward = 0
        # Variables for one epoch
        self.reset_Epoch_Vars()

    def reset_Epoch_Vars(self):
        self.currentReward = 0
        self.steps_taken = 0
        self.measurements_taken = 0
        self.env.reset()
        self.totalReward += self.currentReward
        #TODO: add variable to keep track of 'actual reward' without costs, and see how this gets effected


    def update_TransTable(self,s1, s2, action):
        '''Updates Transition Table by adding transition from s1 to s2 using action'''
        for i in range(self.StateSize):
            previousT, previousTries = self.TransTable[s1,action,i], self.TriesTable[s1,action,i]
            self.TriesTable[s1,action,i] = previousTries+1
            if i == s2:
                self.TransTable[s1,action,i] = (previousT*previousTries+1) / (previousTries+1)
            else:
                self.TransTable[s1,action,i] = (previousT*previousTries) / (previousTries+1)

    def update_QTable(self, s1, action, measure, s2, reward, done):
        '''Updates Q Table according to action and reward'''
        previousQ, previousTries = self.QTable[s1,action,measure], self.QTriesTable[s1,action,measure]
        if done:
            Q_s2 = 0
        else:
            Q_s2 = self.df * np.max(self.QTable[s2])
        self.QTriesTable[s1,action,measure] += 1
        
        if measure:
            self.QTable[s1,action,1] = (1-self.lr) * self.QTable[s1,action,1] + self.lr * (Q_s2 + reward - self.measureCost)
            self.QTable[s1,action,0] = (1-self.lr) * self.QTable[s1,action,0] + self.lr * (Q_s2 + reward)
        else:
            self.QTable[s1,action,1] = (1-self.lr) * self.QTable[s1,action,1] + self.lr * (Q_s2 + reward - self.measureCost)
            self.QTable[s1,action,0] = (1-self.lr) * self.QTable[s1,action,0] + self.lr * (Q_s2 + reward)
            
        # if measure:
        #     self.QTable[s1,action,0] = (previousQ*previousTries + Q_s2 + reward) / (previousTries+1)
        #     self.QTable[s1,action,1] = (previousQ*previousTries + Q_s2 + reward - 0.01) / (previousTries+1)
        # else:
        #     self.QTable[s1,action,0] = (previousQ*previousTries + Q_s2 + reward) / (previousTries+1)
        #     self.QTable[s1,action,1] = (previousQ*previousTries + Q_s2 + reward - 0.01) / (previousTries+1)
        #print("Current Q_table segment:"+str(self.QTable[s1,action,measure]))

    def guess_current_State(self,s,action):
        return (np.argmax(self.TransTable[s,action]))
    
    def find_optimal_actionPair(self,s):
        '''Returns optimal actionPair according to Q-table'''
        return (np.unravel_index(np.argmax(self.QTable[s]), self.QTable[s].shape))

    def find_nonOptimal_actionPair(self,s):
        '''Returns random actionPair'''
        return  ( np.random.randint(0,self.ActionSize), np.random.randint(0,self.MeasureSize) )


    def train_epoch(self):
        '''Training algorithm of AMRL as given in paper'''
        s_current = self.s_init
        done = False
        while not done:
            # Chose and take step:
            if np.random.random(1) < 1-self.eta or self.be_greedy :
                (action,measure) = self.find_optimal_actionPair(s_current) #Choose optimal action
            else:
                (action,measure) = self.find_nonOptimal_actionPair(s_current) #choose non-optimal action
            if self.s_init == -1:
                measure = 1

            # Update reward, Q-table and s_next
            if measure:
                (reward, done) = self.env.step(action)
                (obs, cost) = self.env.measure()
                self.update_TransTable(s_current,obs,action)
                self.measurements_taken += 1
                s_next = obs
            else:
                (reward, done) = self.env.step(action)
                s_next = self.guess_current_State(s_current, action)
            
            self.update_QTable(s_current,action,measure,s_next, reward, done)
            s_current = s_next
            self.currentReward += reward - self.measureCost*measure #this could be cleaner...
            self.steps_taken += 1
        if not done:
            print ("max nmbr of steps exceded!")

        # Reset after epoch, return reward and #steps
        self.totalReward += self.currentReward
        (rew, steps, ms) = self.currentReward, self.steps_taken, self.measurements_taken
        self.reset_Epoch_Vars()
        return (rew,steps,ms)

    def run(self, nmbr_epochs, get_intermediate_results=False):
        self.reset_Run_Variables()
        rewards, steps, ms = np.zeros((nmbr_epochs)), np.zeros((nmbr_epochs)), np.zeros((nmbr_epochs))
        for i in range(nmbr_epochs):
            rewards[i], steps[i], ms[i] = self.train_epoch()
            if self.turn_greedy and i/nmbr_epochs > self.greedy_perc:
                self.be_greedy = True
        print((self.TransTable, self.QTriesTable, self.QTable))    # Debug stuff
        if get_intermediate_results:
            return (self.totalReward, rewards, steps, ms)
        return self.totalReward