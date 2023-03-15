'''
    BAM-QMDP, but bayesian!: 

This file contains an agent for BAM-QMDP, an algorithm to find policies through reinforcement for ACNO-MDP environments.
(In short, these are POMDP-settings where observations are either complete when taking a measurement as some cost, and {} otherwise)
A full and formal description can be found in the accompanying paper. 
'''

from csv import QUOTE_ALL
from functools import total_ordering
import numpy as np
import math as m
import time

from AM_Gyms.AM_Env_wrapper import AM_ENV


class BAM_QMDP:
    "Implementation of BAM-QMPD agent as a python class."

    #######################################################
    ###     INITIALISATION AND DEFINING VARIABLES:      ###
    #######################################################

    def __init__(self, env:AM_ENV, eta = 0.00, nmbr_particles = 100,  offline_training_steps = 0):
        # Environment arguments:
        self.env = env
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()
        
        self.StateSize = self.StateSize + 1         # Adding a Done-state
        self.doneState = self.StateSize -1

        # Meta-variables:
        self.eta = eta                              # Chance of picking a non-greedy action (should be called epsilon...)
        self.nmbr_particles = nmbr_particles        # Number of particles used to represent the belief state.
        self.NmbrOptimiticTries = 20                # Meta variable determining for how many tries a transition should be biased.
        self.selfLoopPenalty = 1                 # Penalty applied to Q-value for self loops (1 means no penalty)
        self.lossBoost = 1                          # Testing variable to boost the effect of Measurement Loss/Regret (1 means no boost)
        self.stopPenalty = 0.0                      # Penalty aplied to Q-values achieved in the last step (0 means no penalty)
        self.max_estimated_loss = self.MeasureCost  # Minimum Measurement Regret for which a measurement is taken (currently equal to measurement cost)
        self.optimisticPenalty = 1                  # Maximum return estimate (Rewards in all environments are normalised such that this is always 1)
        
        self.otsteps = offline_training_steps
        self.offline_eta = 0.5
        self.eta_measure = 0.0
        self.max_steps_without_measuring = self.StateSize 

        self.initPrior = 1/self.StateSize           # Initial alpha-values, as used in the prior for T
        self.optimism_type = "RMAX+"                # RMAX+ (old), UCB, RMAX
        self.UCB_Cp = 0.1
        self.P_noMeasureRate = 0.0
        self.Q_noMeasureRate = 1.0
        self.use_exp = True
        
        self.dynamicLR = False
        self.lr = 0.1                               # Learning rate, as used in standard Q updates. Currently unused, since we use a dynamic learning rate
        self.df = 0.8                              # Discount Factor, as used in Q updates
        
        self.init_run_variables()

    def init_run_variables(self):
        "Initialises all variables that should be reset each run."
        # Arrays keeping track of model:
        
        # Value Estimation Tables
        self.QTable             = np.ones ( (self.StateSize, self.ActionSize), dtype=np.longfloat )* self.optimisticPenalty    # Q-table as used by other functions, includes initial bias

        self.QTableUnbiased     = np.zeros( (self.StateSize, self.ActionSize), dtype=np.longfloat )                             # Q-table based solely on experience (unbiased)
                                                                                         
        self.QTableRewards      = np.zeros( (self.StateSize, self.ActionSize) )                                                 # Record average immidiate reward for (s,a) (called \hat{R} in report)
        self.Qmax               = np.zeros( (self.StateSize), dtype=np.longfloat)                                               # Q-value of optimal action as given by Q (used for readability)
        self.QMaxUnbiased       = np.zeros( (self.StateSize), dtype=np.longfloat)
        self.QCounter           = np.zeros( (self.StateSize, self.ActionSize ))
        
        self.QTableUnbiased[self.doneState] = 0
        self.QTable[self.doneState]         = 0
        
        
        self.alpha              = np.ones ( (self.StateSize, self.ActionSize, self.StateSize) ) * self.initPrior                 # Alpha-values used for dirichlet-distributions.
        self.alpha_sum          = np.ones ( (self.StateSize, self.ActionSize) ) * (self.initPrior * self.StateSize)
        self.ChangedStates      = {}
        self.T                  = np.zeros((self.StateSize, self.ActionSize, self.StateSize), dtype=np.longfloat) # States to be checked in global Q update
        # Other vars:
        self.totalReward        = 0     # reward over all episodes
        self.totalSteps         = 0     # steps over all episodes
        self.init_episode_variables()

    def init_episode_variables(self):
        "Initialises all episode-specific variables"
        # Logging variables
        self.episodeReward          = 0 # episodic reward
        self.steps_taken            = 0 # steps taken this episode
        self.measurements_taken     = 0 # measurements taken this episode

        # State variables        
        self.is_done                = False

        self.env.reset()
    
    #######################################################
    ###                 RUN FUNCTIONS:                  ###
    #######################################################    

    def run_episode(self):
        "Performes one episode of BAM-QMPD algorithm."
        # Initialise all variables:
        self.init_episode_variables()

        # Initialise state, history, and previous state vars
        s = {}
        if self.s_init == -1: # Random start
            for i in range(self.StateSize):
                s[i] = 1.0/self.StateSize
        else:
            s[self.s_init] = 1
        next_action_known = False
 
        ### MAIN LOOP ###
        while not self.is_done:
            
            # 1: Find optimal action:
            if next_action_known:
                action = next_action
            else:
                action = self.get_action(s)
            
            #2: Compute next belief state:
            b_next = self.guess_next_state(s, action)
            
            #3: Decide whether or not to measure:
            next_action = self.get_action(b_next)
            next_action_known = True
            
            Loss = self.get_loss(b_next,next_action)
            TransSupport = self.get_support(s,action)
            
            measure = (Loss > self.MeasureCost) or (TransSupport < self.NmbrOptimiticTries) or self.steps_taken > self.max_steps_without_measuring
            
            #4: Take Action:
            if np.random.rand() < self.eta:
                action = m.floor(np.random.randint(self.ActionSize))
                measure = True
            (reward, self.is_done) = self.env.step(action)
            cost = 0
            
            #5: Measure
            if measure:
                s_next, cost = self.env.measure()
                
                next_action_known = False
                self.measurements_taken += 1
            
            #6: Update b_next
                b_next.clear()
                b_next[s_next] = 1
                
            #7: Update P:
                self.update_T(s,b_next,action, self.is_done)
                
            #8: Update Q

            self.update_Q_lastStep_only(s,b_next,action,reward, isDone=self.is_done)
                
            if self.otsteps > 0:
                for i in range(self.otsteps):
                    self.train_offline()
            
            #9: Update variables for next step:
            #print(s,action,measure, b_next)
            s = self.check_validity_belief(b_next)
            self.episodeReward  += reward - cost
            self.steps_taken    += 1
            self.totalSteps     += 1
            
        ### END LOOP ###
        self.totalReward += self.episodeReward
        returnVars = (self.episodeReward, self.steps_taken, self.measurements_taken)
        return returnVars  
            

    def run(self, nmbr_episodes, get_full_results=False, print_info = False, logmessages = True):
        "Performs the specified number of episodes of BAM-QMDP."
        self.init_run_variables()
        epreward,epsteps,epms = np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes))
        for i in range(nmbr_episodes):
            log_nmbr = 100
            if (i > 0 and i%log_nmbr == 0 and logmessages):
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {}, nmbr measures = {})".format( 
                        i, nmbr_episodes, np.average(epreward[(i-log_nmbr):i]), np.average(epsteps[(i-log_nmbr):i]), np.average(epms[(i-log_nmbr):i]) ) )
                
                    
            epreward[i], epsteps[i], epms[i]  = self.run_episode()
                
        if print_info:
            print("""
Run complete: 
Alpha table: {}
QTable: {}
Rewards Table: {}
Unbiased QTable: {}            
            """.format(self.alpha,  self.QTable, self.QTableRewards, self.QTableUnbiased))
        if get_full_results:
            return(self.totalReward, epreward,epsteps,epms)
        return self.totalReward

    #######################################################
    ###                 HELPER FUNCTIONS:               ###
    #######################################################

    def get_action(self,S):
        "Obtains the most greedy action according to current belief and model."

        #Compute optimal action
        thisQ = np.zeros(self.ActionSize) # weighted value of each action, according to current belief
        for s in S:
            p = S[s]
            thisQ += self.QTable[s]*p
        thisQMax = np.max(thisQ)
        return int(np.random.choice(np.where(np.isclose(thisQ,thisQMax))[0])) #randomize tiebreaks
        
    def get_loss(self, S, action):
        "Returns measure regret of taking given action in given belief state"
        Loss = 0
        for s in S:
            p = S[s]
            QTable_max = np.max(self.QTable[s]) # expected return if we were in state s
            Loss +=  p * max( 0.0, QTable_max - self.QTableUnbiased[s,action] )
        return Loss
    
    def get_support(self, S, action):
        "Compute transition support of current belief-action pair"

        # If belief state becomes too big, always measure
        if len(S) > 0.5*self.nmbr_particles:
            return 0
        
        elif len(S) > 1:
            return self.NmbrOptimiticTries
        
        # Calculate support:
        support = 0
        for s in S:
            support += S[s]*(np.sum(self.alpha[s,action] ) - self.StateSize*self.initPrior )
        return support

    def check_validity_belief(self, b:dict):
        if self.doneState in b:
            p_done = b[self.doneState]
            b.pop(self.doneState)
            scaling_factor = 1 / (1-p_done)
            for s in b:
                b[s] = b[s] * scaling_factor
        return b
    
    
    def _dict_to_arrays_(self, S):
        "Returns array of states and probabilities for belief state"
        array = np.array(list(S.items()))
        return array[:,0], array[:,1] #states, then probs
       
    def _dict_to_particles_(self, S):
        "Returns an array of particles for given belief state"
        states, probs = self._dict_to_arrays_(S)
        particles = []
        for (i,s) in enumerate(states):
            for j in range(round(probs[i]*self.nmbr_particles)):
                particles.append(int(s))
        if len(particles) != self.nmbr_particles:
            print(states, probs, particles, len(particles))
            print("Problem in dict_to_particles!")
        return particles
    
    def sample_T(self,s, action, nmbr=1):
        "Returns a (sampled) transition function according to current dirichlet distribution"
        if self.use_exp:
            if self.alpha_sum[s,action] > 8:
                return (self.alpha[s,action] - self.initPrior) / (self.alpha_sum[s,action] - ( self.StateSize * self.initPrior) )
            return (self.alpha[s,action]) / self.alpha_sum[s,action]
        if nmbr != 1:
            print("Average samples not implemented yet!")
        return self.dirichlet_approx(alpha = self.alpha[s,action], nmbr_samples = 1)
        
        
    def guess_next_state(self, S, action):
        "Samples a next belief state after action a, according to current belief and model."

        # Sample probability distribution for next state 
        probDist = np.zeros(self.StateSize)
        for s in S:
            probDist += S[s] * self.sample_T(s,action)
        
        # Filter all zero-probability states for efficiency
        states = np.arange(self.StateSize)
        filter = np.nonzero(probDist)
        probDist, states = probDist[filter], states[filter]

        # Fix normalisation problems
        if np.sum(probDist) == 0:
            probDist, states = np.ones(self.StateSize)/self.StateSize, np.arange(self.StateSize)
        elif np.sum(probDist) < 1:
            probDist = probDist/np.sum(probDist)
        
        # Sample next states
        SnextArray = np.random.choice(states, size=self.nmbr_particles, p=probDist)

        # Combine states into a probability distr.
        S_next = {}
        states, counts = np.unique(SnextArray, return_counts=True)
        for i in range(len(states)):
            s = states[i]
            S_next[int(s)] = counts[i] * 1/self.nmbr_particles
        return S_next
    
    

    #######################################################
    ###                 MODEL UPDATING:                 ###
    #######################################################

   
    def update_T(self, S1, S2, action, isDone = False):
        'Updates probability function P according to transition (S1, a, S2)'
        
        for s1 in S1:
            p1 = S1[s1]
            
            # If done, we can update alphas regardless of whether we measured.
            if isDone:
                self.alpha[s1,action,self.doneState] += p1
                self.alpha_sum[s1,action] += p1
            
            # Otherwise, update alpha normally when measuring
            elif len(S2) == 1 and len(S1) == 1:
                for s2 in S2:
                    self.alpha[s1,action,s2] += p1
                    self.alpha_sum[s1,action] += p1
            
            # If not measuring
            else:
                pass
                                     
    def update_Q_lastStep_only(self,S1, S2, action, reward, isDone = False, isReal = True):
        'Updates Q-table according to transition (S1, a, S2)'

        for s1 in S1:
            # Unpack some variables
            p1 = S1[s1]
            if p1 < 1:
                    p1 = p1 * self.Q_noMeasureRate
            thisQ = 0
            thisQUnbiased = 0
            
            if not (s1 == self.doneState):
                if not isDone:
                    if len(S2) > 1 :
                        dict_this_s = dict()
                        dict_this_s[s1] = 1
                        S2 = self.guess_next_state(dict_this_s, action)
                    for s2 in S2:
                        #Compute chance of transition:
                        p2 = S2[s2]
                        
                        pt = p1*p2 

                        # Update Q-table according to transition
                        if not isDone:
                            if s1 != s2:
                                thisQ += p2*np.max(self.QTable[s2])
                                thisQUnbiased += p2*np.max(self.QTableUnbiased[s2])
                            elif s1 == s2:
                                thisQ += p2*self.selfLoopPenalty*np.max(self.QTable[s2]) # We dis-incentivize self-loops by applying a small penalty to them
                                thisQUnbiased += p2*self.selfLoopPenalty*np.max(self.QTableUnbiased[s2])
                                

                # Update Q-unbiased
                if self.dynamicLR:
                    print(" UNIMPLEMENTED! ")
                    pass
                else: 
                    thisLR = self.lr * p1
                    totQUnbiased =  (1-thisLR) * self.QTableUnbiased[s1,action] + thisLR * (reward + self.df * thisQ)
                    totQ = (1-thisLR) * self.QTableUnbiased[s1,action] + thisLR * (reward + self.df*thisQ) 

                self.QTableUnbiased[s1,action] =  totQUnbiased
                
                # Update QTries & R only if real action
                if isReal and len(S2) == 1 and len(S1) == 1:
                    prevCounter = self.QCounter[s1,action]
                    self.QCounter[s1,action] += p1
                    self.QTableRewards[s1,action] = (self.QTableRewards[s1,action]*prevCounter + p1 * reward) / (self.QCounter[s1,action])
                
                # Implement bias
                thisAlpha = np.sum(self.alpha[s1,action]) + p1

                if thisAlpha >= self.NmbrOptimiticTries and self.optimism_type != "UCB":
                    self.QTable[s1,action] = totQ
                else:
                    match self.optimism_type:
                        case "RMAX+":
                            self.QTable[s1,action] = totQ + max(0,1-totQ) * ( (self.NmbrOptimiticTries - thisAlpha) / self.NmbrOptimiticTries)
                        case "UCB":
                            optTerm = np.sqrt(2 * np.log10(self.totalSteps+2) / (self.QCounter+1)) # How do we do this +2 cleanly?
                            self.QTableUnbiased[s1,action] = totQ
                            self.QTable = self.QTableUnbiased + ( self.UCB_Cp * optTerm )
                        case "RMAX":
                            self.QTable[s1,action] = 1
    
    def train_offline(self):
        "Performs Dyna-style oflline training of Q-values using current transition function"
        for i in range(self.otsteps):
            # Choose random state and action
            s = np.random.randint(self.StateSize)
            S_dict = {}
            S_dict[s] = 1
            
            if np.random.rand() < self.offline_eta:
                a = np.random.randint(self.ActionSize)
            else:
                a = self.get_action(S_dict)
                
            if np.sum(self.alpha[s,a]) > 5:
                b_next = self.guess_next_state(S_dict,a)
                r = self.QTableRewards[s,a]
                self.update_Q_lastStep_only(S_dict, b_next,a, r, isReal=False)
