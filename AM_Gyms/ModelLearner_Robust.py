import numpy as np
import math as m

from AM_Gyms.ModelLearner_V2 import ModelLearner
from AM_Gyms.AM_Env_wrapper import AM_ENV



class ModelLearner_Robust():
    """Class to determine the ICVaR of a MDP, as represented by an AM-env."""
    
    def __init__(self, Env:AM_ENV, alpha:float, df = 0.95):
        
        # Unpacking variables from environment:
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = self.env.get_vars()
        self.StateSize  += 1 #include done state
        self.doneState  = self.StateSize -1
        
        # Setting up tables:
        self.P          = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )
        self.R          = np.zeros( (self.StateSize, self.ActionSize) )
        self.Q          = np.zeros( (self.StateSize, self.ActionSize) )
        self.Q_max      = np.zeros( self.StateSize)
        
        self.ICVaR      = np.zeros( (self.StateSize, self.ActionSize) )
        self.ICVaR_max  = np.zeros(self.StateSize)
        
        # Other variables:
        self.alpha      = alpha
        self.df         = df
        self.epsilon    = 0.5
        
        
    def learn_model(self, eps, logging = False):
        """Uses the ModelLearner-class to get P, R & Q"""
        
        # Run modelLearner:
        modelLearner = ModelLearner(self.env, df = self.df)
        modelLearner.run_visits()
        
        # Unpack values 
        self.P, self.R, self.Q = modelLearner.get_model()
        self.Q_max = np.max(self.Q, axis=1)
        
        self.ICVaR, self.ICVaR_max = np.copy(self.Q), np.copy(self.Q_max)
        self.DeltaP = {}
        for s in range(self.StateSize):
            self.DeltaP[s] = {}
            for a in range(self.ActionSize):
                self.DeltaP[s][a] = dict(self.P[s][a])
        
    
    def update_Q(self, s, a):
        """Updates Q-table according to (known) model dynamics (currently unused)"""
        self.Q[s,a] = self.df * np.sum( self.P[s,a] * self.Q_max ) + self.R[s,a]
        self.Q_max[s] = np.max(self.Q[s])
    
    def update_ICVaR(self, s, a):
        """Updates ICVaR according to (known) model dynamics"""
        
        # 1) Make a dictionary of all non-zero elements in P (Efficiency!)
        states, probs, icvar_max = [], [], []
        for (state, prob) in self.P[s][a].items():
            states.append(state)
            probs.append(prob)
            icvar_max.append(self.ICVaR_max[state])
        states, probs, icvar_max = np.array(states.copy()), np.array(probs.copy()), np.array(icvar_max)
        r = self.R[s,a]
        
        
        # 2) Get ICVaR values according to custom procedure
        delta_p_new = ModelLearner_Robust.custom_delta_minimize(probs, icvar_max, self.alpha)
        
        # 3) Update deltaP's and ICVaR
        self.ICVaR[s,a] = r
        for (i,snext) in enumerate(states):
            self.DeltaP[s][a][snext] = delta_p_new[i]
            self.ICVaR[s][a] += self.df * delta_p_new[i] *icvar_max[i]
        self.ICVaR_max[s] = np.max(self.ICVaR[s])
    
    @staticmethod    
    def custom_delta_minimize(probs, icvar, alpha):
        """Calculates the worst-case disturbance delta of transition probabilities probs,
        according to next state icvar's and perturbation budget 1/alpha.
        
        The general idea of this method is to repeatedly maximize the probability for
        the worst-case scenerio (for which delta<1/alpha), while simultaiously lowering 
        probabilities for the best-case scenario. By alternating these, we aim to keep the 
        total transition probability equal to 1.
        """
        delta = np.ones(len(probs))
        
        # 1) Sort according to icvar:
        sorted_indices = np.argsort(icvar)
        probs, icvar = probs[sorted_indices], icvar[sorted_indices]
        
        # 2) Repeatedly higher/lower probability of lowest/highest icvar elements
        sum_delta_p = np.sum(probs)
        changable_probs = list(range(len(probs)))   # list of probabilities we have not yet changed
        while changable_probs:
            # Higher probability of worst outcomes
            if sum_delta_p <= 1:
                this_i = changable_probs[0]
                delta[this_i] = 1/alpha
                sum_delta_p += probs[this_i] * (1/alpha - 1)
                lowest_i_highered = this_i
                changable_probs.pop(0)
            # lower probability of best outcomes
            elif sum_delta_p > 1:
                this_i = changable_probs[-1]
                delta[this_i] = 0   # it should already be..
                sum_delta_p -= probs[this_i]
                highest_i_lowered = this_i
                changable_probs.pop(-1)
                
        # 3) Fix probability problems by editing last 'bad' change:
        if m.isclose(sum_delta_p, 1, rel_tol=1e-5):
            pass
        # If our total probability is too low, we must compensate by upping the last probability we lowered.
        elif sum_delta_p < 1:
            required_p = 1-sum_delta_p 
            delta[highest_i_lowered] = required_p / probs[highest_i_lowered]
        # Similarly, if our total probability is too high, we compensate by lowering the last probability upped.
        elif sum_delta_p > 1:
            required_p = 1 - (sum_delta_p - probs[lowest_i_highered]*delta[lowest_i_highered])
            delta[lowest_i_highered] = required_p / probs[lowest_i_highered]

        # 4) Restore original order
        original_indices = np.argsort(sorted_indices)
        delta_p_new = delta[original_indices] * probs[original_indices]
        return delta_p_new
        
        
        
    def pick_action(self,s):
        """Pick higheste icvar-action epsilon-greedily"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.ActionSize)
        return np.argmax(self.ICVaR[s])
    
    def run(self, updates = 1_000, eps_modelLearner = 10_000, logging = True):
        """Calculates model dynamics using eps_modelearning episodes, then ICVaR using
        updates updates per state."""
        
        # Get P, R & Q from model learner
        if logging:
            print("Calculating model dynamics using ModelLearner module:")
        self.learn_model(eps_modelLearner, logging)

        # Learn ICVaR
        if logging:
            print("Calculating ICVaR:")
        for i in range(updates):
            S = np.arange(self.StateSize)
            np.random.shuffle(S)
            for s in S:
                a = self.pick_action(s)
                self.update_ICVaR(s, a)
                
            if (i%(np.min([updates/10, 1000])) == 0 and logging):
                print("Episode {} completed!".format(i+1))

    def get_model(self):
        """Return all model tables (P, R, Q, DeltaP, ICVaR)"""
        return (self.P, self.R, self.Q, self.DeltaP, self.ICVaR)

    def get_model_dictionaries(self):
        """returns both P and DeltaP as dictionaries"""
        return (self.P, self.DeltaP)
    
    
    # def calculate_model_dicts(self):
        
    #     self.P_dict, self.DeltaP_dict = {}, {}
    #     for s in range(self.StateSize):
    #         self.P_dict[s] = {}; self.DeltaP_dict[s] = {}
    #         for a in range(self.ActionSize):
    #             self.P_dict[s][a] = {}; self.DeltaP_dict[s][a] = {}
    #             for (snext,p) in enumerate(self.P[s,a]):
    #                 if p != 0:
    #                     self.P_dict[s][a][snext] = p
    #                     self.DeltaP_dict[s][a][snext] = self.DeltaP[s,a,snext]
                        

# Code for testing:

# from AM_Gyms.frozen_lake import FrozenLakeEnv

# Semi-slippery, larger   
# Env = AM_ENV( FrozenLakeEnv_v2(map_name = "8x8", is_slippery=True), StateSize=64, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# semi-slippery, small
# Env = AM_ENV( FrozenLakeEnv_v2(map_name = "4x4", is_slippery=True), StateSize=16, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# slippery, small
# Env = AM_ENV( FrozenLakeEnv(map_name = "4x4", is_slippery=True), StateSize=16, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# icvar = ICVaR(Env, alpha = 0.3)

# icvar.run(logging=True)
# print(icvar.ICVaR)