import numpy as np

from AM_Gyms.AM_Env_wrapper import AM_ENV

def build_dictionary(statesize, actionsize, array:np.ndarray = None):
    dict = {}
    for s in range(statesize):
        dict[s] = {}
        for a in range(actionsize):
            dict[s][a] = {}
            if array is not None:
                for snext in range(statesize):
                    dict[s][a][snext] = array[s,a,snext]



class ModelLearner():
    """Class for learning ACNO-MDP """

    def __init__(self, env:AM_ENV, df = 0.90):
        # Set up AM-environment
        self.env = env
        
        # Model variables
        self.StateSize, self.CActionSize, self.cost, self.s_init = env.get_vars()
        self.StateSize += 1
        self.ActionSize = self.CActionSize * 2  #both measuring & non-measuring actions
        self.EmptyObservation = self.StateSize +100
        self.doneState = self.StateSize -1
        self.loopPenalty = 0 #self.cost
        self.donePenalty = 0 #self.cost
        self.df_real = df    # discount factor, only impacts Q_real
        
        self.init_model()

    def init_model(self):
        # Model dictionaries:
        self.counter = np.zeros((self.StateSize, self.ActionSize)) + 1
        self.T_counter = np.zeros((self.StateSize, self.ActionSize, self.StateSize)) + 1/self.StateSize
        self.T_counter[self.doneState,:,:] = 0
        self.T_counter[self.doneState,:,self.doneState] = 1
        self.T = self.T_counter / self.counter[:,:,np.newaxis]
        
        self.T_dict = {}
        self.R_dict = {}
        
        self.R_counter = np.zeros((self.StateSize, self.ActionSize))
        self.R = self.R_counter / np.maximum(self.counter-1,1)
        self.R_biased = self.R_counter / np.maximum(self.counter-1,1) #Not yet biased...
        
        self.Q_real = np.ones( (self.StateSize, self.ActionSize )) # The Q-value of the MDP
        self.Q_real_max = np.ones( ( self.StateSize) )
        
        # Variables for learning:
        self.Q = 1/self.counter[:,:self.CActionSize] # The 'value' of checking a transition, i.e. the Q-value of this algorihtm
        self.lr = 0.3
        self.df = 0.99
    
    def get_model(self, transformed = False):
        """Returns T, R, R_biased (Note: actionspace is \tilde{A} = AxM!) """
        if transformed:
            return self.T[:, self.CActionSize:, :], self.R[:, self.CActionSize:], self.R_biased[:,self.CActionSize:]
        return self.T, self.R, self.R_biased
    
    def get_T_dictionary(self):
        """returns T in the form of dictionary"""
        return self.T_dict
    
    def create_dictionaries(self):
        
        for s in range(self.StateSize):
            self.T_dict[s] = {}
            for a in range(self.CActionSize):
                self.T_dict[s][a] = {}
                for (snext,p) in enumerate(self.T[s,a]):
                    if p != 0:
                        self.T_dict[s][a][snext] = p
        

        
    
    def get_Q(self, transformed=False):
        """Returns the Q-value function (NOTE: measuring cost currently unused, and action space \tilde{A}!)"""
        if transformed:
            return self.Q_real[:,self.CActionSize:]
        return self.Q_real
    
    def get_vars(self):
        """returns StateSize, ActionSize, cost, s_init, doneState"""
        return (self.StateSize, self.ActionSize, self.cost, self.s_init, self.doneState)
    
    def remove_done_transitions(self):
        for state in range(self.StateSize-1):
            for action in range(self.ActionSize):
                self.T_counter[state, action, self.doneState] = 0
                self.T[state,action] = self.T_counter[state,action] / np.sum(self.T_counter[state,action])
    
    def filter_T(self):
        """Filters all transitions with p<1/|S| from T"""
        p = min(1/self.StateSize, 0.05)
        # mask = self.T_counter<p*self.counter[:,:,np.newaxis]
        mask = self.T < p
        self.T[mask] = 0
        self.T = self.T / np.sum(self.T, axis =2)[:,:,np.newaxis]
        
    def add_costs(self):
        self.R_biased = np.copy(self.R)
        costs = np.zeros((self.StateSize, self.ActionSize))
        costs[:, :self.CActionSize] -= self.cost    # Measuring cost
        for a in range(self.ActionSize):
            selfprob = np.diagonal(self.T[:,a,:])
            doneprob = self.T[:,a,self.doneState]
            costs[:,a] -= selfprob*self.loopPenalty + doneprob*self.donePenalty
        self.R_biased += costs
    
    def sample(self, N, max_steps = 100, logging = True, includeCosts = True, modify = True):
        """Learns the model using N episodes, returns episodic costs and steps"""
        # Intialisation
        self.init_model()
        self.sampling_rewards = np.zeros(N)
        self.sampling_steps = np.zeros(N)
        
        for eps in range(N):
            self.sample_episode(eps, max_steps)
            if (eps+1) % (N/10) == 0 and logging:
                print("{} exploration episodes completed!".format(eps+1))
        self.update_Q_only(updates = np.min([self.StateSize*self.ActionSize, 100]))
        if modify:
            self.filter_T()
            self.add_costs()
        self.create_dictionaries()
        return self.sampling_rewards, self.sampling_steps
    
    def sample_episode(self, episode, max_steps):
        """Samples one episode, following method proposed in https://hal.inria.fr/hal-00642909"""
        self.env.reset()
        done = False
        (s_prev, _cost) = self.env.measure()
        for step in range(max_steps):
            
            # Greedily pick action from Q
            a = np.argmax(self.Q[s_prev])
            
            # Take step & measurement
            reward, done = self.env.step(a)
            if done:
                s = self.doneState
            else:
                (s, cost) = self.env.measure()
            
            # Update logging variables
            self.sampling_rewards[episode] += reward - self.cost
            self.sampling_steps[episode] += 1
            
            # Update model
            self.update_step(s_prev, a, s, reward) 
            
            # Update learning Q-table
            CAS = self.CActionSize
            Psi = np.sum(self.T[s_prev,:CAS] * np.max(self.Q, axis=1), axis=1) #axis?
            self.Q[s_prev] = (1-self.lr)*self.Q[s_prev] + self.lr*(1/self.counter[s_prev,:CAS] + self.df*Psi )
            s_prev = s
            if done:
                break
    
    def update_step(self, s_prev, a, s_next, reward):
        ac, ao = a % self.CActionSize, a // self.CActionSize
        
        # update measuring action counters
        self.counter[s_prev,ac] += 1
        self.T_counter[s_prev,ac,s_next] += 1
        self.R_counter[s_prev,ac] += reward
        
        self.T[s_prev,ac,s_next] = self.T_counter[s_prev,ac,s_next] / self.counter[s_prev,ac]
        self.R[s_prev,ac] = self.R_counter[s_prev,ac] / np.maximum(self.counter[s_prev,ac]-1,1)
        
        # update non-measuring actions counters
        anm = ac + self.CActionSize
        self.counter[s_prev,anm] += 1
        self.T_counter[s_prev,anm,s_next] += 1
        self.R_counter[s_prev,anm] += reward
        
        self.T[s_prev,anm,s_next] = self.T_counter[s_prev,anm,s_next] / self.counter[s_prev,ac]
        self.R[s_prev,anm] = self.R_counter[s_prev,anm] / np.maximum(self.counter[s_prev,anm]-1,1)
        
        # update Q-values
        self.Q_real[s_prev, ac] = self.df_real * np.sum(self.T[s_prev,ac]*self.Q_real_max) + self.R[s_prev,ac] 
        self.Q_real[s_prev, anm] = self.df_real * np.sum(self.T[s_prev,anm]*self.Q_real_max) + self.R[s_prev,anm]
        self.Q_real_max[s_prev] = np.max(self.Q_real[s_prev])
    
    def update_Q_only(self, updates = 100):
        for i in range(updates):
            saPairs = np.array(np.meshgrid(np.arange(self.StateSize), np.arange(self.ActionSize))).T.reshape(-1,2)
            np.random.shuffle(saPairs)
            for sa in saPairs:
                s,a = sa[0], sa[1]
                self.Q_real[s,a] = self.df_real * np.sum(self.T[s,a]*self.Q_real_max) + self.R[s,a]
            self.Q_real_max = np.max(self.Q_real, axis=1)
    
    def reset_env(self):
        self.env.reset()
        
    def real_step(self, action):
        return self.env.step(action)
    
    def measure_env(self):
        return self.env.measure()
