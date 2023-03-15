


from AM_Gyms.AM_Env_wrapper import AM_ENV
import numpy as np

class ModelLearner():
    
    
    def __init__(self, env:AM_ENV, df = 0.90):
         
         self.env = env
         self.StateSize, self.ActionSize, self.cost, self.s_init = env.get_vars()
         self.doneState = self.StateSize
         self.StateSize += 1
         self.df = df
         self.fullStepUpdate = True
         self.init_model()
    
    
    def init_model(self):
        
        self.counter = np.zeros((self.StateSize, self.ActionSize))
        self.P = build_dictionary(self.StateSize, self.ActionSize)
        self.P_counter = build_dictionary(self.StateSize, self.ActionSize)
        for a in range(self.ActionSize):
            self.P_counter[self.doneState][a][self.doneState] = 1
         
        self.R_counter = np.zeros((self.StateSize, self.ActionSize))
        self.R = np.zeros((self.StateSize, self.ActionSize))
        
        self.Q = np.zeros((self.StateSize, self.ActionSize))
        self.Q[self.doneState, :] = 0
        self.Q_max = np.zeros((self.StateSize))
        
        self.Q_learning = np.ones((self.StateSize, self.ActionSize))
        self.Q_learning[self.doneState, :] = 0
        self.Q_learning_max = np.max(self.Q_learning, axis=1)
        self.df_learning = 0.90
    
    def get_model(self):
        """Returns P, R, Q"""
        return self.P, self.R, self.Q
    
    def run_visits(self, min_visits = 50, max_eps = np.inf, logging = True):
        
        i = 0
        final_updates = 100
        
        done = False
        while not done :
            i += 1
            _r, _s, _m = self.run_episode()
            counter_nonzero = np.nonzero(self.counter)
            done = np.min(self.counter[counter_nonzero]) > min_visits or i > max_eps
            if logging and i%250 == 0:
                print("{} episodes completed!".format(i))
        self.insert_done_transitions()
        print("Nmbr of episodes: {}".format(i))
        
        
        for i in range(final_updates):
            self.update_model()
        
        
    
    def run_episode(self):
        self.env.reset()
        done = False
        (s, _cost) = self.env.measure()
        totalreward, totalsteps = 0, 0
        
        while not done:
            
            a = np.argmax(self.Q_learning[s])
            reward, done = self.env.step(a)
            if done:
                snext = self.doneState
            else:
                (snext,cost) = self.env.measure()
                
            self.update_counters(s, a, snext, reward)
            self.update_model([(s,a)])
            
            totalreward += reward; totalsteps += 1
            s = snext
        return totalreward, totalsteps, totalsteps
        
        
    def update_counters(self, s, a, snext, reward):
        
        # Update counters
        self.counter[s,a] += 1
        if snext in self.P[s][a]:
            self.P_counter[s][a][snext] += 1
        else:
            self.P_counter[s][a][snext] = 1
            self.P[s][a][snext] = 0
        self.R_counter[s,a] += reward
    
    def update_model(self, state_action_pairs = None, full_update = False):
        
        if full_update:
            state_action_pairs = np.meshgrid(np.arange(self.StateSize), np.arange(self.ActionSize))
        
        if state_action_pairs is not None:
            for (s,a) in state_action_pairs:
                self.R[s,a] = self.R_counter[s,a] / self.counter[s,a]
                Psi, Psi_learning = 0, 0
                for (s_next, _p) in self.P[s][a].items():
                    self.P[s][a][s_next] = self.P_counter[s][a][s_next] / self.counter[s,a]
                    Psi += self.P[s][a][s_next] *  self.Q_max[s_next]
                    Psi_learning += self.P[s][a][s_next] * self.Q_learning_max[s_next]
                
                self.Q[s,a] = self.R[s,a] + self.df * Psi
                self.Q_learning[s,a] = 1/self.counter[s,a] + self.df_learning * Psi_learning
                
                self.Q_max[s] = np.max(self.Q[s])
                self.Q_learning_max[s] = np.max(self.Q_learning[s])
    
    def insert_done_transitions(self):

        for s in range(self.StateSize):
            for a in range(self.ActionSize):
                if not self.P[s][a]:
                    self.P[s][a][self.doneState] = 1

def build_dictionary(statesize, actionsize, array:np.ndarray = None):
    dict = {}
    for s in range(statesize):
        dict[s] = {}
        for a in range(actionsize):
            dict[s][a] = {}
            if array is not None:
                for snext in range(statesize):
                    dict[s][a][snext] = array[s,a,snext]
    return dict