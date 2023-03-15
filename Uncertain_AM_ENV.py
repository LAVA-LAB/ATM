
import numpy as np
from dataclasses import dataclass
from AM_Gyms.AM_Env_wrapper import AM_ENV
from AM_Gyms.AM_Tables import AM_Environment_tables,  RAM_Environment_tables

@dataclass
class Distribution:
    type:str
    variance:float

available_distributions = ["uniform_05"]

uniform_05 = Distribution("Uniform", 0.5)

class Uncertain_AM_ENV(AM_ENV):
    
    def __init__(self, env, table:AM_Environment_tables, distribution:Distribution = uniform_05):
        
        self.env = env  #only used for name-stuff
        self.table = table
        self.distribution = distribution
        
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = table.get_vars()
        self.StateSize = self.StateSize + 1
        self.P, self.R, self.Q = table.get_tables()
        
        self.P_this_run = build_dictionary(self.StateSize, self.ActionSize)
        self.P_is_chosen= np.zeros((self.StateSize, self.ActionSize), dtype=bool)
        
        self.state = 0
        self.max_steps = 10_000 # Just guessed...        
        
    def step(self, action):
        
        if not self.P_is_chosen[self.state, action]:
            self.choose_P(self.state,action)
        thisP = self.P_this_run[self.state][action]
        
        states, probs = np.array(list(self.P[self.state][action].keys())), np.array(list(self.P[self.state][action].values()))
        next_state = np.random.choice(a=states, p=probs)
        
        reward = self.R[self.state, action]
        done = (self.state == self.StateSize-1)
        self.state = next_state
        self.obs = self.state
        
        self.steps_taken += 1
        if self.steps_taken > self.max_steps:
            done = True
        
        return (reward, done)
    
    def reset(self):
        self.state = self.s_init
        self.P_this_run = {}
        self.P_is_chosen = np.zeros((self.StateSize, self.ActionSize), dtype=bool)
        self.steps_taken = 0
        
    def choose_P(self, s, a):
        
        variance = self.distribution.variance
        states, probs = np.array(list(self.P[s][a].keys())), np.array(list(self.P[s][a].values()))
        mins, maxs = np.maximum(probs - variance, 0), np.minimum(probs + variance, 1)

        match self.distribution.type:
            case "Uniform":
                thisP = np.random.uniform(mins, maxs)
            case _:
                print("Error: distribution not found!")
        if not s in self.P_this_run:
            self.P_this_run[s] = {}
        self.P_this_run[s][a] = {}
        self.P_is_chosen[s][a] = True
        
        for (snext, pnext) in zip(states, thisP):
            self.P_this_run[s][a][snext] = pnext



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
        