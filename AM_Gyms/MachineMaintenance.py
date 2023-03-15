import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class Machine_Maintenance_Env(gym.Env):
    """
    Machine Maintenance environment.
    This openAI gym environment describes the Machine Maintenance MDP introduced in
    in Delage and Mannor (2010) for testing robust policy finders, with some slight alterations.
    In short, the environment consists of N 'working' states (W1..WN), 
    one 'broken' state (B), and two repair states (R1, R2), representing two 
    difficulties of repair.
    These states are ordered in a chain: R2 -> R1 -> W1 -> ... -> WN -> B
    At each state, the agent can decide between a 'work' and 'repair' action.
    When working, the agent has a 0.8 chance to move up the chain (or selfloop
    if in B), and a 0.2 chance to stay in the current state.
    When repairing, the agent moves to R1 with prob. 0.6, to R2 with prob. 0.1,
    and to the next state with prob. 0.3.
    The agent receives reward -1 for reaching B, reward -0.5 for reaching R2, and
    reward -0.1 for reaching R1.
    """
    Standard_Rewards    = { "R1": -0.1, "R2": -0.5, "B": -1, "W": 0}
    Standard_Probs      = {"Working": {"Next":0.8, "This":0.2, "R1":0.0, "R2":0.0},
                       "Repair" : {"Next":0.3, "This":0.0, "R1":0.6, "R2":0.1}   }
    
    def __init__(self, N:int = 8, termination_prob:float = 0.02, rewards:dict = Standard_Rewards, probs:dict = Standard_Probs ):
        
        self.N          = N
        self.probs      = probs
        self.cum_probs  = Machine_Maintenance_Env.get_cumulative_probs(probs)
        self.rewards    = rewards
        self.done_prob  = termination_prob
        
        self.state              = 0
        self.action_space       = spaces.Discrete(2)
        self.observation_space  = spaces.Discrete(self.N+3)
        
        self.seed()
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        
        # Choose correct transition probs according to action:
        match action:
            case 0: current_cum_probs = self.cum_probs["Working"]
            case 1: current_cum_probs = self.cum_probs["Repair"]
        
        # Transition state:
        rnd = np.random.rand()
        has_transitioned = False
        for transition in current_cum_probs:
            cum_prob = current_cum_probs[transition]
            if cum_prob > rnd and not has_transitioned:
                has_transitioned = True
                match transition:
                    case "Next" : self.state = min(self.state+1, self.N)
                    case "This" : pass
                    case "R1"   : self.state = -1
                    case "R2"   : self.state = -2
                    case _      : print("Transition not recognised: probability dictionary likely set up wrong!")
        
        # Determine reward:
        match self.state:
            case -2         : reward = self.rewards["R2"]
            case -1         : reward = self.rewards["R1"]
            case self.N     : reward = self.rewards["B"]
            case n if n in list(range(self.N)):
                              reward = self.rewards["W"]
            case _          : print("Warning: entered impossible state {}".format(self.state))
        
        done = np.random.rand() < self.done_prob
        return self.state+2, reward, done, {}
    
    def reset(self):
        self.state = 0
        return self.state
    
    @staticmethod
    def get_cumulative_probs(probs):
        for action in probs:
            action_probs = probs[action]
            cum_prob = 0
            for event in action_probs:
                this_prob = action_probs[event]
                cum_prob += this_prob
                action_probs[event] = cum_prob
            
            probs[action] = action_probs
        return probs
    
    def getname(self):
        return "Maintenance_N{}".format(self.N)
        