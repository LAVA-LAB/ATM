import numpy as np
import math as m
import cvxpy as cp


from AM_Gyms.AM_Tables import AM_Environment_tables, RAM_Environment_tables
from AM_Gyms.AM_Env_wrapper import AM_ENV

class ACNO_Planner():
    
    def __init__(self, Env:AM_ENV, tables = AM_Environment_tables, df=0.95):
        
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = tables.get_vars()
        self.P, _R, self.Q = tables.get_tables()

        self.df          = df
        self.epsilon_measuring = 0.01
        self.loopPenalty = 0.95
    
    def run(self, eps, logging=False):
        
        if logging:
            print("starting learning of model...")
        rewards, steps, measurements = np.zeros(eps), np.zeros(eps), np.zeros(eps)
        log_nmbr = 100
        
        if logging:
            print("Start planning:")
        
        for i in range(eps):
            reward, step, measurement = self.run_episode()
            rewards[i], steps[i], measurements[i] = reward, step, measurement
            #print(i)
            if (i > 0 and i%log_nmbr == 0 and logging):
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {}, nmbr measures = {})".format( 
                        i, eps, np.average(rewards[(i-log_nmbr):i]), np.average(steps[(i-log_nmbr):i]), np.average(measurements[(i-log_nmbr):i]) ) )
            
        return (np.sum(rewards), rewards, steps, measurements)
    
    def run_episode(self):
        
        self.env.reset()
        currentBelief, nextBelief = {0:1}, {}
        nextAction:int; currentAction:int; currentMeasuring:bool
        done = False
        total_reward, total_steps, total_measures = 0, 0, 0
        
        currentAction = self.determine_action(currentBelief)
        
        while not done:

            nextBelief = self.compute_next_belief   (currentBelief, currentAction)
            if currentBelief == nextBelief:
                currentMeasuring = True
            else:
                nextAction = self.determine_action      (nextBelief)
                currentMeasuring  = self.determine_measurement (currentBelief, currentAction, nextBelief, nextAction) 
            
            reward, done = self.execute_action(currentAction, currentBelief, currentMeasuring)
            if currentMeasuring or np.random.random() < self.epsilon_measuring:
                nextBelief, cost = self.measure()
                nextAction = self.determine_action (nextBelief)
                total_measures += 1
            else:
                cost = 0
            #print(currentBelief, currentAction, currentMeasuring, nextBelief)
            total_reward += reward - cost
            total_steps += 1
            currentBelief, currentAction = nextBelief, nextAction
        
        return total_reward, total_steps, total_measures
    
    
    def determine_action(self, b):
        return optimal_action(b, self.Q, None)
    
    def compute_next_belief(self, b, a):
        return next_belief(b,a,self.P)
    
    def determine_measurement(self, b, a, b_next=None, a_next=None):
        if b_next is None:
            b_next = self.compute_next_belief(b,a)
        if a_next is None:
            a_next = self.determine_action(b_next)
        return measuring_value(b_next, a_next, self.Q) > self.cost
    
    def execute_action(self, action, belief, measuring):
        reward, done = self.env.step(action)
        return reward, done
    
    def measure(self):
        s, cost = self.env.measure()
        return {s:1}, cost

class ACNO_Planner_Robust(ACNO_Planner):
    
    def __init__(self, Env:AM_ENV, tables = RAM_Environment_tables, df=0.95):
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = tables.get_vars()
        self.PReal, _R, self.QReal = tables.get_tables()
        self.P, self.Q, self.Pmin, self.Pmax = tables.get_robust_MDP_tables()
        
        self.df         = df
        self.epsilon_measuring    = 0.01
    

    
    def determine_action(self, b):
        return optimal_action(b, self.Q, self.QReal)
    
    def compute_next_belief(self, b, a):
        
        b_real = custom_worst_belief(b, a, self.P, self.Pmin, self.Pmax, self.Q)
        
        return b_real
        

class ACNO_Planner_Control_Robust(ACNO_Planner_Robust):

    def determine_measurement(self, b, a, b_next, a_next):
        b_next_real = next_belief(b,a,self.PReal)
        return (measuring_value(b_next_real, a_next, self.QReal, self.Q) > self.cost
                or  measuring_value(b_next, a_next, self.Q) > self.cost )


# Generalized functions:

def optimal_action(b:dict, Q1:np.ndarray, Q2:np.ndarray = None, returnvalue = False):
    """Returns the optimal action for belief state b as given by Q1.
    Ties are broken according to Q2, then randomly."""

    
    actionsize = np.shape(Q1)[1]
    thisQ1 = np.zeros(actionsize)
    
    for (state, prob) in b.items():
        
        thisQ1 += prob * Q1[state]
    
    thisQ1Max = np.max(thisQ1)
    filter = np.isclose(thisQ1, thisQ1Max, rtol=0.001, atol= 1e-15)
    optimal_actions = np.arange(actionsize)[filter]

    
    if np.size(optimal_actions) > 1 and Q2 is not None:
        
        thisQ2 = np.zeros(actionsize)
        for (state, prob) in b.items():
            thisQ2 += prob * Q1[state]
            
        thisQ2Max = np.max(thisQ2[filter])
        filter2 = np.isclose(thisQ2, thisQ2Max, rtol=0.001, atol= 1e-15)
        filtersCombined = np.logical_and(filter, filter2)
        optimal_actions = np.arange(actionsize)[filtersCombined]

    optimal_action = int(np.random.choice(optimal_actions))
    
    if returnvalue:
        return optimal_action, thisQ1[optimal_action]
    return optimal_action

def next_belief(b:dict, a:int, P:dict, min_probability_considered:float = 0.001):
    """computes next belief state, according to current belief b, action a and transition function P.
    All belief probabilities smaller than min_probability_considered are ignored (for efficiency reasons)."""
    
    b_next = {}
    
    for (state, beliefprob) in b.items():
        for (next_state, transitionprob) in P[state][a].items():
            if next_state in b_next:
                b_next[next_state] += beliefprob * transitionprob
            else:
                b_next[next_state] = beliefprob * transitionprob
    
    # TODO: filter states that are too small
    # filter = b_next_array >= min_probability_considered
    # states = np.arange(statesize)[filter]
    # b_next_array = b_next_array[filter]
    # b_next_array = b_next_array / np.sum(b_next_array)

    
    # b_next = dict()
    # for (state, prob) in zip(states, b_next_array):
    #     b_next[state] = prob
    
    return b_next

def next_belief_array(b:dict, a:int, P:np.ndarray, statesize:int, actionsize:int, flat = False):
    b_next = {}
    
    for (s, beliefprob) in b.items():
        for a in range(actionsize):
            for snext in range(statesize):
                if flat:
                    index = index_flatten_P(statesize, actionsize, s, a, snext)
                else:
                    index = (s,snext)
                if snext in b_next:
                    b_next[snext] += beliefprob * P[index]
                else:
                    b_next[snext] = beliefprob * P[index]
                    
    return b_next
    

def measuring_value(b:dict, a_b:int, Q:np.ndarray, Q_decision = None):
    if Q_decision is None:
        Q_decision = Q
    MV = 0
    for (state, prob) in b.items():
        a_m = np.argmax(Q_decision[state])
        MV += prob * max (0.0, Q[state, a_m] - Q[state, a_b])
    return MV


def index_flatten_P(statesize, s, snext):
    return snext + statesize * s

def index_unflatten_P(statesize, index):
    s = index // (statesize)
    snext = (index % statesize) // statesize
    return (s,snext)

def check_valid_P(P, Pmin, Pmax):
    indexsize = np.size(P)
    for i in range(indexsize):
        if not (m.isclose(np.sum(P[i]), 1) and P[i] < Pmax[i] and P[i] > Pmin[i]):
            return False
    return True

def get_partial_b(state_indexes:np.ndarray, b:dict,  flat = False):
    b_array = np.zeros(np.size(state_indexes))
    for (i,s) in enumerate(state_indexes):
        if s in b:
            b_array[i] = b[s]
    return np.array(b_array)
    
def get_partial_P(state_indexes:np.ndarray, P:dict, a:int, flat = False):
    
    statesize = np.size(state_indexes)
    if flat:
        P_array = np.zeros(statesize * statesize)
    else:
        P_array = np.zeros((statesize, statesize))
    
    for (s_i,s) in enumerate(state_indexes):
        for (snext_i, snext) in enumerate(state_indexes):
            if snext in P[s][a]:
                if flat:
                    P_array[index_flatten_P(statesize,s_i, snext_i)] = P[s][a][snext]
                else:
                    P_array[s_i,snext_i] = P[s][a][snext]
    return P_array

def b_array_to_dict(b_array:np.ndarray, indexes:np.ndarray, min_probability_considered:float = 0.001):
    b_array[b_array < min_probability_considered] = 0
    b_array = b_array / np.sum(b_array)
    
    b_dict = {}
    for (index, prob) in enumerate(b_array):
        if prob > 0:
            b_dict[indexes[index]] = prob
    return b_dict
            
def custom_worst_belief(b:dict, a:int, Pguess, Pmin:dict, Pmax:dict, Q:np.ndarray, min_probability_considered:float = 0.01):

    """We rewrite our worst-case probability function as the solution of a linear program of the following form:
    
    minimize b
    
    
    """
    
    # Unpack P & Q into arrays of required sizes
    relevant_current_states, relevant_next_states = [], []
    for s in b.keys():
        relevant_current_states.append(s)
        for snext in Pmax[s][a].keys():
            relevant_next_states.append(snext)
    relevant_next_states = np.unique(relevant_next_states)
    state_indexes = np.unique(np.append(relevant_current_states, relevant_next_states))
    statesize, actionsize = np.size(state_indexes), np.shape(Q)[1]
    
    Pmin_small = get_partial_P(state_indexes, Pmin, a, flat=False)
    Pmax_small = get_partial_P(state_indexes, Pmax, a, flat=False)
    Q_small = Q[state_indexes]
    b_array = get_partial_b(state_indexes, b)
    
    P = cp.Variable( (statesize,statesize) )
    bnext = cp.Variable(statesize)
    
    Qmax = cp.Variable()
    objective = cp.Minimize(Qmax)
    constraints = [bnext == b_array @ P] # @ ?
    for a in range(actionsize):
        constraints.append(Qmax >= bnext @ Q_small[:,a])
    for (index, state) in enumerate(state_indexes):
        if state in b:
            constraints.append(cp.sum(P[index]) == 1)
            constraints.append(P[index] >= Pmin_small[index])
            constraints.append(P[index] <= Pmax_small[index])
    
    problem = cp.Problem(objective, constraints)
    #problem.solve(solver=cp.GLPK)
    problem.solve(solver=cp.GLOP)
    if bnext.value is None:
        print("Error: solver failed!: {}".format(problem.status))
        return next_belief(b,a,Pguess)
    

    b_worst_dict = b_array_to_dict(bnext.value, state_indexes)
    return b_worst_dict
    
    
    
    # Goal = lambda P: optimal_action(next_belief_flat_array(b,a,P,statesize, actionsize), Q, returnvalue=True)[1]

    
    
    # Constraints = {"type":"ineq", "fun": lambda P: float(not check_valid_P(P, Pmin, Pmax))}
    
    # minimize_results = minimize(fun=Goal, x0 = Pguess, constraints = [Constraints])
    # if minimize_results["success"]:
    #     P = minimize_results["x"]
    # else:
    #     print (minimize_results["message"])
    #     P = Pguess
    
    
    # actionsize = 0
    # done = False
    # currentP = np.zeros()
    # states = np.zeros()
    # b_next_array = ...
    # while not done:
    #     b_next = next_belief(b,a,currentP)
        
    #     Q_next = np.zeros(actionsize)
    #     for (state, prob) in b_next.items():
    #         Q_next += prob * Q[state]
    #     a_b, Q_a_b = np.argmax(Q_next), np.max(Q_next)
    #     Q_rest = np.delete(Q_next,a_b, axis=1)
    #     a_b_2nd, Q_a_b_2nd = np.argmax(Q_rest), np.max(Q_rest)
        
    #     s_next_to_minimize = np.argmax(Q[b_next_array, a_b])
    #     Q_s_next_to_minimize = np.max(Q[b_next_array, a_b])
        
    #     delta_probability = (Q_a_b - Q_a_b_2nd) / Q_s_next_to_minimize
    #     sources = np.nonzero(currentP[:,a,s_next_to_minimize])
    #     done2 = False
    #     while not done2:
    #         source_state = sources[1]
            
            
        
        
    #     source_state = np.argmax(currentP[:,a,s_next_highest_Q])
    #     delta_probability_tried = 
    
    
    # solver = pywraplp.Solver.CreateSolver('GLOP')
    
    # nmbr_states = len(b)
    # (_total_states, nmbr_actions) = np.shape(Q)
    # Normalising_constraints = []
    # Goal = ""
    
    # # Set up actions as extra variables:
    # one_action_constraint = ""
    # actions = []
    # for a in range(nmbr_actions):
    #     variable_name = "a_{}".format(a)
    #     actions.append(solver.IntVar(0, 1, variable_name))
    #     one_action_constraint += "{}+".format(actions[a])
    # one_action_constraint = one_action_constraint[:-1] + "=1"
    
    # sa_pairs = []
    # for (s, _prob) in b.items():
    #     b_next = Pmax[s][a]
    #     this_normalisation_constraint = ""
    #     for (s_next, _prob) in b_next.items():
    #         # Add variable
    #         variable_name = "P_{}_{}".format(s,s_next)
    #         sa_pairs.append(solver.NumVar(0,1, variable_name))
    #         # Set up constraints for this P
    #         solver.Add("{} >= {}".format(variable_name, Pmin[s][a]))
    #         solver.Add("{} <= {}".format(variable_name, Pmax[s][a]))
    #         this_normalisation_constraint += "{}+".format(variable_name)
    #         # Set up goal
    #         for action in range(action_names):
    #             Goal += "{} * {} +".format(action, variable_name)
            
            
            
    #     this_normalisation_constraint = this_normalisation_constraint[:-1] + ">=1"
    #     Normalising_constraints.append(this_normalisation_constraint)
            
    # status = solver.Solve()
    # print(status)
        
        
    
    # states, b_array = [], []
    # for (state, prob) in b.items():
    #     states.append(state), b_array.append(prob)
        
        
    
    
    # done = False
    # states_left = states.copy()
    # while not done:
    #     a_b = np.argmax(Q_next)
        
    #     best_current_s = np.argmax()
        
    #     best_next_s, best_next_q   = np.argmax(Q[states_left,a_b]), np.max(Q[states_left,a_b])
        
        
    #     worst_s, worst_q = np.argmin(Q[states_left,a_b]), np.min(Q[states_left,a_b])
        
    #     change_in_b_to_try = 
        
    
    