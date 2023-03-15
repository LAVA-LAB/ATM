#########################################
#       Data classes for POMDP:         #
#########################################

from pomdpy.discrete_pomdp import DiscreteAction
from pomdpy.discrete_pomdp import DiscreteState
from pomdpy.discrete_pomdp import DiscreteObservation
from pomdpy.pomdp import HistoricalData

class PositionData(HistoricalData):
    def __init__(self, model, position, solver):
        self.model = model
        self.solver = solver
        self.position = position
        self.legal_actions = self.generate_legal_actions 

    def generate_legal_actions(self, includeMeasuring = True):
        legal_actions = []
        if includeMeasuring: 
            for i in range(self.model.actions_n):
                legal_actions.append(i)
        else: #as used for rollouts:
            for i in range(self.model.CActionSize):
                legal_actions.append(i+self.model.CActionSize)
        return legal_actions

    def shallow_copy(self):
        return PositionData(self.model, self.position, self.solver)
    
    def copy(self):
        return self.shallow_copy()

    def update(self, other_belief):
        pass

    def create_child(self, action, observation):
        next_data = self.copy() # deep/shallow copy
        next_position, is_legal = self.model.make_next_position(self.position, action.bin_number)
        next_data.position = next_position
        return next_data

class BoxState(DiscreteState):
    def __init__(self, position, is_terminal=False, r=None):
        self.position = position
        self.terminal = is_terminal
        self.final_rew = r

    def __eq__(self, other_state):
        return self.position == other_state.position

    def copy(self):
        return BoxState(self.position, 
                        is_terminal=self.terminal, 
                        r=self.final_rew)
        
    def to_string(self):
        return str(self.position)

    def print_state(self):
        pass

    def as_list(self):
        pass

    def distance_to(self):
        pass

class BoxAction(DiscreteAction):
    def __init__(self, bin_number):
        self.bin_number = bin_number

    def __eq(self, other_action):
        return self.bin_number == other_action.bin_number
    
    def __ge__(self, other_action):
        return self.bin_number > other_action.bin_number

    def print_action(self):
        pass

    def to_string(self):
        return str(self.bin_number)

    def distance_to(self):
        pass

    def copy(self):
        return BoxAction(self.bin_number)

class BoxObservation(DiscreteObservation):
    def __init__(self, position):
        self.position = position
        self.bin_number = position

    def __eq__(self, other_obs):
        return self.position == other_obs.position

    def copy(self):
        return BoxObservation(self.position)

    def to_string(self):
        return str(self.position)

    def print_observation(self):
        pass

    def as_list(self):
        pass

    def distance_to(self):
        pass