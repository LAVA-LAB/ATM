# Wrapper to turn Open AI Gym-environments into active measure environments
import numpy as np
import math as m
import matplotlib.pyplot as plt


class AM_ENV():
    """Wrapper class for openAI-environments for AM algorithms.
    
    Most imporantly, changes step-function to not return an observation,
    and adds a seperate observe-function.
    """

    def __init__(self, env, StateSize, ActionSize, MeasureCost, s_init, log_choices = False, max_steps = 10_000, max_reward = 1):
        self.env = env
        self.StateSize = StateSize
        self.ActionSize = ActionSize    # Is there any way to get these two from the env. itself?
        self.MeasureCost = MeasureCost
        self.s_init = s_init
        self.obs = 0
        self.max_steps = max_steps
        self.steps_taken = 0
        self.reward_factor = 1.0 / max_reward   # makes sure rewards are always 'normalised'

        self.log_choices = log_choices
        if self.log_choices:
            self.choiceTable    = np.zeros( (self.StateSize, self.ActionSize) )
            self.densityTable   = np.zeros( (self.StateSize) )  # This is also just the sum over actions of choice table, but whatever...
            self.accuracyTable  = np.zeros( (self.StateSize) )


    #######################################################
    ###          Environment Wrapper code:              ###
    #######################################################
    def get_vars(self):
        "Returns StateSize, ActionSize, MeasureCost and s_init (-1 if random)"
        return (self.StateSize, self.ActionSize, self.MeasureCost, self.s_init)
    
    def step(self, action, s = None):
        "Perform action on environment, without returning an observation"
        (obs, reward, done, info) = self.env.step(action)
        self.obs = obs
        reward = reward * self.reward_factor

        if done:
            self.obs = 0

        # Log action (if turned on):
        if ( (s == None) & self.log_choices ):
            done
            #print("Warning: Logger is turned on, but not all required arguments are given. No logging will be performed.")
        elif self.log_choices:
            self.log_action(action, obs, s)
        
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            done = True

        return (reward, done)

    def measure(self): #For full version should include m as argument
        'Returns current state of environment'
        return (self.obs, self.MeasureCost)
 
    def reset(self):
        self.env.reset()
        self.steps_taken = 0
        
    def getname(self):
        return self.env.getname()


    #######################################################
    ###                 Logging Code:                   ###
    #######################################################

    # Used for debugging Agents running on Frozen Lake environment.

    def log_action(self, action, obs, s):

        self.choiceTable[obs,action] += 1
        self.densityTable[obs] += 1
        if obs in s:
            self.accuracyTable[obs] = ( self.accuracyTable[obs] * (self.densityTable[obs]-1) + s[obs] ) / self.densityTable[obs]
        else: 
            self.accuracyTable[obs] = self.accuracyTable[obs] * (self.densityTable[obs]-1) / self.densityTable[obs]


class AM_Visualiser(): # Assuming a grid!
    """Class for visualising results on Frozen Lake environments.
    Used for testing only, would not recommend using!"""

    # Winter: Blue  = low, green = high

    def __init__(self, env_wrapper, agent):
        self.StateSize  = agent.StateSize
        self.gridSize   = m.ceil(m.sqrt(self.StateSize))

        self.QTable = agent.QTable

        self.density = env_wrapper.densityTable
        self.accuracy = env_wrapper.accuracyTable
        self.choice = env_wrapper.choiceTable
        return 

    def __action_to_symbol__(self, action):
        match action:
            case 0: return '<'
            case 1: return '.'
            case 2: return '>'
            case 3: return '^'

    def plot_choice_certainty(self):

        # Gather data:
        choice, certainty = np.zeros(self.StateSize, dtype=np.int8), np.zeros(self.StateSize)
        for i in range(self.StateSize):
            choice[i] = np.argmax(self.QTable[i])
            certainty[i] = self.QTable[i,choice[i]] - np.max(np.delete(self.QTable[i], choice[i]))
        
        certainty   = certainty / np.max(certainty) #normalise
        choice      = np.vectorize(self.__action_to_symbol__)(choice)
        
        choice      = choice.reshape(self.gridSize, self.gridSize)
        certainty   = certainty.reshape(self.gridSize, self.gridSize)

        # Create Plot:
        plt.axis([-0.5,self.gridSize-0.5, -0.5, self.gridSize-0.5])
        plt.imshow(np.flipud(certainty), cmap='winter')
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                plt.text(y,x,np.flipud(choice)[x,y])
        
        plt.savefig("Test_choice_certainty")
        plt.clf()


    def plot_choice_density(self):
        choice = np.zeros(self.StateSize, dtype=np.int8)
        for i in range(self.StateSize):
            choice[i] = np.argmax(self.QTable[i])
        
        choice      = np.vectorize(self.__action_to_symbol__)(choice)
        
        choice      = choice.reshape(self.gridSize, self.gridSize)
        print(self.density)
        density     = self.density / np.max([np.max(self.density),1])
        density     = density.reshape(self.gridSize, self.gridSize)

        # Create Plot:
        plt.axis([-0.5,self.gridSize-0.5, -0.5, self.gridSize-0.5])
        plt.imshow(np.flipud(density),cmap='winter')
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                plt.text(y,x,np.flipud(choice)[x,y])
        
        plt.savefig("Test_choice_density")
        plt.clf()

    def plot_choice_maxQ(self):
        choice, maxQ = np.zeros(self.StateSize, dtype=np.int8), np.zeros(self.StateSize)
        for i in range(self.StateSize):
            choice[i] = np.argmax(self.QTable[i])
        
        choice  = np.vectorize(self.__action_to_symbol__)(choice)
        maxQ    = np.amax(self.QTable,1)

        choice  = choice.reshape(self.gridSize, self.gridSize)
        maxQ    = maxQ.reshape(self.gridSize, self.gridSize)

        # Create Plot:
        plt.axis([-0.5,self.gridSize-0.5, -0.5, self.gridSize-0.5])
        plt.imshow(np.flipud(maxQ),cmap='winter')
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                plt.text(y,x,np.flipud(choice)[x,y])
        
        plt.savefig("Test_choice_maxQ")
        plt.clf()
    
    def plot_choice_state_accuracy(self):
        choice, acc = np.zeros(self.StateSize, dtype=np.int8), np.zeros(self.StateSize)
        choice = np.argmax(self.QTable, 1)
        acc = self.accuracy
        
        choice  = np.vectorize(self.__action_to_symbol__)(choice)

        choice  = choice.reshape(self.gridSize, self.gridSize)
        acc    = acc.reshape(self.gridSize, self.gridSize)

        # Create Plot:
        plt.axis([-0.5,self.gridSize-0.5, -0.5, self.gridSize-0.5])
        plt.imshow(np.flipud(acc),cmap='winter')
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                plt.text(y,x,np.flipud(choice)[x,y])
        
        plt.savefig("Test_choice_accuracy")
        plt.clf()