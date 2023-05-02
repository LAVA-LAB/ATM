'''
File for running & gathering data on Active-Measuring algorithms.
For a brief description of how to use it, see the Readme-file in this repo.

'''

######################################################
        ###             Imports                 ###
######################################################


# File structure stuff
import sys
import os
sys.path.append(os.path.join(sys.path[0],"Baselines"))
sys.path.append(os.path.join(sys.path[0],"Baselines", "ACNO_generalised"))

# External modules
import numpy as np
import gym
import matplotlib.pyplot as plt
import time as t
import datetime
import json
import argparse
from typing import List, Optional
import os

# Agents
import Baselines.AMRL_Agent as amrl
from BAM_QMDP import BAM_QMDP
from Baselines.ACNO_generalised.Observe_then_plan_agent import ACNO_Agent_OTP
# from Baselines.DRQN import DRQN_Agent requires torch to be downloaded, so kept turned off
from Baselines.DynaQ import QBasic, QOptimistic, QDyna

# Environments
from AM_Gyms.NchainEnv import NChainEnv
from AM_Gyms.Loss_Env import Measure_Loss_Env
from AM_Gyms.frozen_lake_v2 import FrozenLakeEnv_v2
from AM_Gyms.Sepsis.SepsisEnv import SepsisEnv
from AM_Gyms.Blackjack import BlackjackEnv
from AM_Gyms.frozen_lake import FrozenLakeEnv, generate_random_map, is_valid

# Environment wrappers
from AM_Gyms.AM_Env_wrapper import AM_ENV as wrapper
from AM_Gyms.AM_Env_wrapper import AM_Visualiser as visualiser
from Baselines.ACNO_generalised.ACNO_ENV import ACNO_ENV

# JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

######################################################
        ###       Parsing Arguments           ###
######################################################

parser = argparse.ArgumentParser(description="Run tests on Active Measuring Algorithms")

# Defining all parser arguments:
parser.add_argument('-algo'             , default = 'AMRL',             help='Algorithm to be tested.')
parser.add_argument('-env'              , default = 'Lake',             help='Environment on which to perform the testing')
parser.add_argument('-env_var'          , default = 'None',             help='Variant of the environment to use (if applicable)')
parser.add_argument('-env_gen'          , default = None,               help='Size of the environment to use (if applicable)')
parser.add_argument('-env_size'         , default = 0,                  help='Size of the environment to use (if applicable)')
parser.add_argument('-m_cost'           , default = -1.0,               help='Cost of measuring (default: use as specified by environment)')
parser.add_argument('-nmbr_eps'         , default = 500,                help='nmbr of episodes per run')
parser.add_argument('-nmbr_runs'        , default = 1,                  help='nmbr of runs to perform')
parser.add_argument('-f'                , default = None,               help='File name (default: generated automatically)')
parser.add_argument('-rep'              , default = './Data/',          help='Repository to store data (default: ./Data')
parser.add_argument('-env_remake'       , default=True,                 help='Option to make a new (random) environment each run or not')
parser.add_argument('-save'             , default = True,               help='Option to save or not save data.')

# Unpacking for use in this file:
args             = parser.parse_args()
algo_name        = args.algo
env_name         = args.env
env_variant      = args.env_var
env_size         = int(args.env_size)
env_gen          = str(args.env_gen)
MeasureCost      = float(args.m_cost)
nmbr_eps         = int(args.nmbr_eps)
nmbr_runs        = int(args.nmbr_runs)
file_name        = args.f
rep_name         = args.rep
remake_env_opt   = True

if args.env_remake in  ["False", "false"]:
        remake_env_opt = False

if args.save == "False" or args.save == "false":
        doSave = False
else:
        doSave = True

# Create name for Data file
envFullName = env_name
if env_size != 0:
        envFullName += "_"+env_gen+str(env_size)

if env_variant != 'None':
        envFullName += "_"+env_variant

######################################################
        ###     Intitialise Environment        ###
######################################################

# Lake Envs
s_init                          = 0
MeasureCost_Lake_default        = 0.05
MeasureCost_Taxi_default        = 0.01 / 20
MeasureCost_Chain_default       = 0.05
remake_env                      = False
env_folder_name = os.path.join(os.getcwd(), "AM_Gyms", "Learned_Models")

def get_env(seed = None):
        "Returns AM_Env as specified in global (user-specified) vars"
        global MeasureCost
        global remake_env
        global env_size
        global env_full_name
        
        # Required for making robust env through generic-gym class
        has_terminal_state = True
        terminal_prob = 0.0
        
        np.random.seed(seed)
        
        # Basically, just a big messy pile of if/else statements (Not using match for pre 3.10 python users)
        
        # Loss-environment, called Measure Regret environment in paper.
        if env_name == "Loss":
                env = Measure_Loss_Env()
                StateSize, ActionSize, s_init = 4, 2, 0
                if MeasureCost == -1:
                        MeasureCost = 0.1
                
                # Frozen lake environment (includes all variants)
        elif env_name == "Lake":
                ActionSize, s_init = 4,0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Lake_default
                if env_size == 0:
                        print("Using standard size map (4x4)")
                        env_size = 4
                        StateSize = 4**2
                else:
                        StateSize = env_size**2
                        
                if env_gen == "random":
                        map_name = None
                        desc = generate_random_map(size=env_size)
                elif env_gen == "standard":
                                if env_size != 4 and env_size != 8:
                                        print("Standard map type can only be used for sizes 4 and 8")
                                else:
                                        map_name = "{}x{}".format(env_size, env_size)
                                        desc = None
                else:
                        print("Using random map")
                        map_name = None
                        desc = generate_random_map(size=env_size)
                                
                if map_name is None and remake_env_opt:
                        remake_env = True
                        
                if env_variant == "det":
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                elif env_variant == "slippery":
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=True)
                elif env_variant ==  "semi-slippery":
                        env = FrozenLakeEnv_v2(desc=desc, map_name=map_name)
                elif env_variant ==  None:
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                else: #default = deterministic
                        print("Environment var not recognised! (using deterministic variant)")
                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                        
                # Taxi environment, as used in AMRL-Q paper. Not used in paper           
        elif env_name == "Taxi":
                env = gym.make('Taxi-v3')
                StateSize, ActionSize, s_init = 500, 6, -1
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Taxi_default

                # Chain environment, as used in AMRL-Q paper. Not used in paper
        elif env_name == "Chain":
                if env_size == '10':
                        StateSize = 10
                elif env_size == '20':
                        StateSize = 20
                elif env_size == '30':
                        StateSize = 30
                elif env_size == '50':
                        StateSize = 50
                elif env_size == other: # default
                        print("env_map not recognised!")
                        StateSize = 20
                        
                env = NChainEnv(StateSize)
                ActionSize, s_init = 2, 0
                if MeasureCost == -1:
                                MeasureCost = MeasureCost_Chain_default
              
                # Sepsis environment, as used in ACNO-paper. Not used in paper
        elif env_name == 'Sepsis':
                env = SepsisEnv()
                StateSize, ActionSize, s_init = 720, 8, -1
                if MeasureCost == -1:
                        MeasureCost = 0.05

                # Standard OpenAI Gym blackjack environment. Not used in paper
        elif env_name == 'Blackjack':
                env = BlackjackEnv()
                StateSize, ActionSize, s_init = 704, 2, -1
                if MeasureCost ==-1:
                        MeasureCost = 0.05

        else:
                print("Environment {} not recognised, please try again!".format(env_name))
                return
                        
        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)
        args.m_cost = MeasureCost
        
        return ENV

######################################################
        ###     Defining Agents        ###
######################################################

# Both final names and previous/working names are implemented here
def get_agent(seed=None):
        
        ENV = get_env(seed)
        if algo_name == "AMRL":
                agent = amrl.AMRL_Agent(ENV, turn_greedy=True)
        # AMRL-Q, alter so it is completely greedy in last steps.
        elif algo_name == "AMRL_greedy":
                agent = amrl.AMRL_Agent(ENV, turn_greedy=False)
        # BAM_QMDP, named Dyna-ATMQ in paper. Variant with no offline training
        elif algo_name == "BAM_QMDP":
                agent = BAM_QMDP(ENV, offline_training_steps=0)
        # BAM_QMDP, named Dyna-ATMQ in paper. Variant with 25 offline training steps per real step
        elif algo_name == "BAM_QMDP+":
                agent = BAM_QMDP(ENV, offline_training_steps=25)
        # Observe-then-plan agent from ACNO-paper. As used in paper, slight alterations made from original
        elif algo_name == "ACNO_OTP":
                ENV_ACNO = ACNO_ENV(ENV)
                agent = ACNO_Agent_OTP(ENV_ACNO)
        # A number of generic RL-agents. We did not include these in the paper.
        # elif algo_name == "DRQN":
        #         agent = DRQN_Agent(ENV)
        elif algo_name == "QBasic":
                agent = QBasic(ENV)
        elif algo_name == "QOptimistic":
                agent = QOptimistic(ENV)
        elif algo_name == "QDyna":
                agent = QDyna(ENV)
        else:
                print("Agent {} not recognised, please try again!".format(algo_name))
        return agent

######################################################
        ###     Exporting Results       ###
######################################################

# Automatically creates filename is not specified by user
if file_name == None:
        file_name = 'AMData_{}_{}_{}.json'.format(algo_name, envFullName, str(int(float(args.m_cost)*100)).zfill(3))

# Set measurecost if not set by environment.
if args.m_cost == -1:
        args.m_cost == MeasureCost

def PR_to_data(pr_time):
        "Prints timecode as used in datafiles"
        return (datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=pr_time)).strftime("%d%m%Y%H%M%S")

def export_data(rewards, steps, measures,  t_start):
        "Exports inputted data, as well as user-set variables, to JSON file"
        with open(rep_name+file_name, 'w') as outfile:
                json.dump({
                        'parameters'            :vars(args),
                        'reward_per_eps'        :rewards,
                        'steps_per_eps'         :steps,
                        'measurements_per_eps'  :measures,
                        'start_time'            :t_start,
                        'current_time'          :t.perf_counter()
                }, outfile, cls=NumpyEncoder)


######################################################
        ###     Running Simulations       ###
######################################################

rewards, steps, measures = np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps))
t_start = 0 + t.perf_counter()
rewards_avg, steps_avg, measures_avg = np.zeros(nmbr_runs), np.zeros(nmbr_runs), np.zeros(nmbr_runs)
print("""
Start running agent with following settings:
Algorithm: {}
Environment: {}
nmbr runs: {}
nmbr episodes per run: {}.
""".format(algo_name, envFullName, nmbr_runs, nmbr_eps))

agent = get_agent(0)

for i in range(nmbr_runs):
        t_this_start = t.perf_counter()
        (r_tot, rewards[i], steps[i], measures[i]) = agent.run(nmbr_eps, True) 
        rewards_avg[i], steps_avg[i], measures_avg[i] =np.average(rewards[i]), np.average(steps[i]),np.average(measures[i])
        t_this_end = t.perf_counter()
        if doSave:
                export_data(rewards[:i+1],steps[:i+1],measures[:i+1],t_start)
        print("Run {0} done with average reward {2}! (in {1} s, with {3} steps and {4} measurements avg.)\n".format(i+1, t_this_end-t_this_start, rewards_avg[i], steps_avg[i], measures_avg[i]))
        if remake_env and i<nmbr_runs-1:
                agent = get_agent(i+1)
print("Agent Done! ({0} runs in {1} s, with average reward {2}, steps {3}, measures {4})\n\n".format(nmbr_runs, t.perf_counter()-t_start, np.average(rewards_avg), np.average(steps_avg),np.average(measures_avg)))
