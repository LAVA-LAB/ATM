# AM_Gyms (BAM-QMDP Repository)

This folder contains the following RL-environments (in openAI Gym format):

- Frozen_lake.py      : a copy of the standard openAI Gym environment with the same name;
- Frozen_lake_v2.py   : a 'less random' version of OpenAI's frozen lake environment;
- Loss_Env.py         : a custom environment designed to test Measurement Regret;
- NChainEnv.py        : an implementation of the (discontinued) OpenAI chain environment;
- Blackjack.py        : a version of OpenAI's Blackjack environment, which returns non-factorised states;
- Sepsis folder       : an RL-environment introduced by Nam et al (2021) for Active Measure reinforced learning.

Furthermore, it contains the following additional files:
- AM_Env_wrapper        : a wrapper class to add measuring functionality to openAI gyms.
- ModelLearner.py       : a class to learn the dynamics of an environment (i.e. transition function, rewards, done-states & Q-values) using method described by Delage and Mannor (2010).

For the first three environments mentioned above, data can be found in the data folder. The others were only used for testing.
