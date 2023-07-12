# ATM Repository

Repository containing code for ATM-Q (referred to here as BAM-QMDP), and gathered data, as used in the paper:

> Merlijn Krale, Thiago D. Simão, and Nils Jansen  
> Act-Then-Measure: Reinforcement Learning for Partially Observable Environments with Active Measuring  
> In ICAPS, 2023.

![teaser](https://github.com/lava-lab/ATM/blob/main/assets/teaser.gif?raw=true)

## Contents

This repository contains the following files:

Code:

  - **BAM_QMDP.py**           : The BAM-QMDP (a.k.a. (Dyna-)ATMQ) agent as a python class.
  - **Plot_Data.ipynb**       : Code for plotting data.
  - **Run.py**                : Code for automatically running agents on environments & recording their data.
  - **RunAll.sh**             : Bashfile for running all experiments at once.

Folders:

  - **AM_Gyms**             : Contains Gym environments used for testing, and wrapper class to make generic OpenAI envs into ACNO-MDP envs.
  - **Data**                : Contains gahtered date for BNAIC and ICAPS-paper (including analysed data & standard plots).
  - **Final_Plots**         : Contains compiled plots.
  - **Baselines**           : Contains code for all baseline algorithms used in the paper or in the testing phase.

## Getting started

After cloning this repository:

1. create a virtualenv and activate it
```bash
cd ATM/
python3 -m venv .venv
source .venv/bin/activate
```
2. install the dependencies
```bash
pip install -r requirements.txt
```

## How to run

All algorithms can be run using the Run.py file from command line. Running 'python Run.py -h' gives an overview of the functionaliality.

As an example, starting a run looks something like:

```bash
python Run.py -algo BAM_QMDP -env Lake -env_gen standard -env_size 8 -env_var semi-slippery -nmbr_eps 2500
```

This command runs the BAM-QMDP algorithm on the 8x8 semi-slippery lake environment for 2500 episodes (1 run), then it saves the results in the 'Data' folder.
For convenience, all experiments used in the paper are combined in a bashfile, which can be called using './RunAll.sh'.
