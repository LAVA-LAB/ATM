#!/bin/bash

# This file runs all experiments as used in the ICAPS23 paper, and stores them in the data-folder. For creating plots, see the Plot_Data.ipynb
# Note: in reality, all data has been gathered by manually starting runs. Thus, this code is meant more as a reference and might take a very long time to run!

# Experiments on Measuring value environment, for different measuring values:
echo -e "\n\n============= MEASURING VALUE ENV =============\n\n"
eps=1000
runs=5
for cost in $(seq 0.05 0.01 0.2)
do
    python3 ./Run.py -alg AMRL      -env Loss -nmbr_eps $eps -nmbr_runs $runs -m_cost $cost
    python3 ./Run.py -alg BAM_QMDP  -env Loss -nmbr_eps $eps -nmbr_runs $runs -m_cost $cost
    python3 ./Run.py -alg BAM_QMDP+ -env Loss -nmbr_eps $eps -nmbr_runs $runs -m_cost $cost
    python3 ./Run.py -alg ACNO_OTP  -env Loss -nmbr_eps $eps -nmbr_runs $runs -m_cost $cost
done

# Experiments on small frozen lake, for different variants:
echo -e "\n\n============= Small Frozen Lake =============\n\n"
eps=1000
runs=5
for variant in "det" "semi-slippery" "slippery"
do
    python3 ./Run.py -alg AMRL      -env Lake -env_var $variant -env_gen standard -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 4
    python3 ./Run.py -alg BAM_QMDP  -env Lake -env_var $variant -env_gen standard -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 4
    python3 ./Run.py -alg BAM_QMDP+ -env Lake -env_var $variant -env_gen standard -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 4
    python3 ./Run.py -alg ACNO_OTP  -env Lake -env_var $variant -env_gen standard -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size 4
done

# Experiments on increasingly larger frozen lake environments
echo -e "\n\n============= Large Frozen Lake =============\n\n"
eps=7500
runs=5
for size in $(seq 8 1 20)
do
    python3 ./Run.py -alg AMRL      -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size $size
    python3 ./Run.py -alg BAM_QMDP  -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size $size
    python3 ./Run.py -alg BAM_QMDP+ -env Lake -env_var semi-slippery -env_gen random -m_cost 0.05 -nmbr_eps $eps -nmbr_runs $runs -env_size $size
done

echo -e "\n\n============= RUNS COMPLETED =============\n\n"