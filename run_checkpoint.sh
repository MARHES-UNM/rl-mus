#!/bin/bash

CHECKPOINT="checkpoints/cur_learning/policies/shared_policy"

python run_experiment.py --run PPO --tf 4 test \
    --checkpoint $CHECKPOINT \
    --render \
    --plot_results \
    --seed None
