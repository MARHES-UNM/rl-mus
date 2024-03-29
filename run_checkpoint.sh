#!/bin/bash

CHECKPOINT="checkpoints/cur_learning/policies/shared_policy"

python run_experiment.py test --checkpoint $CHECKPOINT --render --plot_results