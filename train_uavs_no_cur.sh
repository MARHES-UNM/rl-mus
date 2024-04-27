#!/bin/bash

python run_experiment.py --name test_max_std_dt --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
