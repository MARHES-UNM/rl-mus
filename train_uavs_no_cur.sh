#!/bin/bash

python run_experiment.py --name baseline_1_5_stp_reward --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000
