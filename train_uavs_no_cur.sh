#!/bin/bash

python run_experiment.py --name baseline_dt_1_0_stp_reward_mean --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000
