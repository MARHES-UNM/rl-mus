#!/bin/bash

python run_experiment.py --name high_dt_go_reward --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
