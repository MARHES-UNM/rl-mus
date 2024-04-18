#!/bin/bash

python run_experiment.py --name dt_go_fix_rew_long --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
