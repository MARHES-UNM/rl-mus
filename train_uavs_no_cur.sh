#!/bin/bash

python run_experiment.py --name train_subtract_states --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
