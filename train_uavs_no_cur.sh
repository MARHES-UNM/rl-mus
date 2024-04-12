#!/bin/bash

python run_experiment.py --name lower_max_std --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000
