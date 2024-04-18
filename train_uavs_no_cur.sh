#!/bin/bash

python run_experiment.py --name rand_stp --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 15000000
