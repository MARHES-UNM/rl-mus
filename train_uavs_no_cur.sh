#!/bin/bash

python run_experiment.py --name replicate_dc8b9b2_02_05 --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
