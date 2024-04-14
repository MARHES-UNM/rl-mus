#!/bin/bash

python run_experiment.py --name rebaseline_85 --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 30000000
