#!/bin/bash

python run_experiment.py --name replicate_66c4c18 --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 20000000
