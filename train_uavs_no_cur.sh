#!/bin/bash

python run_experiment.py --name replicate_66c4c18_train_bounds --run PPO \
    train \
    --cpu 8 \
    --gpu 0.5 \
    --stop_timesteps 35000000
