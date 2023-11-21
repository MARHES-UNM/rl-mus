#!/bin/bash


python multi_agent_shared_parameter.py --name low_stp --stop-timesteps 20000000
python train_agent.py --name low_stp --stop-timesteps 20000000