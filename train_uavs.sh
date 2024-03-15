#!/bin/bash

python multi_agent_shared_parameter.py --name mulagen_par --stop-timesteps 20000000
# python train_agent.py --name cur_l --stop-timesteps 20000000
python run_experiment.py --name run_exp --stop-timesteps 20000000