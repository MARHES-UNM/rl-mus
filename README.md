# RL Multi-UAV Simulator

## Install Dependencies
conda create --name rl_mus_env --python=3.10
pip install matplotlib chardet qpsolvers quadprog seaborn

install pytorch
for CPU:
pip3 install torch torchvision torchaudio

for GPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

install ray:
pip install ray[rllib,tune]==2.6.3 --force

## TODO: 
Reduce state to just pos and velocity and compare performance.
[] add cbf for getting outside the environment
[x] set environment to match the lab, 1.75 x 1.75 x 1.5
[x] track mean dt_go instead of done_dt
[] use convolution model (objective)
[] create environment where goal is to reach target simultaneously as fast as possible (objective)
[] save best checkpoint after training