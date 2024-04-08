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
[x] use convolution model (objective)
[] create environment where goal is to reach target simultaneously as fast as possible (objective)
    This can use the to drive the cumulative t_go_estimate error to zero
[] write collision reward as exponential function
[] save best checkpoint after training


### RelRen Environment
[] have target sampled from uniform sphere using this equation:
https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/


$$ 1 / n \sum_j^n \text{abs} (t_{\text{go},i} - t_{\text{go},j})$$