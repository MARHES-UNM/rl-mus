# Reinforcement Multi-UAV sim

Mostly taken from master's thesis [Autonomous Navigation and Tracking of Dynamic Surface Targets On-board a Computationally Impoverished Aerial Vehicle](https://s3-us-west-2.amazonaws.com/selbystorage/wp-content/uploads/2016/05/WCSelbyMSThesisFinal.pdf) By William Clayton Selby


The repository uses a trajectory planner of a quintic function. https://en.wikipedia.org/wiki/Quintic_function
https://github.com/AtsushiSakai/PythonRobotics

this paper shows how to model a quadrotor using eular lagrange equations. 
https://sal.aalto.fi/publications/pdf-files/eluu11_public.pdf

they also added Aerodynamical effects to the equation. 

A simulation use RL to flip Quadcopters https://github.com/nikhilkalige/quadrotor

References this [paper](https://arxiv.org/pdf/2111.03915.pdf)

they trained a quadrotor with RL to hover. 
https://github.com/adipandas/gym_multirotor

Use this model for quadrotor simulation: 
https://andrew.gibiansky.com/downloads/pdf/Quadcopter%20Dynamics,%20Simulation,%20and%20Control.pdf  

https://andrew.gibiansky.com/blog/physics/quadcopter-dynamics/


https://github.com/gibiansky/experiments/blob/master/quadcopter/matlab/simulate.m. 
b = drag coefficient
https://github.com/abhijitmajumdar/Quadcopter_simulator

https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=2324&context=facpub

https://andrew.gibiansky.com/blog/physics/quadcopter-dynamics/

use this matlab code to simulate dynamics
https://github.com/gibiansky/experiments/blob/master/quadcopter/matlab/simulate.m


Low level LQR controller
https://github.com/sundw2014/Quadrotor_LQR

## References: 
https://github.com/NishanthARao/Python-Quadrotor-Simulation
https://github.com/AtsushiSakai/PythonRobotics

http://www.kostasalexis.com/simulations-with-simpy.html

Another drone simulation using python gym:
https://github.com/SuhrudhSarathy/drone_sim
git@github.com:NishanthARao/Python-Quadrotor-Simulation.git

# TODO: implement PID controller from UPenn class
* take a look at page 13 of paper: https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y

# use the dynamics in this repo for the drones. Follow the rotation matrix
git@github.com:AtsushiSakai/PythonRobotics.git

# use the gui class in this repo
https://github.com/abhijitmajumdar/Quadcopter_simulator

# derive the lqr k matrix from this repo
https://github.com/sundw2014/Quadrotor_LQR


## MOdeling the UAV
https://andrew.gibiansky.com/blog/physics/quadcopter-dynamics/

the thesis below has different methods for linearization of of the UAV dynamic.
[Implementation and comparison of linearization-based and backstepping controllers for quadcopters](https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y)

[Autonomous Navigation and Tracking of Dynamic Surface Targets On-board a Computationally Impoverished Aerial Vehicle ](https://s3-us-west-2.amazonaws.com/selbystorage/wp-content/uploads/2016/05/WCSelbyMSThesisFinal.pdf)

Use the paper from the Upen Vijay Kumar class
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5569026&casa_token=Ld80GMUrC8cAAAAA:k2y_jt-vTcrl2pOba5m7_29nsADbio0zTnsMUQTGs7aSwVVenSw-xVq1_-7cH-P_TcNPkyHXqNqAvQ&tag=1

## Linearization Techniques for UAVs:
[LQR controller design for quad-rotor helicopters](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/joe.2018.8126)

use linearization from this paper:Different Linearization Control Techniques for a Quadrotor System
https://ieeexplore-ieee-org.libproxy.unm.edu/stamp/stamp.jsp?tp=&arnumber=6417914

Look to using runge-kutta 4 method to simulate step
https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/

Look at this dissertation for the control and trajectory tracking: 
https://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations

Linear Quadratic Regulator for Trajectory Tracking of a Quadrotor
https://reader.elsevier.com/reader/sd/pii/S2405896319311450?token=A623301C4D79726D5E878D5ED45FDEB9B09B01E091E8D096636924105BB74069114A9518838AF42A95C14A4BC20156C7&originRegion=us-east-1&originCreation=20230505012459

## Trajectory Generation
https://repository.upenn.edu/cgi/viewcontent.cgi?article=1705&context=edissertations

Prediction-Based Leader-Follower Rendezvous Model Predictive Control with Robustness to Communication Losses
https://arxiv.org/pdf/2304.01045.pdf


## another repo
This repo is good for cbf for quadrotor
https://github.com/hocherie/cbf_quadrotor
accompanying pdf
https://github.com/hocherie/cbf_quadrotor/blob/master/docs/ensuring-safety-pdf.pdf


### Optimal Control 
https://motion.cs.illinois.edu/RoboticSystems/OptimalControl.html
http://e.guigon.free.fr/rsc/book/BrysonHo75.pdf
lqr controllers:
http://www.mwm.im/lqr-controllers-with-python/
https://github.com/ssloy/tutorials/blob/master/tutorials/pendulum/lqr.py
https://stanford.edu/class/ee363/lectures/clqr.pdf

### solving the time optimal control problem 
1. First create a runge_kata4 function
https://www.youtube.com/watch?v=1FYrnwqWQNY
https://www.geeksforgeeks.org/runge-kutta-4th-order-method-solve-differential-equation/
2. solve the Algebraic Ricatti solution problem
https://www.dynamicpublishers.com/Neural/NPSC2007/08-NPSC-2007-125-136.pdf
Solution of matrix Riccati differential equation for the linear quadratic singular system using neural networks


LQR example:
https://github.com/kowshikchills/LQR_python/blob/main/LQR.ipynb
https://ivanpapusha.com/cds270/lectures/05_DynProgLQR.pdf

#### TODO: turn discrete lqr below to continuous by discretizing the continuous space. 
``` python
def lqr(actual_state_x, desired_state_xf, Q, R, A, B, dt):
    x_error = actual_state_x - desired_state_xf
    N = 100
    P = [None] * (N + 1)
    Qf = Q
    P[N] = Qf
    for i in range(N, 0, -1):
        P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
            R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
    K = [None] * N
    u = [None] * N
    for i in range(N):
        K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
        u[i] = K[i] @ x_error
    u_star = u[N-1]
    return u_star
```
 



Calculate body rates to angle rates: 
https://ocw.mit.edu/courses/2-154-maneuvering-and-control-of-surface-and-underwater-vehicles-13-49-fall-2004/bc67e15b31b4f30aceabef2a66a6229d_lec1.pdf


## Create Neural Network CBF

Construct a network
crate a safe mask
create an unsafe mask

create a linear approximation of the system to sample

$ C = \{ x \in R^n: h(x) \geq 0 \}, $
$ \partial C = \{ x \in R^n: h(x) = 0 \}, $
$ \text{Int}( C ) = \{ x \in R^n: h(x) < 0 \}, $

$\partial C$ is the boundary of the set and $\text{Int}(C)$ is the interial of the set. 

Create loss function: 
``` python 
    loss = (
        0.0
        + h_safe_weight * (F.relu(eps - h_safe)).mean()
        + h_unsafe_weight * (F.relu(eps + h_unsafe)).mean()
        + h_deriv_weight * (F.relu(eps - h_deriv)).mean()
        + 0.01 * ((u - u_nom) ** 2).mean()
    )
```

$\dot h$ is difficult to approximate, instead use a linear approximation. Start with the linear model `f_linear`. Get the next state. 

See : https://github.com/MIT-REALM/sablas/blob/main/modules/trainer.py#L176 for more info.


# implementation of this paper
# https://arxiv.org/abs/2206.03568
# see example: 
# https://github.com/ChoiJangho/CBF-CLF-Helper/blob/master/demos/run_clf_simulation_inverted_pendulum.m
# impolement this from scratch
# https://github.com/MIT-REALM/neural_clbf/blob/main/neural_clbf/systems/inverted_pendulum.py
# https://yachienchang.github.io/NeurIPS2019/

# https://github.com/YaChienChang/Neural-Lyapunov-Control
# https://openreview.net/pdf?id=8K5kisAnb_p


To reference:
https://github.com/AgrawalAmey/safe-explorer
https://github.com/zisikons/deep-rl/tree/main
https://github.com/MIT-REALM/sablas


## TODO: 
update to use ray air session: https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
reference this paper: https://ieeexplore.ieee.org/document/9849119/authors#authors
another cbf resource: https://escholarship.org/uc/item/9dp7v842#page=119
possibly use this as a source: https://ieeexplore-ieee-org.libproxy.unm.edu/stamp/stamp.jsp?tp=&arnumber=6393426

* Review difference between weight decay and regularization
* implement rk4 from scratch [done]
https://github.com/mahaitongdae/Safety_Index_Synthesis/blob/master/envs_and_models/collision_avoidance_env.py
* implement custom ppo for multi agent system
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_continuous_action.py
https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py
* refactor uav class to be modular [done]
* update uav reward to be more in line with: https://arc.aiaa.org/doi/epdf/10.2514/1.I010961


In order to fix this problem, do the following:

1) Run `pip install gymnasium` on your command line.
2) Change all your import statements in your code from
   `import gym` -> `import gymnasium as gym` OR
   `from gym.space import Discrete` -> `from gymnasium.spaces import Discrete`

For your custom (single agent) gym.Env classes:
3.1) Either wrap your old Env class via the provided `from gymnasium.wrappers import
     EnvCompatibility` wrapper class.
3.2) Alternatively to 3.1:
 - Change your `reset()` method to have the call signature 'def reset(self, *,
   seed=None, options=None)'
 - Return an additional info dict (empty dict should be fine) from your `reset()`
   method.
 - Return an additional `truncated` flag from your `step()` method (between `done` and
   `info`). This flag should indicate, whether the episode was terminated prematurely
   due to some time constraint or other kind of horizon setting.

For your custom RLlib `MultiAgentEnv` classes:
4.1) Either wrap your old MultiAgentEnv via the provided
     `from ray.rllib.env.wrappers.multi_agent_env_compatibility import
     MultiAgentEnvCompatibility` wrapper class.
4.2) Alternatively to 4.1:
 - Change your `reset()` method to have the call signature
   'def reset(self, *, seed=None, options=None)'
 - Return an additional per-agent info dict (empty dict should be fine) from your
   `reset()` method.
 - Rename `dones` into `terminateds` and only set this to True, if the episode is really
   done (as opposed to has been terminated prematurely due to some horizon/time-limit
   setting).
 - Return an additional `truncateds` per-agent dictionary flag from your `step()`
   method, including the `__all__` key (100% analogous to your `dones/terminateds`
   per-agent dict).
   Return this new `truncateds` dict between `dones/terminateds` and `infos`. This
   flag should indicate, whether the episode (for some agent or all agents) was
   terminated prematurely due to some time constraint or other kind of horizon setting.


TODO: 
watch this video for hints on clipping the gradient
https://www.youtube.com/watch?v=OAKAZhFmYoI


## TODO: 
* use curriculum learning to get better results
use this example from [Rllib](https://github.com/ray-project/ray/blob/ray-2.6.3/rllib/examples/curriculum_learning.py) to train the agents to define a curriculum function
*  create a wrapper around the multiagent environment to and make it harder and harder each task. 
https://github.com/ray-project/ray/blob/ray-2.6.3/rllib/examples/env/curriculum_capable_env.py
* an example of tasks could be: 
1. define a large radius around the target area
2. reduce the radius around the target area till we get to 0.01
3. now add penalty for getting to the target area too soon

4. use equation to convert from x,yz to spherical coordinate: 
https://en.wikipedia.org/wiki/Spherical_coordinate_system
https://qiita.com/7of9/items/ceea99b35b58a47e5911


Take a look at this paper on MASAC, https://www.frontiersin.org/articles/10.3389/fnbot.2022.932671/full
maybe there's something here to learn and compare against MAPPO.

https://discuss.ray.io/t/how-to-implement-curriculum-learning-as-in-narvekar-and-stone-2018/3124


access model state like so: 
https://docs.ray.io/en/latest/rllib/rllib-training.html

policy_checkpoint = r"/home/prime/Documents/workspace/rl_multi_uav_sim/results/PPO/multi-uav-sim-v0_2023-10-30-05-59_6341c86/beta_0_3_pen_5/PPO_multi-uav-sim-v0_161e9_00003_3_beta=0.3000,d_thresh=0.2000,obstacle_collision_weight=5.0000,tgt_reward=300.0000,uav_collision__2023-10-30_05-59-59/checkpoint_000454/policies/shared_policy"

algo = Policy.from_checkpoint(policy_checkpoint)
https://github.com/ray-project/ray/blob/ray-2.6.3/rllib/examples/restore_1_of_n_agents_from_checkpoint.py

https://discuss.ray.io/t/how-to-use-my-pretrained-model-as-policy-and-value-netwok/11040
https://discuss.ray.io/t/updating-policy-mapping-fn-while-using-tune-run-and-restoring-from-a-checkpoint/11018/4


Joint Synthis of Safety index: 
https://proceedings.mlr.press/v168/ma22a/ma22a.pdf
working on windows: 
https://www3.ntu.edu.sg/home/ehchua/programming/howto/Environment_Variables.html#:~:text=2.2%20Set%2FUnset%2FChange%20an,it%20to%20an%20empty%20string.
"env": {"PL_TORCH_DISTRIBUTED_BACKEND":"gloo"},


custom train function: 
https://github.com/ray-project/ray/blob/ray-2.6.3/rllib/examples/custom_train_fn.py
