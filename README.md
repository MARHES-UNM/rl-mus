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

# TODO: implement PID controller from UPenn class
* take a look at page 13 of paper: https://upcommons.upc.edu/bitstream/handle/2117/112404/Thesis-Jesus_Valle.pdf?sequence=1&isAllowed=y


