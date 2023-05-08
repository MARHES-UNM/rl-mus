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