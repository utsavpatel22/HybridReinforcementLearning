# HybridReinforcementLearning
Package that combines reinforcement learning with Dynamic Window Approach 


# Training iterations 

1) Trained wirth discretized lidar data and policy converged to a stable reward value

2) Traind with DWA costs as input and rewarded for going towards goal and reaching the goal. It was penalized for colliding, diverging from goal and choosing velocities those are not feasible. In this case the model never converged. It might be because of the dynamic nature of the observations and the action space. 

3) In the 3rd iteration, I am sorting the DWA costs so that the network will have a structured input. This training also did not converge. 

4) Now rewarding the robot for going towards goal, executing velocity with linear component. and penalizing for colliding with something. 