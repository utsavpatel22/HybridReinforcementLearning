#!/usr/bin/env python

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import gym
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
from openai_ros.task_envs.turtlebot2.turtlebot2_maze import TurtleBot2MazeEnv
import rospy
import os
from customPolicy import *
from sensor_msgs.msg import LaserScan
import time
# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[144, 144, 144],
                                                          vf=[144, 144, 144])],
                                           feature_extraction="mlp")


if __name__ == '__main__':
    world_file = sys.argv[1]
    number_of_robots = sys.argv[2]
    robot_number = sys.argv[3] # Provide robot number to subscribe to the correct topic  
    max_steps = 900
    max_test_episodes = 50
    min_range = 0.5 # Refer Task environment to get the value of min range
    rospy.init_node('stable_training', anonymous=True, log_level=rospy.WARN)
    env_temp = TurtleBot2MazeEnv
    env = SubprocVecEnv([lambda k=k:env_temp(world_file, k) for k in range(int(number_of_robots))])
    model = PPO2.load("ppo2_turtlebot")

    counter = 0
    collisions = 0
    start_time = rospy.get_time()
    episode_time_list = []
    while(counter < max_test_episodes):
        obs = env.reset()
        # Evaluate the agent
        episode_reward = 0
        for _ in range(max_steps):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            
            if (done):
                print("Done")
                counter += 1
                total_time_episode = rospy.get_time() - start_time
                print("The total time is {}".format(total_time_episode))
                episode_time_list.append(total_time_episode)
                start_time = rospy.get_time()
                break

    print("Total number of collisions {}".format(collisions))
    print(episode_time_list)
        
