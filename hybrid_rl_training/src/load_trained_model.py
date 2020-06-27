#!/usr/bin/env python

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from openai_ros.task_envs.turtlebot2.turtlebot2_maze import TurtleBot2MazeEnv
import rospy
import os
from customPolicy import *
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
    rospy.init_node('stable_training', anonymous=True, log_level=rospy.WARN)
    env_temp = TurtleBot2MazeEnv
    env = SubprocVecEnv([lambda k=k:env_temp(world_file, k) for k in range(int(number_of_robots))])
    model = PPO2.load("ppo2_turtlebot")

    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(900):
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        # if done or info.get('is_success', False):
        #     print("Reward:", episode_reward, "Success?", info.get('is_success', False))
        #     episode_reward = 0.0
        #     obs = env.reset()