#!/usr/bin/env python

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from openai_ros.task_envs.turtlebot2 import turtlebot2_maze
import rospy
import os

rospy.init_node('stable_training', anonymous=True, log_level=rospy.WARN)

env = gym.make('turtlebot-v0')
model = PPO2(MlpPolicy, env, tensorboard_log="../PPO2_turtlebot_tensorboard/", verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_cartpole")
