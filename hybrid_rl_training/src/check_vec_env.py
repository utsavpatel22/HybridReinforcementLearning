#!/usr/bin/env python
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import gym
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.vec_env import SubprocVecEnv
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from openai_ros.task_envs.turtlebot2.turtlebot2_maze import TurtleBot2MazeEnv
import rospy
import random 


if __name__ == '__main__':
	k_n=random.randrange(20, 50, 3)
	rospy.init_node('turtlebot_gym'+str(k_n), anonymous=True, log_level=rospy.DEBUG)
	env_temp = TurtleBot2MazeEnv
	print("If the env is an instance of gym {} /////******//".format((env_temp.__bases__)))
	print("If the env is an instance of gym {} /////******//".format(isinstance(env_temp, gym.Env)))
	env = SubprocVecEnv([lambda k=k:env_temp(k) for k in range(1)])
	
	check_env(env)

