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
import numpy as np
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
import csv
import time
from std_msgs.msg import Float32, Bool, Int64


td = 0
def getDistance(msg):
    global td
    td = msg.data

gr = False
def getGoalReachingStatus(msg):
    global gr
    gr = msg.data

ac = None
def getAction(msg):
    # print("In action callback")
    global ac 
    ac = msg.data

if __name__ == '__main__':
    world_file = sys.argv[1]
    number_of_robots = sys.argv[2]
    robot_number = sys.argv[3]
    rospy.init_node('action_subscriber', anonymous=True, log_level=rospy.WARN)
    # TotalDistance = rospy.Subscriber("/predicted_action",Float32, getAction)
    profiling = True
    max_steps = 900
    max_test_episodes = 1
    min_range = 0.5 # Refer Task environment to get the value of min range
    obs_publisher = rospy.Publisher('/obs_data', numpy_msg(Floats),queue_size=10)
    TotalDistance = rospy.Subscriber("/total_distance",Float32, getDistance)
    action_subscriber = rospy.Subscriber("/predicted_action", Int64, getAction)
    goal_reaching_status = rospy.Subscriber('/turtlebot'+str(robot_number)+'/goal_reaching_status', Bool, getGoalReachingStatus)
    env_temp = TurtleBot2MazeEnv
    env = SubprocVecEnv([lambda k=k:env_temp(world_file, k) for k in range(int(number_of_robots))])
    counter = 0

    while not rospy.is_shutdown() and counter < max_test_episodes :
        start_time = rospy.get_time()
        episode_time_dist_list = []
        start_td = td
        goal_reached = False
        ac = None
        while(counter < max_test_episodes):
            obs = env.reset()
            obs_pub = obs.flatten()
            episode_reward = 0
            while ac == None:
                obs_publisher.publish(obs_pub)
            for _ in range(max_steps):
                action = np.asarray([ac])
                td_before_reset = td
                obs, reward, done, info = env.step(action)
                obs_pub = obs.flatten()
                obs_publisher.publish(obs_pub)
                episode_reward += reward
                goal_reached = gr
                if (done):
                    print("Done")
                    counter += 1
                    if profiling:
                        total_time_episode = rospy.get_time() - start_time
                        total_distance_episode = td_before_reset - start_td
                        print("The total time is {}".format(total_time_episode))
                        print("The distance travelled {}".format(total_distance_episode))
                        episode_time_dist_list.append([total_time_episode, total_distance_episode, goal_reached])
                        start_time = rospy.get_time()
                        start_td = td
                    break
    file = open('trained_model_data_Jetson'+str(world_file)+'.csv', 'w')
    with file:     
      write = csv.writer(file) 
      write.writerows(episode_time_dist_list)
        


        
