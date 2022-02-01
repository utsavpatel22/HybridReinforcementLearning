#!/usr/bin/env python
import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from stable_baselines import PPO2
from std_msgs.msg import Int64
import time

obs = np.array([])
def getObservation(msg):
    global obs
    obs = msg.data

if __name__ == '__main__':
	rospy.init_node('action_publisher', anonymous=True, log_level=rospy.WARN)
	model = PPO2.load("ppo2_turtlebot#5")
	action_publisher = rospy.Publisher("/predicted_action", Int64,queue_size=10)
	observation_subscriber = rospy.Subscriber("/obs_data", numpy_msg(Floats), getObservation)
	n = 0
	total_time = 0
	while not rospy.is_shutdown():
		if obs.shape[0] != 0:
			n = n+1 
			obs_sub = np.reshape(obs, (1,144,10,4))
			start_t = time.time()
			action, _ = model.predict(obs_sub)
			total_time = total_time + (time.time() - start_t)
			avg_time = total_time / n 
			print("The average time for prediction ", avg_time) 
			action_publisher.publish(action[0])
			print("Publishing the action")
