#!/usr/bin/env python
import rospy
import numpy as np
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from stable_baselines import PPO2
from std_msgs.msg import Int64
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import time

obs = np.array([])
def getObservation(msg):
    global obs
    obs = msg.data

if __name__ == '__main__':
	with tf.Session() as sess:
		rospy.init_node('action_publisher', anonymous=True, log_level=rospy.WARN)
		action_publisher = rospy.Publisher("/predicted_action", Int64,queue_size=10)
		observation_subscriber = rospy.Subscriber("/obs_data", numpy_msg(Floats), getObservation)
		# First deserialize your frozen graph:
		with tf.gfile.GFile('best_saved_model.pb', "rb") as f:
			frozen_graph = tf.GraphDef()
			frozen_graph.ParseFromString(f.read())
	    # Now you can create a TensorRT inference graph from your
	    # frozen graph:
		print("---------------Optimizing the graph-----------------")
		converter = trt.TrtGraphConverter(
			input_graph_def=frozen_graph,
			precision_mode='FP16',
			nodes_blacklist=['output/ArgMax:0']) #output nodes
		trt_graph = converter.convert()
		print("-------------- Optimization Done -------------------")
		# converter.save(output_saved_model_dir='/home/thebeast/workspaces/TFSaveTest/TensorRT/') # Does not work
		# Import the TensorRT graph into a new graph and run:
		output_node = tf.import_graph_def(trt_graph, return_elements=['output/ArgMax:0'])
		n = 0
		total_time = 0
		while not rospy.is_shutdown():
			if obs.shape[0] != 0:
				n = n+1
				obs_sub = np.reshape(obs, (1,144,10,4))
				start_t = time.time()
				action = sess.run(output_node, {'import/input/Ob:0': obs_sub})
				total_time = total_time + (time.time() - start_t)
				avg_time = total_time / n
				print("The average time for prediction ", avg_time)
				action_publisher.publish(action[0][0])
				print("Publishing the action")

