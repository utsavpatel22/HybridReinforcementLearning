from stable_baselines.common.policies import *
import stable_baselines.common.policies as common
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
import numpy as np


def modified_nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = conv_to_fc(layer_2)
    return activ(linear(layer_2, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))



class CustomCNNPolicy(common.FeedForwardPolicy):
    
    def __init__(self, *args, **kwargs):
        super(CustomCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_nature_cnn, feature_extraction="cnn")