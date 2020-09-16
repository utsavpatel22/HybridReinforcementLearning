from stable_baselines.common.policies import *
import stable_baselines.common.policies as common
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm
import numpy as np


def modified_shallow_nature_cnn(scaled_images, **kwargs):
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

def modified_deep_nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=6, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = conv_to_fc(layer_2)
    layer_3 = activ(linear(layer_2, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))
    return activ(linear(layer_3, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))

def tiny_filter_deep_nature_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=6, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=8, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=10, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_4 = activ(conv(layer_3, 'c4', n_filters=12, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_5 = activ(conv(layer_4, 'c5', n_filters=14, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_5 = conv_to_fc(layer_5)
    layer_6 = activ(linear(layer_5, 'fc1', n_hidden=256, init_scale=np.sqrt(2)))
    layer_7 = activ(linear(layer_6, 'fc2', n_hidden=128, init_scale=np.sqrt(2)))
    return activ(linear(layer_7, 'fc3', n_hidden=128, init_scale=np.sqrt(2)))


def modified_cnn(scaled_images, **kwargs):
    """
    CNN from Nature paper.
    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=8, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=1, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = conv_to_fc(layer_2)
    return activ(linear(layer_2, 'fc1', n_hidden=128, init_scale=np.sqrt(2)))


class CustomMLP_LSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[144, 144, 'lstm', 144],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


class CustomCNN_LSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, **_kwargs):
        # print(n_env)
        # print(n_steps)
        super(CustomLSTMPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         layer_norm=True, feature_extraction="cnn", cnn_extractor=modified_cnn,**_kwargs)


# def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
#     super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
#                      net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
#                      layer_norm=True, feature_extraction="mlp", **_kwargs)

class CustomShallowCNNPolicy(common.FeedForwardPolicy):
    
    def __init__(self, *args, **kwargs):
        super(CustomShallowCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_shallow_nature_cnn, feature_extraction="cnn")


class CustomDeepCNNPolicy(common.FeedForwardPolicy):
    
    def __init__(self, *args, **kwargs):
        super(CustomDeepCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=modified_deep_nature_cnn, feature_extraction="cnn")

class CustomTinyDeepCNNPolicy(common.FeedForwardPolicy):
    
    def __init__(self, *args, **kwargs):
        super(CustomTinyDeepCNNPolicy, self).__init__(*args, **kwargs, cnn_extractor=tiny_filter_deep_nature_cnn, feature_extraction="cnn")
