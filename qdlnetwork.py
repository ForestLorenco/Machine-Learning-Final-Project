import tensorflow as tf
import random

class DQLnetwork:
    def __init__(self, action_space):
        self.actions = action_space

    def weight_var(self, shape):
        """
        Creates a tensor with shape shape and random numbers around 0 with stdev 0.01
        :param shape:
        :return:
        """

        init = tf.random.truncated_normal(shape, stddev=0.01)
        return tf.Variable(init)

    def bias_var(self, shape):
        """
        Creates a bias variable with shaope shape
        :param shape:
        :return:
        """
        init = tf.constant(0.01, shape=shape)
        return tf.Variable(init)

    def conv2d(self, input, W, stride):
        """
        Creates a conv2d tensor for a layer in a neural network
        :param input:
        :param W:
        :param stride:
        :return:
        """

        return tf.nn.conv2d(input, W, strides=[1,stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        """
        Performs max_pool for the output of the conv layer
        :param x:
        :return:
        """
        return tf.nn.max_pool2d(x, ksize=[1,2,2,1], strides=[1,2,2,1])

    def createNetwork(self):
        """
        Creates and returns a DQN network for agent
        :return:
        """

        #input layer
        inp_layer = tf.compat.v1.placeholder("float", [None, 128, 128, 4])

        #conv layer tensors

        #fully connected layer tensors

        #output layer tensors

        #declaring layer declerations


