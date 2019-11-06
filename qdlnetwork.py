import tensorflow as tf
import random

class DQLnetwork:
    def __init__(self, action_space):
        self.actions = action_space
        self.strides = (4,2,1)

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
        Creates a bias variable with shape shape
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
        conv1_weight = self.weight_var([8, 8, 4, 32])
        conv1_bias = self.bias_var([32])
        conv2_weight = self.weight_var([4, 4, 32, 64])
        conv2_bias = self.bias_var([64])
        conv3_weight = self.weight_var([3, 3, 64, 64])
        con3_bias = self.bias_var([64])

        #fully connected layer tensors
        fc_weight = self.weight_var([256, 256])
        fc_bias = self.bias_var([256])
        output_weight = self.weight_var(256, self.actions)
        output_bias = self.bias_var([self.actions])

        #output layer tensors
        h_conv1 = tf.nn.relu(self.conv2d(inp_layer, conv1_weight, self.strides[0]) + conv1_bias)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, conv2_weight, self.strides[1] + conv2_bias))
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_pool2, conv2_weight, self.strides[1] + conv2_bias))
        h_pool3 = self.max_pool_2x2(h_conv3)

        #declaring layer declerations
        print("I declare a layer declarations")



