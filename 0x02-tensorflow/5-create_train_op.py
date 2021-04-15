#!/usr/bin/env python3
'''Create Train Op'''
import tensorflow as tf


def create_train_op(loss, alpha):
    '''Function that creates
    the training operation
    for the network'''
    return tf.train.GradientDescentOptimizer(
        alpha
    ).minimize(loss)
