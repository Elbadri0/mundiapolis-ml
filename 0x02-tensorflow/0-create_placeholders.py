#!/usr/bin/env python3
'''Create PlaceHolder'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    '''return2 placeholders
    x and y
    '''
    return tf.placeholder(
        "float", [None, nx], name='x'
    ), tf.placeholder(
        "float", [None, classes], name='y'
    )
