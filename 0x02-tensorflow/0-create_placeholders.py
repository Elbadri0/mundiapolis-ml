#!/usr/bin/env python3
'''Create PlaceHolder'''
import tensorflow as tf


def create_placeholders(nx, classes):
    '''Function that returns
    two placeholders, x and y,
    for the neural network'''
    return tf.placeholder(
        "float", [None, nx], name='x'
    ), tf.placeholder(
        "float", [None, classes], name='y'
    )
