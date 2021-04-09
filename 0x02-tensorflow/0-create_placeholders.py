#!/usr/bin/env python3
'''Create PlaceHolder'''
import tensorflow as tf


def create_placeholders(nx, classes):
    '''return2 placeholders
    x and y
    '''
    return tf.placeholder(
        "float", [None, nx], name='x'
    ), tf.placeholder(
        "float", [None, classes], name='y'
    )
