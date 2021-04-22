#!/usr/bin/env python3
'''LeNet-5 (Tensorflow)'''
import tensorflow as tf


def lenet5(x, y):
    '''Function that builds a modified
    version of the LeNet-5 architecture
    using tensorflow'''
    initializer = tf.contrib.layers.variance_scaling_initializer()
    layer = tf.layers.Conv2D(
        6,
        5,
        padding='same',
        activation='relu',
        kernel_initializer=initializer,
    )(x)
    layer = tf.layers.MaxPooling2D(2, 2)(layer)
    layer = tf.layers.Conv2D(
        16,
        5,
        padding='valid',
        activation='relu',
        kernel_initializer=initializer,
    )(layer)
    layer = tf.layers.MaxPooling2D(2, 2)(layer)
    layer = tf.layers.Flatten()(layer)
    layer = tf.layers.Dense(
        120,
        activation='relu',
        kernel_initializer=initializer
    )(layer)
    layer = tf.layers.Dense(
        84,
        activation='relu',
        kernel_initializer=initializer
    )(layer)
    layer = tf.layers.Dense(
        10,
        kernel_initializer=initializer
    )(layer)
    loss = tf.losses.softmax_cross_entropy(
        y,
        layer
    )
    train_op_adamOpt = tf.train.AdamOptimizer().minimize(
        loss
    )
    max_pred = tf.argmax(layer, 1)
    equal = tf.equal(
        tf.argmax(y, 1),
        max_pred
    )
    acc = tf.reduce_mean(
        tf.cast(
            equal,
            tf.float32
        )
    )
    return (
        tf.contrib.layers.softmax(layer),
        train_op_adamOpt,
        loss,
        acc
    )
