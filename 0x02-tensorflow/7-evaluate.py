#!/usr/bin/env python3
'''Evaluate'''
import tensorflow as tf


def evaluate(X, Y, save_path):
    '''Function that evaluates
    the output of a neural network'''
    session = tf.Session()
    saver = tf.train.import_meta_graph("{}.meta".format(save_path))
    saver.restore(session, save_path)

    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    y_pred = tf.get_collection("y_pred")[0]
    acc = tf.get_collection("accuracy")[0]
    loss = tf.get_collection("loss")[0]

    predY = session.run(y_pred, feed_dict={x: X, y: Y})
    accuracy = session.run(accuracy, feed_dict={x: X, y: Y})
    lossE = session.run(loss, feed_dict={x: X, y: Y})

    return predY, accuracy, lossE
