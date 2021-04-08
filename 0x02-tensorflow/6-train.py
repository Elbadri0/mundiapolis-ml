#!/usr/bin/env python3
'''Train'''
import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        layer_sizes,
        activations,
        alpha,
        iterations,
        save_path="/tmp/model.ckpt"
):
    '''Function that builds, trains,
    and saves a neural network classifie'''
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    predY = forward_prop(x, layer_sizes, activations)
    lossE = calculate_loss(y, predY)
    accurancy = calculate_accuracy(y, predY)
    train_op = create_train_op(lossE, alpha)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("predY", predY)
    tf.add_to_collection("lossE", lossE)
    tf.add_to_collection("accuracy", accurancy)
    tf.add_to_collection("train_op", train_op)

    session = tf.Session()
    with session.as_default():
        session.run(tf.global_variables_initializer())
        for i in range(iterations + 1):
            if (
                (
                    i % 100 == 0
                ) or (
                    i == iterations
                )
            ):
                cost_train = lossE.eval(
                    {x: X_train, y: Y_train}, session
                )
                accuracy_train = accurancy.eval(
                    {x: X_train, y: Y_train}, session
                )
                cost_valid = lossE.eval(
                    {x: X_valid, y: Y_valid}, session
                )
                accuracy_valid = accurancy.eval(
                    {x: X_valid, y: Y_valid}, session
                )
                print("After {} iterations:".format(i)
                      + "\n\tTraining Cost: {}".format(cost_train)
                      + "\n\tTraining Accuracy: {}".format(accuracy_train)
                      + "\n\tValidation Cost: {}".format(cost_valid)
                      + "\n\tValidation Accuracy: {}".format(accuracy_valid))
            if i != iterations:
                session.run(
                    train_op,
                    feed_dict={x: X_train, y: Y_train}
                )
        trainSaver = tf.train.Saver()
        return trainSaver.save(session, save_path)
