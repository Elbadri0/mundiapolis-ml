#!/usr/bin/env python3
'''Create Confusion'''
import numpy as np


def create_confusion_matrix(labels, logits):
    '''Function that creates
    a confusion matrix'''
    result = np.zeros(
        (
            labels.shape[1],
            labels.shape[1]
        )
    )
    prevY = np.argmax(logits, axis=1)
    trueY = np.argmax(labels, axis=1)
    for i in range(labels.shape[0]):
        result[
            trueY[i]
        ][
            prevY[i]
        ] += 1
    return (result)
