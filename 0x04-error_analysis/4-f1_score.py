#!/usr/bin/env python3
'''f1 Score'''
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    '''Function that calculates the
    F1 score of a confusion matrix'''
    callNd = sensitivity(confusion)
    precis = precision(confusion)
    return 2 * (
        callNd * precis
    ) / (
        callNd + precis
    )
