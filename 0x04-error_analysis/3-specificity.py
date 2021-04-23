#!/usr/bin/env python3
'''Specificity'''
import numpy as np


def specificity(confusion):
    '''Function that calculates
    the specificity for each class
    in a confusion matrix'''
    posTrue = np.diag(confusion)
    posFalse = np.sum(
        confusion, axis=0
    ) - posTrue
    negFalse = np.sum(
        confusion, axis=1
    ) - posTrue
    negTrue = np.sum(
        confusion
    ) - negFalse - posTrue - posFalse
    return (negTrue / (negTrue + posFalse))
