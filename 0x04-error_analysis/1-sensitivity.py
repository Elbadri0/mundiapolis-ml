#!/usr/bin/env python3
'''Sensitivity'''
import numpy as np


def sensitivity(confusion):
    '''Function that calculates
    the sensitivity for each class
    in a confusion matrix'''
    return np.diag(
        confusion
    ) / np.sum(
        confusion, axis=1
    )
