#!/usr/bin/env python3
'''creting my first neruon class'''

import numpy as np


class Neuron:
    '''neuron's Class'''

    def __init__(self, nx):
        '''function for the Neuron class

        conditions
            nx: nx is the number of input features to the neuron.
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
