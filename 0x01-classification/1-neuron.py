#!/usr/bin/env python3
'''Module containing the neruon class
'''

import numpy as np


class Neuron:
    '''Class that defines a neuron
    '''

    def __init__(self, nx):
        '''Initialization function for the Neuron class

        Args.
            nx: The number of input features to the neuron.
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''Returns the value of __W
        '''

        return self.__W

    @property
    def b(self):
        '''Returns the value of __b
        '''

        return self.__b

    @property
    def A(self):
        '''Returns the value of __A
        '''

        return self.__A
