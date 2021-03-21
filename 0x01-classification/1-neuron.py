#!/usr/bin/env python3
'''the neruon class
'''

import numpy as np


class Neuron:
    '''Class that initialized a neuron
    '''

    def __init__(self, nx):
        '''the function for the Neuron class
        
        input.
            nx: The number of input of the neuron.
        '''

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0

    @property
    def W(self):
        '''Returns __W
        '''

        return self.__W

    @property
    def b(self):
        '''Returns __b
        '''

        return self.__b

    @property
    def A(self):
        '''Returns __A
        '''

        return self.__A
