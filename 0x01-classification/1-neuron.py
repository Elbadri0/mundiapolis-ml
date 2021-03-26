#!/usr/bin/env python3
'''Neuron'''
import numpy as np


class Neuron:
    '''Neuron defines a single neuron
    performing binary classification'''

    def __init__(self, nx):
        '''Class Constructor'''
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')
        else:
            self.__W = np.random.normal(size=(1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        '''W'''
        return self.__W

    @property
    def b(self):
        '''b'''
        return self.__b

    @property
    def A(self):
        '''A'''
        return self.__A
