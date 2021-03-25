#!/usr/bin/env python3
'''the neruon class
'''

import numpy as np


class Neuron:
    '''Class that initialized a neuron
    '''

    def __init__(self, nx):
        '''the function for the Neuron class
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
        '''Returns the value of __W.
        '''

        return self.__W

    @property
    def b(self):
        '''Returns the value of __b.
        '''

        return self.__b

    @property
    def A(self):
        '''Returns the value of __A.
        '''

        return self.__A

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron.
        '''

        x = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-x))
        return self.__A

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression.
        '''
        loss_sum = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        self.__cost = -(1 / A.size) * loss_sum
        return self.__cost

    def evaluate(self, X, Y):
        '''Evaluates the neuron’s predictions.
        '''

        self.forward_prop(X)
        pred = np.where(self.__A >= 0.5, 1, 0)
        cost = self.cost(Y, self.__A)
        return pred, cost
