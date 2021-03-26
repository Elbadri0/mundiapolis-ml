#!/usr/bin/env python3
'''Neural Network'''
import numpy as np


class NeuralNetwork():
    '''Neural Network defines a neural
    network with one hidden layer
    performing binary classification'''

    def __init__(self, nx, nodes):
        '''Class Constructor'''
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        elif type(nodes) != int:
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")
        else:
            self.W1 = np.random.normal(size=(nodes, nx))
            self.b1 = np.array([[np.array(0.)]] * nodes)
            self.A1 = 0
            self.W2 = np.random.normal(size=(1, nodes))
            self.b2 = 0
            self.A2 = 0
