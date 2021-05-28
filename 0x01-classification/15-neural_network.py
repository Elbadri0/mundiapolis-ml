#!/usr/bin/env python3
"""Class NeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """ defines a neural network with one hidden layer performing
    binary classification"""
    def __init__(self, nx, nodes):
        """
        class constructor
        :param nx: is the number of input features
        :param nodes: is the number of nodes found in the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        W1 attribute getter.
        :return: The weights vector for the hidden layer
        """
        return self.__W1

    @property
    def b1(self):
        """
        b1 attribute getter.
        :return: The bias for the hidden layer
        """
        return self.__b1

    @property
    def A1(self):
        """
        A1 attribute getter.
        :return: The Activation output for the inner layer
        """
        return self.__A1

    @property
    def W2(self):
        """
        W1 attribute getter.
        :return: The weights vector for the output neuron
        """
        return self.__W2

    @property
    def b2(self):
        """
        b2 attribute getter.
        :return: The bias for the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """
        A2 attribute getter.
        :return: The activated output for the output neuron (prediction)
        """
        return self.__A2

    def forward_prop(self, X):
        """
        defines a neural network with one hidden layer performing
        binary classification
        :param X: is a ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :return: the private attributes __A1 and __A2, respectively
        """
        z = np.matmul(self.__W1, X) + self.__b1
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A1 = sigmoid
        z = np.matmul(self.__W2, self.A1) + self.__b2
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A2 = sigmoid
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param A: is a numpy.ndarray with shape (1, m) containing
        the activated output of the neuron for each example
        :return: the cost
        """
        summatory = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        constant = -(1/A.shape[1])
        return constant * summatory.sum()

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        :param X: numpy.ndarray with shape (nx, m) that contains
        the input data
        :param Y: is a numpy.ndarray with shape (1, m) that
        contains
        the correct labels for the input data
        :return: the neuron’s prediction and the cost of the
        network
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        :param X: is a ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param A1: is the output of the hidden layer
        :param A2: is the predicted output
        :param alpha: is the learning rate
        :return: Updates the private attributes __W1, __b1, __W2, and __b2
        """
        dz2 = A2 - Y
        m = Y.shape[1]
        dw2 = np.matmul(A1, dz2.T) / m  # A, Y are transpose for operation
        db2 = dz2.sum(axis=1, keepdims=True) / m

        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = np.matmul(X, dz1.T) / m  # A, Y are transpose for operation
        db1 = dz1.sum(axis=1, keepdims=True) / m

        self.__W2 = self.__W2 - (alpha * dw2.T)
        self.__b2 = self.__b2 - (alpha * db2)
        self.__W1 = self.__W1 - (alpha * dw1.T)
        self.__b1 = self.__b1 - (alpha * db1)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neural network
        :param X: is a ndarray with shape (nx, m) that contains the input data
                  nx is the number of input features to the neuron
                  m is the number of examples
        :param Y: is a numpy.ndarray with shape (1, m) that contains
        the correct labels for the input data
        :param iterations: is the number of iterations to train over
        :param alpha: is the learning rate
        :param verbose: is a boolean that defines whether or not to print
        information about the training.
        :param graph:  is a boolean that defines whether or not to graph
        information about the training once the training has completed
        :param step: step iterations
        :return: the evaluation of the training data after
        iterations of training have occurred
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True and graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if type(step) is not int:
            raise TypeError("step must be an integer")
        graph_iteration = []
        graph_cost = []

        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A2)

            if step and (i % step == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
                graph_iteration.append(i)
                graph_cost.append(cost)

            if i < iterations:
                self.gradient_descent(X, Y, self.__A1, self.A2, alpha)

        if graph is True:
            plt.plot(graph_iteration, graph_cost)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)
