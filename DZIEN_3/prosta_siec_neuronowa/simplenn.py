import numpy as np

class SimpleNeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.weights = 2*np.random.random((3,1))-1
