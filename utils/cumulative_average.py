import numpy as np


class CumulativeAverage:
    def __init__(self):
        self.values = []
        self.weights = []

    def append(self, value, weight):
        self.values.append(value)
        self.weights.append(weight)

    def average(self, axis=None):
        return np.average(self.values, axis=axis, weights=self.weights)