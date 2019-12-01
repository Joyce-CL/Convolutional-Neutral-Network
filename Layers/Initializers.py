import numpy as np


class Constant:
    def __init__(self):
        self.weight_initialization = 0.1

    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = self.weight_initialization * np.ones((fan_out, fan_in))
        return initialized_tensor


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = np.random.rand(fan_out, fan_in)
        return initialized_tensor


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        theta = np.sqrt(np.divide(2, (fan_in + fan_out)))
        initialized_tensor = np.random.normal(0, theta, weights_shape)
        return initialized_tensor


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        theta = np.sqrt(np.divide(2, fan_in))
        initialized_tensor = np.random.normal(0, theta, weights_shape)
        return initialized_tensor