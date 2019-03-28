import numpy as np


class BaseEnvironment(object):
    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def input_dim(self):
        return self.bounds.shape[0]

    def __call__(self, x):
        raise NotImplementedError


class Kink1D(BaseEnvironment):
    bounds = np.array([[-2, 2]])

    def __call__(self, x):
       return 1 / (10 ** (-4) + x ** 2)


class Kink2D(BaseEnvironment):
    bounds = np.array([[0, 1], [0, 1]])

    def __call__(self, x):
        y = 1 / (np.abs(0.5 - x[..., 0] ** 4 - x[..., 1] ** 4) + 0.1)
        return y[..., None]

