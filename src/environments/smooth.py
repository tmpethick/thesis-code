import numpy as np
from numpy import where

from src.environments.core import BaseEnvironment


class Sin(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def _call(self, x):
        return np.sin(30 * x) #+ np.sin(60 * x)


class Sinc(BaseEnvironment):
    bounds = np.array([[-20, 20]])
    x_opt = 0

    def _call(self, x):
        x = np.asanyarray(x)
        y = where(x == 0, 1.0e-20, x)
        return np.sin(y)/y

    def derivative(self, x):
        # TODO: fix devision by zero if it becomes a problem.
        return (x * np.cos(x) - np.sin(x)) / (x ** 2)

    def hessian(self, x):
        return -((x ** 2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / (x ** 3)


class BigSinc(Sinc):
    bounds = np.array([[-0.2, 0.2]])

    def _call(self, x):
        return super()._call(100 * x)


class NegSinc(Sinc):
    def _call(self, x):
        return -super()._call(x)


class Sin2D(BaseEnvironment):
    bounds = np.array([[0,1], [0,1]])

    def _call(self, x):
        return (0.5 * np.sin(13 * x[..., 0]) * np.sin(27 * x[..., 0]) + 0.5) * (0.5 * np.sin(13 * x[..., 1]) * np.sin(27 * x[..., 1]) + 0.5)[..., None]


class CosProd2D(BaseEnvironment):
    bounds = np.array([[-1,1], [-1,1]])
    def _call(self, X):
        return (np.cos(0.5 * np.pi * X[..., 0]) * np.cos(0.5 * np.pi * X[..., 1]))[..., None]


class Sinc2D(BaseEnvironment):
    bounds = np.array([[-20, 20], [-20, 20]])

    def _call(self, x):
        r = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        return (np.sin(r) / r)[..., None]

    def derivative(self, x):
        r_pow = x[..., 0] ** 2 + x[..., 1] ** 2
        r = np.sqrt(r_pow)
        return (x * (np.cos(r) / r) - (np.sin(r) / r_pow)) / r

    def hessian(self, x):
        pass


__all__ = [
    'Sin',
    'Sinc',
    'BigSinc',
    'NegSinc',
    'Sin2D',
    'CosProd2D',
    'Sinc2D',
]
