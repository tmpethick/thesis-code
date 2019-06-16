import numpy as np

from src.environments.core import BaseEnvironment


class GenzContinuous(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10, **kwargs):
        super().__init__(**kwargs)
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.ones((1, D)) * 0.5

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5

        assert self.a.shape[0] == D
        assert self.u.shape[1] == D

    def _call(self, X):
        Z = np.abs(X - self.u)
        return np.exp(-np.einsum("i,ji->j", self.a, Z))[:, None]


class GenzCornerPeak(BaseEnvironment):
    bounds = None

    def __init__(self, a=None, D=10, **kwargs):
        super().__init__(**kwargs)
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5

        assert self.a.shape[0] == D

    def _call(self, X):
        return np.power(1 + np.einsum("i,ji->j", self.a, X), -(self.D + 1))[:, None]


class GenzDiscontinuous(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10, **kwargs):
        super().__init__(**kwargs)
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.array([0.5, 0.5])


        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5

        assert self.a.shape[0] == D

    def _call(self, X):
        Z = np.exp(np.einsum("i,ji->j", self.a, X))
        zero = np.zeros(X.shape[0])
        return np.where((X[..., 0] > self.u[0]) | (X[..., 1] > self.u[1]), zero, Z)[:, None]


class GenzGaussianPeak(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10, **kwargs):
        super().__init__(**kwargs)
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.ones((1, D)) * 0.5

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5

        assert self.a.shape[0] == D
        assert self.u.shape[1] == D


    def _call(self, X):
        Z = np.power(X - self.u, 2)
        return np.exp(-np.einsum("i,ji->j", self.a ** 2, Z))[:, None]


class GenzOscillatory(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10, **kwargs):
        super().__init__(**kwargs)
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.array([0.5])

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5

        assert self.a.shape[0] == D
        assert self.u.shape[0] == 1


    def _call(self, X):
        return np.cos(2 * np.pi * self.u[0] + np.einsum("i,ji->j", self.a, X))[:, None]


class GenzProductPeak(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10, **kwargs):
        super().__init__(**kwargs)
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.ones((1, D)) * 0.5

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5

        assert self.a.shape[0] == D
        assert self.u.shape[1] == D

    def _call(self, X):
        # Introduce new axis so addition with X works.
        a_pow = np.power(self.a, -2)[None, :]

        return np.product(1 / (a_pow + np.power(X - self.u, 2)), axis=-1)[:,None]


__all__ = [
    'GenzContinuous',
    'GenzCornerPeak',
    'GenzDiscontinuous',
    'GenzGaussianPeak',
    'GenzOscillatory',
    'GenzProductPeak',
]
