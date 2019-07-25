import numpy as np

from src.environments.core import BaseEnvironment
from src.environments.smooth import Sinc


class Exp(BaseEnvironment):
    bounds = np.array([[-1, 1]])

    def _call(self, X):
        return np.exp(X)
    
    def derivative(self, X):
        return np.exp(X)


class ActiveSubspaceTest(BaseEnvironment):
    """"
    Function in 10D input space with only a 1D active subspace
    i.e. after a suitable rotation it only has 1 dimension in which the function varies significantly.

    Code:
    https://github.com/sischei/global_solution_yale19/blob/master/Lecture_6/code/AS_ex2.py

    Paper (fig. 4):
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
    """
    bounds = np.array([[-1, 1]] * 10)

    def _call(self, X):
        y = np.exp(0.01 * X[..., 0] + 0.7 * X[..., 1] + 0.02 * X[..., 2] + 0.03 * X[..., 3] + 0.04 * X[..., 4] +
                   0.05 * X[..., 5] + 0.06 * X[..., 6] + 0.08 * X[..., 7] + 0.09 * X[..., 8] + 0.1 * X[..., 9])
        return y[..., None]

    def derivative(self, X):
        Y = self(X)
        coefs = np.array([0.01, 0.7, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1])
        return Y * coefs[None, :]


class Embedding(BaseEnvironment):
    bounds = None

    def __init__(self, base_env, D=10, is_random=False, **kwargs):
        self.base_env = base_env
        self.bounds = np.repeat(self.base_env.bounds, D, axis=0)

        if is_random:
            rand = np.random.RandomState(42)
            self.coefs = rand.uniform(0,1, size=D)

        n_inactive = D - 1
        active = 0.7
        ratio = 1
        self.coefs = np.linspace(0, 1, n_inactive + 1)[1:]

        # Find out what to scale coefficient with so sum is 1 times the active dim.
        a = active * ratio / np.sum(self.coefs)
        self.coefs = a * self.coefs

        # Add the active dim
        self.coefs = np.insert(self.coefs, len(self.coefs) // 2, active)
        
        super().__init__(**kwargs)

    def transform(self, X):
        return np.einsum('...i,i->...', X, self.coefs)[:, None]

    def _call(self, X):
        Z = self.transform(X)
        return self.base_env(Z)

    def derivative(self, X):
        Z = self.transform(X)
        Y = self.base_env.derivative(Z)
        return Y * self.coefs[None, :]

    # TODO: implement from_config to setup base_env


class CircularEmbedding(BaseEnvironment):
    # Hack to make it settable in `__init__`.
    bounds = None

    def __init__(self, base_env, D=10, **kwargs):
        super().__init__(**kwargs)
        self.base_env = base_env
        self.bounds = np.repeat(self.base_env.bounds, D, axis=0)
        self._D = D

    def transform(self, X):
        # if X.ndim == 1:
        #     return X ** 2
        return np.sqrt(np.sum(X ** 2, axis=-1))[:,None]

    def noiseless(self, X):
        Z = self.transform(X)
        return self.base_env.noiseless(Z)

    def _call(self, X):
        Z = self.transform(X)
        return self.base_env(Z)

    def derivative(self, X):
        # a little chain rule..
        Z = self.transform(X)
        f_diff = self.base_env.derivative(Z)
        Z_diff = 2 * X

        return f_diff * Z_diff


class ActiveManifoldArbitrary1D(CircularEmbedding):
    def __init__(self, D=10, **kwargs):
        super().__init__(base_env=Sinc(), D=D, **kwargs)
    

class ActiveSubspaceArbitrary1D(Embedding):
    """"
    Function in `D` input space with only a 1D active subspace
    i.e. after a suitable rotation it only has 1 dimension in which the function varies significantly.

    Code:
    https://github.com/sischei/global_solution_yale19/blob/master/Lecture_6/code/AS_ex2.py

    Paper (fig. 4):
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
    """

    def __init__(self, D=10, is_random=False, **kwargs):
        super().__init__(base_env=Exp(), D=D, is_random=is_random, **kwargs)


__all__ = [
    'ActiveSubspaceTest',
    'ActiveSubspaceArbitrary1D',
    'ActiveManifoldArbitrary1D',
    'CircularEmbedding',
    'Embedding',
]
