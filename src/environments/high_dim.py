import numpy as np

from src.environments.core import BaseEnvironment


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


class ActiveSubspaceArbitrary1D(BaseEnvironment):
    """"
    Function in `D` input space with only a 1D active subspace
    i.e. after a suitable rotation it only has 1 dimension in which the function varies significantly.

    Code:
    https://github.com/sischei/global_solution_yale19/blob/master/Lecture_6/code/AS_ex2.py

    Paper (fig. 4):
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
    """
    bounds = None

    def __init__(self, D=10, **kwargs):
        self.bounds = np.array([[-1, 1]] * D)
        
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

    def _call(self, X):
        Z = np.einsum('...i,i->...', X, self.coefs)
        y = np.exp(Z)
        return y[..., None]

    def derivative(self, X):
        Y = self(X)
        return Y * self.coefs[None, :]


# class ActiveSubspaceModifiedTest(BaseEnvironment):
#     bounds = np.array([[-1,1]] * 10)

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._f = ActiveSubspaceTest()

#     def _call(self, X):
#         return X[..., 1]*X[..., 2] * self._f(X)

#     def derivative(self, X):
#         """TODO: Fix derivative
#         """
#         _test_old = self._f.derivative(X)
#         val = np.atleast_1d(X[..., 1] * X[..., 2] * _test_old)
#         coefs = np.array([0.01, 0.7, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1])
#         out = val[:, None] * coefs[None, :]
#         out[:, 1] += X[..., 2] * _test_old
#         out[:, 2] += X[..., 1] * _test_old
#         return out

__all__ = [
    'ActiveSubspaceTest',
    'ActiveSubspaceArbitrary1D',
]
