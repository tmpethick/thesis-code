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


# import growth_model_GPR.nonlinear_solver_initial as solver_initializor
# import growth_model_GPR.nonlinear_solver_iterate as solver_iterator


# class DynamicBell(BaseEnvironment):
#     # bounds are np.array([k_bar, k_up] * dim)

#     def __init__(self, input_dim=10):
#         self.dim = input_dim

#         self._is_initialized = False
#         self.n_agents = 2
#         # TODO: consolidate with n_agents in parameters.py

#     def _call(self, X, prob_model):
#         """OBS: every call will iterate.
#         We assume that self.prob_model is updated with new observations (discarding the old.)
#         """
#         n = X.shape[0]
#         y = np.zeros(n, float)

#         if self._is_initialized:
#             for i in range(n):
#                 y[i] = solver_iterator.iterate(X[i], self.n_agents, prob_model)[0]
#         else:
#             self._is_initialized = True
#             for i in range(n):
#                 y[i] = solver_initializor.initial(X[i], self.n_agents)[0]


#         return y[:, None]


__all__ = ['ActiveSubspaceTest']
