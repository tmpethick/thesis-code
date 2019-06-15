import numpy as np
import scipy
from scipy.linalg import cho_solve
from scipy.optimize import fmin_l_bfgs_b

from src.models import ProbModel


class LowRankGPModel(ProbModel):
    def __init__(self, kernel, noise=0.01, n_features=10, do_optimize=False, n_restarts_optimizer=10):
        super().__init__()
        self.noise = noise
        self.m = n_features
        self.kernel_ = kernel

        self.is_noise_free = float(noise) == 0.0

        self.do_optimize = do_optimize
        self.K_noisy_inv = None
        self._X = None
        self._Y = None

        self.n_restarts_optimizer = n_restarts_optimizer

    def max_log_marginal_likelihood(self):
        # TODO: implement opt strategy for MLM
        obj_func = lambda theta: -self.log_marginal_likelihood(theta)
        optima = [(self._constrained_optimization(
            obj_func, self.kernel_.theta, self.kernel_.bounds))]

        if self.n_restarts_optimizer > 0:
            bounds = self.kernel_.bounds
            for iteration in range(self.n_restarts_optimizer):
                theta_initial = np.random.uniform(bounds[:, 0], bounds[:, 1])
                optima.append(
                    self._constrained_optimization(obj_func, theta_initial, bounds))

        from operator import itemgetter
        lml_values = list(map(itemgetter(1), optima))
        return optima[np.argmin(lml_values)][0]


    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        """Shameless stealing from scipy.
        """
        theta_opt, func_min, convergence_dict = \
            fmin_l_bfgs_b(obj_func, initial_theta, bounds=bounds, approx_grad=True)
        if convergence_dict["warnflag"] != 0:
            import warnings
            warnings.warn("fmin_l_bfgs_b terminated abnormally with the state: {}".format(convergence_dict))

        return theta_opt, func_min

    def log_marginal_likelihood(self, theta):
        """Watch out! It updates the hyperparms of the kernel.
        """
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/gaussian_process/gpr.py
        # GPML p. 131 (pdf)
        # log evidence / log marginal likelihood: p(y|theta)

        self.kernel_.theta = theta
        self._recalc_spectral_kernel = True

        # Recompute the design matrix given theta
        Y = self.Y
        Phi, A, chol, L = self.compute_matrices(self.X)
        alpha = (Phi.T.dot(Y))[:, 0]

        # print(self.Y.shape)
        # print(Phi.shape, A.shape, chol)
        # print(alpha.shape)

        n, d = self.X.shape
        m = self.m

        B = cho_solve(chol, alpha)
        L_dia = np.diagonal(L)
        log_det_A = np.sum(np.log(L_dia))

        mll = \
            - 1/(2 * self.noise) * (Y.T.dot(Y) - B.T.dot(B)) \
            - log_det_A \
            - 1/2 * (n-m) * np.log(self.noise) \
            - n/2 * np.log(2 * np.pi) \

        return mll

    def compute_matrices(self, X):
        Phi = self.feature_map(X)
        A = self.noise * np.identity(self.m) + Phi.T @ Phi
        chol = scipy.linalg.cho_factor(A, lower=True)
        L = scipy.linalg.cholesky(A)

        return Phi, A, chol, L

    def _fit(self, X, Y, Y_dir=None):
        # Hack
        self._X = X
        if self.do_optimize:
            self.kernel_.theta = self.max_log_marginal_likelihood()
            self._recalc_spectral_kernel = True

        n, d = X.shape
        Phi, A, chol, L = self.compute_matrices(X)

        # L_inv = scipy.linalg.solve_triangular(L, np.eye(L.shape[0]), lower=True)
        # A_inv = np.dot(L_inv.T, L_inv)
        A_inv = scipy.linalg.cho_solve(chol, np.eye(A.shape[0]))

        noise_inv = (1 / self.noise)
        if self.is_noise_free:
            raise NotImplementedError
        else:
            self.K_noisy_inv = noise_inv * np.identity(n) - noise_inv * Phi.dot(A_inv.dot(Phi.T))

    def feature_map(self, X):
        """Maps from n observations to m feature space.

        Arguments:
            X {[type]} -- (n, d)

        Return:
            numpy.array -- (m, n)
        """

        raise NotImplementedError

    def kernel(self, X1, X2=None):
        if X2 is None:
            Q1 = self.feature_map(X1)
            return Q1.dot(Q1.T)
        else:
            Q1 = self.feature_map(X1)
            Q2 = self.feature_map(X2)
            return Q1.dot(Q2.T)

    def _get_statistics(self, X, full_cov=True):
        assert self.K_noisy_inv is not None, "`self.fit` needs to be called first."

        # TODO: solve linear system instead of precomputing inverse.
        kern = self.kernel
        k = kern(X, self.X)
        mu =  k @ self.K_noisy_inv @ self.Y
        cov = kern(X,X) - k @ self.K_noisy_inv @ k.T

        # (stats, obs, obj_dim)
        if full_cov:
            return mu, cov
        else:
            return mu, np.diagonal(cov)[:, None]


class EfficientLinearModel(LowRankGPModel):
    def __init__(self, noise=0.01, n_features=None):
        super(EfficientLinearModel, self).__init__(noise=noise, n_features=n_features)

    def feature_map(self, X):
        n, d = X.shape
        return X