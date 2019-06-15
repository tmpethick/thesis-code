import numpy as np
from scipy.linalg import cho_factor
from scipy.spatial.distance import cdist

from src.models import ProbModel


class GPVanillaModel(ProbModel):
    def __init__(self, lengthscale=0.1, noise=0.01):
        super(GPVanillaModel, self).__init__()
        self.K_noisy_inv = None
        self.lengthscale = lengthscale
        self.noise = noise

    def _fit(self, X, Y, Y_dir=None):
        n, d = X.shape
        kern = self.kernel(X,X) + self.noise * np.identity(n)

        self.K_noisy_inv = np.linalg.inv(kern)
        self.cho_decomp = cho_factor(kern)

    def kernel(self, X1, X2):
        # SE kernel
        pairwise_sq_dists = cdist(X1, X2, 'sqeuclidean')
        K = np.exp(-pairwise_sq_dists / self.lengthscale ** 2)
        return K

    def _get_statistics(self, X, full_cov=True):
        assert self.K_noisy_inv is not None, "`self.fit` needs to be called first."

        # Compute over multiple rounds
        # kern = self.kernel
        # k = kern(X, self.X)
        # alpha = cho_solve(self.cho_decomp, self.Y)
        # mu = k @ alpha
        # v = np.linalg.solve(L, k.T)
        # cov = kern(X,X) + v.T @ v

        kern = self.kernel
        k = kern(X, self.X)
        mu =  k @ self.K_noisy_inv @ self.Y
        cov = kern(X,X) + self.noise - k @ self.K_noisy_inv @ k.T

        # (stats, obs, obj_dim)
        if full_cov:
            return mu, cov
        else:
            return mu, np.diagonal(cov)[:, None]


class GPVanillaLinearModel(GPVanillaModel):
    def kernel(self, X1, X2):
        return X1 @ X2.T