import numpy as np

from src.models import LowRankGPModel


class RandomFourierFeaturesModel(LowRankGPModel):
    """Based on analysis in [1] we choose the unbiased variant as it has strictly smaller variance for the Squared Exponential.

    [1]: https://www.cs.cmu.edu/~dsutherl/papers/rff_uai15.pdf
    """

    def __init__(self, kernel, noise=0.01, n_features=10, do_optimize=False):
        assert n_features % 2 == 0, "`n_features` has to be even."

        super().__init__(kernel, noise=noise, n_features=n_features,
        do_optimize=do_optimize)

        # `self.kernel` is already reserved by LowRankGPModel
        self.kernel_ = kernel
        self.W = None
        self._recalc_spectral_kernel = False

    def __repr__(self):
        return "RFF"

    def spectral_kernel(self, size):
        # It is important that these are fixed across feature_map calls.
        # However it should be recomputed when kernel_.theta changes!
        # But then we're optimizing over stochastic function... (we could be sampling another set when returning to the same hyperparameters).
        if self._recalc_spectral_kernel or self.W is None:
            self.W = self.kernel_.sample(size)
            self._recalc_spectral_kernel = False
        return self.W

    def feature_map(self, X):
        n, d = X.shape

        # sample omegas
        W = self.spectral_kernel(size=(self.m // 2, d))

        # Compute m x n feature map
        Z = W @ X.T
        uniform_weight = np.sqrt(2.0 / self.m * self.kernel_.variance)
        Q_cos = uniform_weight * np.cos(Z)
        Q_sin = uniform_weight * np.sin(Z)

        # n x m
        return np.concatenate((Q_cos, Q_sin), axis=0).T


class QuadratureFourierFeaturesModel(LowRankGPModel):

    def __init__(self, lengthscale=0.1, noise=0.01, n_features=100):
        assert n_features % 2 == 0, "`n_features` has to be even."

        super(QuadratureFourierFeaturesModel, self).__init__(noise=noise, n_features=n_features)

        # Not used since the structure is implicit in the particular use of the Gauss-Hermite Scheme.
        self.lengthscale = lengthscale
        self.spectral_kernel_pdf = lambda w: np.exp(- np.square(w).dot(np.square(self.lengthscale))/ 2)

    def feature_map(self, X):
        # m is now the number of nodes in the dense grid.
        # compute the number of nodes in 1D by the relationship m = 2 * m_bar^d.
        n, d = X.shape
        self.m_bar = int(np.power(self.m // 2, 1. / d))
        self.m = 2 * self.m_bar ** d

        # Compute (nodes, weights) in 1D
        W_bar, weights = self.gauss_hermite_1d(self.m_bar)
        weights = np.sqrt(2) / np.power(self.lengthscale, 2) * weights

        # Construct cartesian grid (m_bar**d, d) of weights
        # and eventually get (m_bar**d,) vector of weights.
        # Contrary to the mathematical presentation in the paper
        # we compute the cartesian product over weights as well, for convenience. (equation 5)
        weights = cartesian_product(*((weights,) * d))
        weights = np.prod(weights, axis=1)
        weights = np.sqrt(weights)

        # (m_bar**d, d)
        # From paper:
        # j is dense grid points (split into cos and sin)
        # i is contribution from each dimension
        W = cartesian_product(*((W_bar, ) * d))

        # (m, n) = (m, d) . (n, d)^T
        Q = W @ X.T

        # Since numpy multiple along the 1 axis we have to transpose back and forth.
        # (m,) * (m, n)
        Q_cos = (weights * np.cos(Q).T).T
        Q_sin = (weights * np.sin(Q).T).T
        Q = np.concatenate((Q_cos, Q_sin), axis=0)
        return Q.T

    def gauss_hermite_1d(self, m_bar):
        W_bar, weights = np.polynomial.hermite.hermgauss(m_bar)

        # Compute weights (hermgauss already does this...)
        # H = np.polynomial.hermite.Hermite(m_bar - 1)
        # weights = np.sqrt(2) / self.lengthscale * ((2 ** (m_bar-1) *  np.math.factorial(m_bar) * np.sqrt(np.pi)) / m_bar ** # 2 * H(W_bar) ** 2)
        return W_bar, weights


def cartesian_product(*arrays):
    """Pulled from (stackoverflow)[https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points].

    Returns:
        numpy.array -- shape (len(a_1) * ... * len(a_n), n)
    """

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)