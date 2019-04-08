import numpy as np
import matplotlib.pyplot as plt

import GPy
from scipy.spatial.distance import cdist, squareform
from scipy.linalg import cho_factor, cho_solve


class Transformer(object):
    @property
    def output_dim(self):
        raise NotImplementedError

    def fit(self, X, Y, Y_dir=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class ActiveSubspace(Transformer):
    """
    Requires feeding `M = α k log(m)` samples to `self.fit` 
    where α=5..10, m is actually dim, and k<m.
    """

    def __init__(self, k=10, output_dim=None):
        # How many eigenvalues are considered. We do not consider all 
        # eigenvalues (i.e. k=m) as the samples required increases in k.
        self.k = k

        # Uses a fixed output dim if not zero
        self._output_dim = output_dim

        # Used to decide when a big change occurs (eig[i] > thresholds_factor * eig[i+1])
        self.threshold_factor = 10
        
        self.vals = None
        self.W = None

    @property
    def output_dim(self):
        if self._output_dim is not None:
            return self._output_dim
        else:
            raise Exception("No promises can be made about `output_dim` since it is dynamically determind.")

    def _get_active_subspace_index(self, vals):
        """ Given list of eigenvectors sorted in ascented order (e.g. `vals = [1,2,30,40,50]`) return the index `i` being the first occurrence of a big change in value (in the example `i=2`).
        """
        if self._output_dim is not None:
            return -self._output_dim

        # Only consider k largest
        vals = vals[-self.k:]

        for i in reversed(range(len(vals))):
            big = vals[i]
            small = vals[i - 1]
            if (big / self.threshold_factor > small):
                return i
        return 0

    def fit(self, X, Y, Y_dir):
        """[summary]

        Arguments:
            X {[type]} -- input
            Y {[type]} -- (unused) function evaluation
            Y_dir {[type]} -- function gradient
        """

        N = X.shape[0]
        CN = (Y_dir.T @ Y_dir) / N

        # find active subspace
        vals, vecs = np.linalg.eigh(CN)
        self.vals = vals

        i = self._get_active_subspace_index(vals)

        self.W = vecs[:, i:]

    def transform(self, X):
        assert self.W is not None, "Call `self.fit` first"
        # (W.T @ X.T).T
        return X @ self.W

    def plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(9, 5))
        x = np.arange(1, 11)
        ax[0].plot(x, self.vals, ".", ms=15)
        ax[0].set_xlabel("Eigenvalues")
        ax[0].set_xticks(x)
        ax[0].set_ylabel("$\lambda$")

        ax[1].plot(x, self.W, ".", ms=15)
        ax[1].set_xlabel("Input dimension")
        ax[1].set_xticks(x)
        ax[1].set_ylabel("Magnitude of W")

        fig.tight_layout()
        return fig


class BaseModel(object):
    def __repr__(self):
        return "{}".format(type(self).__name__)

    def get_incumbent(self):
        i = np.argmax(self.Y)
        return self.X[i], self.Y[i]

    def init(self, X, Y, Y_dir=None, train=True):
        self.X = X
        self.Y = Y
        self.Y_dir = Y_dir
        if train:
            self._fit(self.X, self.Y, Y_dir=self.Y_dir, is_initial=True)

    def add_observations(self, X_new, Y_new, Y_dir_new=None):
        # Update data
        self.X = np.concatenate([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])

        if self.Y_dir is not None:
            self.Y_dir = np.concatenate([self.Y_dir, Y_dir_new])
        
        self._fit(self.X, self.Y, Y_dir=self.Y_dir, is_initial=False)

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        raise NotImplementedError

    def get_statistics(self, X, full_cov=True):
        raise NotImplementedError

    def plot(self, X_line, ax=None):
        import matplotlib.pyplot as plt
        ax = ax if ax is not None else plt

        assert X_line.shape[1] is 1, "GPModel can currently only plot on 1-dim domains."

        mean, var = self.get_statistics(X_line, full_cov=False)

        # Add hyperparam dim if not already there.
        if mean.ndim == 2:
            mean = mean[None, :]
            var = var[None, :]

        n_hparams = mean.shape[0]
        # TODO: MULTIOBJ Plot only first objective for now

        # Iterate stats under different hyperparameters
        ax.scatter(self.X, self.Y)
        for (mean, var) in ((mean[i], var[i]) for i in range(n_hparams)):
            ax.plot(X_line, mean)
            ax.fill_between(X_line.reshape(-1), 
                            (mean + np.sqrt(var)).reshape(-1), 
                            (mean - np.sqrt(var)).reshape(-1), alpha=.2)


class ProbModel(BaseModel):
    def get_statistics(self, X, full_cov=True):
        raise NotImplementedError


class LinearInterpolateModel(ProbModel):
    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        pass

    def get_statistics(self, X, full_cov=True):
        assert X.shape[1] == 1, "LinearInterpolateModel only works in 1D."

        from scipy.interpolate import interp1d
        f = interp1d(self.X[:,0], self.Y[:,0], bounds_error=False, fill_value="extrapolate")
        return f(X), np.zeros(f(X).shape)


class GPModel(ProbModel):
    def  __init__(self, 
            kernel, 
            noise_prior=None,
            do_optimize=False, 
            num_mcmc=0, 
            n_burnin=100,
            subsample_interval=10,
            step_size=1e-1,
            leapfrog_steps=20):

        self.kernel = kernel
        self.noise_prior = noise_prior
        self.do_optimize = do_optimize
        self.num_mcmc = num_mcmc

        self.n_burnin = n_burnin
        self.subsample_interval = subsample_interval
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps

        self.gpy_model = None
        self.has_mcmc_warmup = False 
        self.output_dim = None

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        assert X.shape[0] == Y.shape[0], \
            "X and Y has to match size. It was {} and {} respectively".format(X.shape[0], Y.shape[0])

        self.output_dim = Y.shape[-1]

        if self.gpy_model is None:
            self.gpy_model = GPy.models.GPRegression(X, Y, self.kernel)
            if self.noise_prior:
                if isinstance(self.noise_prior, float):
                    self.gpy_model.Gaussian_noise.fix(self.noise_prior)
                else:
                    self.gpy_model.Gaussian_noise.variance.set_prior(self.noise_prior)
        else:
            self.gpy_model.set_XY(X, Y)

        if self.do_optimize:
            if self.num_mcmc > 0:
                if not self.has_mcmc_warmup:
                    # Most likely hyperparams given data
                    self.hmc = GPy.inference.mcmc.HMC(self.gpy_model, stepsize=self.step_size)
                    self.has_mcmc_warmup = True

                ss = self.hmc.sample(num_samples=self.n_burnin + self.num_mcmc * self.subsample_interval, hmc_iters=self.leapfrog_steps)
                self._current_thetas = ss[self.n_burnin::self.subsample_interval]
            else:
                self.gpy_model.randomize()
                self.gpy_model.optimize()
                self._current_thetas = [self.gpy_model.param_array]
        else:
            self._current_thetas = [self.gpy_model.param_array]

    def get_statistics(self, X, full_cov=True):
        """[summary]
        
        Arguments:
            X {[type]} -- [description]
        
        Returns:
            numpy.array -- shape (hyperparams, stats, obs, obj_dim)
        """

        num_X = X.shape[0]
        num_obj = self.output_dim
        mean = np.zeros((len(self._current_thetas), num_X, num_obj))
        covar = None
        var = None

        if full_cov:
            covar = np.zeros((len(self._current_thetas), num_X, num_X, num_obj)) 
        else:
            var = np.zeros((len(self._current_thetas), num_X, num_obj))
     
        for i, theta in enumerate(self._current_thetas):
            self.gpy_model[:] = theta
            mean_i, covar_i = self.gpy_model.predict(X, full_cov=True)
            covar_i = covar_i if len(covar_i.shape) == 3 else covar_i[:, :, None] 

            mean[i, :] = mean_i

            if full_cov:
                covar[i, :] = covar_i
            else:
                var_i = np.diagonal(covar_i, axis1=0, axis2=1)
                var_i = np.moveaxis(var_i, -1, 0)
                var[i, :] = var_i

        if full_cov:
            return mean, covar
        else:
            return mean, var


class DerivativeGPModel(GPModel):
    """Fits a GP to the derivative of a function.
    OBS: only support 1D. 
    Exists only to easy integration into the current configuration setup.
    """

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        return super(DerivativeGPModel, self)._fit(X, Y_dir, Y_dir=None, is_initial=is_initial)


class TransformerModel(ProbModel):
    """Proxy a ProbModel through a Transformer first.
    """


    def __init__(self, *, transformer: Transformer, prob_model: ProbModel):
        self.transformer = transformer
        self.prob_model = prob_model

    def __repr__(self):
        return "{}<{},{}>".format(type(self).__name__, 
                                  type(self.transformer).__name__, 
                                  type(self.prob_model).__name__)

    # TODO: self.kernel

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        self.transformer.fit(X, Y, Y_dir=Y_dir)
        X = self.transformer.transform(X)
        return self.prob_model._fit(X, Y, Y_dir=Y_dir, is_initial=is_initial)

    def get_statistics(self, X, full_cov=True):
        X = self.transformer.transform(X)
        return self.prob_model.get_statistics(X, full_cov=full_cov)


class GPVanillaModel(ProbModel):
    def __init__(self, gamma=0.1, noise=0.01):
        self.K_noisy_inv = None
        self.gamma = gamma
        self.noise = noise

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        n, d = X.shape
        kern = self.kernel(X,X) + self.noise * np.identity(n)
        
        self.K_noisy_inv = np.linalg.inv(kern)
        self.cho_decomp = cho_factor(kern)

    def kernel(self, X1, X2):
        # SE kernel
        pairwise_sq_dists = cdist(X1, X2, 'sqeuclidean')
        K = np.exp(-pairwise_sq_dists / self.gamma ** 2)
        return K
    
    def get_statistics(self, X, full_cov=True):
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


class LowRankGPModel(ProbModel):
    def __init__(self, gamma=0.1, noise = 0.01, n_features=10):
        self.noise = noise
        self.gamma = gamma
        self.m = n_features
        
        self.K_noisy_inv = None
        self.X = None
        self.Y = None

    def maximum_likelihood(self):
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/gaussian_process/gpr.py
        # GPML p. 131 (pdf)
        pass

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        n, d = X.shape 
        Q = self.feature_map(X)
        noise_inv = (1 / self.noise)
        small_kernel = self.noise * np.identity(self.m) + Q.T @ Q
        small_kernel_inv = np.linalg.inv(small_kernel)
        self.K_noisy_inv = noise_inv * np.identity(n) - noise_inv * (Q @ small_kernel_inv @ Q.T)

        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        # L_inv = solve_triangular(self.L_.T,
        #                             np.eye(self.L_.shape[0]))
        # self._K_inv = L_inv.dot(L_inv.T)

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
            return Q1 @ Q1.T
        else:
            Q1 = self.feature_map(X1)
            Q2 = self.feature_map(X2)
            return Q1 @ Q2.T

    def get_statistics(self, X, full_cov=True):
        assert self.K_noisy_inv is not None, "`self.fit` needs to be called first."

        kern = self.kernel
        k = kern(X, self.X)
        mu =  k @ self.K_noisy_inv @ self.Y
        cov = kern(X,X) + self.noise - k @ self.K_noisy_inv @ k.T

        # (stats, obs, obj_dim)
        if full_cov:
            return mu, cov
        else:
            return mu, np.diagonal(cov)[:, None]


class RandomFourierFeaturesModel(LowRankGPModel):
    """Based on analysis in [1] we choose the unbiased variant as it has strictly smaller variance for the Squared Exponential.

    [1]: https://www.cs.cmu.edu/~dsutherl/papers/rff_uai15.pdf
    """

    def __init__(self, gamma=0.1, noise=0.01, n_features=10):
        assert n_features % 2 == 0, "`n_features` has to be even."
        
        super(RandomFourierFeaturesModel, self).__init__(gamma=gamma, noise=noise, n_features=n_features)

        # TODO: right now only SE is supported.
        self.W = None

    def spectral_kernel(self, size):
        if self.W is not None: 
            return self.W 
        else:
            self.W = np.random.normal(size=size) * (1.0 / self.gamma)
            return self.W

    def feature_map(self, X):
        n, d = X.shape 

        # sample omegas
        W = self.spectral_kernel(size=(self.m // 2, d))

        # Compute m x n feature map
        Z = W @ X.T
        uniform_weight = np.sqrt(2.0 / self.m)
        Q_cos = uniform_weight * np.cos(Z)
        Q_sin = uniform_weight * np.sin(Z)

        # n x m
        return np.concatenate((Q_cos, Q_sin), axis=0).T


class EfficientLinearModel(LowRankGPModel):
    def __init__(self, gamma=0.1, noise=0.01, n_features=None):
        super(EfficientLinearModel, self).__init__(gamma=gamma, noise=noise, n_features=n_features)

    def feature_map(self, X):
        n, d = X.shape
        return X


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


class QuadratureFourierFeaturesModel(LowRankGPModel):

    def __init__(self, gamma=0.1, noise=0.01, n_features=100):
        assert n_features % 2 == 0, "`n_features` has to be even."

        super(QuadratureFourierFeaturesModel, self).__init__(gamma=gamma, noise=noise, n_features=n_features)
        
        # Not used since the structure is implicit in the particular use of the Gauss-Hermite Scheme.
        self.spectral_kernel_pdf = lambda w: np.exp(- np.square(w).dot(np.square(self.gamma))/ 2)

    def feature_map(self, X):
        # m is now the number of nodes in the dense grid.
        # compute the number of nodes in 1D by the relationship m = 2 * m_bar^d.
        n, d = X.shape
        self.m_bar = int(np.power(self.m // 2, 1. / d))
        self.m = 2 * self.m_bar ** d

        # Compute (nodes, weights) in 1D
        W_bar, weights = self.gauss_hermite_1d(self.m_bar)
        weights = np.sqrt(2) / np.power(self.gamma, 2) * weights

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
        # weights = np.sqrt(2) / self.gamma * ((2 ** (m_bar-1) *  np.math.factorial(m_bar) * np.sqrt(np.pi)) / m_bar ** # 2 * H(W_bar) ** 2)
        return W_bar, weights
