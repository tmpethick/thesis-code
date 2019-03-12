import numpy as np
from scipy.spatial.distance import cdist, squareform

import GPy 


class BaseModel(object):
    def get_incumbent(self):
        i = np.argmax(self.Y)
        return self.X[i], self.Y[i]

    def init(self, X, Y, train=True):
        self.X = X
        self.Y = Y
        if train:
            self.fit(self.X, self.Y, is_initial=True)

    def add_observations(self, X_new, Y_new):
        # Update data
        self.X = np.concatenate([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])
        
        self.fit(self.X, self.Y, is_initial=False)

    def fit(self, X, Y, is_initial=True): 
        raise NotImplementedError

    def get_statistics(self, X):
        raise NotImplementedError


class GPModel(BaseModel):
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

    def fit(self, X, Y, is_initial=True):
        if self.gpy_model is None:
            self.gpy_model = GPy.models.GPRegression(X, Y, self.kernel)
            if self.noise_prior:
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

    def get_statistics(self, X):
        """[summary]
        
        Arguments:
            X {[type]} -- [description]
        
        Returns:
            numpy.array -- shape (hyperparams, stats, obs, obj_dim)
        """

        num_X = X.shape[0]
        num_obj = self.Y.shape[1]
        stats = np.zeros((len(self._current_thetas), 2, num_X, num_obj))
        for i, theta in enumerate(self._current_thetas):
            self.gpy_model[:] = theta
            mean, var = self.gpy_model.predict(X)

            stats[i, 0, :] = mean
            stats[i, 1, :] = var
        return stats

    def plot(self, X_line):
        import matplotlib.pyplot as plt

        assert X_line.shape[1] is 1, "GPModel can currently only plot on 1-dim domains."

        stats_all = self.get_statistics(X_line)
        # TODO: MULTIOBJ Plot only first objective for now
        stats_all = stats_all[:, :, :, 0]

        # Iterate stats under different hyperparameters
        for stats in stats_all:
            mean = stats[0]
            var = stats[1]
            plt.fill_between(X_line.reshape(-1), (mean + np.sqrt(var)).reshape(-1), (mean - np.sqrt(var)).reshape(-1), alpha=.2)
            plt.plot(X_line, mean)
            plt.scatter(self.X, self.Y)


class FourierFeatureModel(object):
    # SE specifically

    def __init__(self, gamma=0.1, noise = 0.01, n_features=10):
        self.noise = noise
        self.gamma = gamma
        self.m = n_features
        
        # TODO: right now only SE is supported.
        self.dist = lambda size: np.random.normal(size=size) * (1. / self.gamma)

        self.K_noisy_inv = None
        self.X = None
        self.Y = None

    def fit(self, X, Y):
        n, d = X.shape 
        Q = self.feature_map(X)
        noise_inv = (1 / self.noise)
        small_kernel = self.noise * np.identity(self.m * 2) + Q.T @ Q
        small_kernel_inv = np.linalg.inv(small_kernel)
        self.K_noisy_inv = noise_inv * np.identity(n) - noise_inv * (Q @ small_kernel_inv @ Q.T)
        
        self.X = X
        self.Y = Y

        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        # L_inv = solve_triangular(self.L_.T,
        #                             np.eye(self.L_.shape[0]))
        # self._K_inv = L_inv.dot(L_inv.T)

    def feature_map(self, X):
        n, d = X.shape 
        
        # sample omegas
        W = self.dist(size=(self.m, d))

        # Compute m x n feature map
        Q = X @ W.T
        Q_cos = np.sqrt(2. / self.m) * np.cos(-2 * np.pi * Q)
        Q_sin = np.sqrt(2. / self.m) * np.sin(-2 * np.pi * Q)
        return np.concatenate((Q_cos, Q_sin), axis=1)

    def kernel(self, X1, X2):
        # SE 
        pairwise_sq_dists = cdist(X1, X2, 'sqeuclidean')
        K = np.exp(-pairwise_sq_dists / self.gamma ** 2)
        return K

    def get_statistics(self, X):
        assert self.K_noisy_inv is not None, "`self.fit` needs to be called first."

        kern = self.kernel
        k = kern(X, self.X)
        mu =  k @ self.K_noisy_inv @ self.Y
        cov = kern(X,X) + self.noise ** 2 - k @ self.K_noisy_inv @ k.T

        # (stats, obs, obj_dim)
        return mu, cov

    def plot(self, X_line):
        import matplotlib.pyplot as plt

        assert X_line.shape[1] is 1, "GPModel can currently only plot on 1-dim domains."

        mean, covar = self.get_statistics(X_line)
        std = np.sqrt(np.diagonal(covar))

        X_line = X_line[:,0]
        mean = mean[:,0]

        plt.fill_between(X_line.reshape(-1), (mean + std).reshape(-1), (mean - std).reshape(-1), alpha=.2)
        plt.plot(X_line, mean)
        plt.scatter(self.X, self.Y)



# Quadrature
# p = lambda omega: np.exp(-np.sum(omega ** 2, axis=1).reshape(-1, 1) / 2 * (self.gamma ** 2)) * np.power(
#           (self.gamma / np.sqrt(2 * np.pi)), 1.)*np.power(np.pi / 2,1.)
# Lay out grid:
# ....
# NodesandWeights:
# (nodes, weights) = np.polynomial.hermite.hermgauss(q)
# nodes = np.sqrt(2) * nodes / self.gamma
# #weights = np.sqrt(2) * weights / self.gamma
# weights = weights/np.sqrt(np.pi)
