import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import fmin_l_bfgs_b

import GPy
from scipy.spatial.distance import cdist
from scipy.linalg import cho_factor, cho_solve

from src.kernels import GPyRBF


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean


class Normalizer(object):
    def __init__(self, X_train):
        self.X, self._X_mean, self._X_std = zero_mean_unit_var_normalization(X_train)

    def get_transformed(self):
        return self.X

    def normalize(self, X):
        """Transform new X into rescaling constructed by X_train.
        """
        X, _, _ = zero_mean_unit_var_normalization(X, mean=self._X_mean, std=self._X_std)
        return X

    def denormalize(self, X):
        """Take normalized X and turn back into unnormalized version.
        (used for turning predicted output trained on normalized version into original domain.)
        """
        X = zero_mean_unit_var_unnormalization(X, self._X_mean, self._X_std)
        return X

    def denormalize_variance(self, X_var):
        return (X_var*(self._X_std**2))

    def denormalize_covariance(self, covariance):
        return (covariance[..., None]*(self._X_std**2))


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

    def __init__(self, k=10, output_dim=None, threshold_factor=10):
        # How many eigenvalues are considered. We do not consider all 
        # eigenvalues (i.e. k=m) as the samples required increases in k.
        self.k = k

        # Uses a fixed output dim if not zero
        self._output_dim = output_dim

        # Used to decide when a big change occurs (eig[i] > thresholds_factor * eig[i+1])
        self.threshold_factor = threshold_factor
        
        self.vals = None
        self.W = None

    @property
    def output_dim(self):
        if self._output_dim is not None:
            return self._output_dim
        elif self.W is not None:
            return self.W.shape[-1]
        else:
            raise Exception("No promises can be made about `output_dim` since it is dynamically determind.")

    def _get_active_subspace_index(self, vals):
        """ Given list of eigenvectors sorted in ascended order (e.g. `vals = [1,2,30,40,50]`) return the index `i` being the first occurrence of a big change in value (in the example `i=2`).
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
        x = np.arange(1, self.vals.shape[0] + 1)
        ax[0].plot(x, self.vals, ".", ms=15)
        ax[0].set_xlabel("Eigenvalues")
        ax[0].set_xticks(x)
        ax[0].set_ylabel("$\lambda$")

        ax[1].plot(x, np.linalg.norm(self.W, axis=1), ".", ms=15)
        ax[1].set_xlabel("Input dimension")
        ax[1].set_xticks(x)
        ax[1].set_ylabel("Magnitude of W")

        fig.tight_layout()
        return fig


class BaseModel(object):
    def __init__(self):
        self._X = None
        self._Y = None

    def __repr__(self):
        return "{}".format(type(self).__name__)

    def get_incumbent(self):
        i = np.argmax(self._Y)
        return self._X[i], self._Y[i]

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def init(self, X, Y, Y_dir=None):
        self._X = X
        self._Y = Y
        self.Y_dir = Y_dir

        self._fit(X, Y, Y_dir=self.Y_dir)

    def add_observations(self, X_new, Y_new, Y_dir_new=None):
        assert self._X is not None, "Call init first"

        # Update data
        X = np.concatenate([self._X, X_new])
        Y = np.concatenate([self._Y, Y_new])

        if self.Y_dir is not None:
            Y_dir = np.concatenate([self.Y_dir, Y_dir_new])

        self.init(X, Y, Y_dir)

    def _fit(self, X, Y, Y_dir=None):
        # TODO: get rid of this dirty hack.
        raise NotImplementedError

    def _get_statistics(self, X, full_cov=True):
        # TODO: get rid of this dirty hack.
        raise NotImplementedError

    def get_statistics(self, X, full_cov=True):
        return self._get_statistics(X, full_cov=full_cov)

    def get_mean(self, X):
        return self.get_statistics(X, full_cov=False)[0]

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
        ax.scatter(self._X, self._Y)
        for (mean, var) in ((mean[i], var[i]) for i in range(n_hparams)):
            ax.plot(X_line, mean)
            ax.fill_between(X_line.reshape(-1), 
                            (mean + np.sqrt(var)).reshape(-1), 
                            (mean - np.sqrt(var)).reshape(-1), alpha=.2)
        X_line


class NormalizerModel(BaseModel):
        def __init__(self, model, normalize_input=True, normalize_output=True):
            super().__init__()
            self.model = model
            self.X_normalizer = None
            self.Y_normalizer = None
            self._normalize_input = normalize_input
            self._normalize_output = normalize_output

        def _normalize(self, X, Y):
            if self._normalize_input:
                self.X_normalizer = Normalizer(X)
                X = self.X_normalizer.get_transformed()

            if self._normalize_output:
                self.Y_normalizer = Normalizer(Y)
                Y = self.Y_normalizer.get_transformed()

            return X, Y

        def init(self, X, Y, Y_dir=None):
            # TODO: do not "copy/paste" behaviour from BaseModel.
            self._X = X
            self._Y = Y
            self.Y_dir = Y_dir

            X, Y = self._normalize(X, Y)
            self.model.init(X, Y, Y_dir)

        def add_observations(self, X_new, Y_new, Y_dir_new=None):
            # TODO: do not "copy/paste" behaviour from BaseModel.
            assert self._X is not None, "Call init first"

            # Update data
            X = np.concatenate([self._X, X_new])
            Y = np.concatenate([self._Y, Y_new])

            if self.Y_dir is not None:
                Y_dir = np.concatenate([self.Y_dir, Y_dir_new])

            self.init(X, Y, Y_dir)

        def get_statistics(self, X, full_cov=True):
            if self._normalize_input:
                X = self.X_normalizer.normalize(X)

            mean, covar = self.model.get_statistics(X, full_cov=full_cov)

            if self._normalize_output:
                mean = self.Y_normalizer.denormalize(mean)
                if full_cov:
                    covar = self.Y_normalizer.denormalize_covariance(covar)
                else:
                    covar = self.Y_normalizer.denormalize_variance(covar)
            return mean, covar



class ProbModel(BaseModel):
    def _get_statistics(self, X, full_cov=True):
        raise NotImplementedError


class LinearInterpolateModel(ProbModel):
    def _fit(self, X, Y, Y_dir=None):
        pass

    def _get_statistics(self, X, full_cov=True):
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
            leapfrog_steps=20,
            mean_prior=None):
        super().__init__()

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
        self.mean_prior = mean_prior

    def __repr__(self):
        return "ExactGP"

    def get_common_hyperparameters(self):
        return {
            'lengthscale': self.kernel.lengthscale,
            'noise': self.gpy_model.Gaussian_noise.variance,
        }

    def _fit(self, X, Y, Y_dir=None):
        assert X.shape[0] == Y.shape[0], \
            "X and Y has to match size. It was {} and {} respectively".format(X.shape[0], Y.shape[0])

        self.output_dim = Y.shape[-1]

        if self.mean_prior is not None:
            Y_mean = np.mean(Y)
            mean_function = GPy.mappings.constant.Constant(X.shape[1], Y.shape[1], Y_mean)
        else:
            mean_function = None

        if self.gpy_model is None:
            self.gpy_model = GPy.models.GPRegression(X, Y, self.kernel, mean_function=mean_function)
            if self.noise_prior:
                if isinstance(self.noise_prior, float) or isinstance(self.noise_prior, int):
                    assert self.noise_prior != 0, "Zero noise is ignored. Use e.g. 1e-10 instead."
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

    def _predict(self, X, func, full_cov=True):
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

        # TODO: could be very inefficient if kernel is recomputed...
        if full_cov:
            covar = np.zeros((len(self._current_thetas), num_X, num_X, num_obj)) 
        else:
            var = np.zeros((len(self._current_thetas), num_X, num_obj))
     
        for i, theta in enumerate(self._current_thetas):
            self.gpy_model[:] = theta
            mean_i, covar_i = func(X, full_cov=True)

            if len(covar_i.shape) != 3:
                covar_i = covar_i[:, :, None] 

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

    def _get_statistics(self, X, full_cov=True):
        return self._predict(X, self.gpy_model.predict, full_cov=full_cov)

    # def predict_jacobian_1sample(self, X, full_cov=True):
    #     N_new = 1
    #     N, D = self.X.shape
    #     x = X[0]

    #     self.gpy_model[:] = self._current_thetas[0]

    #     l_inv = 1 / self.gpy_model.kern.lengthscale
    #     Lambda_inv = np.diag(l_inv.repeat(D))

    #     X_tilde = x - self.X
    #     np.testing.assert_array_equal(X_tilde.shape, [N,D])

    #     alpha = self.gpy_model.posterior.woodbury_vector
    #     k_Xx = self.gpy_model.kern.K(self.X, np.array([x]))
    #     np.testing.assert_array_equal(k_Xx.shape, [N, D])

    #     interm = np.dot(X_tilde.T, (k_Xx * alpha))
    #     return - Lambda_inv @ interm, None
    
    # def predict_jacobian_1sample(self, X, full_cov=True):
    #     N_new = 1
    #     N, D = self.X.shape
    #     x = X[0]

    #     self.gpy_model[:] = self._current_thetas[0]

    #     l_inv = 1 / self.gpy_model.kern.lengthscale
    #     Lambda_inv = np.diag(l_inv.repeat(D))

    #     X_tilde = x - self.X
    #     np.testing.assert_array_equal(X_tilde.shape, [N,D])

    #     alpha = self.gpy_model.posterior.woodbury_vector
    #     k_Xx = self.gpy_model.kern.K(self.X, np.array([x]))
    #     np.testing.assert_array_equal(k_Xx.shape, [N, N_new])

    #     interm = np.einsum("jk,ji->k", X_tilde, (k_Xx * alpha))
    #     return - np.einsum("kk,k->k", Lambda_inv, interm), None

    def predict_jacobian(self, X, full_cov=True):
        # Normalize the input (normalization of output is dealt with by GPy)
        X = self.normalizer.normalize_X(X)

        N_new = X.shape[0]
        N, D = self.X.shape

        self.gpy_model[:] = self._current_thetas[0]

        l_inv = 1 / self.gpy_model.kern.lengthscale
        Lambda_inv = np.diag(l_inv.repeat(D))

        X_tilde = X[None, :, :] - self.X[:, None, :]
        np.testing.assert_array_equal(X_tilde.shape, [N, N_new, D])

        alpha = self.gpy_model.posterior.woodbury_vector
        k_Xx = self.gpy_model.kern.K(self.X, X)

        interm = np.einsum("jik,ji->ik", X_tilde, (k_Xx * alpha))
        return - np.einsum("kk,ik->ik", Lambda_inv, interm), None

    def predict_hessian(self, X, full_cov=True):
        """Does not support variance yet.
        Does not support hyperparameters either...

        returns np.ndarray (N*, D, D)
        """
        # Normalize the input (normalization of output is dealt with by GPy)
        X = self.normalizer.normalize_X(X)

        assert isinstance(self.gpy_model.kern, GPyRBF), "Only support RBF for now"

        N_new = X.shape[0]
        N, D = self.X.shape

        self.gpy_model[:] = self._current_thetas[0]

        # (D, D)
        l_inv = 1 / self.gpy_model.kern.lengthscale
        Lambda_inv = np.diag(l_inv.repeat(D))

        # (N, O)
        alpha = self.gpy_model.posterior.woodbury_vector

        X_tilde = X[None, :, :] - self.X[:, None, :]
        np.testing.assert_array_equal(X_tilde.shape, [N, N_new, D])

        k_Xx = self.gpy_model.kern.K(self.X, X)
        np.testing.assert_array_equal(k_Xx.shape, [N, N_new])

        #dK_Xx = self.gpy_model.kern.dK_dr_via_X(self.X, X)
        # dK_Xx = self.gpy_model.kern.gradients_X(alpha, X, self.X)
        # TODO: X_hat is just -X_tilde
        X_hat = -X_tilde
        np.testing.assert_array_equal(X_hat.shape, [N, N_new, D])

        dK_Xx = np.einsum("ji,jik->jik", (k_Xx * alpha), (X_hat @ Lambda_inv))
        np.testing.assert_array_equal(dK_Xx.shape, [N, N_new, D])

        Ones = np.ones((N_new, N, D, D))

        # Take dot product along N axis.
        prod_rule = np.einsum("ijkl,ji->ikl", Ones, k_Xx * alpha) \
            + np.einsum("jik,jil->ikl", X_tilde, dK_Xx)

        # TODO: what axis is it multiplying?
        hessian_mean = -np.einsum("kk,ikl->ikl", Lambda_inv, prod_rule)
        return hessian_mean, None


class DerivativeGPModel(GPModel):
    """Fits a GP to the derivative of a function.
    OBS: only support 1D. 
    Exists only to easy integration into the current configuration setup.
    """

    def _fit(self, X, Y, Y_dir=None):
        return super(DerivativeGPModel, self)._fit(X, Y_dir, Y_dir=None, is_initial=is_initial)


class TransformerModel(ProbModel):
    """Proxy a ProbModel through a Transformer first.
    """

    def __init__(self, *, transformer: Transformer, prob_model: ProbModel):
        super(TransformerModel, self).__init__()
        self.transformer = transformer
        self.prob_model = prob_model

    def __repr__(self):
        return "{}<{},{}>".format(type(self).__name__, 
                                  type(self.transformer).__name__, 
                                  type(self.prob_model).__name__)

    # TODO: proxy the rest of the interface to prob_model (incl self.X and self.Y)

    def init(self, X, Y, Y_dir=None, train=True):
        self.X = X
        self.Y = Y
        self.Y_dir = Y_dir
        
        self.transformer.fit(self.X, self.Y, Y_dir=self.Y_dir)
        X = self.transformer.transform(self.X)

        self.prob_model.init(X, Y, Y_dir=Y_dir, train=train)

    def add_observations(self, X_new, Y_new, Y_dir_new=None):
        self.X = np.concatenate([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])

        if self.Y_dir is not None:
            self.Y_dir = np.concatenate([self.Y, Y_dir_new])

        self.transformer.fit(self.X, self.Y, Y_dir=self.Y_dir)
        X = self.transformer.transform(self.X)

        # Necessary to call init again since we do not know if the transformation of previous observation stayed the same.
        self.prob_model.init(X, self.Y, Y_dir=self.Y_dir, train=True)
        # self.prob_model.add_observations(X_new, Y_new, Y_dir_new=Y_dir_new)
  
    def _get_statistics(self, X, full_cov=True):
        X = self.transformer.transform(X)
        mean, covar = self.prob_model.get_statistics(X, full_cov=full_cov)
        return mean, covar


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


class LowRankGPModel(ProbModel):
    def __init__(self, kernel, noise = 0.01, n_features=10, do_optimize=False, n_restarts_optimizer=10):
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


class EfficientLinearModel(LowRankGPModel):
    def __init__(self, noise=0.01, n_features=None):
        super(EfficientLinearModel, self).__init__(noise=noise, n_features=n_features)

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
