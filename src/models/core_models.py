import numpy as np

import GPy

from src.experiment.config_helpers import ConfigMixin, lazy_construct_from_module
from src.kernels import GPyRBF


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
        else:
            Y_dir = None

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


class ProbModel(BaseModel):
    def _get_statistics(self, X, full_cov=True):
        raise NotImplementedError


class LinearInterpolateModel(ConfigMixin, ProbModel):
    def _fit(self, X, Y, Y_dir=None):
        pass

    def _get_statistics(self, X, full_cov=True):
        assert X.shape[1] == 1, "LinearInterpolateModel only works in 1D."

        from scipy.interpolate import interp1d
        f = interp1d(self.X[:,0], self.Y[:,0], bounds_error=False, fill_value="extrapolate")
        return f(X), np.zeros(f(X).shape)


class GPModel(ConfigMixin, ProbModel):
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
        self.kernel_constructor = kernel
        self.kernel = None
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

    @classmethod
    def process_config(cls, *, kernel=None, **kwargs):
        import src.kernels as kernels_module
        kernel = lazy_construct_from_module(kernels_module, kernel)
        return dict(
            kernel=kernel,
            **kwargs,
        )

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
        input_dim = X.shape[-1]

        self.kernel = self.kernel_constructor(input_dim)

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
        return super(DerivativeGPModel, self)._fit(X, Y_dir, Y_dir=None)


