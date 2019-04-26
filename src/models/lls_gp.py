import numpy as np
from scipy.optimize import differential_evolution

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ConstantKernel as C, Matern


from src.models.models import BaseModel


class LocalLengthScaleGPModel(BaseModel):
    def __init__(self, n_optimizer_iter=20, l_samples=5):
        super(LocalLengthScaleGPModel, self).__init__()
        self.model: GaussianProcessRegressor = None
        self.l_samples = l_samples
        self.lls_kernel = None
        self.n_optimizer_iter = n_optimizer_iter

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        from gp_extras.kernels import LocalLengthScalesKernel
        # Define custom optimizer for hyperparameter-tuning of non-stationary kernel
        # This is required here because the log-marginal-likelihood for the LocalLengthScalesKernel is highly
        # multi-modal, which is problematic for gradient-based methods like L-BFGS.
        def de_optimizer(obj_func, initial_theta, bounds):
            res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                         bounds, maxiter=self.n_optimizer_iter, disp=True, polish=False)
            return res.x, obj_func(res.x, eval_gradient=False)

        self.lls_kernel = LocalLengthScalesKernel.construct(X, l_L=0.1, l_U=2.0, l_samples=self.l_samples)
        kernel_lls = C(1.0, (1e-10, 1000)) * self.lls_kernel
        gp_lls = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer)

        # Fit GPs
        gp_lls.fit(X, Y)
        self.model = gp_lls

        print("Learned kernel LLS: %s" % gp_lls.kernel_)
        print("Log-marginal-likelihood LLS: %s" \
              % gp_lls.log_marginal_likelihood(gp_lls.kernel_.theta))

    def get_statistics(self, X, full_cov=True):
        assert full_cov is not True, "Full covariance is not supported yet."
        assert self.model is not None, "Call `self.fit` before predicting."

        y_mean_lls, y_std_lls = self.model.predict(X, return_std=True)
        return y_mean_lls, (y_std_lls ** 2)[:, None]

    def get_lengthscale(self, X):
        # TODO: shouldn't we get a GP (with var) over the length scale?
        kern = self.model.kernel_
        return kern.k2.theta_gp * 10 ** kern.k2.gp_l.predict(X)


class LocalLengthScaleGPBaselineModel(BaseModel):
    def __init__(self):
        super(LocalLengthScaleGPBaselineModel, self).__init__()
        self.model = None

    def _fit(self, X, Y, Y_dir=None, is_initial=True):
        kernel_matern = C(1.0, (1e-10, 1000)) \
            * Matern(length_scale_bounds=(1e-1, 1e3), nu=1.5)
        gp_matern = GaussianProcessRegressor(kernel=kernel_matern)

        gp_matern.fit(X, Y)
        self.model = gp_matern

        print("Learned kernel Matern: %s" % gp_matern.kernel_)
        print("Log-marginal-likelihood Matern: %s" \
              % gp_matern.log_marginal_likelihood(gp_matern.kernel_.theta))

    def get_statistics(self, X, full_cov=True):
        assert full_cov is not True, "Full covariance is not supported yet."
        assert self.model is not None, "Call `self.fit` before predicting."

        y_mean, y_std = self.model.predict(X, return_std=True)
        return y_mean, (y_std ** 2)[:, None]

    def get_lengthscale(self, X):
        return np.ones_like(X) * self.model.kernel_.k2.length_scale


