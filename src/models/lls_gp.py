import numpy as np
from scipy.optimize import differential_evolution

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ConstantKernel as C, Matern
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from gp_extras.kernels import LocalLengthScalesKernel

from src.models.models import BaseModel


class LocalLengthScaleGPModel(BaseModel):
    def __init__(self):
        super(LocalLengthScaleGPModel, self).__init__()
        self.model: GaussianProcessRegressor = None

    def fit(self, X, Y, is_initial=True):
        super(LocalLengthScaleGPModel, self).fit(X, Y, is_initial=is_initial)

        # Define custom optimizer for hyperparameter-tuning of non-stationary kernel
        def de_optimizer(obj_func, initial_theta, bounds):
            res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                         bounds, maxiter=20, disp=False, polish=False)
            return res.x, obj_func(res.x, eval_gradient=False)

        kernel_lls = C(1.0, (1e-10, 1000)) \
                     * LocalLengthScalesKernel.construct(X, l_L=0.1, l_U=2.0, l_samples=5)
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
        return y_mean_lls, y_std_lls ** 2

    def get_lengthscale(self, X):
        kern = self.model.kernel_
        return kern.k2.theta_gp * 10 ** kern.k2.gp_l.predict(X)

class LocalLengthScaleGPBaselineModel(BaseModel):
    def __init__(self):
        super(LocalLengthScaleGPBaselineModel, self).__init__()
        self.model = None

    def fit(self, X, Y, is_initial=True):
        super(LocalLengthScaleGPBaselineModel, self).fit(X, Y, is_initial=True)

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

        y_mean_lls, y_std_lls = self.model.predict(X, return_std=True)
        return y_mean_lls, y_std_lls ** 2

    def get_lengthscale(self, X):
        # TODO: shouldn't we get a GP (with var) over the length scale?
        return np.ones_like(X) * self.model.kernel_.k2.length_scale


