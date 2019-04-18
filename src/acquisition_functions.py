import numpy as np
from GPyOpt.acquisitions import AcquisitionBase as GPyOptAcquisitionBase


class AcquisitionBase(object):
    pass

class AcquisitionModelMismatch(AcquisitionBase):
    def __init__(self, *models, beta=2):
        assert len(models) == 2, "It can only compute difference between two models."

        self.model = models[0]
        self.model_compare = models[1]
        self.beta = beta

    def __call__(self, X):
        mean, var = self.model.get_statistics(X, full_cov=False)
        mean2, var2 = self.model_compare.get_statistics(X, full_cov=False)

        # aggregate hyperparameters dimension
        if var.ndim == 3:
            mean = np.mean(mean, axis=0)
            var = np.mean(var, axis=0)

        if mean2.ndim == 3:
            mean2 = np.mean(mean2, axis=0)

        return np.abs(mean - mean2) + self.beta * np.sqrt(var)


class QuadratureAcquisition(AcquisitionBase):
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        mean, var = self.model.get_statistics(X, full_cov=False)

        # aggregate hyperparameters dimension
        if var.ndim == 3:
            var = np.mean(var, axis=0)

        return np.sqrt(var)


# class StableOptAcq(AcquisitionBase):
#     """The derivative approach can be seen as the first taylor approximation of this 
#     (or as the limiting case when delta -> 0).

#     So useful when derivatives are not present or when correlation is across bigger distances.
#     """

#     def __init__(self, *models, beta=2):
#         self.model = models[0]
#         self.beta = beta

#     def __call__(self, X):
#         # TODO: maximize perturbation

#         mean, var = self.model.get_statistics(X, full_cov=False)

#         # aggregate hyperparameters dimension
#         if var.ndim == 3:
#             mean = np.mean(mean, axis=0)
#             var = np.mean(var, axis=0)

#         # Notice that we are interested in |∇f(x)|.
#         return np.abs(mean) + self.beta * np.sqrt(var)


class CurvatureAcquisition(AcquisitionBase):
    def __init__(self, model, use_var=True, beta=0):
        self.model = model
        self.use_var = use_var
        self.beta = beta

    def __call__(self, X):
        mean, var = self.model.get_statistics(X, full_cov=False)

        # aggregate hyperparameters dimension
        if var.ndim == 3:
            mean = np.mean(mean, axis=0)
            var = np.mean(var, axis=0)

        hessian_mean, hessian_var = self.model.predict_hessian(X, full_cov=False)
        hess_norm = np.linalg.norm(hessian_mean, ord='fro', axis=(-2, -1))
        hess_norm = hess_norm[:, None]

        # Remove output dimensions
        if self.use_var:
            return hess_norm * np.sqrt(var) + self.beta * np.sqrt(var)
        else:
            return hess_norm


class CurvatureAcquisitionDistribution(AcquisitionBase):
    """Use this with MCMC sampling (not maximization).
    """

    def __init__(self, model, beta=2):
        self.model = model
        self.beta = beta

    def __call__(self, X):
        mean, var = self.model.get_statistics(X, full_cov=False)

        # aggregate hyperparameters dimension
        if var.ndim == 3:
            mean = np.mean(mean, axis=0)
            var = np.mean(var, axis=0)

        hessian_mean, hessian_var = self.model.predict_hessian(X, full_cov=False)
        hess_norm = np.linalg.norm(hessian_mean, ord='fro', axis=(-2,-1))

        return hess_norm + self.beta * np.sqrt(var)


class DerivativeAcquisition(AcquisitionBase):
    """Redundant but helpful for structure.
    Should be extracted into a multi-object GP.

    Usually we are optimizing R^D -> R.
    Now R^D -> R^D...
    
    First consider low dim:
        - How do we weight the D partial gradients? 
            Using max we will explore peaks in any direction.
            Assume we used mean. If all "the action" was along one dimension this might not be explored.
        
        - Alternatively we could do the aggregation BEFORE fitting a GP thus avoiding multi-objectivity...

        - What does exploration/exploitation mean in this setting?
            Exploration will now learn an accurately representation of the gradient.
            Seems like we will cover the whole domain well in the long run (i.e. not get stuck exploiting).

    Now consider high-dim:
        (Note: This approach becomes infeasible if dimensionality is not reduced (multi-objective GP is expensive I think!).
        For Manifold learning we need gradients to flow through the function transformation.)

    Conclusion:
        Let's try with multi-objective GP on derivatives and max{mu + beta * var}.
        First: 1D.
    """

    def __init__(self, *models, beta=2):
        self.model = models[0]
        self.derivative_model = models[1]
        self.beta = beta

    def __call__(self, X):
        mean, var = self.derivative_model.get_statistics(X, full_cov=False)

        # aggregate hyperparameters dimension
        if var.ndim == 3:
            mean = np.mean(mean, axis=0)
            var = np.mean(var, axis=0)

        # Notice that we are interested in |∇f(x)|.
        return np.abs(mean) + self.beta * np.sqrt(var)

# ------------------ GPyOpt --------------------


class GPyOptQuadratureAcquisition(GPyOptAcquisitionBase):
    """
    GP-Confidence Bound acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost
    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(GPyOptQuadratureAcquisition, self).__init__(model, space, optimizer)
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound
        """
        m, s = self.model.predict(x)
        f_acqu = s
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        f_acqu = s
        df_acqu = dsdx
        return f_acqu, df_acqu


class GPyOptAcquisitionModelMismatch(GPyOptAcquisitionBase):
    """
    GP-Confidence Bound acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: does not allow to be used with cost
    """

    analytical_gradient_prediction = False

    def __init__(self, model, model2, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(GPyOptAcquisitionModelMismatch, self).__init__(model, space, optimizer)
        if cost_withGradients is not None:
            print('The set cost function is ignored! LCB acquisition does not make sense with cost.')
        self.model2 = model2
        self.exploration_weight = exploration_weight

    def _compute_acq(self, x):
        """
        Computes the GP-Lower Confidence Bound
        """
        m, s = self.model.predict(x)
        m2, s2 = self.model2.predict(x)

        f_acqu = np.abs(m - m2) + self.exploration_weight * np.sqrt(s)
        return f_acqu

    # def _compute_acq_withGradients(self, x):
    #     """
    #     Computes the GP-Lower Confidence Bound and its derivative
    #     """
    #     m, s, dmdx, dsdx = self.model.predict_withGradients(x)
    #     m2, s2, dmdx2, dsdx2 = self.model2.predict_withGradients(x)
    #
    #     f_acqu = s
    #     df_acqu = dsdx
    #     return f_acqu, df_acqu

