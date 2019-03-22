import numpy as np

class AcquisitionRelative(object):
    def __init__(self, *models, beta=2):
        assert len(models) == 2, "It can only compute difference between two models."

        self.model = models[0]
        self.model_compare = models[1]
        self.beta = beta

    def __call__(self, X):
        mean, var = self.model.get_statistics(X, full_cov=False)
        mean2, var2 = self.model_compare.get_statistics(X, full_cov=False)
        # aggregate hyperparameters dimension
        mean = np.mean(mean, axis=0)
        mean2 = np.mean(mean2, axis=0)
        var = np.mean(var, axis=0)
        return np.abs(mean - mean2) + self.beta * np.sqrt(var)


class QuadratureAcquisition(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        mean, var = self.model.get_statistics(X, full_cov=False)
        # aggregate hyperparameters dimension
        var = np.mean(var, axis=0)
        return np.sqrt(var)
