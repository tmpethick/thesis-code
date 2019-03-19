import numpy as np

class QuadratureAcquisition(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        mean, var = self.model.get_statistics(X, full_cov=False)
        # aggregate hyperparameters dimension
        var = np.mean(var, axis=0)
        return np.sqrt(var)
