import numpy as np

class QuadratureAcquisition(object):
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        stats = self.model.get_statistics(X)
        var = stats[:,1,:, :]
        # aggregate hyperparameters dimension
        var = np.mean(var, axis=0)
        return np.sqrt(var)
