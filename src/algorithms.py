import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def constrain_points(x, bounds):
    dim = x.shape[0]
    minx = np.repeat(bounds[:, 0][None, :], dim, axis=0)
    maxx = np.repeat(bounds[:, 1][None, :], dim, axis=0)
    return np.clip(x, a_min=minx, a_max=maxx)


def random_hypercube_samples(n_samples, bounds, rng=None):
    """Random sample from d-dimensional hypercube (d = bounds.shape[0]).

    Returns: (n_samples, dim)
    """
    if rng is None:
        rng = np.random.RandomState()

    dims = bounds.shape[0]
    a = rng.uniform(0, 1, (dims, n_samples))
    bounds_repeated = np.repeat(bounds[:, :, None], n_samples, axis=2)
    samples = a * np.abs(bounds_repeated[:,1] - bounds_repeated[:,0]) + bounds_repeated[:,0]
    samples = np.swapaxes(samples, 0, 1)

    # This handles the case where the sample is slightly above or below the bounds
    # due to floating point precision (leading to slightly more samples from the boundary...).
    return constrain_points(samples, bounds)


class AcquisitionAlgorithm(object):
    def __init__(self, 
        f, 
        model, 
        acquisition_function, 
        n_init=20,
        n_iter=100,
        n_acq_max_starts=200,
        f_opt=None, 
        bounds=np.array([[0,1]]), 
        rng=None):

        self.f = f
        self.model = model
        # An acq func is defined on a model and define how it uses that model 
        # (be it through sampling or mean/variance).
        self.acquisition_function = acquisition_function(model)

        self.n_iter = n_iter
        self.n_init = n_init
        self.n_acq_max_starts = n_acq_max_starts
        self.bounds = bounds
        self.f_opt = f_opt

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

    def max_acq(self):
        min_y = float("inf")
        min_x = None

        def min_obj(x):
            """Lift into array and negate.
            """
            X = np.array([x])
            return -self.acquisition_function(X)[0]

        for x0 in random_hypercube_samples(self.n_acq_max_starts, self.bounds, rng=self.rng):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B') 

            if res.fun < min_y:
                min_y = res.fun
                min_x = res.x 
            
            # fix if no point is <inf
            elif min_x is None:
                min_x = res.x

        return min_x

    def run(self):
        X = random_hypercube_samples(self.n_init, self.bounds, rng=self.rng)
        Y = self.f(X)
        self.model.init(X,Y)

        for i in range(0, self.n_iter):
            print("... starting round", i, "/", self.n_iter)

            # new datapoint from acq
            x_new = self.max_acq()
            X_new = np.array([x_new])
            X_new = constrain_points(X_new, self.bounds)
            Y_new = self.f(X_new)

            self.model.add_observations(X_new, Y_new)

    def plot(self):
        # (obs, 1)
        X_line = np.linspace(self.bounds[0,0], self.bounds[0,1], 500)[:,None]

        self.model.plot(X_line)
        plt.plot(X_line, self.f(X_line))
        plt.show()
        plt.plot(X_line, self.acquisition_function(X_line))
