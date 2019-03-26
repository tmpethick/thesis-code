import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns

from src.acquisition_functions import AcquisitionBase


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
        models, 
        acquisition_function, 
        n_init=20,
        n_iter=100,
        n_acq_max_starts=200,
        f_opt=None, 
        bounds=np.array([[0,1]]), 
        rng=None):

        self.f = f
        self.models = models
        # An acq func is defined on a model and define how it uses that model 
        # (be it through sampling or mean/variance).
        if isinstance(acquisition_function, AcquisitionBase):
            self.acquisition_function = acquisition_function
        else:
            self.acquisition_function = acquisition_function(*models)
  
        self.n_iter = n_iter
        self.n_init = n_init
        self.n_acq_max_starts = n_acq_max_starts
        self.bounds = bounds
        self.f_opt = f_opt

        # Keep a local reference to X,Y for convinience.
        self.X = None
        self.Y = None

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

        # Sample 5000. Pick 100 max and minimize those.
        for x0 in random_hypercube_samples(self.n_acq_max_starts, self.bounds, rng=self.rng):
            res = minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')

            if res.fun < min_y:
                min_y = res.fun
                min_x = res.x 
            
            # fix if no point is <inf
            elif min_x is None:
                min_x = res.x

        return min_x

    def run(self, callback=None):
        X = random_hypercube_samples(self.n_init, self.bounds, rng=self.rng)
        Y = self.f(X)

        self.X = X
        self.Y = Y
        
        for model in self.models:
            model.init(X,Y)

        for i in range(0, self.n_iter):
            print("... starting round", i, "/", self.n_iter)

            # new datapoint from acq
            x_new = self.max_acq()
            X_new = np.array([x_new])
            X_new = constrain_points(X_new, self.bounds)
            Y_new = self.f(X_new)

            for model in self.models:
                model.add_observations(X_new, Y_new)
                self.X = np.concatenate([self.X, X_new])
                self.Y = np.concatenate([self.Y, Y_new])

            if callable(callback):
                callback(self)

    def plot(self):
        # (obs, 1)
        dims = self.bounds.shape[0]

        if dims == 1:
            X_line = np.linspace(self.bounds[0,0], self.bounds[0,1], 500)[:,None]

            fig = plt.figure()
            ax = fig.add_subplot(221)            
            ax.set_title('Ground truth')
            ax.plot(X_line, self.f(X_line))
            
            ax = fig.add_subplot(222)
            ax.set_title('Acq func')
            ax.plot(X_line, self.acquisition_function(X_line))

            ax = fig.add_subplot(223)
            ax.set_title('Estimate')
            for model in self.models:
                  model.plot(X_line, ax=ax)
            
            ax = fig.add_subplot(224)
            ax.set_title('Sample density')
            ax.hist(self.X)

        elif dims == 2:
            from mpl_toolkits import mplot3d
            fig = plt.figure()

            XY, X, Y = self.construct_2D_grid()
            
            ax = fig.add_subplot(221)
            ax.set_title('Ground truth')
            Z = self.call_function_on_grid(self.f, XY)
            ax.contour(X,Y,Z, 50)

            ax = fig.add_subplot(222)
            ax.set_title('Acq func')
            Z = self.call_function_on_grid(self.acquisition_function, XY)
            ax.contour(X,Y,Z, 50)

            ax = fig.add_subplot(223)
            ax.set_title('Estimate')
            Z = self.call_function_on_grid(self.evaluate_f_estimate, XY)
            ax.contour(X,Y,Z, 50)

            ax = fig.add_subplot(224)
            ax.set_title('Sample density')
            sns.kdeplot(data=self.X[..., 0], data2=self.X[..., 1], ax=ax, clip=self.bounds, shade=True, cbar=True, cmap="Reds", shade_lowest=False)
            sns.scatterplot(self.X[...,0], self.X[...,1], ax=ax)
        else:
            warnings.warn("Cannot plot above 2D.", Warning)

    def evaluate_f_estimate(self, X):
        mean, var = self.models[0].get_statistics(X)
        return mean

    def construct_2D_grid(self):
        x_bounds = self.bounds[0]
        y_bounds = self.bounds[1]
        X = np.linspace(x_bounds[0], x_bounds[1], 50)
        Y = np.linspace(y_bounds[0], y_bounds[1], 50)
        X, Y = np.meshgrid(X, Y)
        XY = np.stack((X,Y), axis=-1)
        
        return XY, X, Y

    def call_function_on_grid(self, func, XY):
        original_grid_size = XY.shape[0]
        XY = XY.reshape((-1, 2)) # remove grid
        Z = func(XY)
        Z = Z.reshape((original_grid_size, original_grid_size)) # recreate grid
        return Z
