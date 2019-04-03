import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns

from src.acquisition_functions import AcquisitionBase
from src.models.models import BaseModel

from src.plot_utils import construct_2D_grid, call_function_on_grid
from src.utils import random_hypercube_samples, constrain_points


class AcquisitionAlgorithm(object):
    def __init__(self, 
        f, 
        models, 
        acquisition_function, 
        n_init=20,
        n_iter=100,
        n_acq_max_starts=200,
        f_opt=None,
        rng=None):

        self.f = f

        if isinstance(models, BaseModel):
            self.models = [models]
        else:
            self.models = models

        # An acq func is defined on a model and define how it uses that model 
        # (be it through sampling or mean/variance).
        if isinstance(acquisition_function, AcquisitionBase):
            self.acquisition_function = acquisition_function
        else:
            self.acquisition_function = acquisition_function(*self.models)
  
        self.n_iter = n_iter
        self.n_init = n_init
        self.n_acq_max_starts = n_acq_max_starts
        self.bounds = f.bounds
        self.f_opt = f_opt

        # Keep a local reference to X,Y for convinience.
        self.X = None
        self.Y = None

        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.RandomState()

    def max_acq(self):
        def min_obj(x):
            """Lift into array and negate.
            """
            X = np.array([x])
            return -self.acquisition_function(X)[0]

        if False: 
            # Code version that can eventually be parallized
            def minimize(x0):
                res = optimize.minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')
                return [res.x, res.fun]

            # Sample 5000. Pick 100 max and minimize those.
            x0_batch = list(random_hypercube_samples(self.n_acq_max_starts, self.bounds, rng=self.rng))

            # p = Pool(4)
            res = map(minimize, x0_batch)
            res = np.array(list(res)) # (n_acq_max_starts, 2)
            ymin_idx = np.argmin(res[:, 1])
            min_x, min_y = res[ymin_idx]

            return min_x
        else:
            min_y = float("inf")
            min_x = None

            for x0 in random_hypercube_samples(self.n_acq_max_starts, self.bounds, rng=self.rng):
                res = optimize.minimize(min_obj, x0=x0, bounds=self.bounds, method='L-BFGS-B')

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
            model.init(X, Y)

        for i in range(0, self.n_iter):
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
                callback(self, i)

    def plot(self):
        # (obs, 1)
        dims = self.bounds.shape[0]

        if dims == 1:
            X_line = np.linspace(self.bounds[0,0], self.bounds[0,1], 500)[:,None]

            fig = plt.figure()
            ax = fig.add_subplot(221)
            ax.set_title('Ground truth')
            ax.scatter(self.X, self.Y)
            ax.plot(X_line, self.f(X_line))
            
            ax = fig.add_subplot(222)
            ax.set_title('Acq func')
            ax.plot(X_line, self.acquisition_function(X_line))

            ax = fig.add_subplot(223)
            ax.scatter(self.X, self.Y)
            ax.set_title('Estimate')
            for model in self.models:
                  model.plot(X_line, ax=ax)
            
            ax = fig.add_subplot(224)
            ax.set_title('Sample density')
            ax.hist(self.X)

        elif dims == 2:
            fig = plt.figure()

            XY, X, Y = construct_2D_grid(self.bounds)
            
            ax = fig.add_subplot(221)
            ax.set_title('Ground truth')
            Z = call_function_on_grid(self.f, XY)
            cont = ax.contourf(X,Y,Z, 50)
            fig.colorbar(cont)
            ax.plot(self.X[:, 0], self.X[:, 1], '.', markersize=10)

            ax = fig.add_subplot(222)
            ax.set_title('Estimate')
            Z = call_function_on_grid(self.model_estimate, XY)
            cont = ax.contourf(X,Y,Z, 50)
            fig.colorbar(cont)
            ax.plot(self.X[:, 0], self.X[:, 1], '.', markersize=10)

            ax = fig.add_subplot(223)
            ax.set_title('Acq func')
            Z = call_function_on_grid(self.acquisition_function, XY)
            cont = ax.contourf(X,Y,Z, 50)
            fig.colorbar(cont)
            ax.plot(self.X[:, 0], self.X[:, 1], '.', markersize=10)

            ax = fig.add_subplot(224)
            ax.set_title('Sample density')
            sns.kdeplot(data=self.X[..., 0], data2=self.X[..., 1], ax=ax,
                        clip=self.bounds, shade=True, cbar=True, cmap="Reds", shade_lowest=False)
            sns.scatterplot(self.X[...,0], self.X[...,1], ax=ax)
        else:
            warnings.warn("Cannot plot above 2D.", Warning)
            fig = None
        return fig

    def model_estimate(self, X):
        mean, var = self.models[0].get_statistics(X, full_cov=False)
        return mean


def bo_plot_callback(bq: AcquisitionAlgorithm, i: int):
   # print("Round {} using model {}".format(i, bq.models[0]))
   if i % 5 == 0 or i == bq.n_iter:
      bq.plot()
      plt.show()
