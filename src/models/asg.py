import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.utils import construct_2D_grid, call_function_on_grid

import sys 
import os 

dir_path = os.getcwd()
python_path = os.path.join(dir_path, 'SparseGridCode/TasmanianSparseGrids/InterfacePython')
C_LIB = os.path.join(dir_path, 'SparseGridCode/TasmanianSparseGrids/libtasmaniansparsegrid.so')

sys.path.append(python_path)

import TasmanianSG
import numpy as np

# imports specifically needed by the examples
import math
from datetime import datetime

print("TasmanianSG version: {0:s}".format(TasmanianSG.__version__))
print("TasmanianSG license: {0:s}".format(TasmanianSG.__license__))


class AdaptiveSparseGrid(object):
    def __init__(self, f, depth=1, refinement_level=5, f_tol=1e-5, point_tol=None):
        self.depth = depth
        self.refinement_level = refinement_level
        self.f_tol = f_tol
        self.point_tol = point_tol
        self.f = f

        in_dim = self.f.input_dim
        out_dim = 1
        which_basis = 1

        self.grid  = TasmanianSG.TasmanianSparseGrid(tasmanian_library=C_LIB)
        self.grid.makeLocalPolynomialGrid(in_dim, out_dim, self.depth, which_basis, "localp")
        self.grid.setDomainTransform(self.f.bounds)

        # This will only be able to change if point_tol is set.
        self.early_stopping_level = refinement_level


    def fit(self, callback=None):
        X_train = self.grid.getPoints()
        Y_train = self.f(X_train)
        self.grid.loadNeededPoints(Y_train)

        if callable(callback):
            callback(i=-1, model=self)

        for iK in range(self.refinement_level):
            self.grid.setSurplusRefinement(self.f_tol, -1, "classic")
            X_train = self.grid.getNeededPoints()
            Y_train = self.f(X_train)
            self.grid.loadNeededPoints(Y_train)

            if self.point_tol is not None:
                if self.grid.getNumPoints() > self.point_tol:
                    self.early_stopping_level = iK
                    break

            if callable(callback):
                callback(i=iK, model=self)

    def evaluate(self, X):
        Y_hat = self.grid.evaluateBatch(X)

        # Raise warning if the estimate is undefined on part of the domain.
        # This happened for certain threshold values.
        if np.any(np.isnan(Y_hat)):
            warnings.warn("The Sparse Grid estimate something to NaN.")
        return Y_hat

    def calc_error(self, X_test, Y_test):
        N_test = X_test.shape[-1]
        Y_hat = self.evaluate(X_test)
        Loo_err = np.max(np.fabs(Y_hat - Y_test))
        L2_err = np.sqrt(np.sum((Y_hat - Y_test) ** 2) / N_test)
        return Loo_err, L2_err

    def plot(self):
        X_train = self.grid.getLoadedPoints()

        fig = plt.figure()
        XY, X, Y = construct_2D_grid(self.f.bounds)
        
        ax = fig.add_subplot(221)
        ax.set_title("f")
        Z1 = call_function_on_grid(self.f, XY)[...,0]
        cont = ax.contourf(X,Y,Z1, 50)
        fig.colorbar(cont)

        ax = fig.add_subplot(222)
        ax.set_title("ASG Estimate $\hat{f}$")
        Z2 = call_function_on_grid(self.evaluate, XY)[...,0]
        cont = ax.contourf(X,Y,Z2, 50)
        fig.colorbar(cont)
        sns.scatterplot(X_train[...,0], X_train[...,1], ax=ax, size=2, alpha=0.5, legend=False)

        ax = fig.add_subplot(223)
        ax.set_title("$|f - \hat{f}|$")
        Z = call_function_on_grid(self.evaluate, XY)[...,0]
        cont = ax.contourf(X,Y,np.fabs(Z1 - Z2), 50)
        fig.colorbar(cont)

        plt.tight_layout()

        # asg.grid.plotResponse2D()
        # asg.grid.plotPoints2D()

        return fig


