import matplotlib.pyplot as plt
import seaborn as sns

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
    def __init__(self, f, depth=1, refinement_level=5, f_tol=1e-5):
        self.depth = depth
        self.refinement_level = refinement_level
        self.f_tol = f_tol
        self.f = f

        in_dim = self.f.input_dim
        out_dim = 1
        which_basis = 1

        self.grid  = TasmanianSG.TasmanianSparseGrid(tasmanian_library=C_LIB)            
        self.grid.makeLocalPolynomialGrid(in_dim, out_dim, self.depth, which_basis, "localp")
        self.grid.setDomainTransform(f.bounds)


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

            if callable(callback):
                callback(i=iK, model=self)

    def evaluate(self, X):
        return self.grid.evaluateBatch(X)

    def plot(self):
        X_train = self.grid.getLoadedPoints()

        fig = plt.figure()
        
        ax = fig.add_subplot(121)
        ax.set_title("f")
        XY, X, Y = construct_2D_grid(self.f.bounds)
        Z = call_function_on_grid(self.f, XY)[...,0]
        cont = ax.contourf(X,Y,Z, 50)
        fig.colorbar(cont)

        ax = fig.add_subplot(122)
        ax.set_title("ASG Estimate")
        XY, X, Y = construct_2D_grid(self.f.bounds)
        Z = call_function_on_grid(self.evaluate, XY)[...,0]
        cont = ax.contourf(X,Y,Z, 50)
        fig.colorbar(cont)
        sns.scatterplot(X_train[...,0], X_train[...,1], ax=ax)

        return fig


