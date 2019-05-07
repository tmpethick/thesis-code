#%%

%load_ext autoreload
%autoreload 2

from runner import notebook_run, notebook_run_CLI, notebook_run_server, execute

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from src.utils import *
from src.plot_utils import *
from src.kernels import *
from src.models.models import *
from src.models.dkl_gp import *
from src.models.lls_gp import *
from src.environments import *
from src.acquisition_functions import *
from src.algorithms import *

#%%

# Make embedding
TwoKink1D().plot()
TwoKink1D().plot_derivative()
TwoKink2D().plot(projection="3d")
TwoKink2D().plot_derivative(projection="3d")
TwoKinkDEmbedding(D=2).plot(projection="3d")
TwoKinkDEmbedding(D=2).plot_derivative(projection="3d")

#%% ------------------ Testing Non-stationarity ----------------------------

# ExactGP
run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': 0.001,
            'do_optimize': True,
            'num_mcmc': 0,
        }
    },
    'gp_samples': 100
})


#%%
# DKL
run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'noise': 1e-5,
            'n_iter': 150,
            'nn_kwargs': {
                'layers': (500, 50, 2),
            }
        }
    },
    'gp_samples': 20
})

#%%

# LLS
run = notebook_run(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 200,
            'n_optimizer_iter': 10,
        }
    },
    'gp_samples': 100,
})

#%% ------------------ Testing Scaling ----------------------------

#RFF (fixed l)
run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'RandomFourierFeaturesModel',
        'kwargs': {
            'kernel': {
                'name': 'RFFRBF',
                'kwargs': {
                    'lengthscale': 0.1,
                    'variance': 0.5
                }
            },
            'noise': 0.001,
            'do_optimize': False,
            'n_features': 50,
        }
    },
    'gp_samples': 100
})

#%% ------------------ Testing High-dim ----------------------------

run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKinkDEmbedding',
        'kwargs': {
            'D': 2
        }
    },
    'model': {
        'name': 'TransformerModel',
        'kwargs': {
            'transformer': {
                'name': 'ActiveSubspace',
                'kwargs': {
                    'output_dim': 1
                }
            },
            'prob_model': {
                'name': 'GPModel',
                'kwargs': {
                    'kernel': {
                        'name': 'GPyRBF',
                        'kwargs': {
                            'lengthscale': 1
                        }
                    },
                    'noise_prior': 0.001,
                    'do_optimize': True,
                    'num_mcmc': 0,
                },
            },
        },
    },
    'gp_use_derivatives': True,
    'gp_samples': 50,
})


#%% ------------------ Adaptive Sampling -----------------------

run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': 0.01,
            'do_optimize': True,
            'num_mcmc': 0,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
        'kwargs': {
            'beta': 0,
            'use_var': True,
        }
    },
    'gp_samples': 30
})


    # 'acquisition_function': {
    #     'name': 'CurvatureAcquisition',
    # },

#%% ------------------ Setup ASG ------------------

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

#%%

# True depth = refinement_level+ depth

class AdaptiveSparseGrid(object):
    def __init__(self, depth=1, refinement_level=5, f_tol=1e-5):
        self.depth = depth
        self.refinement_level = refinement_level
        self.f_tol = f_tol

        self.grid  = TasmanianSG.TasmanianSparseGrid(tasmanian_library=C_LIB)
    
    def fit(self, f, callback=None):
        in_dim = f.input_dim
        out_dim = 1
        which_basis = 1
        
        self.grid.makeLocalPolynomialGrid(in_dim, out_dim, self.depth, which_basis, "localp")

        X_train = self.grid.getPoints()
        Y_train = f(X_train)
        self.grid.loadNeededPoints(Y_train)

        if callable(callback):
            callback(i=-1, model=self)

        for iK in range(self.refinement_level):
            self.grid.setSurplusRefinement(self.f_tol, -1, "classic")
            X_train = self.grid.getNeededPoints()
            Y_train = f(X_train)
            self.grid.loadNeededPoints(Y_train)

            if callable(callback):
                callback(i=iK, model=self)

    def evaluate(self, X):
        return self.grid.evaluateBatch(X)

#%%

from src.environments import BaseEnvironment

class CosProd2D(BaseEnvironment):
    bounds = np.array([[0,1], [0,1]])
    def __call__(self, X):
        return (np.cos(0.5 * np.pi * X[..., 0]) * np.cos(0.5 * np.pi * X[..., 1]))[..., None]

f = CosProd2D()
X_test = np.random.uniform(-1.0, 1.0, size=(1000, 2))
Y_test = f(X_test)

def calc_error(i, model):
    Y_hat = model.evaluate(X_test)
    max_error = np.max(np.fabs(Y_hat - Y_test))
    print("{0:9d} {1:9d}  {2:1.2e}".format(i+1, model.grid.getNumPoints(), max_error))

# Without Adaptive
asg = AdaptiveSparseGrid(depth=5, refinement_level=0)
asg.fit(f, callback=calc_error)

#%%
# Adaptive
asg = AdaptiveSparseGrid(depth=1, refinement_level=5, f_tol=1e-5)
asg.fit(f, callback=calc_error)


#%%

