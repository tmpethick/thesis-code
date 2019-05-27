#%%
%load_ext autoreload
%autoreload 2

from runner import notebook_run, notebook_run_CLI, notebook_run_server, execute

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import *
from src.plot_utils import *
from src.kernels import *
from src.models.models import *
from src.models.dkl_gp import *
from src.models.lls_gp import *
from src.models.asg import *
from src.environments import *
from src.acquisition_functions import *
from src.algorithms import *

latexify(columns=1)

f = TwoKink2D()
X_test = random_hypercube_samples(1000, f.bounds)
# Uses uniform since bounds are [0,1] to ensure implementation is not broken...
X_test = np.random.uniform(size=(1000,2))
N_test = X_test.shape[-1]
Y_test = f(X_test)

def calc_error(i, model):
    max_error, L2_err = model.calc_error(X_test, Y_test)
    print("{0:9d} {1:9d}  Loo={2:1.2e}  L2={3:1.2e}".format(i+1, model.grid.getNumPoints(), max_error, L2_err))


# Motivation:
# Will lengthscale and DKL help us with max_err for kinks?
# See if meaningfull representations
# Then look at error.

# Varying Lengthscale suggests we should sample more around kinks.
# Lets do that and see if it improves Active Sampling.


#%%

config = {
    'obj_func': {'name': 'Kink1DShifted'},
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'mean_prior': True,
            'noise_prior': 100,
            'do_optimize': True,
            'num_mcmc': 0,
        },
    },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)
print(run.result)


#%% DKL

# DKL does not learn meaningful representation...

config = {
    'obj_func': {'name': 'TwoKink1D'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.1,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': 0.001}
        },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)
print(run.result)

#%%
# - LLS lengthscale on Kink2D and Kink1D (how does the lengthscale look?)

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 1000,
            'n_optimizer_iter': 5,
        }
    },
    'gp_samples': 1000,
})


#%%

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 200,
            'n_optimizer_iter': 5,
        }
    },
    'gp_samples': 1000,
})
