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

#%%

run = execute(config_updates={
    'obj_func': {
        'name': 'Sinc',
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
        },
    },
    'gp_samples': 100
})
print("RMSE: {:1.2e}".format(run.result))

#%% Train a DKL without feature mapping (for testing purposes)

run = execute(config_updates={
    'obj_func': {
        'name': 'Sinc',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'noise': 0.001,
            'n_iter': 100,
            'nn_kwargs': {
                'layers': None
            }
        }
    },
    'gp_samples': 100
})
print("RMSE: {:1.2e}".format(run.result))

#%%
model = run.interactive_stash['model']
model.model.covar_module.lengthscale
model.model.covar_module.variance

# very little noise in the data and the inferred noise level eventually gets so small that you run into numerical errors.
# All our problems seems to be fixed if we can fix the noise when it is known to be very very small.

#%% ----------------------- Step function ------------------

run = execute(config_updates={
    'obj_func': {
        'name': 'Step',
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
        },
    },
    'gp_samples': 100
})
print("RMSE: {:1.2e}".format(run.result))

#%%

run = execute(config_updates={
    'obj_func': {
        'name': 'Step',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'noise': 1e-2,
            'n_iter': 100,
            'nn_kwargs': {
                'layers': None, #(1, 30, 2), #(1000, 1000, 500, 50, 2),
            }
        }
    },
    'gp_samples': 100
})
print("RMSE: {:1.2e}".format(run.result))

#%%

run = execute(config_updates={
    'obj_func': {
        'name': 'Step',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'noise': 1e-2,
            'n_iter': 1000,
            'nn_kwargs': {
                'layers': (100, 50, 2), #(1000, 1000, 500, 50, 2),
            }
        }
    },
    'gp_samples': 100
})
print("RMSE: {:1.2e}".format(run.result))


#%%

run = execute(config_updates={
    'obj_func': {
        'name': 'Step',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'noise': 1e-2,
            'n_iter': 1000,
            'nn_kwargs': {
                'layers': (1, 50, 2), #(1000, 1000, 500, 50, 2),
            }
        }
    },
    'gp_samples': 100
})
print("RMSE: {:1.2e}".format(run.result))


#%% 

obj_funs = [
    {'name': 'TwoKink1D'},
    {'name': 'TwoKink2D'},
    {'name': 'TwoKinkDEmbedding', 'kwargs': {'D': 2}},
]

# ExactGP: Matern32, RBF
exactGP = [{
    'name': 'GPModel',
    'kwargs': {
        'kernel': {
            'name': kernel,
            'kwargs': {
                'lengthscale': 1
            }
        },
        'noise_prior': 1e-2,
        'do_optimize': True,
        'num_mcmc': 0,
    },
} for kernel in ['GPyRBF', 'GPyMatern32']]


# DKL "ExactGP": RBF
n_iters = [100, 1000]
DKL_exact = [{
    'name': 'DKLGPModel',
    'kwargs': {
        'noise': 1e-2,
        'n_iter': n_iter,
        'nn_kwargs': {
            'layers': None, 
        }
    }
} for n_iter in n_iters]

# DKL:  n_iter, learning_rate, layers, (noise)
import itertools
n_iters = [100, 1000]
learning_rates = [0.05, 0.01, 0.1]
layerss = [(100, 50, 2), (50, 2)]
parameters = itertools.product(n_iters, learning_rates, layerss)
DKL = [{
    'name': 'DKLGPModel',
    'kwargs': {
        'noise': 1e-2,
        'learning_rate': learning_rate,
        'n_iter': n_iter,
        'nn_kwargs': {
            'layers': layers, 
        }
    }
} for (n_iter, learning_rate, layers) in parameters]

model_types = [exactGP, DKL_exact, DKL]


#%%
i = 0

for obj_fun in obj_funs:
    for model_type in model_types:
        for model in model_type:
            i += 1
            print(i)
            config = {
                'obj_func': obj_fun,
                'model': model,
                'gp_samples': 1000,
            }
            # import traceback
            # try:
            execute(config_updates=config)
            # except Exception as exc:
            #     print(traceback.format_exc())
            #     print(exc)


# Install ASG on server
# ASG: threshold
# Test effect of normalization.
# Add max_error/Loo_err


#%%

# Make TwoKink Steeper / concave
# # Fix pos def error
# Understand how #parameters and DKL interacts

# config = {
#     'obj_func': {'name': 'TwoKink2D'},
#     'model': {
#         'name': 'DKLGPModel',
#         'kwargs': {'noise': 0.01,
#             'learning_rate': 0.05,
#             'n_iter': 1,
#             'nn_kwargs': {'layers': (1000, 1000, 50, 2)}
#         }
#     },
#     'gp_samples': 100,
# }
# run = execute(config_updates=config)

# plot_model(run.interactive_stash['model'], run.interactive_stash['f'])

