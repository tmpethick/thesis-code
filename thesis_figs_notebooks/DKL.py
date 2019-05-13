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


#%% Kink2D learned feature mapping
# \label{fig:dkl-kink2D-manifold}

config = {
    'obj_func': {'name': 'Kink2D'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 2]},
            'noise': 0.01
        },
    },
    'gp_samples': 1000,
}
run = execute(config_updates=config)

#%%

model = run.interactive_stash['model']
f = run.interactive_stash['f']

latexify(columns=2)
fig = f.plot()
savefig(fig, 'DKL/dkl-kink2D-manifold-f.pdf')
plt.show()

fig, ax = plt.subplots()
ax.set_title('f in feature space')

XY, X, Y = construct_2D_grid(f.bounds)
Z = call_function_on_grid(model.get_features, XY)
O = call_function_on_grid(f, XY)
ax.contourf(Z[...,0], Z[...,1], O[...,0], 50)

plt.tight_layout()
savefig(fig, 'DKL/dkl-kink2D-manifold-features.pdf')


#%% IncreasingOscillation lengthscale change
# \label{fig:dkl-lengthscale}

config = {
    'obj_func': {'name': 'IncreasingOscillation'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 1000,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': 0.01
        },
    },
    'gp_samples': 1000,
}
run = execute(config_updates=config)

#%% Plot DKL varying lengthscale features

model = run.interactive_stash['model']
f = run.interactive_stash['f']

latexify(columns=2)
fig = f.plot()
savefig(fig, 'DKL/dkl-lengthscale-f.pdf')
plt.show()

X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 1000)[:,None]
Z = model.get_features(X_line)
O = f(X_line)

fig, ax = plt.subplots()
ax.set_title('f in feature space')
ax.plot(Z.flatten(), O.flatten())

plt.tight_layout()
savefig(fig, 'DKL/dkl-lengthscale-features.pdf')


#%%
run.interactive_stash['model'].plot_features(run.interactive_stash['f'])

#%%
# Reproducability

# run two models and compare differences

config = {
    'obj_func': {'name': 'Sinc'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 1,
            'nn_kwargs': {'layers': None},
            'noise': 0.001
        },
    },
    'model2': {
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
    'gp_samples': 10,
    'model_compare': True
}
run = execute(config_updates=config)


#%% 

# Install ASG on server
# ASG: threshold
# Test effect of normalization.
# Add max_error/Loo_err
# Fix pos def error
# Understand how #parameters and DKL interacts

# Aggregate RMSE
# Click to open plot.

# Question::
# Is DKL model working? DKLGPModel vs GPModel NaN (ensure model works) √
# Does *some* DKL model gain performance? DKLGPModel vs GPModel
# How does learning rate influence DKL?
# How does iterations influence DKL?
# How does DKL behave on Kink2D and TwoKink1D?

# What if we took a Kink2D that was not bend? Would it learn to stretch the domain?
# How does it even help on step function?

# Clickable exploration


# Define trickier problems (variance is to small across models)
