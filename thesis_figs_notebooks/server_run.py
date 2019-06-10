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

NOISE_LEVELS = {
    'Step':                   1e-2,
    'Kink1D':                 10,
    'Kink2D':                 1e-2,
    'TwoKink1D':              1e-2,
    'TwoKink2D':              1e-2,
    'TwoKinkDEmbedding':      1e-2,
    'Sinc':                   1e-2,
    'Branin':                 1e-1,
    'KinkDCircularEmbedding': 1e-2,
    'KinkDCircularEmbedding': 1e-2,
    'KinkDCircularEmbedding': 1e-2,
    'ActiveSubspaceTest':     1e-2,
    'TwoKinkDEmbedding':      1e-2,
}

def add_noiselevels(obj_funs):
    for obj_fun in obj_funs:
        level = NOISE_LEVELS.get(obj_fun['name'])
        kwargs = obj_fun.get('kwargs', {})

        # Only add it if its not been set
        if 'noise' not in kwargs:
            kwargs['noise'] = level
        obj_fun['kwargs'] = kwargs

#%% Non-stationarity (Step)


import itertools
n_iters = [1000]
learning_rates = [0.005, 0.01]
parameters = itertools.product(n_iters, learning_rates)
models = [{
    'name': 'DKLGPModel',
    'kwargs': {
        'noise': None,
        'learning_rate': learning_rate,
        'n_iter': n_iter,
        'nn_kwargs': {
            'layers': [5,5,2],
        }
    }
} for (n_iter, learning_rate) in parameters]

for function_name in ['Step', 'SingleStep']:
    for model in models:
        config = {
            'tag': 'step',
            'obj_func': {'name': function_name, 'kwargs': {'noise': 0.01}},
            'model': model,
            'gp_samples': 1000,
        }
        run = execute(config_updates=config)


#%% Test robustness of ExactDKL (how closely does it match ExactGP)


obj_funs = [
    {'name': 'Sinc'},
    {'name': 'Branin'},
    {'name': 'TwoKink1D'},
]
add_noiselevels(obj_funs)

exactGP = {
    'name': 'GPModel',
    'kwargs': {
        'kernel': {
            'name': 'GPyRBF',
            'kwargs': {
                'lengthscale': 1
            }
        },
        'noise_prior': 1e-2,
        'do_optimize': True,
        'num_mcmc': 0,
    },
}


# DKL "ExactGP": RBF
import itertools
n_iters = [100, 1000]
learning_rates = [0.05, 0.01, 0.1]
parameters = itertools.product(n_iters, learning_rates)

DKL_exact = [{
    'name': 'DKLGPModel',
    'kwargs': {
        'noise': 1e-2,
        'n_iter': n_iter,
        'learning_rate': learning_rate,
        'nn_kwargs': {
            'layers': None, 
        }
    }
} for (n_iter, learning_rate) in parameters]

for obj_fun in obj_funs:
    for model in DKL_exact:
        config = {
            'tag': 'certify-ExactDKL',
            'obj_func': obj_fun,
            'model': model,
            'model2': exactGP,
            'gp_samples': 1000,
            'model_compare': True
        }
        execute(config_updates=config)


#%% Property tests on discontinous and kinks in Low dim

obj_funs = [
    {'name': 'Step'},
    {'name': 'Kink1D'},
    {'name': 'Kink2D'},
    {'name': 'TwoKink1D'},
    {'name': 'TwoKink2D'},
    {'name': 'TwoKinkDEmbedding', 'kwargs': {'D': 2}},
]
add_noiselevels(obj_funs)

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
layerss = [(100, 50, 1), (100, 50, 2), (50, 2)]
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
                'tag': 'DKL-properties',
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




#%% Small High-dim
 
# Assuming we know the output_dim so we don't have to learn it...
# For ActiveSubspaceTest we expect AS-GP to be best (it is tailored to the problem). 
# But we hope DKL to recover it.

exactGP = {
    'name': 'GPModel',
    'kwargs': {
        'kernel': {
            'name': 'GPyRBF',
            'kwargs': {
                'lengthscale': 1
            }
        },
        'noise_prior': 1e-2,
        'do_optimize': True,
        'num_mcmc': 0,
    },
}


DKLModel = {
    'name': 'DKLGPModel',
    'kwargs': {
        'learning_rate': 0.01,
        'n_iter': 1000,
        'nn_kwargs': {'layers': [100, 50, 2]},
        'noise': 1e-2
    },
}

transformer = {
    'name': 'ActiveSubspace',
    'kwargs': {
        'output_dim': 1
    }
}

models = [
    {
        'name': 'TransformerModel',
        'kwargs': {
            'transformer': transformer,
            'prob_model': exactGP
        },
    },
    {
        'name': 'TransformerModel',
        'kwargs': {
            'transformer': transformer,
            'prob_model': DKLModel
        },
    },
    DKLModel
]

# Alpha = TwoKinkDEmbedding.generate_alpha(D=10)
Alpha = [
    [0.78695576],
    [0.70777112],
    [0.34515641],
    [0.20288506],
    [0.52388727],
    [0.2025096 ],
    [0.31752746],
    [0.24497726],
    [0.89249818],
    [0.64264009]]

functions = [
    {'name': 'ActiveSubspaceTest'},
    {'name': 'TwoKinkDEmbedding', 'kwargs': {'Alpha': Alpha}},
    {'name': 'Kink2D'},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 2}},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 5}},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 10}},
 ]
add_noiselevels(functions)

# Test that it is indeed active subspace of dim 1:
f = TwoKinkDEmbedding(Alpha=Alpha)
#f = KinkDCircularEmbedding(D=10)
X = random_hypercube_samples(1000, f.bounds)
G = f.derivative(X)
model = ActiveSubspace()
model.fit(X, f(X), G)
model.W.shape[-1]

# model.plot()
# Z = model.transform(X)
# plt.scatter(Z, f(X))


#%%

assert model.W.shape[1] == 1, "Subspace Dimensionality should be 1 since it is assumed by the model."

run = None

for func in functions:
    for model in models:
        run = execute(config_updates={
            'tag': 'embedding',
            'obj_func': func,
            'model': model,
            'gp_use_derivatives': model.get('name') == 'TransformerModel',
            'gp_samples': 1000,
        })


#%% Genz1984!
from src.environments import *

# Plot all genz
# GenzContinuous(D=2).plot(projection="3d")
# GenzCornerPeak(D=2).plot(projection="3d")
# GenzDiscontinuous(D=2).plot(projection="3d")
# GenzGaussianPeak(D=2).plot(projection="3d")
# GenzOscillatory(D=2).plot(projection="3d")
# GenzProductPeak(D=2).plot(projection="3d")
#%%
Ds = [2,5,10,50]
functions = ['GenzContinuous', 'GenzCornerPeak', 'GenzDiscontinuous', 'GenzGaussianPeak', 'GenzOscillatory', 'GenzProductPeak']

for func in functions:
    for D in Ds:
        run = execute(config_updates={
            'tag': 'genz',
            'obj_func': {'name': func, 'kwargs': {'D': D}},
            'model': {
                'name': 'DKLGPModel',
                'kwargs': {
                    'learning_rate': 0.01,
                    'n_iter': 1000,
                    'nn_kwargs': {'layers': [100, 50, 1]},
                    'noise': 1e-1,
                },
            },
            'gp_samples': 1000,
        })

#%% Genz now with 2D embedding

Ds = [2,5,10,50]
functions = ['GenzContinuous', 'GenzCornerPeak', 'GenzDiscontinuous', 'GenzGaussianPeak', 'GenzOscillatory', 'GenzProductPeak']

for func in functions:
    for D in Ds:
        run = execute(config_updates={
            'tag': 'genz',
            'obj_func': {'name': func, 'kwargs': {'D': D}},
            'model': {
                'name': 'DKLGPModel',
                'kwargs': {
                    'learning_rate': 0.01,
                    'n_iter': 1000,
                    'nn_kwargs': {'layers': [100, 50, 2]},
                    'noise': 1e-1,
                },
            },
            'gp_samples': 1000,
        })


#%% Scalability

# DKL, RFF, SparseGP
# TwoKink1D, Kink1D, Sinc

# In 1D will more points improve accuracy? (and what number of inducing points are necessary)
# Try for different number of inducing points



#%%


RFF = [{
    'name': 'GPModel',
    'kwargs': {
        'kernel': {
            'name': 'GPyRBF',
            'kwargs': {
                'lengthscale': 1
            }
        },
        'noise_prior': 1e-2,
        'do_optimize': True,
        'num_mcmc': 0,
    },
}]

grid_sizes = [100, 1000, 10000]
DKLModel = [{
    'name': 'DKLGPModel',
    'kwargs': {
        'learning_rate': 0.01,
        'n_iter': 1000,
        'nn_kwargs': {'layers': [100, 50, 2]},
        'noise': 1e-2,
        'n_grid': n_grid,
    },
}, ]

functions = [
    {'name': 'TwoKink1D'},
    {'name': 'Kink1D'},
    {'name': 'Sinc'},
 ]
add_noiselevels(functions)


#%%

models = RFF + DKLModel
sample_sizes = [100, 1000, 10000, 1000000]

run = None


for sample_size in sample_sizes:
    for func in functions:
        for model in models:
            run = execute(config_updates={
                'tag': 'embedding',
                'obj_func': func,
                'model': model,
                'gp_use_derivatives': model.get('name') == 'TransformerModel',
                'gp_samples': sample_size,
            })


#%%
functions = ['GenzContinuous', 'GenzCornerPeak', 'GenzDiscontinuous', 'GenzGaussianPeak', 'GenzOscillatory', 'GenzProductPeak']
Ds = [2,3,4,5,10,20]


for D in Ds:
    for function in functions:
        config = {
            'tag': 'genz-dkl-stability',
            'obj_func': {'name': function, 'kwargs': {'D': D, 'noise': 0.01}},
            'model': {
                'name': 'DKLGPModel',
                'kwargs': {
                    'learning_rate': 0.01,
                    'n_iter': 300,
                    'do_pretrain': False,
                    'nn_kwargs': {'layers': [100, 50, 2],
                                'normalize_output': True},
                    'use_cg': True,
                    'precond_size': 15,             # To ensure
                    'noise': 0.01,                  # it does
                    'use_double_precision': True,   # not crash.
                },
            },
            'gp_samples': 1000,
        }
        normalize_config(config)
        run = execute(config_updates=config)
