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


#%% Test robustness of ExactDKL (how closely does it match ExactGP)


obj_funs = [
    {'name': 'Sinc'},
    {'name': 'Branin'},
    {'name': 'TwoKink1D'},
]

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



#%%

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
    {'name': 'TwoKinkDEmbedding', 'kwargs': {'Alpha': Alpha}}
    {'name': 'Kink2D'},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 2}},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 5}},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 10}},
 ]


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
# plt.(Z, f(X))


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


#%%
