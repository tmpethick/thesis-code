#%%
%load_ext autoreload
%autoreload 2
from notebook_header import *


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
                'lengthscale': 0.6
            }
        },
        'noise_prior': None,
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
        'use_cg': False,
        'noise': None
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


functions = [
    {'name': 'ActiveSubspaceArbitrary1D', 'kwargs': {'D': 2}},
    {'name': 'ActiveSubspaceArbitrary1D', 'kwargs': {'D': 5}},
    {'name': 'ActiveSubspaceArbitrary1D', 'kwargs': {'D': 10}},
    {'name': 'ActiveSubspaceArbitrary1D', 'kwargs': {'D': 50}},
    {'name': 'ActiveSubspaceArbitrary1D', 'kwargs': {'D': 100}},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 2}},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 5}},
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 10}},
 ]
#add_noiselevels(functions)

#%%

run = None

for func in functions:
    for model in models:
        config = normalize_config({
            'tag': 'embedding',
            'obj_func': func,
            'model': model,
            'gp_use_derivatives': model.get('name') == 'TransformerModel',
            'gp_samples': 1000,
        })
        run = execute(config_updates=config)


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
            config = normalize_config({
                'tag': 'scalability',
                'obj_func': func,
                'model': model,
                'gp_use_derivatives': model.get('name') == 'TransformerModel',
                'gp_samples': sample_size,
            })
            run = execute(config_updates=config)


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

#%% Scalability

noise = 0.0001

RFF = [{
    'name': 'RandomFourierFeaturesModel',
    'kwargs': {
        'kernel': {
            'name': 'RFFRBF',
            'kwargs': {
                'lengthscale': 1.0,
                'variance': 0.5
            }
        },
        'noise': noise,
        'do_optimize': False,
        'n_features': 500,
    }
}]

exactGP = [{
    'name': 'GPModel',
    'kwargs': {
        'kernel': {
            'name': 'GPyRBF',
            'kwargs': {
                'lengthscale': 1
            }
        },
        'noise_prior': noise,
        'do_optimize': True,
        'num_mcmc': 0,
    },
}]


Ms = [None, int(np.sqrt(1000)), int(np.sqrt(10000))]
DKLModel = [{
    'name': 'DKLGPModel',
    'kwargs': {
        'learning_rate': 0.1,
        'n_iter': 150,
        'nn_kwargs': {'layers': None, 
                      'normalize_output': False},
        'gp_kwargs': {'n_grid': M},
        'use_cg': True,
        'precond_size': 50,
        'max_cg_iter': 2000,
        'use_double_precision': False, 
        'noise': noise,
    }
} for M in Ms]


transformer_model = {
        'name': 'TransformerModel',
        'kwargs': {
            'transformer': {
                'name': 'ActiveSubspace',
                'kwargs': {
                'output_dim': 1
            },
            'prob_model': exactGP,
        },
    },
}

#%% Find a good test function

#models = exactGP + RFF + DKLModel
models = exactGP + [DKLModel[0]]
Ns = [100, 500, 1000, 2000, 3000, 4000]

for N in Ns:
    for model in models:
        for function in [{'name': 'Branin'}, {'name': 'Kink2D'}, {'name': 'GenzContinuous', 'kwargs': {'D': 2}}, {'name': 'Sin2DRotated'}, {'name': 'Sin2D'}]:
            config = {
                'tag': 'scalability-finding-functionsM',
                'obj_func': function,
                'model': model,
                'gp_samples': N,
            }
            normalize_config(config)
            run = execute(config_updates=config)

#%% Scalability in N (fixed M)

#models = exactGP + RFF + DKLModel
models = exactGP + DKLModel
Ns = [1000, 3000, 5000, 10000]

for N in Ns:
    for model in models:
        for function in [{'name': 'GenzContinuous', 'kwargs': {'D': 2}}]:
            config = {
                'tag': 'scalability-fixed-M',
                'obj_func': function,
                'model': model,
                'gp_samples': N,
            }
            normalize_config(config)
            run = execute(config_updates=config)

#%% Scalability in M (fixed N)

#Ms = range(500, 2001, 150)
Ms = [100, 200, 500, 1000, 2000, 3000]

DKLModel = [{
    'name': 'DKLGPModel',
    'kwargs': {
        'learning_rate': 0.1,
        'n_iter': 200,
        'nn_kwargs': {'layers': None,
                      'normalize_output': False},
        'gp_kwargs': {'n_grid': int(np.sqrt(M))},
        'use_cg': True,
        'precond_size': 50,
        'max_cg_iter': 2000,
        'use_double_precision': False, 
        'noise': noise,
    }
} for M in Ms]

models = DKLModel

for model in models:
    #for function in [{'name': 'GenzDiscontinuous', 'kwargs': {'D': 2}}, {'name': 'GenzContinuous', 'kwargs': {'D': 2}}, {'name': 'Branin'}, {'name': 'Kink2D'}]:
    for function in [{'name': 'Branin'}]:
        config = {
            'tag': 'scalability-fixed-N',
            'obj_func': function,
            'model': model,
            'gp_samples': 1000,
        }
        normalize_config(config)
        run = execute(config_updates=config)


#%%

# noise = 0.001

# DKLModel = [{
#     'name': 'DKLGPModel',
#     'kwargs': {
#         'learning_rate': 0.1,
#         'n_iter': 200,
#         'nn_kwargs': {'layers': [100, 500, 1],
#                       'normalize_output': False},
#         'use_cg': True,
#         'precond_size': 50,
#         'max_cg_iter': 2000,
#         'use_double_precision': False, 
#         'noise': noise,
#     }
# }]

# models = DKLModel

# for model in models:
#     for function in [{'name': 'Branin'}, {'name': 'Kink2D'}, {'name': 'Step'}, {'name': 'StepConcave'}]:
#     for function in [{'name': 'Step'}]:
#         config = {
#             'tag': 'discont',
#             'obj_func': function,
#             'model': model,
#             'gp_samples': 1000,
#         }
#         normalize_config(config)
#         run = execute(config_updates=config)

#%%

# With DKL 1D

training_size_to_total_size = lambda x: int(x * 1/(0.8*0.8))

Ds = [1,2,3,4]
Ns = [1000, 10000, 20000]
Ms = [1000, 10000]
# M can be big since the feature space is one dimensional.

# With DKL
for D in Ds:
    for N in Ns:
        for M in Ms:
            run = execute(config_updates={
                'tag': 'SPXOptions',
                'obj_func': {
                    'name': 'SPXOptions',
                    'kwargs': {'D': D, 'subset_size': training_size_to_total_size(N)},
                },
                'model': {
                    'name': 'NormalizerModel',
                    'kwargs': {
                        'model': {
                            'name': 'DKLGPModel',
                            'kwargs': {
                                'learning_rate': 0.1,
                                'n_iter': 100,
                                'nn_kwargs': {'layers': [100, 50, 1]},
                                'gp_kwargs': {'n_grid': M},
                                'use_cg': True,
                                'noise': None
                            }
                        }
                    }
                },
            })


#%%
# With without DKL

training_size_to_total_size = lambda x: int(x * 1/(0.8*0.8))

Ds = [1,2,3,4]
Ns = [1000, 10000, 20000]
Ms = [10000, 100, 22, 10]

# Without DKL
# With DKL
for i, D in enumerate(Ds):
    for N in Ns:
        run = execute(config_updates={
            'tag': 'SPXOptions',
            'obj_func': {
                'name': 'SPXOptions',
                'kwargs': {'D': D, 'subset_size': training_size_to_total_size(N)},
            },
            'model': {
                'name': 'NormalizerModel',
                'kwargs': {
                    'model': {
                        'name': 'DKLGPModel',
                        'kwargs': {
                            'learning_rate': 0.1,
                            'n_iter': 100,
                            'nn_kwargs': {'layers': None},
                            'gp_kwargs': {'n_grid': Ms[i]},
                            'use_cg': True,
                            'noise': None
                        }
                    }
                }
            },
        })


#%%
# Learning rate

from notebook_header import *

Ds = [1,2,3,4]
N = 10000
Ms = [10000, 100, 22, 10] # such that inducing point matches approximately N*0.8^2

# Without DKL
# With DKL
for rate in [0.1, 0.01]:
    for i, D in enumerate(Ds):
        run = execute(config_updates={
            'tag': 'SPXOptions',
            'obj_func': {
                'name': 'SPXOptions',
                'kwargs': {'D': D, 'subset_size': N},
            },
            'model': {
                'name': 'NormalizerModel',
                'kwargs': {
                    'model': {
                        'name': 'DKLGPModel',
                        'kwargs': {
                            'learning_rate': rate,
                            'n_iter': 200,
                            'nn_kwargs': {'layers': None},
                            'gp_kwargs': {'n_grid': Ms[i]},
                            'use_cg': True,
                            'noise': None
                        }
                    }
                }
            },
        })

#%%
# Dims without DKL

from notebook_header import *

Ds = [1,2,3,4]
N = SPXOptions.max_train_size()
Ms = [583104, 763, 84, 28] # Such that inducing point matches approximately N*0.8^2

for i, D in enumerate(Ds):
    run = execute(config_updates={
        'tag': 'SPXOptions',
        'obj_func': {
            'name': 'SPXOptions',
            'kwargs': {'D': D, 'subset_size': N},
        },
        'model': {
            'name': 'NormalizerModel',
            'kwargs': {
                'model': {
                    'name': 'DKLGPModel',
                    'kwargs': {
                        'learning_rate': 0.1,
                        'n_iter': 30,
                        'nn_kwargs': {'layers': None},
                        'gp_kwargs': {'n_grid': Ms[i]},
                        'use_cg': True,
                        'noise': None
                    }
                }
            }
        },
    })

#%%
# Dims without DKL 1D

# Pick M as big as our training set.
from notebook_header import *

Ds = [1,2,3,4,10]
M = int(N * 0.8 * 0.8)

for i, D in enumerate(Ds):
    run = execute(config_updates={
        'tag': 'SPXOptions',
        'obj_func': {
            'name': 'SPXOptions',
            'kwargs': {'D': D, 'subset_size': N},
        },
        'model': {
            'name': 'NormalizerModel',
            'kwargs': {
                'model': {
                    'name': 'DKLGPModel',
                    'kwargs': {
                        'learning_rate': 0.1,
                        'n_iter': 30,
                        'nn_kwargs': {'layers': [100, 50, 1]},
                        'gp_kwargs': {'n_grid': M},
                        'use_cg': True,
                        'noise': None
                    }
                }
            }
        },
    })

#%%
# Dims without DKL 1D

# Pick M as big as our training set.
from notebook_header import *


Ns = [10, 32, 50, 100, 140]

gp = lambda N: {
    'name': 'NormalizerModel',
    'kwargs': {
        'model': {
            'name': 'GPModel',
            'kwargs': dict(
                kernel=dict(
                    name='GPyRBF',
                    kwargs={'lengthscale': 0.6, 'ARD': True},
                ),
                noise_prior=None,
                do_optimize=True,
                num_mcmc=0,
            )
        },
    }
}

kissmodels = lambda N: {
    'name': 'NormalizerModel',
    'kwargs': {
        'model': {
            'name': 'DKLGPModel',
            'kwargs': {
                'learning_rate': 0.1,
                'n_iter': 300,
                'nn_kwargs': {'layers': None},
                'gp_kwargs': {'n_grid': N * 2},
                'use_cg': True,
                'noise': None
            }
        }
    }
}

models = [gp, kissmodels]

for model in models:
    for N in Ns:
        run = execute(config_updates={
            'tag': 'heston',
            'obj_func': {
                'name': 'HestonOptionPricer',
            },
            'model': model(N),
            'gp_samples': N,
            'use_sample_grid': True,
        })

for d in range(1,5):
    run = execute(config_updates={
        'tag': 'heston',
        'obj_func': {
            'name': 'HestonOptionPricer',
        },
        'model': {
            'name': 'AdaptiveSparseGrid',
            'kwargs': dict(
                depth=d, 
                refinement_level=0,
                f_tol=1e-3,
            )
        },
    })
