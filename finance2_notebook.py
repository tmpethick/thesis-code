#%%

%load_ext autoreload
%autoreload 2

from runner import notebook_run, execute

import seaborn as sns
sns.set_style("white")

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
            'n_features': 500,
        }
    },
    'gp_samples': 100
})

#%% SSGP

run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'SSGPModel',
        'kwargs': {
            'noise': 0.01,
            'n_features': 10,
            'do_optimize': True,
        }
    },
    'gp_samples': 1000
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
