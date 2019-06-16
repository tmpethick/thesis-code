#%%
%load_ext autoreload
%autoreload 2

from runner import notebook_run, execute

#%%
# --------------------- Active sampling -----------------------

from src.environments.nonstationary import IncreasingOscillation
import matplotlib.pyplot as plt

fig = IncreasingOscillation().plot()
plt.show()

#%% Model mismatch against linear interpolation
run = notebook_run(config_updates={
    'obj_func': {
        'name': 'IncreasingOscillation',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': dict(
            kernel=dict(
                name='GPyRBF',
                kwargs={'lengthscale': 1},
            ),
            noise_prior=None,
            do_optimize=True,
            num_mcmc=0,
            n_burnin=100,
            subsample_interval=10,
            step_size=1e-1,
            leapfrog_steps=20
        )
    },
    'model2': {
        'name': 'LinearInterpolateModel',
        'kwargs': {},
    },
    'acquisition_function': {
        'name': 'AcquisitionModelMismatch',
        'kwargs': {'beta': 1}
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 50,
            'n_acq_max_starts': 2,            
        }
    },

})

#%%
# -------------------------- Derivative Exploitation ------------------------

# 1D
from src.environments.smooth import Sinc
import numpy as np
f = Sinc()
#f.plot()
f.plot_derivative()
f._plot(lambda x: np.abs(f.derivative(x)))

#%% 2D
from src.environments.discontinous import Kink2D

f = Kink2D()
f.plot_derivative()

#%% Sinc
run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Sinc',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': dict(
            kernel=dict(
                name='GPyRBF',
                kwargs={'lengthscale': 1},
            ),
            noise_prior=None,
            do_optimize=True,
            num_mcmc=0,
        )
    },
    'model2': {
        'name': 'DerivativeGPModel',
        'kwargs': dict(
            kernel=dict(
                name='GPyRBF',
                kwargs={'lengthscale': 1},
            ),
            noise_prior=None,
            do_optimize=True,
            num_mcmc=0,
        ),
    },
    'acquisition_function': {
        'name': 'DerivativeAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'uses_derivatives': (1,),
            'n_init': 5,
            'n_iter': 50,
            'n_acq_max_starts': 2,            
        }
    },
})

#%%
run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': dict(
            kernel=dict(
                name='GPyRBF',
                kwargs={'lengthscale': 1},
            ),
            noise_prior=None,
            do_optimize=True,
            num_mcmc=0,
        )
    },
    'model2': {
        'name': 'GPModel',
        'kwargs': dict(
            kernel=dict(
                name='GPyRBF',
                kwargs={'lengthscale': 1},
            ),
            noise_prior=None,
            do_optimize=True,
            num_mcmc=0,
        ),
    },
    'acquisition_function': {
        'name': 'DerivativeAcquisition',
        'kwargs': {'beta': 0},
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'uses_derivatives': (1,),
            'n_init': 5,
            'n_iter': 50,
            'n_acq_max_starts': 2,            
        }
    },
})




#%%
# ----------------- LocalLengthScaleGPModel (LLS) -------------------

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'ActiveSubspaceTest',
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
                'name': 'LocalLengthScaleGPModel',
                'kwargs': {
                    'l_samples': 200,
                    'n_optimizer_iter': 10,
                },
            },
        },
    },
    'gp_use_derivatives': True,
    'gp_samples': 50,
})


#%%

# Works very well when the lengthscale indeed has a smooth functional form of x.
run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink2D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 200,
            'n_optimizer_iter': 10,
        }
    },
    'gp_samples': 50,
})

#%% LLS also correctly has higher variance in high frequency area (necessary for correct active sampling.)

run = execute(config_updates={
    'obj_func': {
        'name': 'IncreasingOscillation',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 5,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'gp_samples': 50,
})

#%% It has a tendency to overfit however...

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink2D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 5,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 50,
            'n_acq_max_starts': 2,            
        }
    },
})

#%%
run = execute(config_updates={
    'obj_func': {
        'name': 'Kink2D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 5,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 50,
            'n_acq_max_starts': 2,            
        }
    },
})

#%% AS
from src.utils import random_hypercube_samples
from src.environments.high_dim import ActiveSubspaceTest
from src.models import ActiveSubspace

Alpha = np.array([[0.78695576],
       [0.70777112],
       [0.34515641],
       [0.20288506],
       [0.52388727],
       [0.2025096 ],
       [0.31752746],
       [0.24497726],
       [0.89249818],
       [0.64264009]])

f = ActiveSubspaceTest()
#f = TwoKinkDEmbedding(Alpha=Alpha)
X = random_hypercube_samples(100, f.bounds)
G = f.derivative(X)
model = ActiveSubspace()
model.fit(X, f(X), G)
model.plot()
plt.show()
X_test = random_hypercube_samples(1, f.bounds)
model.transform(X_test)


#%%
# ----------------- DKLModel -------------------

#%%

# DKL can learn mapping when there is actually an underlying functional 
# transformation that makes it stationary.
run = notebook_run(config_updates={
    'obj_func': {
        'name': 'IncreasingOscillation',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'n_iter': 500,
            'nn_kwargs': {
                'layers': (1000, 500, 50, 2),
            }
        }
    },
    'gp_samples': 100,
})

#%%
run = notebook_run(config_updates={
    'obj_func': {
        'name': 'ActiveSubspaceTest',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'n_iter': 500,
            'nn_kwargs': {
                'layers': (1000, 500, 50, 2),
            }
        }
    },
    'gp_samples': 100,
})


#%%

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink2D',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'n_iter': 50,
            'nn_kwargs': {
                'layers': (1000, 500, 50, 2),
            }
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 50,
            'n_acq_max_starts': 2,            
        }
    },
})


#%%

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink2D',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'n_iter': 50,
            'nn_kwargs': {
                'layers': (1000, 500, 50, 2),
            }
        }
    },
    'gp_samples': 10,
})


#%%
# model, model2, acq, bo = run.interactive_stash['model'], \
#                          run.interactive_stash['model2'], \
#                          run.interactive_stash['acq'], \
#                          run.interactive_stash['bo']

#%%
# ------------------- High Dim Fast ----------------------




#%%
# ------------------ Templates --------------------

# %% Fully specified GP without acq
notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
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
            'noise_prior': None,
            'do_optimize': False,
            'num_mcmc': 10,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        },
    },
    'gp_samples': 5,
})

# %% Fully specified GP with acq
notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
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
            'noise_prior': None,
            'do_optimize': False,
            'num_mcmc': 10,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'gp_samples': 5,
})

# %% Fully specified BO

notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
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
            'noise_prior': None,
            'do_optimize': False,
            'num_mcmc': 10,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 10,
            'n_acq_max_starts': 10,            
        }
    },
})

#%%

notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 5,
        }
    },
    'acquisition_function': {
        'name': 'QuadratureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 10,
            'n_acq_max_starts': 10,            
        }
    },
})


# %% Small modifications

notebook_run(config_updates={},
       named_configs=['configs/default.yaml', 'configs/gpy.yaml'],
       options={'--force': True})


# %%
# --------------- LinearInterpolate -----------------

# Ground truth for Kink2D (sample proportional to value).
# to compare: kde vs f(x)/int f(x)

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'IncreasingOscillation',
    },
    'model': {
        'name': 'LinearInterpolateModel',
        'kwargs': {}
    },
    'gp_samples': 50,
})

# %%
# --------------- Utils ----------------

# %% Save configs
notebook_run('save_config', named_configs=['gpy'], config_updates={'config_filename': 'configs/gpy.yaml'},
       options={'--force': True})

# %% Print configs

notebook_run('print_config', named_configs=['config/gpy.yaml'])

# %% YAML to python dict

run = notebook_run('print_config', named_configs=['configs/gpy.yaml'])
run.config
