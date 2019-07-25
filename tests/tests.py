#%%
from src.environments.financial import SPXOptions
from src.environments.nonstationary import IncreasingOscillationDecreasingAmplitude, Sin2DRotated
from src.environments.smooth import CosProd2D, Sinc2D
%load_ext autoreload
%autoreload 2
#%%
from runner import notebook_run, execute

import seaborn as sns
sns.set_style('white')

from src.plot_utils import *
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

def normalize_config(config):
    config['model'] = {
        'name': 'NormalizerModel',
        'kwargs': {
            'model': config['model'],
            'normalize_input': True,
            'normalize_output': True,
        }
    }

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
        'name': 'Kink2D',
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

config = {
    'obj_func': {'name': 'Sinc'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 100,
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
    'gp_samples': 100,
    'model_compare': True
}
run = execute(config_updates=config)

#%%

import matplotlib.pyplot as plt

IncreasingOscillationDecreasingAmplitude().plot()
plt.show(block=False)
Sin2DRotated().plot()
plt.show(block=False)
CosProd2D().plot()
plt.show(block=False)
Sinc2D().plot()
plt.show(block=False)
