from runner import notebook_run, notebook_run_CLI, notebook_run_server, execute

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

from src.utils import *
from src.plot_utils import *
from src.kernels import *
from src.models import *
from src.environments import *
from src.acquisition_functions import *
from src.algorithms import *

latexify(columns=1)

# f = TwoKink2D()
# X_test = random_hypercube_samples(1000, f.bounds)
# # Uses uniform since bounds are [0,1] to ensure implementation is not broken...
# X_test = np.random.uniform(size=(1000,2))
# N_test = X_test.shape[-1]
# Y_test = f(X_test)

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
    return config

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
