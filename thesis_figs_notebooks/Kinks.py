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
# SG does backly around kink

SG = AdaptiveSparseGrid(f, depth=15, refinement_level=0)
SG.fit(callback=calc_error)
fig = SG.plot()
SG_Loo_err, SG_L2_err = SG.calc_error(X_test, Y_test)

#%%

# ASG Will correctly sample around Kinks and do better.

def hyperparam_test(f_tol=0.001):
    ASG = AdaptiveSparseGrid(f, depth=1, refinement_level=30, f_tol=f_tol, point_tol=1e5)
    ASG.fit(callback=calc_error)
    fig = ASG.plot()
    ASG_Loo_err, ASG_L2_err = ASG.calc_error(X_test, Y_test)
    return ASG_L2_err

# f_tols = 10 ** (-np.linspace(1, 4, 15))
# V = np.empty(f_tols.shape)
# for i, f_tol in enumerate(f_tols):
#     print(f_tol)
#     V[i] = hyperparam_test(f_tol)

# idx = np.nanargmin(V)
# print(V[idx], f_tols[idx])

# 0.0001 best candidate yielding L2≈0.001617 (using 25k points but not significantly better than just using 11k)
hyperparam_test(0.0001)

#%% Let's see if we can beat this with GP

run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKink2D',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyMatern32',
                'kwargs': {
                    'lengthscale': 0.1
                }
            },
            'noise_prior': 0.001,
            'do_optimize': True,
            'num_mcmc': 0,
        }
    },
    'gp_samples': 1000
})

#%%
run = execute(config_updates={
    'obj_func': {
        'name': 'TwoKink2D',
    },
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'noise': 1e-3,
            'n_iter': 100,
            'nn_kwargs': {
                'layers': (1000, 500, 50, 2),
            }
        }
    },
    'gp_samples': 1000
})
print("RMSE: {:1.2e}".format(run.result))


#%%
# Increase accuracy with RFF, KISS-GP

#%% We care about clock-time...


#%%

print("{0} points: Loo={1:1.2e}  L2={2:1.2e}".format(SG.grid.getNumPoints(), SG_Loo_err, SG_L2_err))
print("{0} points: Loo={1:1.2e}  L2={2:1.2e}".format(ASG.grid.getNumPoints(), ASG_Loo_err, ASG_L2_err))


#%%
