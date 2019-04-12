#%%
%load_ext autoreload
%autoreload 2

from runner import notebook_run

# --------------- TODO: ----------------

# Testing sampling pattern:
# 2D sin (with hessian)
# See that sampling matches hessian
    # Run BO and plot samples against hessian. (kde? requires too many samples...)

# Methods:
    # sample w.r.t. to the true hessian
    # Sample based on GP hessian.
    # Use derivative info to fit GP. (how does it improve hessian estimate?)
    # Use hessian variance?

# Ultimately:
    # Hessian for sampling
    # DKL vs. AS + Lengthscale
    # QFF/KISS for scaling both
    # Use derivatives
# Requires:
    # Implement hessian for DKL and lengthscale

# TODO:
# Compare: uniform, GP variance, hessian
# Find functions for which the sampling stategy is especially effective
# Find models for which the sampling strategy is especially effective

# Implement for DKL
# Implement for Lengthscale
# How do we verify behavior for high-dim? (construct?)


# --------- Old comments -----------
# Codebase:
# Annotate dimensions for each function.


# it becomes messy dealing with multi-objective GP if gradients are observed. Instead funnel through a (better fitted) GP using the gradients and derive the posterior gradient/hessian from the GP. (how much information do we loose?)

# What norm to use for gradient acquisition?

# Big problem: the exploitation is not reduced with samples (the hessian will continue to remain high). We somehow need a tradeoff hyperparameter beta (coin flip of whether to sample uniformly?).
# Maybe instead of maximizing acq we should be sampling from (mu_hessian + var_hessian) + const. We now have a moving distribution (over BO iterations) that we are sampling...
# Maybe MCMC "reusing" the previous step as burn-in (even though it is a different distribution). This is similar to what is done for hyperparameter marginalization.

# hmm relevant? https://arxiv.org/pdf/1802.03479.pdf

# Diffing GPs! http://mlg.eng.cam.ac.uk/mchutchon/DifferentiatingGPs.pdf

#%%
# First see that hessian is correctly implementing by
# plotting Hessian of GP.
import numpy as np
import matplotlib.pyplot as plt

from src.environments import Sinc
from src.models import GPModel
from src.kernels import GPyRBF
from src.utils import random_hypercube_samples

f = Sinc()
X = random_hypercube_samples(50, f.bounds)
Y = f(X)

kernel = GPyRBF(f.input_dim)
model = GPModel(kernel, do_optimize=True)
model.init(X,Y)
X_line = np.linspace(f.bounds[0,0], f.bounds[0,1], 100)[:,None]
mean, var = model.get_statistics(X_line)
jac, jac_var = model.predict_jacobian(X_line)
hess, hess_var = model.predict_hessian(X_line)

model.plot(X_line)
plt.plot(X_line, mean[0])
plt.legend()
plt.show()

plt.plot(X_line, f.derivative(X_line), label="True jac")

plt.plot(X_line, jac[:,0], label="Jac")

# jac_simple = np.zeros(len(X_line))
# for i in range(len(X_line)):
#     p, v = model.predict_jacobian_1sample(np.array([X_line[i]]))
#     jac_simple[i] = p

# Scaling issue (no biggy)
# plt.plot(X_line, jac_simple, label="Jac 1 sample")
plt.legend()
plt.show()

plt.plot(X_line, hess[:, 0, 0], label="hessian")
plt.plot(X_line, f.hessian(X_line), label="True hessian")
plt.legend()
plt.show()

#%%

run = notebook_run(through_CLI=True, config_updates={
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
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': 60,
            'n_acq_max_starts': 10,
        }
    },
})


#%% See that sampling strategy indeed matches curvature

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

f = run.interactive_stash['f']
model = run.interactive_stash['model']
plt.plot(model.X)
plt.scatter(np.arange(len(model.X)), model.X)
plt.show()

f._plot(lambda X: np.abs(f.hessian(X)))
plt.hist(model.X, bins=50, normed=True)
#sns.distplot(model.X)
plt.show()


#%% Run uniform sampling, variance sampling, curvature sampling
N_SAMPLES = 65
FUNC = 'IncreasingOscillationDecreasingAmplitude'

run = notebook_run(through_CLI=True, config_updates={
    'obj_func': {
        'name': FUNC,
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
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'gp_samples': N_SAMPLES,
}

run = notebook_run(through_CLI=True, config_updates={
    'obj_func': {
        'name': FUNC,
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
            'do_optimize': True,
            'num_mcmc': 0,
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
            'n_iter': N_SAMPLES - 5,
            'n_acq_max_starts': 10,
        }
    },
})

run = notebook_run(through_CLI=True, config_updates={
    'obj_func': {
        'name': FUNC,
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
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisition',
    },
    'bo': {
        'name': 'AcquisitionAlgorithm',
        'kwargs': {
            'n_init': 5,
            'n_iter': N_SAMPLES - 5,
            'n_acq_max_starts': 10,
        }
    },
})


# %%
# What function do we expect it to be good on?
# If model is specified correctly we will sample "uniformly" (true)
# So only if mis-specified model.

#%% Debugging MCMC... (depends on a BO constructed with a run)

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Sinc',
    },
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 0.1
                }
            },
            'noise_prior': None,
            'do_optimize': True,
            'num_mcmc': 0,
            'subsample_interval': 10,
            'step_size': 1e-1,
            'leapfrog_steps': 20,
            'n_burnin': 100,
        }
    },
    'acquisition_function': {
        'name': 'CurvatureAcquisitionDistribution',
        'kwargs': {
            'beta': 0
        }
    },
    'bo': {
        'name': 'SampleAlgorithm',
        'kwargs': {
            'n_init': 50,
            'n_iter': 0,
            'n_acq_max_starts': 10,
        }
    },
})


import matplotlib.pyplot as plt
import numpy as np
import emcee

bo = run.interactive_stash['bo']
bo.models[0].init(bo.X, bo.Y)
bo.plot()
plt.show()
hess_samples = np.array([bo._next_x() for i in range(100)])
plt.hist(hess_samples[:, 0])
plt.show()

bo.plot()
plt.show()

def acq_one(x):
    return bo.acquisition_function(np.array([x]))[0]

walkers = 20
sampler = emcee.EnsembleSampler(walkers, 1, acq_one)
sampler.run_mcmc([np.array([0]) for i in range(walkers)], 100, storechain=False)
sampler.run_mcmc(None, 100)
plt.hist(sampler.flatchain)
plt.show()
