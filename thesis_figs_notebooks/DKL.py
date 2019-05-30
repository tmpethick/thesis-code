#%%
%load_ext autoreload
%autoreload 2

from runner import notebook_run, notebook_run_CLI, notebook_run_server, execute

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

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
            'learning_rate': 0.01,
            'noise': 1e-1,
            'n_iter': 1000,
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
            'noise': 1e-1,
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
O = call_function_on_grid(f.noiseless, XY)
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

#%% Construct Frequency spectrum to show that it is more compact.

from scipy.fftpack import fft

model = run.interactive_stash['model']
f = run.interactive_stash['f']
f.plot()
plt.show()
N = 1000

X = np.linspace(f.bounds[0,0], f.bounds[0,1], N)[:, None]
y = f.noiseless(X)[:, 0]
sp = np.fft.fft(y)
freq = np.fft.fftfreq(N)
plt.plot(freq, np.abs(sp.real))
plt.show()

# TODO: freq. spectrum for feature domain
# # Naive: grid in feature space -> requires inverse mapping.
# phi = model.get_features(X)[:,0]
# sp = np.fft.fft(phi)
# freq = np.fft.fftfreq(N)
# plt.plot(freq, np.abs(sp.real))
# plt.show() 
# # We see how it focuses the spectrum to a more compact freq. support.


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

config = {
    'tag': 'certify-ExactDKL',
    'obj_func': {'name': 'Sinc'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'noise': 1e-2,
            'n_iter': 1000,
            'learning_rate': 0.01, 
            'nn_kwargs': {
                'layers': None, 
            }
        }
    },
    'model2': exactGP,
    'gp_samples': 20,
    'model_compare': True,
}
run = execute(config_updates=config)
model = run.interactive_stash['model']
model2 = run.interactive_stash['model2']
print(run.interactive_stash['model'].model.covar_module.lengthscale)



#%% Non-stationary in low dim

# Compare DKL with GP for low dim non-stationary (see A-SG further down)

config = {
    'obj_func': {'name': 'TwoKink1D'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': 0.01}
        },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)
print(run.result)

#%%

config = {
    'obj_func': {'name': 'TwoKink1D'},
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'noise_prior': 1e-4,
            'do_optimize': True,
            'num_mcmc': 0,
        },
    },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)
print(run.result)

#%% DNGO without BLR is even unstable...
# n_iter=1000 => stuck
# n_iter=10000 => always escapes it seems

# We first need LinearFromFeatureExtractor to train

config = {
    'obj_func': {'name': 'Step', 'kwargs': {'noise': 0.01}},
    'model': {
        'name': 'LinearFromFeatureExtractor',
        'kwargs': {
            'normalize_input': True,
            'normalize_output': True,
            'learning_rate': 0.01,
            'n_iter': 1000,
            'layers': [50, 5],
            'data_dim': 1,
            }
        },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)


#%% 1D Step in 2D feature space
# Reproducing DKL (and Manifold GP)

# Fix Unstable
# Recreate nice behaving feature map with [5,5,1]
# Inspect GP prediction in feature space

config = {
    'obj_func': {'name': 'SingleStep', 'kwargs': {'noise': 0.01}},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 1000,
            'pretrain_n_iter': 10000,
            'do_pretrain': True,
            'nn_kwargs': {'layers': [5, 5, 1]},
            # 'gp_kwargs': {'n_grid': 1000},
            'noise': None, #0.01,
            }
        },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)


#%%

# Very tricky beating A-SG in low dim... 
# So only useful to consider DKL because A-SG breaks down in high-dim.
# (keep as baseline anyway)


f = TwoKink1D()
X_test = random_hypercube_samples(1000, f.bounds)
N_test = X_test.shape[-1]
Y_test = f(X_test)

def calc_error(i, model):
    max_error, L2_err = model.calc_error(X_test, Y_test)
    print("{0:9d} {1:9d}  Loo={2:1.2e}  L2={3:1.2e}".format(i+1, model.grid.getNumPoints(), max_error, L2_err))

#asg = AdaptiveSparseGrid(f, depth=1, refinement_level=20, f_tol=1e-3, point_tol=1000)
asg = AdaptiveSparseGrid(f, depth=10, refinement_level=0) # 2^10 = 1024 points
asg.fit(callback=calc_error)
fig = asg.plot()


#%%

KinkDCircularEmbedding(D=1).plot()
KinkDCircularEmbedding(D=2).plot()

f = KinkDCircularEmbedding(D=10)
X = random_hypercube_samples(10000, f.bounds)
G = f.derivative(X)
model = ActiveSubspace(threshold_factor=4)
model.fit(X, f(X), G)
print(model.W.shape[-1])
model.plot()
Z = model.transform(X)
plt.show()
plt.scatter(Z, f(X))

#%%
# How does the transformation look in 2D? (we fix it to 2D even though we only care about 1 eigenvector)

f = KinkDCircularEmbedding(D=2)
X = random_hypercube_samples(1000, f.bounds)
G = f.derivative(X)
model = ActiveSubspace(threshold_factor=4, output_dim=2)
model.fit(X, f(X), G)
print(model.W.shape[-1])
model.plot()

XY, X, Y = construct_2D_grid(f.bounds)
Z = call_function_on_grid(f.noiseless, XY)[..., 0]
F = call_function_on_grid(model.transform, XY)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.contourf(F[...,0], F[...,1], Z, 50)
plt.show()

#%% Plot example of Active Subspace failing (Circular5D)

f = KinkDCircularEmbedding(D=5)
X = random_hypercube_samples(1000, f.bounds)
G = f.derivative(X)
model = ActiveSubspace(threshold_factor=4)
model.fit(X, f(X), G)
Z = model.transform(X)
latexify(columns=3)
fig = plt.figure()
plt.ylabel("$Y$")
plt.xlabel("$\Phi(X)$")
plt.scatter(Z, f(X), s=0.4)
plt.tight_layout()
savefig(fig, 'DKL/AS-circular5D-feature.pdf')

#%%
D = 2
f = KinkDCircularEmbedding(D=D, bounds=np.array([[-1, 1]] * D))
X = random_hypercube_samples(1000, f.bounds)
G = f.derivative(X)
model = ActiveSubspace(threshold_factor=4)
model.fit(X, f(X), G)
Z = model.transform(X)


fig = plt.figure()
plt.ylabel("$Y$")
plt.xlabel("$\Phi(X)$")
plt.scatter(Z, f(X), s=0.4)
plt.tight_layout()


#%% Investigating Embeddings

exactGP = {
    'name': 'GPModel',
    'kwargs': {
        'kernel': {
            'name': 'GPyRBF',
            'kwargs': {
                'lengthscale': 1
            }
        },
        'noise_prior': None,#1e-2,
        'do_optimize': True,
        'num_mcmc': 0,
        'mean_prior': True,
    },
}


DKLModel = {
    'name': 'DKLGPModel',
    'kwargs': {
        'learning_rate': 0.01,
        'n_iter': 1000,
        'nn_kwargs': {'layers': [100, 50, 1]},
        'noise': None, #1e-2
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
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 5}},
]
model = models[2]
run = execute(config_updates={
    'tag': 'embedding',
    'obj_func': functions[0],
    'model': model,
    'gp_use_derivatives': model.get('name') == 'TransformerModel',
    'gp_samples': 1000,
})

#%% Test mean/0 estimator

model = run.interactive_stash['model']
f = run.interactive_stash['f']
Y = model.Y
Y_est = np.mean(Y, axis=0)
Y_est_RMSE = np.sum((Y_est - Y) ** 2) / Y_est.shape[0]
print("const mean est:", Y_est_RMSE)
print("const 0 est:", np.sum((Y) ** 2) / Y_est.shape[0])

RMSE = calc_errors(model, f, rand_N=2500)
RMSE / Y_est_RMSE

# 3, 5, 10, 20
# 0.15725597810108272, 0.10239141449920977, 0.0014818985149986441, 0.0001217382706791325

#%%
# Scale DKL to more samples

functions = [
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 6}},
]
model = models[2]
run = execute(config_updates={
    'tag': 'embedding',
    'obj_func': functions[0],
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.1,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'gp_kwargs': {'n_grid': 1000},
            'noise': None, #1e-2,
        },
    },
    'gp_use_derivatives': model.get('name') == 'TransformerModel',
    'gp_samples': 10000,
})

# We need more samples to learn meaningful mapping (which in turn require us to scale the GP)
# Could we come up with a better high-dim test than CircularD?

#%%



#%%
# RMSE as function of #points (for ExactGP)

# Could KISS-GP be better than ExactGP if it allowed for more points?
# (it could be a requirement if the join model requires many points because feature mapping does!)
# Assume we've learned correct mapping... so we can safely consider the 1D case in isolation.

tests = range(0, 4)

RMSEs = np.zeros(len(tests))

for i in tests:
    functions = [
        {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 1}},
    ]
    model = models[2]
    run = execute(config_updates={
        'tag': 'embedding',
        'obj_func': functions[0],
        'model': {
            'name': 'DKLGPModel',
            'kwargs': {
                'learning_rate': 0.1,
                'n_iter': 100,
                'nn_kwargs': {'layers': None},
                # 'gp_kwargs': {'n_grid': 1000},
                'noise': None, #1e-2,
            },
        },
        'gp_use_derivatives': model.get('name') == 'TransformerModel',
        'gp_samples': 10 ** (i + 1),
    })

    RMSEs[i] = run.result['rmse']
plt.plot(RMSEs)

#%% Why do we not learn to smooth out the kink? (but it works for IncreasingOscillation)

# Sensitive to hyperparams: gp_samples=100, lr=0.1 fails (not pos.def.) for Kink2D while lr=0.01 works. (Only a problem if noiseless it seems)
# linear_cg error when gp_samples=1000

functions = [
    {'name': 'IncreasingOscillation', 'kwargs': {'noise': 1e-1}},
    #{'name': 'Kink2D', 'kwargs': {'noise': 1e-2}},
    #{'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 2, 'noise': 1e-1}},
]
run = execute(config_updates={
    'tag': 'embedding',
    'obj_func': functions[0],
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': 1e-1,
        },
    },
    'gp_use_derivatives': False,
    'gp_samples': 100,
})

#%%
# Trying to see if many samples help
# Problem: breaks with "CG terminated in 1000 iterations with average residual norm 1.3085484504699707 which is larger than the tolerance of 1 specified by gpytorch.settings.cg_tolerance.""

functions = [
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 1}},
]
model = models[2]
run = execute(config_updates={
    'tag': 'embedding',
    'obj_func': functions[0],
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.1,
            'n_iter': 100,
            'nn_kwargs': {'layers': None},
            'gp_kwargs': {'n_grid': 5000},
            'noise': None, #1e-2,
        },
    },
    'gp_use_derivatives': model.get('name') == 'TransformerModel',
    'gp_samples': 5000,
})



#%% 

# Normalize (instead of prior)
# Raise error if gpytorch warning (if log has warning?)

# What if we took a Kink2D that was not bend? Would it learn to stretch the domain?
# How does it even help on step function?


#%% Scalability (take hardest problems of Ganz1984)

# Do this with fixed hyperparameters... (which one to choose?)
# First function that can comes from the kernel (to avoid approximation error from other factors)
# Also let it be low dimensional.

# Then function that does not (model mis-match). (still in low dim)

# Move to higher dimensions where many points are actually needed. (Genz1984)
    # effect of m
    # effect of n

# scale exact
# scale DKL
# (see performance difference and when exact breaks)


#%% 
# Genz1984 2D for active sampling


#%% Manifold GP

import numpy as np
import pylab

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gp_extras.kernels import ManifoldKernel

np.random.seed(1)

# Specify Gaussian Process
kernel = C(1.0, (0.01, 100)) \
    * ManifoldKernel.construct(base_kernel=RBF(0.1), architecture=((1, 5, 2),),
                               transfer_fct="relu", max_nn_weight=1)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                              n_restarts_optimizer=10)


X_ = np.linspace(-0.1, 1.1, 100)

def f(X_):
    X_steps = np.array([0.0, 0.5])
    Y_values = np.array([0, 1])
    # X_steps = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    # Y_values = np.array([0, 1, 0.2, 0.8, 1])

    condlist = [X_ > threshold for threshold in X_steps]
    return np.piecewise(X_, condlist, Y_values)
y_ = f(X_)

# Visualization of prior
pylab.figure(0, figsize=(10, 8))
X_nn = gp.kernel.k2._project_manifold(X_[:, None])
pylab.subplot(3, 2, 1)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i], label="Manifold-dim %d" % i)
pylab.legend(loc="best")
pylab.xlim(-0.1, 1.1)
pylab.title("Prior mapping to manifold")

pylab.subplot(3, 2, 2)
y_mean, y_std = gp.predict(X_[:, None], return_std=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X_, y_mean - y_std, y_mean + y_std,
                   alpha=0.5, color='k')
y_samples = gp.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.plot(X_, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.legend(loc="best")
pylab.xlim(-0.1, 1.1)
pylab.ylim(-4, 3)
pylab.title("Prior samples")


# Generate data and fit GP
X = np.random.uniform(0, 1, 100)[:, None]
y = f(X)[:,0]
gp.fit(X, y)

# Visualization of posterior
X_nn = gp.kernel_.k2._project_manifold(X_[:, None])

pylab.subplot(3, 2, 3)
for i in range(X_nn.shape[1]):
    pylab.plot(X_, X_nn[:, i], label="Manifold-dim %d" % i)
pylab.xlim(-0.1, 1.1)
pylab.legend(loc="best")
pylab.title("Posterior mapping to manifold")

pylab.subplot(3, 2, 4)
y_mean, y_std = gp.predict(X_[:, None], return_std=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X_, y_mean - y_std, y_mean + y_std,
                   alpha=0.5, color='k')
y_samples = gp.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.plot(X_, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.plot(X_, y_, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-0.1, 1.1)
pylab.ylim(-4, 3)
pylab.legend(loc="best")
pylab.title("Posterior samples")

# For comparison a stationary kernel
kernel = C(1.0, (0.01, 100)) * RBF(0.1)
gp_stationary = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                                         n_restarts_optimizer=1)
gp_stationary.fit(X, y)

pylab.subplot(3, 2, 6)
y_mean, y_std = gp_stationary.predict(X_[:, None], return_std=True)
pylab.plot(X_, y_mean, 'k', lw=3, zorder=9, label="mean")
pylab.fill_between(X_, y_mean - y_std, y_mean + y_std,
                   alpha=0.5, color='k')
y_samples = gp_stationary.sample_y(X_[:, None], 10)
pylab.plot(X_, y_samples, color='b', lw=1)
pylab.plot(X_, y_samples[:, 0], color='b', lw=1, label="samples") # just for the legend
pylab.scatter(X[:, 0], y, c='r', s=50, zorder=10)
pylab.plot(X_, y_, 'r', lw=1, zorder=9, label="true function")
pylab.xlim(-0.1, 1.1)
pylab.ylim(-4, 3)
pylab.legend(loc="best")
pylab.title("Stationary kernel")

pylab.tight_layout()
pylab.show()
