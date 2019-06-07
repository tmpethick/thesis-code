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

def normalize_config(config):
    config['model'] = {
        'name': 'NormalizerModel',
        'kwargs': {
            'model': config['model'],
            'normalize_input': True,
            'normalize_output': True,
        }
    }

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
#savefig(fig, 'DKL/dkl-kink2D-manifold-f.pdf')
plt.show()

fig, ax = plt.subplots()
ax.set_title('f in feature space')

XY, X, Y = construct_2D_grid(f.bounds)
Z = call_function_on_grid(model.get_features, XY)
O = call_function_on_grid(f.noiseless, XY)
ax.contourf(Z[...,0], Z[...,1], O[...,0], 50)

plt.tight_layout()

#savefig(fig, 'DKL/dkl-kink2D-manifold-features.pdf')
    

#%% IncreasingOscillation lengthscale change
# \label{fig:dkl-lengthscale}

config = {
    'obj_func': {'name': 'IncreasingOscillation'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': None
        },
    },
    'gp_samples': 1000,
}
normalize_config(config)
run = execute(config_updates=config)

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

#%% Plot the feature space

model = run.interactive_stash['model']
f = run.interactive_stash['f']
normalized_f = EnvironmentNormalizer(f, model.X_normalizer, model.Y_normalizer)
model.model.plot_features(normalized_f)

#%% TODO: Construct Frequency spectrum to show that it is more compact.

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

import scipy.signal as signal
phi = model.get_features(X)[:,0]
f = np.linspace(-20, 20, 1000)
pgram = signal.lombscargle(phi, y, f, normalize=True)
plt.plot(f, pgram)

# TODO: freq. spectrum for feature domain
# # Naive: grid in feature space -> requires inverse mapping.
# phi = model.get_features(X)[:,0]
# sp = np.fft.fft(phi)
# freq = np.fft.fftfreq(N)
# plt.plot(freq, np.abs(sp.real))
# plt.show() 
# # We see how it focuses the spectrum to a more compact freq. support.


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
# Create normalizing plot
from src.utils import _calc_errors

Ds = range(2, 50)
functions = [GenzContinuous, GenzCornerPeak, GenzDiscontinuous, GenzGaussianPeak, GenzOscillatory, GenzProductPeak]

fig, axes = plt.subplots(2, 3, figsize=(5, 3))
axes = axes.flatten()

import src.latex as figuretex
figuretex.use_config(width_scale=1.0, height_scale=2)

for i, function in enumerate(functions):
    errs = np.empty(len(Ds))
    ax = axes[i]
    
    for i, D in enumerate(Ds):
        def test():
            f = function(D=D)

            n_samples = 1000
            X_train = random_hypercube_samples(n_samples, f.bounds)
            Y_train = f(X_train)

            Y_est = np.mean(Y_train, axis=0)
            mean_estimator = lambda X: np.repeat(Y_est[None,:], X.shape[0], axis=0)

            rmse, max_err = _calc_errors(mean_estimator, f.noiseless, f, rand=True, rand_N=10000)
            return rmse

        rmses = [test() for i in range(5)]
        errs[i] = np.mean(rmses)
    ax.plot(Ds, errs, label=function.__name__)
    ax.set_yscale('log')
    ax.set_title(function.__name__)

plt.tight_layout()

# TODO: how quickly do we converge across different functions with DKL?


#%%
# Influence on initial lengthscale on log likelihood

bounds = np.array([[0,1]])
train_x = torch.linspace(bounds[0,0], bounds[0,1], 100)
train_y = np.sin(60 * train_x ** 4)

def gpy_model(noise=0.5, lengthscale=2.5, variance=2):
    import math
    import torch
    import gpytorch
    from matplotlib import pyplot as plt

    import GPy

    kernel = GPy.kern.RBF(1, ARD=False)
    model = GPy.models.GPRegression(train_x.numpy()[:,None], train_y.numpy()[:,None], kernel=kernel)
    model.Gaussian_noise = noise
    model.kern.lengthscale = lengthscale
    model.kern.variance = variance

    model.optimize()

    return model, {
        'lengthscale': kernel.lengthscale,
        'noise':model.Gaussian_noise.variance,
        'variance': kernel.variance
    }

model, params = gpy_model()


#%%
# Make log-vs-lengthscale

latexify(columns=2, fig_height=4)

inits = np.exp(np.arange(-10, 10))
ls = np.empty(inits.shape[0])
ll = np.empty(inits.shape[0])
for i, init in enumerate(inits):
    model, params = gpy_model(noise=1, lengthscale=init, variance=1.5)
    ls[i] = params['lengthscale']
    ll[i] = model.log_likelihood()

fig, ax = plt.subplots()
ax.set_xscale('log')
ax.set_ylabel('Log marginal likelihood')
ax.set_xlabel('Initial lengthscale')
ax.plot(inits, -ll)

savefig(fig, 'DKL/lml(lengthscale)-a.pdf')

#%%

distinct_inits = [
    inits[0],
    inits[inits.shape[0] // 2],
    inits[-1],
]

fig, axes = plt.subplots(3, 1)

for i, init in enumerate(distinct_inits):
    model, params = gpy_model(noise=1, lengthscale=init, variance=1.5)

    test_x = np.linspace(bounds[0,0], bounds[0,1], 1000)
    mean, var = model.predict(test_x[:, None])
    mean = mean[:,0]
    var = var[:,0]

    ax = axes[i]
    lower, upper = mean - 2 * np.sqrt(var), mean + 2 * np.sqrt(var)
    ax.plot(train_x.numpy(), train_y.numpy(), '*')
    ax.plot(test_x, mean)
    ax.fill_between(test_x, lower, upper, alpha=0.5)
    #ax.legend(['Observed Data', 'Mean', 'Confidence'])

savefig(fig, 'DKL/lml(lengthscale)-b.pdf')


#%%
# Working DKL model for IncreasingOscillation
config = {
    'obj_func': {'name': 'IncreasingOscillation'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.1,
            'n_iter': 100,
            'nn_kwargs': {'layers': [500, 100, 1],
                          'normalize_output': True},
            'use_cg': True,
            'initial_parameters': {
                'lengthscale': 0.1,
                'noise': 0.1,
                'outputscale': 1
            },
            'noise': None,
        },
    },
    'gp_samples': 1000,
}
normalize_config(config)
run = execute(config_updates=config)

model = run.interactive_stash['model']
