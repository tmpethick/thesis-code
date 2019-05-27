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

latexify(columns=3)

# model = GPModel(kernel=kernel, noise_prior=0.1, do_optimize=False)

# N = 3
# X_line = np.linspace(0,1, 100)[:, None]
# samples = model.gpy_model.posterior_samples(X_line, size=N)

# for i in range(N):
#     plt.plot(X_line, samples[i])
# plt.show()

#%% Lengthscales


for lengthscale in [0.1, 0.5, 1]:
    kernel = GPy.kern.RBF(1)
    kernel.lengthscale = lengthscale

    N = 100
    M = 3
    X_line = np.linspace(-1,1, N)[:, None]

    covar = kernel.K(X_line, X_line)
    samples = np.random.multivariate_normal(np.zeros(N), covar, size=M)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(M):
        ax.plot(X_line, samples[i])
    plt.show()

    savefig(fig, 'GP/kernel-{}.pdf'.format(lengthscale))


#%% Kernel types

kernels = {
    'RBF': GPy.kern.RBF(1),
    'Matern52': GPy.kern.Matern52(1),
    'Linear': GPy.kern.Linear(1)
}

for (name, kernel) in kernels.items():
    kernel.lengthscale = 0.1

    N = 100
    M = 3
    X_line = np.linspace(-1,1, N)[:, None]

    covar = kernel.K(X_line, X_line)
    samples = np.random.multivariate_normal(np.zeros(N), covar, size=M)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(M):
        ax.plot(X_line, samples[i])
    plt.show()

    savefig(fig, 'GP/kernel-{}.pdf'.format(name))

#%% Covariance matrix

kernels = {
    'RBF': GPy.kern.RBF(1),
    'Linear': GPy.kern.Linear(1),
}
N = 100

for name, kernel in kernels.items():
    X_line = np.linspace(0,1, N)[:, None]
    covar = kernel.K(X_line, X_line)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(covar)

#%% Conditioned covariance matrix

from GPy.models import GPRegression

kernels = {
    'RBF': GPy.kern.RBF(1),
    'Linear': GPy.kern.Linear(1),
}

N = 100
T = 3

def f(X):
    return np.sin(X)

for name, kernel in kernels.items():
    X_line = np.linspace(0,1, N)[:, None]

    X_train = np.random.uniform(-1, 1, T)[:, None]
    Y_train = f(X_train)

    covar = kernel.K(X_line, X_line)

    model = GPRegression(X_train, Y_train, kernel=kernel)
    mean, covar = model.predict(X_line, full_cov=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(covar)


#%% Active subspaces
# 


