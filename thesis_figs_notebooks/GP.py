#%%
#%load_ext autoreload
#%autoreload 2

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

X_train = np.array([-0.6, -0.5, 0.2, 0.8])[:, None]
Y_train = np.array([-1.5, 0.5, 0, -0.3])[:, None]

#%% Lengthscales

latexify(columns=3, fig_height=1)
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

with sns.dark_palette(color[0], reverse=True):

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

        mean = np.zeros(X_line.shape[0])
        var = np.diagonal(covar)
        CI = 2 * np.sqrt(var)
        ax.plot(X_line, mean, linewidth=2, c='C0')
        ax.fill_between(X_line[:,0], -CI, CI, alpha=0.3, color='C0')

        for i in range(M):
            ax.plot(X_line, samples[i])
        plt.show()

        plt.tight_layout()
        savefig(fig, 'GP/kernel-{}.pdf'.format(lengthscale))


#%% Kernel types

latexify(columns=3, fig_height=2.5)
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

kernels = {
    'RBF': GPy.kern.RBF(1),
    'Matern52': GPy.kern.Matern52(1),
    'Linear': GPy.kern.Linear(1)
}

with sns.dark_palette(color[0], reverse=True):

    for (name, kernel) in kernels.items():
        kernel.lengthscale = 0.1

        fig = plt.figure()

        # Prior
        ax = fig.add_subplot(211)

        N = 100
        M = 3
        X_line = np.linspace(-1,1, N)[:, None]

        covar = kernel.K(X_line, X_line)
        samples = np.random.multivariate_normal(np.zeros(N), covar, size=M)

        mean = np.zeros(X_line.shape[0])
        var = np.diagonal(covar)
        CI = 2 * np.sqrt(var)
        ax.plot(X_line, mean, linewidth=2, c='C0')
        ax.fill_between(X_line[:,0], -CI, CI, alpha=0.3, color='C0')

        for i in range(M):
            ax.plot(X_line, samples[i])

        # Plot posterior
        ax = fig.add_subplot(212)

        T = X_train.shape[0]

        covar = kernel.K(X_line, X_line)
        model = GPRegression(X_train, Y_train, kernel=kernel)
        model.Gaussian_noise.fix(0.001)
        mean, covar = model.predict(X_line, full_cov=True)
        samples = model.posterior_samples_f(X_line, size=T)

        var = np.diagonal(covar)
        CI = 2 * np.sqrt(var)
        ax.scatter(X_train, Y_train)
        ax.plot(X_line, mean, linewidth=2, c='C0')
        ax.fill_between(X_line[:,0], mean[:,0] - CI, mean[:,0] + CI, alpha=0.3, color='C0')

        for i in range(M):
            ax.plot(X_line, samples[:, 0, i])
        
        plt.tight_layout()
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
    #'Linear': GPy.kern.Linear(1),
}

N = 100
T = 3

def f(X):
    return np.sin(X)

for name, kernel in kernels.items():
    X_line = np.linspace(0,1, N)[:, None]

    # covar = kernel.K(X_line, X_line)

    model = GPRegression(X_train, Y_train, kernel=kernel)
    mean, covar = model.predict(X_line, full_cov=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.matshow(covar)
    plt.show()




#%%
