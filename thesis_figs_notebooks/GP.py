#%%
#%load_ext autoreload
#%autoreload 2

from src.models import RandomFourierFeaturesModel
from src.plot_utils import *
from src.kernels import *
from src.models.core_models import *
from src.models.dkl_gp import *
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
    'Linear': GPy.kern.Linear(1),
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

def matshow(K):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    m = ax.matshow(K)
    ax.axis('off')
    fig.colorbar(m)
    plt.tight_layout()
    return fig

latexify(columns=3)
kernel = GPy.kern.RBF(1, lengthscale=1)
X = np.random.normal(0, 1, size=1000)[:,None]
X = np.sort(X, axis=0)
K = kernel.K(X,X)
fig = matshow(K)
savefig(fig, 'DKL/kernel-rbf.pdf')

#%%
# RFF
kernel = RFFRBF(lengthscale=1, variance=1)
model = RandomFourierFeaturesModel(kernel, n_features=40)
K_ssgp = model.kernel(X, X)
fig, ax = plt.subplots()
plt.matshow(K_ssgp)
fig = matshow(K_ssgp)
savefig(fig, 'DKL/kernel-rff.pdf')
fig = matshow(np.abs(K - K_ssgp))
savefig(fig, 'DKL/kernel-rff-diff.pdf')


#%%
# SKI
kernel = gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.RBFKernel(), grid_size=40, num_dims=1)
X_torch = torch.Tensor(X)
K = gpytorch.kernels.RBFKernel()(X_torch, X_torch).numpy()
K_ski = kernel(X_torch, X_torch).numpy()
fig = matshow(K_ski)
savefig(fig, 'DKL/kernel-ski.pdf')
fig = matshow(np.abs(K - K_ski))
savefig(fig, 'DKL/kernel-ski-diff.pdf')


#%% Error

# SKI
tests = range(10, 25, 2)
error = np.empty(len(tests))

fig, ax = plt.subplots()

MARKER = cycle_markers()

for i, M in enumerate(tests):
    aver_err = np.empty(1)
    for j in range(len(aver_err)):
        kernel = gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.RBFKernel(), grid_size=M, num_dims=1)
        X_torch = torch.Tensor(X)
        K = gpytorch.kernels.RBFKernel()(X_torch, X_torch).numpy()
        K_ski = kernel(X_torch, X_torch).numpy()
        aver_err[j] = np.linalg.norm(K - K_ski) / K.size
    error[i] = np.mean(aver_err)
ax.plot(tests, error, label="SKI", marker=next(MARKER))

#RFF error
tests = range(10, 25, 2)
error = np.empty(len(tests))

kernel = GPy.kern.RBF(1, lengthscale=1)
K = kernel.K(X,X)

# kernel = RFFRBF(lengthscale=1, variance=1)
# model = RandomFourierFeaturesModel(kernel, n_features=100000)
# K = model.kernel(X, X)

for i, M in enumerate(tests):
    aver_err = np.empty(10)
    for j in range(len(aver_err)):
        kernel = RFFRBF(lengthscale=1, variance=1)
        model = RandomFourierFeaturesModel(kernel, noise=0, n_features=M)
        K_ssgp = model.kernel(X, X)
        aver_err[j] = np.linalg.norm(K - K_ssgp) / K.size
    error[i] = np.mean(aver_err)

ax.plot(tests, error, label="RFF", marker=next(MARKER))
plt.legend(loc='upper right')
plt.xlabel("m")
plt.ylabel("Error")
plt.tight_layout()
savefig(fig, "DKL/kernel-err-vs-m.pdf")

#%% Active subspace

latexify(columns=2)
def f(X):
    return np.exp(0.7 * X[..., 0] + 0.3 * X[...,1])

bounds = np.array([[-1,1], [-1,1]])
XY, X, Y = construct_2D_grid(bounds, N=1000)
Z = call_function_on_grid(f, XY)

fig, ax = plt.subplots()
m = ax.imshow(Z, extent=list(bounds.flatten()), origin='lower')
fig.colorbar(m)
ax.axis(aspect='image')
savefig(fig, "AS/AS-example.pdf")

#%%
