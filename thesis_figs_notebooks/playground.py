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


f = Kink2D()
X_test = random_hypercube_samples(1000, f.bounds)
# Uses uniform since bounds are [0,1] to ensure implementation is not broken...
X_test = np.random.uniform(size=(1000,2))
N_test = X_test.shape[-1]
Y_test = f(X_test)

def calc_error(i, model):
    max_error, L2_err = model.calc_error(X_test, Y_test)
    print("{0:9d} {1:9d}  Loo={2:1.2e}  L2={3:1.2e}".format(i+1, model.grid.getNumPoints(), max_error, L2_err))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# A-SG 
########################################################################

# I should be able to get good performance with non-adaptive SG with 3e5 points but can't.

# (bad estimate around kink.)
asg = AdaptiveSparseGrid(f, depth=15, refinement_level=0)
asg.fit(callback=calc_error)
fig = asg.plot()


#%% A-SG Extremely sensitive to f_tol (0.0099 works, >=0.01 breaks)

# (samples better around kink however)
asg = AdaptiveSparseGrid(f, depth=1, refinement_level=30, f_tol=0.01)
asg.fit(callback=calc_error)
fig = asg.plot()

#%%

f2 = Kink2D()

for i in [0, -0.5, -0.8]:
    f2.bounds = np.array([[-1,1], [i,1]])
    asg = AdaptiveSparseGrid(f2, depth=1, refinement_level=10, f_tol=1e-2)
    asg.fit(callback=calc_error)
    fig = asg.plot()
    plt.show()


#%% L2 and Loo as function of #points

refinement_levels = 30
f_tol = 0.0099

def test_depth_to_error(ASG_creator, max_points=4e5):
    N = np.empty(refinement_levels)
    Loo_err = np.empty(refinement_levels)
    L2_err = np.empty(refinement_levels)

    for i in range(refinement_levels):
        ASG = ASG_creator(i)
        ASG.fit()

        N[i] = ASG.grid.getNumPoints()
        Loo_err[i], L2_err[i] = ASG.calc_error(X_test, Y_test)
        
        if N[i] > max_points:
            break

    return N[:i], Loo_err[:i], L2_err[:i]

N, Loo_err, L2_err = test_depth_to_error(lambda i: AdaptiveSparseGrid(f, depth=i, refinement_level=0, f_tol=f_tol))
plt.plot(N, Loo_err, label="$L_\infty$ error - SG", marker='*', c='black')
plt.plot(N, L2_err, label="$L_2$ error - SG", marker=11, c='black')

N, Loo_err, L2_err = test_depth_to_error(lambda i: AdaptiveSparseGrid(f, depth=1, refinement_level=i, f_tol=f_tol))
plt.plot(N, Loo_err, label="$L_\infty$ error - ASG", marker='*', dashes=[2,2], c='black')
plt.plot(N, L2_err, label="$L_2$ error - ASG", marker=11,  dashes=[2,2], c='black')

plt.xlabel('\#Points N')
plt.ylabel('Error')
plt.yscale('log')
plt.xscale('log')
plt.legend()
savefig(fig, 'ASG/depth_to_error.pgf')

#


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# DKL
########################################################################

# We learn the correct feature mapping

config = {
    'obj_func': {'name': 'Kink2D'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 200,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': 0.01
        },
    },
    'gp_samples': 1000,
}
run = execute(config_updates=config)

#%%
# Problematic in higher dim
# Why does it find 1D subspace?

KinkDCircularEmbedding(D=1).plot()
KinkDCircularEmbedding(D=2).plot()

f = KinkDCircularEmbedding(D=5)
X = random_hypercube_samples(100000, f.bounds)
G = f.derivative(X)
model = ActiveSubspace(threshold_factor=4)
model.fit(X, f(X), G)
print(model.W.shape[-1])
model.plot()
Z = model.transform(X)
plt.show()
plt.ylabel("$Y$")
plt.xlabel("$\Phi(X)$")
plt.scatter(Z, f(X))

#%% Mean/zero estimator as function of dimensions.

M = 20
Y_est_RMSE = np.zeros(M)
zero_RMSE = np.zeros(M)

for D in range(0, M):
    f = KinkDCircularEmbedding(D=D+1)
    X = random_hypercube_samples(1000, f.bounds)
    Y = f(X)
    Y_est = np.mean(Y, axis=0)
    Y_est_RMSE[D] = np.sum((Y_est - Y) ** 2) / Y.shape[0]
    zero_RMSE[D] = np.sum((Y) ** 2) / Y.shape[0]

plt.plot(Y_est_RMSE, label="mean estimate")
plt.plot(zero_RMSE, label="zero estimate")
plt.yscale('log')
plt.legend()


#%% DKL in high dim
# Learns meaningful mapping in D=5 but breaks with D=7.

run = execute(config_updates={
    'obj_func': {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 5}},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 200, 
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': None, #1e-2,
        },
    },
    'gp_use_derivatives': False,
    'gp_samples': 1000,
})

#%% More samples (incomplete)

run = execute(config_updates={
    'obj_func': {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 7}},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 200,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'gp_kwargs': {'n_grid': 5000},
            'noise': None, #1e-2,
        },
    },
    'gp_use_derivatives': False,
    'gp_samples': 10000,
})

#%% Sample from manifold 

run = execute(config_updates={
    'obj_func': {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 7}},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 200,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'gp_kwargs': {'n_grid': 5000},
            'noise': None, #1e-2,
        },
    },
    'gp_use_derivatives': False,
    'gp_samples': 10000,
})

#%% Confident we can learn feature mapping => look into many test points in low dimensions

functions = [
    {'name': 'KinkDCircularEmbedding', 'kwargs': {'D': 1}},
]

run = execute(config_updates={
    'tag': 'embedding',
    'obj_func': functions[0],
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 200,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'gp_kwargs': {'n_grid': 1000},
            'noise': None, #1e-2,
        },
    },
    'gp_use_derivatives': False,
    'gp_samples': 10000,
})

# Add noise to functions
# Normalize (instead of prior)
# Raise error if gpytorch warning (if log has warning?)
# sample from manifold
# Implement Genz function


# https://arxiv.org/pdf/1601.02557.pdf (Bayesian subset simulation)


#%%

# # Maybe not so simple... sampling from manifold

# import scipy

# D = 2
# size = 100

# x = np.empty((size, D))
# z = np.random.uniform(0, 1, size)

# # Put down on line
# x[:,0] = z

# # Random rotation in D dim
# theta = np.random.uniform(0, 1, D)

# # Form rotation matrix
# R = scipy.stats.special_ortho_group.rvs(D)
# X = np.tensordot(R, x, axes=0)

# plt.scatter(X[:,0], X[:,1])
