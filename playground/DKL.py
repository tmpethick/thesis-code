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

#######################################################
# DKL
#######################################################



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

#######################################################
# DKL - debug like any other NN
# see: https://cs231n.github.io/neural-networks-3/
#######################################################

# Observations (issues)
#######################



# Fit is identical if initial values are the correct ones.
# But if we are optimizing hyperparameters we cannot expect this.
# Initiallization is important => Is mml convex? Or do we get trapped in local minimum? (otherwise is it an analytical solution to a non-convex problem?)
    # Will same initial lead to same opt?
        # no: model.initialize(**{
            #     'likelihood.noise': 0.3701965415310207,
            #     'covar_module.base_kernel.lengthscale': 2.5120422203935733,
            #     'covar_module.outputscale': 1.3978480431756182
            # }) => mean prediction!! (this is the bad behaviour we have observed!)
        # How can we make sure we do well even with weird initial configuration? (like above).
        # Same initilization as for ExactGP does not work for DKL. Why can the optimization more easily get stuck in local minimum/saddlepoints?
# Further: Some configuration with low noise will break (so initialize correctly).

# Fix GP noise (at high so it does break DKL) and try to recreate.
    # ExactGP is consistent even with random initialization.
    # 

# DKLGP Identity
    # Unstable oscillating variance on IncrOsc (seems to only be a problem for IncrOsc and not Sin.)
        # First: DKL noise is var and GP is std (DKL is square root smaller). So how can we get smaller error?
        # Posterior variance is unstable. How to fix?
            # Even breaks with blow up norm "average residual norm 8816.3291015625 which is larger than the tolerance of 0.01" when N=500, noise=0.0001
        # Add noise for stability: https://github.com/cornellius-gp/gpytorch/issues/703
        # Float vs. doubles
    # Fixed noise? (potential problem with SKI)

    # Learning rate find correct noise level. but noise level is low. Which causes jitter in posterior variance. (when many observations!)
    # Fixed noise courses quick drop in loss (loss starts out much higher...)

# DKLGP
    # Without normalization: error blows up (assuming feature space is too big)
        # (Bug: Crash when n_iter=1000) wait
    # Even with normalization: suddent peak in loss during training (not very stable...)
    # Suddent drop in loss in 20 first step (afterwards almost no progress.) (can we recreate bad performance on feature mapping for IncreOsc?)


# Investigate
#############
# DKLLGP Identity
    # On easy/smooth functions
# DKL
    # Persistant results for IncOsc and Step.


#%%

# Convergence time for each function (with cap)

# sensitivity analysis of the learning rate U shape.
# https://www.deeplearningbook.org/contents/guidelines.html (434)

#%% 

# Stable GP
    # loss over time? (variance of solution)
    # Bassin for learning rate (how does learning rate influence)
        # figure out

# Sometimes didn't learn correct feature mapping for Oscillation..
    # Modification of DKL Feature Normalization (footnote: in practise use batch norm with momentum. even if computational more expensive it doesn't show in practise.)

# How can we consistently learn this mapping for step and increasing oscillation?
    # How do we check if converged?
    # How quickly do each converge?
    # How much does it depend on random initialization. (do we simply compare loss?)
    # How much does it effect the last convergence?
# When constantly learning one mapping: what learning rate works across problems.
    # plot loss
# Normalize feature space...

#%%

config = {
    'obj_func': {'name': 'IncreasingOscillation'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': None,
            'use_double_precision': True,
            'use_cg': False,
        },
    },
    'gp_samples': 1000,
}
normalize_config(config)
run = execute(config_updates=config)


#%%

#%%

model = run.interactive_stash['model']
fig = plt.plot(model.model.training_loss)


#%%

#######################################################
# DKL - Non-stationarity
#######################################################

#%% Non-stationary in low dim

# Compare DKL with GP for low dim non-stationary (see A-SG further down)

config = {
    'obj_func': {'name': 'Kink1D'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 1000,
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

model = {
        'name': 'LinearFromFeatureExtractor',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 1000,
            'layers': [100, 50, 1],
            'data_dim': 1,
            }
        }
config = {
    'obj_func': {'name': 'Step', 'kwargs': {'noise': 0.01}},
    'model': {
        'name': 'NormalizerModel',
        'kwargs': {
            'model': model,
            'normalize_input': True,
            'normalize_output': True,
        }
    },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)


#%%

model = run.interactive_stash['model']
f = run.interactive_stash['f']
normalized_f = EnvironmentNormalizer(f, model.X_normalizer, model.Y_normalizer)
model.model.plot_features(normalized_f)

#%% 1D Step in 2D feature space
# Reproducing DKL (and Manifold GP)

config = {
    'obj_func': {'name': 'Step', 'kwargs': {'noise': 0.01}},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.01,
            'n_iter': 1000,
            'do_pretrain': True,
            'pretrain_n_iter': 1000,
            'nn_kwargs': {'layers': [100, 50, 1]},
            # 'gp_kwargs': {'n_grid': 1000},
            'noise': None, #0.01,
            }
        },
    'gp_samples': 1000,
}
normalize_config(config)
run = execute(config_updates=config)

# Need more stable model: Gets stuck even for SingleStep with NN
    # Try initialization

# We want to ensure that it does not just use "Linear" kernel. Otherwise our expressively (computationally costly) is not needed.
    # Not only stretching of the input (it could move it to a completely different part of the domain)
    # So not only able to learn monotonically increase or decreasing function with linear kernel.
    # It can map it to a completely different part of the domain... 



#%%
model = run.interactive_stash['model']
f = run.interactive_stash['f']
normalized_f = EnvironmentNormalizer(f, model.X_normalizer, model.Y_normalizer)
model.model.plot_features(normalized_f)


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

#######################################################
# DKL - Embeddings
#######################################################


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

#######################################################
# DKL - Scalability
#######################################################



#%% 
# Genz1984 2D for active sampling

# Motivation:
# Will lengthscale and DKL help us with max_err for kinks?
# See if meaningfull representations
# Then look at error.

# Varying Lengthscale suggests we should sample more around kinks.
# Lets do that and see if it improves Active Sampling.


#%%

config = {
    'obj_func': {'name': 'Kink1DShifted'},
    'model': {
        'name': 'GPModel',
        'kwargs': {
            'kernel': {
                'name': 'GPyRBF',
                'kwargs': {
                    'lengthscale': 1
                }
            },
            'mean_prior': True,
            'noise_prior': 100,
            'do_optimize': True,
            'num_mcmc': 0,
        },
    },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)
print(run.result)


#%% DKL

# DKL does not learn meaningful representation...

config = {
    'obj_func': {'name': 'TwoKink1D'},
    'model': {
        'name': 'DKLGPModel',
        'kwargs': {
            'learning_rate': 0.1,
            'n_iter': 100,
            'nn_kwargs': {'layers': [100, 50, 1]},
            'noise': 0.001}
        },
    'gp_samples': 1000,
 }
run = execute(config_updates=config)
print(run.result)

#%%
# - LLS lengthscale on Kink2D and Kink1D (how does the lengthscale look?)

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'TwoKink1D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 1000,
            'n_optimizer_iter': 5,
        }
    },
    'gp_samples': 1000,
})


#%%

run = notebook_run(config_updates={
    'obj_func': {
        'name': 'Kink1D',
    },
    'model': {
        'name': 'LocalLengthScaleGPModel',
        'kwargs': {
            'l_samples': 200,
            'n_optimizer_iter': 5,
        }
    },
    'gp_samples': 1000,
})

#%%
