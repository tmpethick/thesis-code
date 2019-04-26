#%%
%load_ext autoreload
%autoreload 2

import numpy as np
from src.environments import NegSinc, Kink1D
from src.models import GPModel
from src.kernels import GPyRBF

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

#%% PDF of max of gaussians
import scipy
import numpy as np


mu_1 = 1
sigma_1 = 1
mu_2 = 0
sigma_2 = 1
corr = 0

mu = mu_1 - mu_2
sigma = np.sqrt ( sigma_1 ** 2 + sigma_2 ** 2 - 2 * corr * sigma_1 * sigma_2 )

# x1 > x2
1 - scipy.stats.norm(mu, sigma).cdf(0)

#%% Testing out Frank-Wolfe

f = NegSinc()
kernel = GPyRBF(1)
model = GPModel(kernel=kernel, noise_prior=None, do_optimize=True, num_mcmc=0, normalize_input=False, normalize_output=False)

N_INIT = 2
X = np.random.uniform(f.bounds[0,0], f.bounds[0,1], N_INIT)[:, None]
Y = f(X)
model.init(X, Y)

M = 100

p_t = np.ones(M) / M
X_line = np.linspace(f.bounds[0,0], f.bounds[0,1], M)[:, None]


T = 20
for t in range(1, T + 1):
    #St = t ** 2
    St = 1

    sample_path = model.gpy_model.posterior_samples_f(X_line, size=St)
    sample_path = np.mean(sample_path, axis=-1)

    gamma = 2 / (t + 1)

    # p_idx = np.argmin(sample_path)
    p_idx = np.random.choice(np.flatnonzero(sample_path == sample_path.min()))

    w = np.zeros(M)
    w[p_idx] = 1
    
    p_t = p_t + gamma * (w - p_t)

    X_new = X_line[p_idx][:, None]
    Y_new = f(X_new)

    model.add_observations(X_new, Y_new)

    # plot
    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(221)
    ax.set_title("GP")
    model.plot(X_line, ax=ax)

    ax = fig.add_subplot(222)
    ax.set_title("Regret")
    y_opt = np.minimum.accumulate(model.Y[N_INIT:])
    simple_regret = np.abs(f.f_opt - y_opt)
    ax.set_yscale('log')
    ax.plot(np.arange(simple_regret.size), simple_regret)

    ax = fig.add_subplot(223)
    ax.set_title("$p_t$")
    ax.bar(X_line[:,0], p_t, 0.4)

    ax = fig.add_subplot(224)
    ax.set_title("Sample history")
    ax.scatter(np.arange(len(model.X)), model.X)
    ax.plot(np.arange(len(model.X)), model.X)
    
    plt.tight_layout()
    plt.show()


#%% Model setup

f = Sinc()
kernel = GPyRBF(1)
model = GPModel(kernel=kernel, noise_prior=None, do_optimize=True, num_mcmc=0, normalize_input=False, normalize_output=False)

X = np.random.uniform(f.bounds[0,0], f.bounds[0,1], 2)[:, None]
Y = f(X)
model.init(X, Y)

# update p_t instead and have it converge to delta function.
# by using (grid) samples of GP. 
# When do we add new points from true f?

#%%

# Discrete (grid) case
T = 20
use_UCB = False
use_mean = False
for t in range(T):
    # Sample estimate
    GRID_SIZE = 100
    X_line = np.linspace(f.bounds[0,0], f.bounds[0,1], GRID_SIZE)[:,None]
    if use_UCB:
        mean, var = samples_t = model.gpy_model.predict(X_line) # N x D
        mean, var = mean[:,0], var[:,0]
        samples_t = mean + np.sqrt(var)
    elif use_mean:
        mean, var = samples_t = model.gpy_model.predict(X_line) # N x D
        mean, var = mean[:,0], var[:,0]
        samples_t = mean
    else:
        samples_t = model.gpy_model.posterior_samples_f(X_line, size=1) # N x D x samples paths
        samples_t = samples_t[:,0,0]

    # Construct GRID_SIZE dim p
    alpha_t = 1 / (np.sqrt(t + 1))
    p_t = np.exp(samples_t  / alpha_t) / np.sum(np.exp(samples_t / alpha_t))

    # Pick sample (from p?)
    p_idx = np.random.choice(len(p_t), 1, p=p_t)[0]
    x_new = X_line[p_idx]
    X_new = x_new[:, None]

    # Update GP with sample
    Y_new = f(X_new)
    model.add_observations(X_new, Y_new)
    
    # plot
    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(221)
    model.plot(X_line, ax=ax)

    # regret plot
    ax = fig.add_subplot(222)
    y_opt = np.maximum.accumulate(model.Y)
    simple_regret = np.abs(f.f_opt - y_opt)
    ax.set_yscale('log')
    ax.plot(np.arange(simple_regret.size), simple_regret)

    ax = fig.add_subplot(223)
    plt.bar(X_line[:,0], p_t, 0.4)

    ax = fig.add_subplot(224)
    ax.scatter(np.arange(len(model.X)), model.X)
    ax.plot(np.arange(len(model.X)), model.X)

    plt.show()

#%%


#%% Thompson sampling

f = Sinc()
kernel = GPyRBF(1)
model = GPModel(kernel=kernel, noise_prior=None, do_optimize=True, num_mcmc=0, normalize_input=True, normalize_output=True)

X = np.random.uniform(f.bounds[0,0], f.bounds[0,1], 2)[:, None]
Y = f(X)
model.init(X, Y)

# Discrete (grid) case
T = 20
for t in range(T):
    # Sample estimate
    GRID_SIZE = 100
    X_line = np.linspace(f.bounds[0,0], f.bounds[0,1], GRID_SIZE)[:,None]
    samples_t = model.gpy_model.posterior_samples_f(X_line, size=1) # N x D x samples paths
    samples_t = samples_t[:,0,0]

    max_idx = np.argmax(samples_t)
    x_new = X_line[max_idx]
    X_new = x_new[:, None]

    # Update GP with sample
    Y_new = f(X_new)
    model.add_observations(X_new, Y_new)
    
    # plot
    fig = plt.figure(figsize=(12,4))
    ax = fig.add_subplot(221)
    model.plot(X_line, ax=ax)

    # regret plot
    ax = fig.add_subplot(222)
    y_opt = np.maximum.accumulate(model.Y)
    simple_regret = np.abs(f.f_opt - y_opt)
    ax.set_yscale('log')
    ax.plot(np.arange(simple_regret.size), simple_regret)

    ax = fig.add_subplot(223)
    ax.scatter(np.arange(len(model.X)), model.X)
    ax.plot(np.arange(len(model.X)), model.X)
    plt.show()


#%% ----------------------- Continuous -----------------------



#%% Contiunous case with SGLD

def SGLD():
    # Given theta, prior p(theta), data X of size N
    # sample from p(theta | X)
    # by updating chain: 
    # delta theta_t = epsilon_t / 2 * (gradient log p(theta_t) + N/n * sum gradient log p(x | theta_t)) + eta
    # eta is ~N(0, epsilon_t).
    pass

f = Sinc()
kernel = GPyRBF(1)
model = GPModel(kernel=kernel, noise_prior=None, do_optimize=True, num_mcmc=0, normalize_input=True, normalize_output=True)

X = np.random.uniform(f.bounds[0,0], f.bounds[0,1], 2)[:, None]
Y = f(X)
model.init(X, Y)

T = 20
for t in range(T):
    a = 100
    alpha_t = 1 / (np.sqrt(a * (t + 1)))
    # TODO: instead of computing p_t vector we sample from it directly using Langevin dynamics.

    # SGLD
    L = 1000
    z_t = np.random.uniform(f.bounds[0,0], f.bounds[0,1], 1)[:, None]
    # z_t = np.array([[-15]]) # fixed start
    Z_t = z_t
    for l in range(L):
        # Use only the gradient of the mean for now
        # N*, Q ,D
        # z_grad, _ = model.gpy_model.predictive_gradients(z_t)
        # z_grad = z_grad[:,0,:]
        
        # (N*, Q ,D), (N*, Q, D, D)
        z_grad_mean, z_grad_covar = model.gpy_model.predict_jacobian(z_t, full_cov=True)
        
        # Remove #samples and output axes.
        z_grad_mean = z_grad_mean[0,0,:]
        z_grad_covar = z_grad_covar[0,0,:,:]

        # Sample a gradient
        z_grad = np.random.multivariate_normal(z_grad_mean, z_grad_covar, size=1)

        # LD step
        epsilon_t = alpha_t #1
        eta_t = np.random.normal(0, epsilon_t, 1)
        z_step_t = epsilon_t / 2 * (z_grad) + eta_t
        z_t = z_t + z_step_t
        
        # Projection step
        z_t = np.clip(z_t, f.bounds[0,0], f.bounds[0,1])
        
        Z_t = np.concatenate([Z_t, z_t])

    X_new = z_t

    # Update GP with sample
    Y_new = f(X_new)
    model.add_observations(X_new, Y_new)

    # Sample estimate
    GRID_SIZE = 100
    X_line = np.linspace(f.bounds[0,0], f.bounds[0,1], GRID_SIZE)[:,None]

    # plot
    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(221)
    ax.set_title("True f")
    model.plot(X_line, ax=ax)

    ax = fig.add_subplot(222)
    ax.set_title("Regret")
    y_opt = np.maximum.accumulate(model.Y)
    simple_regret = np.abs(f.f_opt - y_opt)
    ax.set_yscale('log')
    ax.plot(np.arange(simple_regret.size), simple_regret)

    ax = fig.add_subplot(223)
    ax.set_title("$p_t$")
    plt.hist(Z_t, bins=40, density=1)

    ax = fig.add_subplot(224)
    ax.set_title("Sample history")
    ax.scatter(np.arange(len(model.X)), model.X)
    ax.plot(np.arange(len(model.X)), model.X)
    
    plt.tight_layout()
    plt.show()
