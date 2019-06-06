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
