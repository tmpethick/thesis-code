#%%

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import ConstantKernel as C, Matern
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

from gp_extras.kernels import LocalLengthScalesKernel

np.random.seed(42)

import seaborn as sns
sns.set_style("darkgrid")

#%% ------------------- 1D ----------------------
n_samples = 50

def f(x):
   return 1 / (10 ** (-4) + x ** 2)

data_dim = 1
bounds = np.array([[-2,2]])
X = np.random.uniform(bounds[0,0], bounds[0,1], (n_samples, 1))
y = f(X)[:, 0]

#%% -------------------- 2D -----------------------
from src.algorithms import construct_2D_grid, call_function_on_grid
from src.algorithms import random_hypercube_samples

data_dim = 2
n_samples = 15 ** data_dim

def f(x):
   y = 1 / (np.abs(0.5 - x[...,0] ** 4 - x[...,1] ** 4) + 0.1)
   return y[...,None]

bounds = np.array([[0,1],[0,1]])

X = random_hypercube_samples(n_samples, bounds)
y = f(X)[:,0]


#%% -----------------------------------------------

# Define custom optimizer for hyperparameter-tuning of non-stationary kernel
def de_optimizer(obj_func, initial_theta, bounds):
    res = differential_evolution(lambda x: obj_func(x, eval_gradient=False),
                                 bounds, maxiter=20, disp=False, polish=False)
    return res.x, obj_func(res.x, eval_gradient=False)

# Specify stationary and non-stationary kernel
kernel_matern = C(1.0, (1e-10, 1000)) \
    * Matern(length_scale_bounds=(1e-1, 1e3), nu=1.5)
gp_matern = GaussianProcessRegressor(kernel=kernel_matern)

kernel_lls = C(1.0, (1e-10, 1000)) \
  * LocalLengthScalesKernel.construct(X, l_L=0.1, l_U=2.0, l_samples=5)
gp_lls = GaussianProcessRegressor(kernel=kernel_lls, optimizer=de_optimizer)

# Fit GPs
gp_matern.fit(X, y)
gp_lls.fit(X, y)

print("Learned kernel Matern: %s" % gp_matern.kernel_)
print("Log-marginal-likelihood Matern: %s" \
    % gp_matern.log_marginal_likelihood(gp_matern.kernel_.theta))


print("Learned kernel LLS: %s" % gp_lls.kernel_)
print("Log-marginal-likelihood LLS: %s" \
    % gp_lls.log_marginal_likelihood(gp_lls.kernel_.theta))

#%% -------------- Test 1D -----------------
# Compute GP mean and standard deviation on test data
X_ = np.linspace(-1, 1, 500)

y_mean_lls, y_std_lls = gp_lls.predict(X_[:, np.newaxis], return_std=True)
y_mean_matern, y_std_matern = \
    gp_matern.predict(X_[:, np.newaxis], return_std=True)

plt.figure(figsize=(7, 7))
plt.subplot(2, 1, 1)
plt.plot(X_, f(X_), c='k', label="true function")
plt.scatter(X[:, 0], y, color='k', label="samples")
plt.plot(X_, y_mean_lls, c='r', label="GP LLS")
plt.fill_between(X_, y_mean_lls - y_std_lls, y_mean_lls + y_std_lls,
                 alpha=0.5, color='r')
plt.plot(X_, y_mean_matern, c='b', label="GP Matern")
plt.fill_between(X_, y_mean_matern - y_std_matern, y_mean_matern + y_std_matern,
                 alpha=0.5, color='b')
plt.legend(loc="best")
plt.title("Comparison of learned models")
plt.xlim(-1, 1)

plt.subplot(2, 1, 2)
plt.plot(X_, gp_lls.kernel_.k2.theta_gp
             * 10**gp_lls.kernel_.k2.gp_l.predict(X_[:, np.newaxis]),
         c='r', label="GP LLS")
plt.plot(X_, np.ones_like(X_) * gp_matern.kernel_.k2.length_scale,
         c='b', label="GP Matern")
plt.xlim(-1, 1)
plt.ylabel("Length-scale")
plt.legend(loc="best")
plt.title("Comparison of length scales")
plt.tight_layout()
plt.show()


#%% -------------- Test 2D -----------------

XY, X, Y = construct_2D_grid(bounds)

def predict(X):
    y_mean_lls, y_std_lls = gp_lls.predict(X, return_std=True)
    return y_mean_lls, y_std_lls

Z = call_function_on_grid(f, XY)
Z_pred_mean = call_function_on_grid(lambda XY: predict(XY)[0], XY)
Z_pred_var = call_function_on_grid(lambda XY: predict(XY)[1], XY)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.contour(X, Y, Z, 50)
ax = fig.add_subplot(222)
ax.contour(X, Y, Z_pred_mean, 50)
ax = fig.add_subplot(223)
ax.contour(X, Y, Z_pred_var, 50)
ax = fig.add_subplot(224, projection='3d')
ax.contour3D(X,Y,np.abs(Z_pred_mean-Z), 50, cmap='binary')


#%%

XY, X, Y = construct_2D_grid(bounds)

def predict(X):
    y_mean_matern, y_std_matern = \
    gp_matern.predict(X, return_std=True)
    return y_mean_matern, y_std_matern

Z = call_function_on_grid(f, XY)
Z_pred_mean = call_function_on_grid(lambda XY: predict(XY)[0], XY)
Z_pred_var = call_function_on_grid(lambda XY: predict(XY)[1], XY)

fig = plt.figure()
ax = fig.add_subplot(221)
ax.contour(X, Y, Z, 50)
ax = fig.add_subplot(222)
ax.contour(X, Y, Z_pred_mean, 50)
ax = fig.add_subplot(223)
ax.contour(X, Y, Z_pred_var, 50)
ax = fig.add_subplot(224, projection='3d')
ax.contour3D(X,Y,np.abs(Z_pred_mean-Z), 50, cmap='binary')
