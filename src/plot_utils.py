import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

from src.environments import BaseEnvironment
from src.models.models import BaseModel


def plot_function(f: BaseEnvironment, func, title="Function"):
    if f.input_dim == 1:
        X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:, None]
        Y_line = f(X_line)[:, 0]

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_title('Ground truth f')
        ax.plot(X_line, Y_line)

        ax = fig.add_subplot(122)
        ax.set_title(title)
        ax.plot(X_line, func(X_line))
    elif f.input_dim == 2:
        XY, X, Y = construct_2D_grid(f.bounds)
        Z = call_function_on_grid(f, XY)
        Z_hat = call_function_on_grid(func, XY)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_title('Ground truth f')
        ax.contour(X, Y, Z, 50)

        ax = fig.add_subplot(122)
        ax.set_title(title)
        ax.contour(X, Y, Z_hat, 50)
    else:
        raise ValueError("Cannot plot in input dim above 2.")

def plot1D(model, f):
    X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:, None]
    Y_line = f(X_line)[:, 0]

    mean, var = model.get_statistics(X_line, full_cov=False)

    plt.scatter(model.X.reshape(-1), model.Y)
    # TODO: DKL model was misbehaving throwing out negative variance
    #plt.fill_between(X_line.reshape(-1), (mean + 2 * np.sqrt(var)).reshape(-1), (mean - 2 * np.sqrt(var)))
    plt.fill_between(X_line.reshape(-1), (mean + var).reshape(-1), (mean - var).reshape(-1))
    plt.plot(X_line, mean)
    plt.plot(X_line, Y_line)


def plot2D(model: BaseModel, f: BaseEnvironment):
    XY, X, Y = construct_2D_grid(f.bounds)

    # remove grid
    original_grid_size = XY.shape[0]
    XY = XY.reshape((-1, 2))

    mean, var = model.get_statistics(XY, full_cov=False)
    ground_truth = f(XY)

    # recreate grid
    mean = mean.reshape((original_grid_size, original_grid_size))
    var = var.reshape((original_grid_size, original_grid_size))
    ground_truth = ground_truth.reshape((original_grid_size, original_grid_size))

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.set_title('Ground truth f')
    ax.contour(X, Y, ground_truth, 50)

    ax = fig.add_subplot(222)
    ax.set_title('Mean estimate m')
    ax.contour(X, Y, mean, 50)

    ax = fig.add_subplot(223)
    ax.set_title('Model Variance')
    ax.contour(X, Y, var, 50)

    ax = fig.add_subplot(224, projection='3d')
    ax.set_title('Estimate Error |f-m|')
    ax.contour3D(X, Y, np.abs(mean - ground_truth), 50, cmap='binary')


def construct_2D_grid(bounds):
    x_bounds = bounds[0]
    y_bounds = bounds[1]
    X = np.linspace(x_bounds[0], x_bounds[1], 50)
    Y = np.linspace(y_bounds[0], y_bounds[1], 50)
    X, Y = np.meshgrid(X, Y)
    XY = np.stack((X,Y), axis=-1)

    return XY, X, Y


def call_function_on_grid(func, XY):
    # remove grid
    original_grid_size = XY.shape[0]
    XY = XY.reshape((-1, 2))

    Z = func(XY)

    # recreate grid
    Z = Z.reshape((original_grid_size, original_grid_size))
    return Z