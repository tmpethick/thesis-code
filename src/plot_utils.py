import numpy as np
from matplotlib import pyplot as plt

from src.environments import BaseEnvironment
from src.models.models import BaseModel
import seaborn as sns

from src.utils import construct_2D_grid, call_function_on_grid


def plot_function(f: BaseEnvironment, func, title="Function", points=None):
    if f.input_dim == 1:
        X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:, None]
        Y_line = f(X_line)[:, 0]

        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.set_title('Ground truth f')
        ax.plot(X_line, Y_line)

        ax = fig.add_subplot(212)
        ax.set_title(title)
        ax.plot(X_line, func(X_line))
        if points is not None:
            sns.scatterplot(points[:, 0], np.zeros(points.shape[0]), ax=ax)

    elif f.input_dim == 2:
        XY, X, Y = construct_2D_grid(f.bounds)
        Z = call_function_on_grid(f, XY)[..., 0]
        Z_hat = call_function_on_grid(func, XY)[..., 0]

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_title('Ground truth f')
        ax.contourf(X, Y, Z, 50)

        ax = fig.add_subplot(122)
        ax.set_title(title)
        ax.contourf(X, Y, Z_hat, 50)
        if points is not None:
            sns.scatterplot(points[:, 0], points[:, 1], ax=ax)
    else:
        raise ValueError("Cannot plot in input dim above 2.")

    plt.tight_layout()
    
    return fig

def plot_model(model: BaseModel, f: BaseEnvironment):
    if f.bounds.shape[0] == 1:
        return plot1D(model, f)
    elif f.bounds.shape[0] == 2:
        return plot2D(model, f)
    else:
        return None


def plot1D(model: BaseModel, f: BaseEnvironment) -> plt.Figure:
    X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:, None]
    Y_line = f(X_line)[:, 0]

    mean, var = model.get_statistics(X_line, full_cov=False)

    # aggregate hyperparameters dimension
    if var.ndim == 3:
        mean = np.mean(mean, axis=0)
        var = np.mean(var, axis=0)

    fig, ax = plt.subplots()
    ax.scatter(model.X.reshape(-1), model.Y)
    ax.plot(X_line, Y_line)
    ax.plot(X_line, mean)
    ax.fill_between(X_line.reshape(-1),
                    (mean + 2 * np.sqrt(var)).reshape(-1),
                    (mean - 2 * np.sqrt(var)).reshape(-1), alpha=0.5)
    
    plt.tight_layout()
    
    return fig


def plot2D(model: BaseModel, f: BaseEnvironment) -> plt.Figure:
    XY, X, Y = construct_2D_grid(f.bounds)

    # remove grid
    original_grid_size = XY.shape[0]
    XY = XY.reshape((-1, 2))

    mean, var = model.get_statistics(XY, full_cov=False)
    ground_truth = f(XY)

    # aggregate hyperparameters dimension
    if var.ndim == 3:
        mean = np.mean(mean, axis=0)
        var = np.mean(var, axis=0)

    # recreate grid
    mean = mean.reshape((original_grid_size, original_grid_size))
    var = var.reshape((original_grid_size, original_grid_size))
    ground_truth = ground_truth.reshape((original_grid_size, original_grid_size))

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.set_title('Ground truth f')
    cont = ax.contourf(X, Y, ground_truth, 50)
    fig.colorbar(cont)
    ax.plot(model.X[:, 0], model.X[:, 1], '.', markersize=10)

    ax = fig.add_subplot(222)
    ax.set_title('Mean estimate m')
    cont = ax.contourf(X, Y, mean, 50)
    fig.colorbar(cont)
    # ax.plot(model.X[:, 0], model.X[:, 1], '.', markersize=10)

    ax = fig.add_subplot(223)
    ax.set_title('Model std')
    cont = ax.contourf(X, Y, np.sqrt(var), 50, vmin=0)
    fig.colorbar(cont)
    # ax.plot(model.X[:, 0], model.X[:, 1], '.', markersize=10)

    ax = fig.add_subplot(224)
    ax.set_title('Estimate Error |f-m|')
    conf = ax.contourf(X, Y, np.abs(mean - ground_truth), 50)
    fig.colorbar(cont)

    plt.tight_layout()

    return fig

