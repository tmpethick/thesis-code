import numpy as np

# from src.environments import BaseEnvironment
from src.models.models import BaseModel


def mean_square_error(model: BaseModel, f):
    if f.input_dim == 1:
        X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 500)[:, None]
    elif f.input_dim == 2:
        XY, X, Y = construct_2D_grid(f.bounds)
        X_line = XY.reshape((-1, 2))
    else:
        # TODO: put down grid instead.
        X_line = random_hypercube_samples(100000, f.bounds)
        #raise ValueError("Does not support dim above 2.")

    Y = f(X_line)
    Y_hat, var = model.get_statistics(X_line, full_cov=False)

    # average over hyperparameters if there.
    if Y_hat.ndim == 3:
        Y_hat = np.mean(Y_hat, axis=0)

    mse = np.sqrt(np.sum(np.square(Y - Y_hat)))

    return mse


def random_hypercube_samples(n_samples, bounds, rng=None):
    """Random sample from d-dimensional hypercube (d = bounds.shape[0]).

    Returns: (n_samples, dim)
    """
    if rng is None:
        rng = np.random.RandomState()

    dims = bounds.shape[0]
    a = rng.uniform(0, 1, (dims, n_samples))
    bounds_repeated = np.repeat(bounds[:, :, None], n_samples, axis=2)
    samples = a * np.abs(bounds_repeated[:,1] - bounds_repeated[:,0]) + bounds_repeated[:,0]
    samples = np.swapaxes(samples, 0, 1)

    # This handles the case where the sample is slightly above or below the bounds
    # due to floating point precision (leading to slightly more samples from the boundary...).
    return constrain_points(samples, bounds)


def constrain_points(x, bounds):
    dim = x.shape[0]
    minx = np.repeat(bounds[:, 0][None, :], dim, axis=0)
    maxx = np.repeat(bounds[:, 1][None, :], dim, axis=0)
    return np.clip(x, a_min=minx, a_max=maxx)


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
    Z = Z.reshape((original_grid_size, original_grid_size) + Z.shape[1:])
    return Z
