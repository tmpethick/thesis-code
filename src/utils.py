import math 
import numpy as np
import pandas as pd

# from src.environments import BaseEnvironment
# from src.models.models import BaseModel


def calc_errors(model, f, rand=False, rand_N=2500):
    est1 = lambda X_line: model.get_statistics(X_line, full_cov=False)[0]
    est2 = lambda X_line: f.noiseless(X_line)
    return _calc_errors(est1, est2, f, rand=rand, rand_N=rand_N)


def calc_errors_model_compare_mean(model1, model2, f, rand=False, rand_N=2500):
    est1 = lambda X_line: model1.get_statistics(X_line, full_cov=False)[0]
    est2 = lambda X_line: model2.get_statistics(X_line, full_cov=False)[0]
    return _calc_errors(est1, est2, f, rand=rand, rand_N=rand_N)


def calc_errors_model_compare_var(model1, model2, f, rand=False, rand_N=2500):
    est1 = lambda X_line: model1.get_statistics(X_line, full_cov=False)[1]
    est2 = lambda X_line: model2.get_statistics(X_line, full_cov=False)[1]
    return _calc_errors(est1, est2, f, rand=rand, rand_N=rand_N)


def _calc_errors(est1, est2, f, rand=False, rand_N=2500):
    rng = np.random.RandomState(42)
    if rand:
        N = rand_N
        X_line = random_hypercube_samples(N, f.bounds, rng=rng)
    elif f.input_dim == 1:
        N = 500
        X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], N)[:, None]
    elif f.input_dim == 2:
        N = 2500
        XY, X, Y = construct_2D_grid(f.bounds, N=N)
        X_line = XY.reshape((-1, 2))
    else:
        # TODO: put down grid instead.
        N = rand_N
        X_line = random_hypercube_samples(N, f.bounds, rng=rng)

    Y = est2(X_line)
    Y_hat = est1(X_line)

    # average over hyperparameters if they are there.
    if Y_hat.ndim == 3:
        Y_hat = np.mean(Y_hat, axis=0)

    # average over hyperparameters if they are there.
    if Y.ndim == 3:
        Y = np.mean(Y, axis=0)

    assert Y.ndim == 2, "est1 has to return two dimensional"
    assert Y_hat.ndim == 2, "est2 has to return two dimensional"

    Y_diff = Y - Y_hat
    rmse = np.sqrt(np.sum(np.square(Y_diff)) / N)
    max_err = np.max(np.fabs(Y_diff))

    return rmse, max_err


def errors(Y_pred_mean, Y_pred_var, Y_true_mean, training_mean):
    Y_diff = Y_pred_mean - Y_true_mean
    N = Y_pred_mean.size
    return dict(
        mae = np.average(np.fabs(Y_diff)),
        max_err = np.max(np.fabs(Y_diff)),
        rmse = np.sqrt(np.sum(np.square(Y_diff)) / N),
        mnlp = 1/N*np.sum((Y_diff) ** 2 / Y_pred_var + 1/2 * np.log(2*np.pi*Y_pred_var)),
        nmse = np.sum(np.square(Y_diff)) / np.sum(np.square(Y_pred_mean - training_mean))
    )


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


def construct_2D_grid(bounds, N=2500):
    n = int(math.sqrt(N))
    x_bounds = bounds[0]
    y_bounds = bounds[1]
    X = np.linspace(x_bounds[0], x_bounds[1], n)
    Y = np.linspace(y_bounds[0], y_bounds[1], n)
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


def average_at_locations(X, Y):
    """Takes all Ys associated with the same X location. Then:
    1) for each group takes their mean,
    2) and stores the mean at each of the locations.
    
    Arguments:
        X {[type]} -- [description]
        Y {[type]} -- [description]
    """
    if Y.shape[-1] != 1:
        raise ValueError("Y has to be 1-dimensional")
    Y = Y[:,0]
    I = [tuple(i) for i in X]

    df = pd.DataFrame({'X': I, 'Y': Y})
    df = df.groupby(['X']).agg({'Y': 'mean'})
    Y_mean = df['Y']

    Y_meaned = np.empty(len(Y))
    for i, x in enumerate(I):
        Y_meaned[i] = Y_mean[x]
    
    return Y_meaned[:, None]
