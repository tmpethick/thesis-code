import math
import os
import pathlib
import itertools

import matplotlib
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import src.experiment.settings as settings
from src.environments.core import BaseEnvironment
from src.models.core_models import BaseModel
from src.utils import call_function_on_grid, construct_2D_grid


MARKER = ('o', 's', '^', '*', '>', '<', '8', 'p')
CMAP = 'viridis'
COLORS = sns.color_palette('colorblind').as_hex()


def cycle_markers():
    return itertools.cycle(MARKER)


def savefig(fig, filepath, **kwargs):
    filepath = os.path.join(settings.THESIS_FIGS_DIR, filepath)
    dir_name = os.path.dirname(filepath)
    pathlib.Path(dir_name).mkdir(parents=True, exist_ok=True) 
    fig.savefig(filepath, transparent=False, **kwargs)


def latexify(fig_width=None, fig_height=None, columns=3):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2,3])

    # width in inches
    if fig_width is None:
        if columns==1:
            fig_width = 6.9
        elif columns==2:
            fig_width = 3.39 
        else:
            fig_width = 2.2

    if fig_height is None:
        golden_mean = (math.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height + 
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    fig_size = [fig_width,fig_height]

    params = {
            'backend': 'pgf',
            'pgf.preamble': [
                r'\usepackage[utf8x]{inputenc}',
                r'\usepackage[T1]{fontenc}',
                r'\usepackage{gensymb}',
            ],
            'axes.labelsize': 8, # fontsize for x and y labels (was 10)
            'axes.titlesize': 8,
            #'font.fontsize': 8, # was 10
            'legend.fontsize': 8, # was 10
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.usetex': True,
            'figure.figsize': fig_size,
            'font.family': 'serif',
        
            'legend.markerscale': .9,
            'legend.numpoints': 1,
            'legend.handlelength': 2,
            'legend.scatterpoints': 1,
            'legend.labelspacing': 0.5,
            'legend.facecolor': '#eff0f1',
            # 'legend.edgecolor': 'none',
            'legend.handletextpad': 0.5,  # pad between handle and text
            'legend.borderaxespad': 0.5,  # pad between legend and axes
            'legend.borderpad': 0.5,  # pad between legend and legend content
            'legend.columnspacing': 1,  # pad between each legend column
            'axes.spines.left': True,
            'axes.spines.top': True,
            'axes.titlesize': 'medium',
            'axes.spines.bottom': True,
            'axes.spines.right': True,
            'axes.axisbelow': True,
            'axes.grid': True,
            'grid.linewidth': 0.5,
            'grid.linestyle': '-',
            'grid.alpha': .6,
            'lines.linewidth': 1,
            'lines.markersize': 4,
            'lines.markeredgewidth': 1,

            # Force white to avoid transparent turning into black in latex.
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',

            # 'axes.prop_cycle': cycler('linestyle', LINES),

        # 'path.simplify': True,
        # 'path.simplify_threshold': 0.1,
        # 'pgf.preamble': [
        #     r'\usepackage[utf8x]{inputenc}',
        #     r'\usepackage[T1]{fontenc}',
        #     #rf'\usepackage{{{typeface}}}'
        # ],
        # Colors
    }

    matplotlib.rcParams.update(params)

    sns.set_style("whitegrid")
    sns.set_palette(COLORS)

    matplotlib.rcParams.update({'image.cmap': CMAP})


SPINE_COLOR = 'gray'

def format_axes(ax):
  
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def plot_function(f: BaseEnvironment, func, title="Function", points=None):
    if f.input_dim == 1:
        X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:, None]
        Y_line = f.noiseless(X_line)[:, 0]

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
        Z = call_function_on_grid(f.noiseless, XY)[..., 0]
        Z_hat = call_function_on_grid(func, XY)[..., 0]

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_title('Ground truth f')
        ax.contourf(X, Y, Z, 50)

        ax = fig.add_subplot(122)
        ax.set_title(title)
        ax.contourf(X, Y, Z_hat, 50)
        if points is not None:
            sns.scatterplot(points[:, 0], points[:, 1], size=2, ax=ax)
    else:
        raise ValueError("Cannot plot in input dim above 2.")

    plt.tight_layout()
    
    return fig

def plot_model_unknown_bounds(model: BaseModel):
    X_min = np.min(model.X)
    X_max = np.max(model.X)
    X_line = np.linspace(X_min, X_max, 100)[:, None]

    mean, var = model.get_statistics(X_line, full_cov=False)

    # aggregate hyperparameters dimension
    if var.ndim == 3:
        mean = np.mean(mean, axis=0)
        var = np.mean(var, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(model.X.reshape(-1), model.Y)
    ax.plot(X_line, mean)
    ax.fill_between(X_line.reshape(-1),
                    (mean + 2 * np.sqrt(var)).reshape(-1),
                    (mean - 2 * np.sqrt(var)).reshape(-1), alpha=0.5)    
    plt.tight_layout()
    
    return fig



def plot_model(model: BaseModel, f: BaseEnvironment):
    if f.bounds.shape[0] == 1:
        return plot1D(model, f)
    elif f.bounds.shape[0] == 2:
        return plot2D(model, f)
    else:
        return None


def plot1D(model: BaseModel, f: BaseEnvironment):  # -> plt.Figure:
    X_line = np.linspace(f.bounds[0, 0], f.bounds[0, 1], 100)[:, None]
    Y_line = f.noiseless(X_line)[:, 0]

    mean, var = model.get_statistics(X_line, full_cov=False)

    # aggregate hyperparameters dimension
    if var.ndim == 3:
        mean = np.mean(mean, axis=0)
        var = np.mean(var, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.scatter(model.X.reshape(-1), model.Y)
    ax.plot(X_line, Y_line)
    ax.plot(X_line, mean)
    ax.fill_between(X_line.reshape(-1),
                    (mean + 2 * np.sqrt(var)).reshape(-1),
                    (mean - 2 * np.sqrt(var)).reshape(-1), alpha=0.5)
    
    ax = fig.add_subplot(122)
    ax.set_title('Difference $|f-\hat{f}|$')
    ax.plot(X_line, np.fabs(Y_line-mean[:,0]))
    
    plt.tight_layout()
    
    return fig


def plot2D(model: BaseModel, f: BaseEnvironment): # -> plt.Figure:
    XY, X, Y = construct_2D_grid(f.bounds)

    # remove grid
    original_grid_size = XY.shape[0]
    XY = XY.reshape((-1, 2))

    mean, var = model.get_statistics(XY, full_cov=False)
    ground_truth = f.noiseless(XY)

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
    ax.set_title('Ground truth $f$')
    cont = ax.contourf(X, Y, ground_truth, 50)
    fig.colorbar(cont)
    ax.plot(model.X[:, 0], model.X[:, 1], '.', markersize=2)

    ax = fig.add_subplot(222)
    ax.set_title('Mean estimate $m$')
    cont = ax.contourf(X, Y, mean, 50)
    fig.colorbar(cont)
    # ax.plot(model.X[:, 0], model.X[:, 1], '.', markersize=10)

    ax = fig.add_subplot(223)
    ax.set_title('Model std')
    cont = ax.contourf(X, Y, np.sqrt(var), 50, vmin=0)
    fig.colorbar(cont)
    # ax.plot(model.X[:, 0], model.X[:, 1], '.', markersize=10)

    ax = fig.add_subplot(224)
    ax.set_title('Estimate Error $|f-m|$')
    conf = ax.contourf(X, Y, np.abs(mean - ground_truth), 50)
    fig.colorbar(cont)

    plt.tight_layout()

    return fig
