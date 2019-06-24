import numpy as np

from src.experiment.config_helpers import ConfigMixin
from src.utils import construct_2D_grid, call_function_on_grid


class BaseEnvironment(ConfigMixin):
    x_opt = None
    is_expensive = False

    def __init__(self, noise=None):
        self.noise = noise

    def __repr__(self):
        return "{}".format(type(self).__name__)

    def _call(self, x):
        raise NotImplementedError

    def noiseless(self, x):
        return self._call(x)

    def __call__(self, x):
        if self.noise is not None:
            noise = np.random.normal(0, self.noise, size=x.shape[0])[:, None]
            y = self._call(x) + noise
        else:
            y = self._call(x)

        return y

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def input_dim(self):
        return self.bounds.shape[0]

    @property
    def f_opt(self):
        if self.x_opt is None:
            raise Exception("Function does not have an optimum defined. Please specify `x_opt`.")
        return self(np.array([self.x_opt]))[0]

    def derivative(self, x):
        raise NotImplementedError

    def hessian(self, x):
        raise NotImplementedError

    def plot(self, projection=None, title="$f$"):
        return self._plot(self.noiseless, projection=projection, title=title)

    def plot_derivative(self, projection=None, title="$\\nabla f$"):
        return self._plot(self.derivative, projection=projection, title=title)

    def plot_curvature(self, norm="fro", projection=None, title="$\\nabla^2 f$"):
        if self.bounds.shape[0] != 1:
            def hess(x):
                H = self.hessian(x)
                return np.linalg.norm(H, ord=norm, axis=(-2, -1))[...,None]
        else:
            hess = self.hessian
        return self._plot(hess, projection=projection, title=title)

    def _plot(self, func, projection=None, title=None):
        assert self.bounds.shape[0] in [1,2], "Only support 1D/2D plots."

        if self.bounds.shape[0] == 1:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            X = np.linspace(self.bounds[0,0], self.bounds[0,1], 1000)[:, None]
            if title is not None:
                ax.set_title(title)
            Y = func(X)
            ax.plot(X, Y)

            # Show the 2*std for the noice
            # if self.noise is not None:
            #     plt.fill_between(X, Y - 2 * self.noise, Y + 2 * self.noise, alpha=0.2)

            return fig

        elif self.bounds.shape[0] == 2:
            import matplotlib.pyplot as plt
            XY, X, Y = construct_2D_grid(self.bounds)
            Z = call_function_on_grid(func, XY)[...,0]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection=projection)
            ax.contourf(X, Y, Z, 50)
            return fig


__all__ = ['BaseEnvironment']
