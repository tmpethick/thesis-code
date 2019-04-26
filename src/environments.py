import numpy as np
from numpy.core.numeric import where

from src.utils import construct_2D_grid, call_function_on_grid


class BaseEnvironment(object):
    x_opt = None

    def __repr__(self):
        return "{}".format(type(self).__name__)

    def __call__(self, x):
        raise NotImplementedError

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

    def plot(self):
        return self._plot(self)

    def plot_derivative(self):
        return self._plot(self.derivative)

    def plot_curvature(self):
        assert self.bounds.shape[0] == 1, "curvature only supported in 1D"
        return self._plot(self.hessian)

    def _plot(self, func):
        assert self.bounds.shape[0] in [1,2], "Only support 1D/2D plots."

        if self.bounds.shape[0] == 1:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            X = np.linspace(self.bounds[0,0], self.bounds[0,1], 1000)
            ax.plot(X, func(X))
            return fig

        elif self.bounds.shape[0] == 2:
            import matplotlib.pyplot as plt
            XY, X, Y = construct_2D_grid(self.bounds)
            Z = call_function_on_grid(func, XY)[...,0]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.contourf(X, Y, Z, 50)
            return fig


class Jump1D(BaseEnvironment):
    bounds = np.array([[-1, 1]])

    def __call__(self, x):
        return np.sin(5*x) + np.sign(x)


class IncreasingOscillation(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def __call__(self, x):
        return np.sin(60 * x ** 4)


class IncreasingOscillationDecreasingAmplitude(IncreasingOscillation):
    bounds = np.array([[0, 1]])

    def __call__(self, x):
        import scipy.stats
        return super(IncreasingOscillationDecreasingAmplitude, self).__call__(x) \
             * scipy.stats.norm.pdf(x, 0.5, 0.3)


class Kink1D(BaseEnvironment):
    bounds = np.array([[-2, 2]])

    def __call__(self, x):
       return 1 / (10 ** (-4) + x ** 2)

    def derivative(self, x):
        return (2 * x) / ((x ** 2 + 1 / 10000) ** 2)


class Sin2D(BaseEnvironment):
    def __call__(self, x):
        return (0.5 * np.sin(13 * x[..., 0]) * np.sin(27 * x[..., 0]) + 0.5) * (0.5 * np.sin(13 * x[..., 1]) * np.sin(27 * x[..., 1]) + 0.5)


class Sinc(BaseEnvironment):
    bounds = np.array([[-20, 20]])
    x_opt = 0

    def __call__(self, x):
        x = np.asanyarray(x)
        y = where(x == 0, 1.0e-20, x)
        return np.sin(y)/y

    def derivative(self, x):
        # TODO: fix devision by zero if it becomes a problem.
        return (x * np.cos(x) - np.sin(x)) / (x ** 2)

    def hessian(self, x):
        return -((x ** 2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / (x ** 3)


class BigSinc(Sinc):
    bounds = np.array([[-0.2, 0.2]])

    def __call__(self, x):
        return super().__call__(100 * x)


class NegSinc(Sinc):
    def __call__(self, x):
        return -super().__call__(x)


class Sinc2D(BaseEnvironment):
    bounds = np.array([[-20, 20], [-20, 20]])

    def __call__(self, x):
        r = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        return np.sin(r) / r

    def derivative(self, x):
        r_pow = x[..., 0] ** 2 + x[..., 1] ** 2
        r = np.sqrt(r_pow)
        return (x * (np.cos(r) / r) - (np.sin(r) / r_pow)) / r

    def hessian(self, x):
        pass


class Kink2D(BaseEnvironment):
    bounds = np.array([[0, 1], [0, 1]])

    def __call__(self, x):
        y = 1 / (np.abs(0.5 - x[..., 0] ** 4 - x[..., 1] ** 4) + 0.1)
        return y[..., None]

    def derivative(self, X):
        # TODO: fix devision by zero if it becomes a problem.
        x = X[...,0]
        y = X[...,1]
        dx = -(4 * x ** 3 * (-0.5 + x ** 4 + y ** 4)) / (np.abs((0.5 - x ** 4 - y ** 4)) * (np.abs(0.5 - x ** 4 - y ** 4) + 0.1) ** 2)
        dy = -(4 * y ** 3 * (-0.5 + x ** 4 + y ** 4)) / (np.abs((0.5 - x ** 4 - y ** 4)) * (np.abs(0.5 - x ** 4 - y ** 4) + 0.1) ** 2)
        dd = np.stack((dx, dy), axis=-1)
        return dd


class ActiveSubspaceTest(BaseEnvironment):
    """"    
    Code:
    https://github.com/sischei/global_solution_yale19/blob/master/Lecture_6/code/AS_ex2.py

    Paper (fig. 4):
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
    """
    bounds = np.array([[-1,1]] * 10)

    def __call__(self, X):
        y = np.exp(0.01*X[...,0] + 0.7*X[...,1] + 0.02*X[...,2] + 0.03*X[...,3] + 0.04*X[...,4] + 
                    0.05*X[...,5] + 0.06*X[...,6] + 0.08*X[...,7] + 0.09*X[...,8] + 0.1*X[...,9])
        return y[...,None]

    def derivative(self, X):
        Y = self(X)
        coefs = np.array([0.01, 0.7, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1])
        return Y * coefs[None, :]


class ActiveSubspaceModifiedTest(ActiveSubspaceTest):
    def __call__(self, X):
        return X[..., 1]*X[..., 2] * super(ActiveSubspaceModifiedTest, self).__call__(X)

    def derivative(self, X):
        _test_old = super().derivative(X)
        val = np.atleast_1d(X[..., 1] * X[..., 2] * _test_old)
        coefs = np.array([0.01, 0.7, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1])
        out = val[:, None] * coefs[None, :]
        out[:, 1] += X[..., 2] * _test_old
        out[:, 2] += X[..., 1] * _test_old
        return out


class DynamicBell(BaseEnvironment):
    def __init__(self, input_dim=10):
        self.dim = input_dim

    def __call__(self, X):
        n = x.shape[0]
        y = np.zeros(n, float)
        
        # solve bellman equations at training points
        for i in range(n):
            y[i] = solver.initial(X[i], self.dim)[0] 
        return y[:, None]


from GPyOpt.objective_examples import experiments2d


class GPyOptEnvironment(BaseEnvironment):
    Func = None

    def __init__(self, *args, **kwargs):
        self._gpyopt_func = self.Func(*args, **kwargs)

    def __call__(self, x):
        return self._gpyopt_func.f(x)

    @property
    def bounds(self):
        return np.array(self._gpyopt_func.bounds)


class Beale(GPyOptEnvironment): Func = experiments2d.beale
class Branin(GPyOptEnvironment): Func = experiments2d.branin
class Cosines(GPyOptEnvironment): Func = experiments2d.cosines
class Dropwave(GPyOptEnvironment): Func = experiments2d.dropwave
class Eggholder(GPyOptEnvironment): Func = experiments2d.eggholder
class Goldstein(GPyOptEnvironment): Func = experiments2d.goldstein
class Mccormick(GPyOptEnvironment): Func = experiments2d.mccormick
class Powers(GPyOptEnvironment): Func = experiments2d.powers
class Rosenbrock(GPyOptEnvironment): Func = experiments2d.rosenbrock
class Sixhumpcamel(GPyOptEnvironment): Func = experiments2d.sixhumpcamel


def to_gpyopt_bounds(bounds):
    return [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': bounds[i]} for i in range(bounds.shape[0])]
