import numpy as np


class BaseEnvironment(object):
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

    def plot(self):
        assert self.bounds.shape[0] == 1, "Only support 1D plots."

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        X = np.linspace(self.bounds[0,0], self.bounds[0,1], 1000)
        ax.plot(X, self(X))
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


class Kink2D(BaseEnvironment):
    bounds = np.array([[0, 1], [0, 1]])

    def __call__(self, x):
        y = 1 / (np.abs(0.5 - x[..., 0] ** 4 - x[..., 1] ** 4) + 0.1)
        return y[..., None]


class ActiveSubspaceTest(BaseEnvironment):
    bounds = np.array([[-1,1]] * 10)

    def __call__(self, X):
        y = np.exp(0.01*X[...,0] + 0.7*X[...,1] + 0.02*X[...,2] + 0.03*X[...,3] + 0.04*X[...,4] + 
                    0.05*X[...,5] + 0.06*X[...,6] + 0.08*X[...,7] + 0.09*X[...,8] + 0.1*X[...,9])
        return y[...,None]

    def derivative(self, X):
        Y = self(X)
        coefs = np.array([0.01, 0.7, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1])
        return Y * coefs[None, :]


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
