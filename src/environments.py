import numpy as np
from numpy.core.numeric import where

from src.utils import construct_2D_grid, call_function_on_grid

# TODO: latent Dirichlet allocation

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

    def plot(self, projection=None):
        return self._plot(self, projection=projection, title="$f$")

    def plot_derivative(self, projection=None):
        return self._plot(self.derivative, projection=projection, title="$\\nabla f$")

    def plot_curvature(self, norm="fro", projection=None):
        if self.bounds.shape[0] != 1:
            def hess(x):
                H = self.hessian(x)
                return np.linalg.norm(H, ord=norm, axis=(-2, -1))[...,None]
        else:
            hess = self.hessian
        return self._plot(hess, projection=projection, title="$\\nabla^2 f$")

    def _plot(self, func, projection=None, title=None):
        assert self.bounds.shape[0] in [1,2], "Only support 1D/2D plots."

        if self.bounds.shape[0] == 1:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            X = np.linspace(self.bounds[0,0], self.bounds[0,1], 1000)
            if title is not None:
                ax.set_title(title)
            ax.plot(X, func(X))
            return fig

        elif self.bounds.shape[0] == 2:
            import matplotlib.pyplot as plt
            XY, X, Y = construct_2D_grid(self.bounds)
            Z = call_function_on_grid(func, XY)[...,0]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection=projection)
            ax.contourf(X, Y, Z, 50)
            return fig


class TwoKink1D(BaseEnvironment):
    bounds = np.array([[0,1]])

    def __init__(self, *args, **kwargs):
        self.alpha = 5
        self.beta = 1
        self.x_1 = 0.3
        self.x_2 = 0.6

        self.fst = lambda x: self.alpha * x ** 2

        self.b = self.fst(self.x_1)
        self.snd = lambda x: self.b

        self.c = self.snd(self.x_2)
        self.trd = lambda x: self.beta * np.log(x) + (self.c - self.beta * np.log(self.x_2))
    
    def __call__(self, X):
        return np.piecewise(X, [X < self.x_1, X > self.x_1, X >= self.x_2], [self.fst, self.snd, self.trd])

    def derivative(self, X):
        fst = lambda x: 2 * self.alpha * x
        snd = lambda x: 0
        trd = lambda x: self.beta / x
        return np.piecewise(X, [X < self.x_1, X > self.x_1, X >= self.x_2], [fst, snd, trd])




class Step(BaseEnvironment):
    bounds = np.array([[0,1]])
    
    def __init__(self):
        # N_steps = 5
        # X_steps = np.random.uniform(0, 1, size=N_steps)
        # X_steps = np.sort(X_steps)
        # Y_values = 10*np.cos(2.5*X_steps-4) + 20 + np.random.uniform(-2, 2, size=N_steps)
        self.X_steps = np.array([0.1059391 , 0.23668238, 0.38007559, 0.47764559, 0.62765332, 0.87921645, 0.93967713, 0.98301519])
        self.Y_values = np.array([11.46027201,  9.59505656,  8.71181213, 11.93343655, 13.55133013, 18.15854289, 18.4201603 , 18.78589584])

    def __call__(self, X):
        condlist = [X > threshold for threshold in self.X_steps]
        return np.piecewise(X, condlist, self.Y_values)


class TwoKink2D(TwoKink1D):
    bounds = np.array([[0,1], [0,1]])

    def _transform(self, X):
        return (X[...,0]**2 + X[...,1]**2)[..., None]

    def __call__(self, X):
        z = self._transform(X)
        return super(TwoKink2D, self).__call__(z)

    def derivative(self, X):
        z = self._transform(X)
        return super(TwoKink2D, self).derivative(z) * 2 * X


class TwoKinkDEmbedding(TwoKink1D):
    def __init__(self, D=10, Alpha=None):
        super().__init__()
        if Alpha is not None:
            self.Alpha = np.array(Alpha)
            self.D = self.Alpha.shape[0]
        else:
            self.Alpha = TwoKinkDEmbedding.generate_alpha(D)
            self.D = D
        self.bounds = np.array([[0,1]] * D)

    @classmethod
    def generate_alpha(cls, D):
        """Useful if we want to fix Alpha across instances.
        """
        return np.random.uniform(size=(D,1))

    def _transform(self, X):
        # Apply elementwise linear transformation
        return X.dot(self.Alpha)

    def __call__(self, X):
        assert X.shape[1] == self.D, "X does not match the required input dim."        
        
        z = self._transform(X)
        return super().__call__(z)

    def derivative(self, X):
        z = self._transform(X)
        return super().derivative(z) * self.Alpha[:,0]


class Jump1D(BaseEnvironment):
    bounds = np.array([[-1, 1]])

    def __call__(self, x):
        return np.sin(5*x) + np.sign(x)


class Sin(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def __call__(self, x):
        import scipy.stats
        return np.sin(30 * x)


class IncreasingOscillation(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def __call__(self, x):
        return np.sin(60 * x ** 4)

class IncreasingAmplitude(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def __call__(self, x):
        import scipy.stats
        return np.sin(60 * x) * scipy.stats.norm.pdf(x, 1, 0.3)


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


class Sin2D(BaseEnvironment):
    def __call__(self, x):
        return (0.5 * np.sin(13 * x[..., 0]) * np.sin(27 * x[..., 0]) + 0.5) * (0.5 * np.sin(13 * x[..., 1]) * np.sin(27 * x[..., 1]) + 0.5)


class CosProd2D(BaseEnvironment):
    bounds = np.array([[-1,1], [-1,1]])
    def __call__(self, X):
        return (np.cos(0.5 * np.pi * X[..., 0]) * np.cos(0.5 * np.pi * X[..., 1]))[..., None]


class Sinc2D(BaseEnvironment):
    bounds = np.array([[-20, 20], [-20, 20]])

    def __call__(self, x):
        r = np.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)
        return (np.sin(r) / r)[..., None]

    def derivative(self, x):
        r_pow = x[..., 0] ** 2 + x[..., 1] ** 2
        r = np.sqrt(r_pow)
        return (x * (np.cos(r) / r) - (np.sin(r) / r_pow)) / r

    def hessian(self, x):
        pass

class Kink2DStraight(BaseEnvironment):
    """Kink2D but ignores y.
    """
    bounds = np.array([[0, 1], [0, 1]])

    def __call__(self, x):
        y = 1 / (np.abs(0.5 - x[..., 0] ** 4) + 0.1)
        return y[..., None]


class KinkDCircularEmbedding(BaseEnvironment):
    # Hack to make it settable in `__init__`.
    bounds = None

    def __init__(self, D=10):
        self.bounds = np.array([[0, 1]] * D)
        self._D = D

    def _transform(self, X):
        return np.sum(X ** 2, axis=-1)[:,None]

    def __call__(self, X):
        Z = self._transform(X)
        Y = 1 / (np.abs(0.5 - Z) + 0.1)
        return Y

    def derivative(self, X):
        # a little chain rule..
        Z = self._transform(X)
        f_diff = (0.5 - Z) / (np.abs(0.5 - Z) * (np.abs(0.5 - Z) + 0.1) ** 2)
        Z_diff = 2 * X

        return f_diff * Z_diff


class Kink2D(BaseEnvironment):
    """To generate derivative and hessian we use sympy:

    ```
    from sympy import *
    from sympy.utilities.lambdify import lambdify, implemented_function,lambdastr
    x, y, z = symbols('x y z', real=True)
    z=1 / (abs(0.5 - x ** 4 - y ** 4) + 0.1)
    z.diff(x)
    hess = [simplify(z.diff(x0).diff(x1)) for x0 in [x,y] for x1 in [x,y]]
    lambdastr(x, hess[0])
    ```
    """
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
    
    def hessian(self, X):
        x = X[...,0]
        y = X[...,1]
        a=np.where(x**4 + y**4 - 0.5 == 0, 0.0, np.copysign(1, x**4 + y**4 - 0.5))
        DiracDelta = lambda x: np.where(x == 0, 1e20, 0)
        h11 = 4*x**2*(8*x**4*a**2 - (8*x**4*DiracDelta(x**4 + y**4 - 0.5) + 3*a)*(np.abs(x**4 + y**4 - 0.5) + 0.1))/(np.abs(x**4 + y**4 - 0.5) + 0.1)**3
        h12 = 32*x**3*y**3*(-(np.abs(x**4 + y**4 - 0.5) + 0.1)*DiracDelta(x**4 + y**4 - 0.5) + a**2)/(np.abs(x**4 + y**4 - 0.5) + 0.1)**3
        h22 = 32*x**3*y**3*(-(np.abs(x**4 + y**4 - 0.5) + 0.1)*DiracDelta(x**4 + y**4 - 0.5) + a**2)/(np.abs(x**4 + y**4 - 0.5) + 0.1)**3
        h21 = 4*y**2*(8*y**4*a**2 - (8*y**4*DiracDelta(x**4 + y**4 - 0.5) + 3*a)*(np.abs(x**4 + y**4 - 0.5) + 0.1))/(np.abs(x**4 + y**4 - 0.5) + 0.1)**3
        hess = np.array([[h11, h12],
                         [h21, h22]])
        return np.moveaxis(hess, -1, 0)


class Kink2DShifted(Kink2D):
    bounds = np.array([[0, 1], [0, 1]])

    def __call__(self, x):
        return super().__call__(x) + 20


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


class ActiveSubspaceModifiedTest(BaseEnvironment):
    bounds = np.array([[-1,1]] * 10)

    def __init__(self):
        self._f = ActiveSubspaceTest()
    
    def __call__(self, X):
        return X[..., 1]*X[..., 2] * self._f(X)

    def derivative(self, X):
        """Fix derivative
        """
        _test_old = self._f.derivative(X)
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
