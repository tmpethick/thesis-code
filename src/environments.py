import numpy as np
from numpy.core.numeric import where

from src.utils import construct_2D_grid, call_function_on_grid


class BaseEnvironment(object):
    x_opt = None

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
            return self._call(x) + noise
        else:
            return self._call(x) 

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
        return self._plot(self.noiseless, projection=projection, title="$f$")

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
            Y = func(X)
            ax.plot(X, Y)

            # Show the 2*std for the noice
            #plt.fill_between(X, Y - 2 * self.noise, Y + 2 * self.noise, alpha=0.2)
            
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
        super().__init__(*args, **kwargs)

        self.alpha = 5
        self.beta = 1
        self.x_1 = 0.3
        self.x_2 = 0.6

        self.fst = lambda x: self.alpha * x ** 2

        self.b = self.fst(self.x_1)
        self.snd = lambda x: self.b

        self.c = self.snd(self.x_2)
        self.trd = lambda x: self.beta * np.log(x) + (self.c - self.beta * np.log(self.x_2))
    
    def _call(self, X):
        return np.piecewise(X, [X < self.x_1, X > self.x_1, X >= self.x_2], [self.fst, self.snd, self.trd])

    def derivative(self, X):
        fst = lambda x: 2 * self.alpha * x
        snd = lambda x: 0
        trd = lambda x: self.beta / x
        return np.piecewise(X, [X < self.x_1, X > self.x_1, X >= self.x_2], [fst, snd, trd])




class Step(BaseEnvironment):
    bounds = np.array([[0,1]])
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # N_steps = 5
        # X_steps = np.random.uniform(0, 1, size=N_steps)
        # X_steps = np.sort(X_steps)
        # Y_values = 10*np.cos(2.5*X_steps-4) + 20 + np.random.uniform(-2, 2, size=N_steps)
        self.X_steps = np.array([0.1059391 , 0.23668238, 0.38007559, 0.47764559, 0.62765332, 0.87921645, 0.93967713, 0.98301519])
        self.Y_values = np.array([11.46027201,  9.59505656,  8.71181213, 11.93343655, 13.55133013, 18.15854289, 18.4201603 , 18.78589584])

    def _call(self, X):
        condlist = [X > threshold for threshold in self.X_steps]
        return np.piecewise(X, condlist, self.Y_values)


class TwoKink2D(TwoKink1D):
    bounds = np.array([[0,1], [0,1]])

    def _transform(self, X):
        return (X[...,0]**2 + X[...,1]**2)[..., None]

    def _call(self, X):
        z = self._transform(X)
        return super(TwoKink2D, self)._call(z)

    def derivative(self, X):
        z = self._transform(X)
        return super(TwoKink2D, self).derivative(z) * 2 * X


class TwoKinkDEmbedding(TwoKink1D):
    def __init__(self, D=10, Alpha=None, **kwargs):
        super().__init__(**kwargs)
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

    def _call(self, X):
        assert X.shape[1] == self.D, "X does not match the required input dim."        
        
        z = self._transform(X)
        return super()._call(z)

    def derivative(self, X):
        z = self._transform(X)
        return super().derivative(z) * self.Alpha[:,0]


class Jump1D(BaseEnvironment):
    bounds = np.array([[-1, 1]])

    def _call(self, x):
        return np.sin(5*x) + np.sign(x)


class Sin(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def _call(self, x):
        import scipy.stats
        return np.sin(30 * x)


class IncreasingOscillation(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def _call(self, x):
        return np.sin(60 * x ** 4)

class IncreasingAmplitude(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def _call(self, x):
        import scipy.stats
        return np.sin(60 * x) * scipy.stats.norm.pdf(x, 1, 0.3)


class IncreasingOscillationDecreasingAmplitude(IncreasingOscillation):
    bounds = np.array([[0, 1]])

    def _call(self, x):
        import scipy.stats
        return super(IncreasingOscillationDecreasingAmplitude, self)._call(x) \
             * scipy.stats.norm.pdf(x, 0.5, 0.3)


class Kink1D(BaseEnvironment):
    bounds = np.array([[-2, 2]])

    def _call(self, x):
       return 1 / (10 ** (-4) + x ** 2)

    def derivative(self, x):
        return (2 * x) / ((x ** 2 + 1 / 10000) ** 2)


class Sinc(BaseEnvironment):
    bounds = np.array([[-20, 20]])
    x_opt = 0

    def _call(self, x):
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

    def _call(self, x):
        return super()._call(100 * x)


class NegSinc(Sinc):
    def _call(self, x):
        return -super()._call(x)


class Sin2D(BaseEnvironment):
    def _call(self, x):
        return (0.5 * np.sin(13 * x[..., 0]) * np.sin(27 * x[..., 0]) + 0.5) * (0.5 * np.sin(13 * x[..., 1]) * np.sin(27 * x[..., 1]) + 0.5)


class CosProd2D(BaseEnvironment):
    bounds = np.array([[-1,1], [-1,1]])
    def _call(self, X):
        return (np.cos(0.5 * np.pi * X[..., 0]) * np.cos(0.5 * np.pi * X[..., 1]))[..., None]


class Sinc2D(BaseEnvironment):
    bounds = np.array([[-20, 20], [-20, 20]])

    def _call(self, x):
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

    def _call(self, x):
        y = 1 / (np.abs(0.5 - x[..., 0] ** 4) + 0.1)
        return y[..., None]


class KinkDCircularEmbedding(BaseEnvironment):
    # Hack to make it settable in `__init__`.
    bounds = None

    def __init__(self, D=10, bounds=None, **kwargs):
        super().__init__(**kwargs)
        if bounds is None:
            self.bounds = np.array([[0, 1]] * D)
        else:
            self.bounds = bounds
        self._D = D

    def _transform(self, X):
        if X.ndim == 1:
            return X ** 2

        return np.sum(X ** 2, axis=-1)[:,None]

    def _call(self, X):
        Z = self._transform(X)
        Y = 1 / (np.abs(0.5 - Z) + 0.1)
        return Y

    def derivative(self, X):
        # a little chain rule..
        Z = self._transform(X)
        f_diff = (0.5 - Z) / (np.abs(0.5 - Z) * (np.abs(0.5 - Z) + 0.1) ** 2)
        Z_diff = 2 * X

        return f_diff * Z_diff
    
    # def uniform_manifold_sample(self, size):
    #     D = self._D

    #     x = np.empty((size, D))
    #     z = np.random.uniform(0, 1, (size, 1))
        
    #     # Put down on line
    #     x[:,0] = z

    #     # Random rotation in D dim
    #     theta = np.random.uniform(0, 1, D)

    #     # Form rotation matrix
    #     R = scipy.stats.special_ortho_group(D)
    #     R.dot(x)

    #     # How to project into x?


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

    def _call(self, x):
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


class Kink1DShifted(Kink1D):
    def _call(self, x):
        return super()._call(x) + 1000


class Kink2DShifted(Kink2D):
    bounds = np.array([[0, 1], [0, 1]])

    def _call(self, x):
        return super()._call(x) + 20


class ActiveSubspaceTest(BaseEnvironment):
    """"    
    Code:
    https://github.com/sischei/global_solution_yale19/blob/master/Lecture_6/code/AS_ex2.py

    Paper (fig. 4):
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2927400
    """
    bounds = np.array([[-1,1]] * 10)

    def _call(self, X):
        y = np.exp(0.01*X[...,0] + 0.7*X[...,1] + 0.02*X[...,2] + 0.03*X[...,3] + 0.04*X[...,4] + 
                    0.05*X[...,5] + 0.06*X[...,6] + 0.08*X[...,7] + 0.09*X[...,8] + 0.1*X[...,9])
        return y[...,None]

    def derivative(self, X):
        Y = self(X)
        coefs = np.array([0.01, 0.7, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1])
        return Y * coefs[None, :]


# class ActiveSubspaceModifiedTest(BaseEnvironment):
#     bounds = np.array([[-1,1]] * 10)

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._f = ActiveSubspaceTest()
    
#     def _call(self, X):
#         return X[..., 1]*X[..., 2] * self._f(X)

#     def derivative(self, X):
#         """TODO: Fix derivative
#         """
#         _test_old = self._f.derivative(X)
#         val = np.atleast_1d(X[..., 1] * X[..., 2] * _test_old)
#         coefs = np.array([0.01, 0.7, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09, 0.1])
#         out = val[:, None] * coefs[None, :]
#         out[:, 1] += X[..., 2] * _test_old
#         out[:, 2] += X[..., 1] * _test_old
#         return out


# import growth_model_GPR.nonlinear_solver_initial as solver_initializor
# import growth_model_GPR.nonlinear_solver_iterate as solver_iterator


# class DynamicBell(BaseEnvironment):
#     # bounds are np.array([k_bar, k_up] * dim)

#     def __init__(self, input_dim=10):
#         self.dim = input_dim

#         self._is_initialized = False
#         self.n_agents = 2
#         # TODO: consolidate with n_agents in parameters.py

#     def _call(self, X, prob_model):
#         """OBS: every call will iterate. 
#         We assume that self.prob_model is updated with new observations (discarding the old.)
#         """
#         n = X.shape[0]
#         y = np.zeros(n, float)

#         if self._is_initialized:
#             for i in range(n):
#                 y[i] = solver_iterator.iterate(X[i], self.n_agents, prob_model)[0]
#         else:
#             self._is_initialized = True
#             for i in range(n):
#                 y[i] = solver_initializor.initial(X[i], self.n_agents)[0]


#         return y[:, None]


from GPyOpt.objective_examples import experiments2d


class GPyOptEnvironment(BaseEnvironment):
    Func = None

    def __init__(self, *args, noise=None, **kwargs):
        super().__init__(*args, noise=noise, **kwargs)
        self._gpyopt_func = self.Func(*args, **kwargs)

    def _call(self, x):
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


class GenzContinuous(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10):
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.ones((1, D)) * 0.5

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5
        
        assert self.a.shape[0] == D
        assert self.u.shape[1] == D

    def _call(self, X):
        Z = np.abs(X - self.u)
        return np.exp(-np.einsum("i,ji->j", self.a, Z))[:, None]


class GenzCornerPeak(BaseEnvironment):
    bounds = None

    def __init__(self, a=None, D=10):
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5
        
        assert self.a.shape[0] == D

    def _call(self, X):
        return np.power(1 + np.einsum("i,ji->j", self.a, X), -(self.D + 1))[:, None]


class GenzDiscontinuous(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10):
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.array([0.5, 0.5])


        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5

        assert self.a.shape[0] == D

    def _call(self, X):
        Z = np.exp(np.einsum("i,ji->j", self.a, X))
        zero = np.zeros(X.shape[0])
        return np.where((X[..., 0] > self.u[0]) | (X[..., 1] > self.u[1]), zero, Z)[:, None]


class GenzGaussianPeak(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10):
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.ones((1, D)) * 0.5

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5
        
        assert self.a.shape[0] == D
        assert self.u.shape[1] == D


    def _call(self, X):
        Z = np.power(X - self.u, 2)
        return np.exp(-np.einsum("i,ji->j", self.a ** 2, Z))[:, None]


class GenzOscillatory(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10):
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.array([0.5])

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5
        
        assert self.a.shape[0] == D
        assert self.u.shape[0] == 1


    def _call(self, X):
        return np.cos(2 * np.pi * self.u[0] + np.einsum("i,ji->j", self.a, X))[:, None]


class GenzProductPeak(BaseEnvironment):
    bounds = None

    def __init__(self, u=None, a=None, D=10):
        self.bounds = np.array([[0,1]] * D)
        self.D = D

        if u is not None:
            self.u = u
        else:
            self.u = np.ones((1, D)) * 0.5

        if a is not None:
            self.a = a
        else:
            self.a = np.ones(D) * 5
        
        assert self.a.shape[0] == D
        assert self.u.shape[1] == D

    def _call(self, X):
        # Introduce new axis so addition with X works.
        a_pow = np.power(self.a, -2)[None, :]

        return np.product(1 / (a_pow + np.power(X - self.u, 2)), axis=-1)[:,None]
