import numpy as np

from src.environments.core import BaseEnvironment


class SingleStep(BaseEnvironment):
    bounds = np.array([[0, 1]])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X_steps = np.array([0.0, 0.5])
        self.Y_values = np.array([-1, 1])

    def _call(self, X):
        condlist = [X >= threshold for threshold in self.X_steps]
        return np.piecewise(X, condlist, self.Y_values)


class Step(BaseEnvironment):
    bounds = np.array([[0,1]])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # N_steps = 5
        # X_steps = np.random.uniform(0, 1, size=N_steps)
        # X_steps = np.sort(X_steps)
        # Y_values = 10*np.cos(2.5*X_steps-4) + 20 + np.random.uniform(-2, 2, size=N_steps)
        self.X_steps = np.array([0.1059391 , 0.23668238, 0.38007559, 0.47764559, 0.62765332, 0.87921645, 0.93967713, 0.98301519])
        self.Y_values = np.array([11.46027201,  9.59505656,  8.71181213, 11.93343655, 15.55133013, 18.15854289, 18.4201603 , 18.78589584]) / 17

    def _call(self, X):
        condlist = [X > threshold for threshold in self.X_steps]
        return np.piecewise(X, condlist, self.Y_values)


class StepConcave(BaseEnvironment):
    bounds = np.array([[0,1]])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # x = np.sort(np.random.uniform(0, 1, 8))
        # plt.plot(x, np.sin(np.pi*x))
        self.X_steps = np.array([0.1, 0.33, 0.5 , 0.7, 0.8, 0.9, 0.95 ])
        self.Y_values = np.sin(np.pi*self.X_steps)

    def _call(self, X):
        condlist = [X > threshold for threshold in self.X_steps]
        return np.piecewise(X, condlist, self.Y_values)


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


class Kink1D(BaseEnvironment):
    bounds = np.array([[-2, 2]])

    def _call(self, x):
       return 1 / (10 ** (-4) + x ** 2)

    def derivative(self, x):
        return (2 * x) / ((x ** 2 + 1 / 10000) ** 2)


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


__all__ = [
    'SingleStep',
    'Step',
    'StepConcave',
    'TwoKink1D',
    'TwoKink2D',
    'TwoKinkDEmbedding',
    'Jump1D',
    'Kink1D',
    'Kink2DStraight',
    'KinkDCircularEmbedding',
    'Kink2D',
    'Kink1DShifted',
    'Kink2DShifted',
]
