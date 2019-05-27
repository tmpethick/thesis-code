class GenzContinuous(AbstractModel):
    """
    N-dimensional "continuous" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].
    .. math::  y = \\exp{\\left(- \sum_{i=1}^{N} a_i | x_i - u_i | \\right)}
    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]
    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output
    Notes
    -----
    .. plot::
       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict
       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)
       plot("GenzContinuous", parameters)
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.
    .. [2] https://www.sfu.ca/~ssurjano/cont.html
    """

    def __init__(self, p, context=None):
        super(GenzContinuous, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum in exponent
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * np.abs(self.p[key] - u[i])

        # determine output
        y = np.exp(-s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzCornerPeak(AbstractModel):
    """
    N-dimensional "CornerPeak" Genz function [1,2]. It is defined in the interval [0, 1] x ... x [0, 1].
    Used by [3] as testfunction.
    .. math:: y = \\left( 1 + \sum_{i=1}^N a_i x_i\\right)^{-(N + 1)}
    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]
    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output
    Notes
    -----
    .. plot::
       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict
       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)
       plot("GenzCornerPeak", parameters)
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.
    .. [2] https://www.sfu.ca/~ssurjano/copeak.html
    .. [3] Jakeman, J. D., Eldred, M. S., & Sargsyan, K. (2015).
       Enhancing ℓ1-minimization estimates of polynomial chaos expansions using basis selection.
       Journal of Computational Physics, 289, 18-34.
    """

    def __init__(self, p, context=None):
        super(GenzCornerPeak, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id=None):
        n = len(self.p.keys())

        # set constants
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = (1 + s) ** -(n + 1)

        y_out = y[:, np.newaxis]

        return y_out


class GenzDiscontinuous(AbstractModel):
    """
    N-dimensional "Discontinuous" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].
    .. math:: y = \exp\\left( \sum_{i=1}^N a_i x_i\\right) \quad \mathrm{if} \quad x_i < u_i \quad \mathrm{else} \quad 0
    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]
    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output
    Notes
    -----
    .. plot::
       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict
       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)
       plot("GenzDiscontinuous", parameters)
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.
    .. [2] https://www.sfu.ca/~ssurjano/disc.html
    """

    def __init__(self, p, context=None):
        super(GenzDiscontinuous, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id=None):

        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        mask = np.zeros((len(self.p[self.p.keys()[0]]), n))

        for i, key in enumerate(self.p.keys()):
            mask[:, i] = self.p[key] > u[i]
        mask = mask.any(axis=1)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = np.exp(s)
        y[mask] = 0.

        y_out = y[:, np.newaxis]

        return y_out


class GenzGaussianPeak(AbstractModel):
    """
    N-dimensional "GaussianPeak" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].
    .. math:: y = \exp\\left( - \sum_{i=1}^{N} a_i ^2 (x_i - u_i)^2\\right)
    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]
    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output
    Notes
    -----
    .. plot::
       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict
       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)
       plot("GenzGaussianPeak", parameters)
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.
    .. [2] https://www.sfu.ca/~ssurjano/gaussian.html
    """

    def __init__(self, p, context=None):
        super(GenzGaussianPeak, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] ** 2 * (self.p[key] - u[i]) ** 2

        # determine output
        y = np.exp(-s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzOscillatory(AbstractModel):
    """
    N-dimensional "Oscillatory" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].
    .. math:: y = \cos \\left( 2 \pi u_1 + \sum_{i=1}^{N}a_i x_i \\right)
    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]
    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output
    Notes
    -----
    .. plot::
       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict
       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)
       plot("GenzOscillatory", parameters)
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.
    .. [2] https://www.sfu.ca/~ssurjano/oscil.html
    """

    def __init__(self, p, context=None):
        super(GenzOscillatory, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine sum
        s = np.zeros(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            s += a[i] * self.p[key]

        # determine output
        y = np.cos(2 * np.pi * u[0] + s)

        y_out = y[:, np.newaxis]

        return y_out


class GenzProductPeak(AbstractModel):
    """
    N-dimensional "ProductPeak" Genz function [1]. It is defined in the interval [0, 1] x ... x [0, 1].
    .. math:: y = \prod_{i=1}^{N} \\left( a_i^{-2} + (x_i - u_i)^2 \\right)^{-1}
    Parameters
    ----------
    p["x1"]: float or ndarray of float [n_grid]
        First parameter defined in [0, 1]
    p["xi"]: float or ndarray of float [n_grid]
        i-th parameter defined in [0, 1]
    p["xN"]: float or ndarray of float [n_grid]
        Nth parameter defined in [0, 1]
    Returns
    -------
    y: ndarray of float [n_grid x 1]
        Output
    Notes
    -----
    .. plot::
       import numpy as np
       from pygpc.testfunctions import plot_testfunction as plot
       from collections import OrderedDict
       parameters = OrderedDict()
       parameters["x1"] = np.linspace(0, 1, 100)
       parameters["x2"] = np.linspace(0, 1, 100)
       plot("GenzProductPeak", parameters)
    .. [1] Genz, A. (1984), Testing multidimensional integration routines.
       Proc. of international conference on Tools, methods and languages for scientific
       and engineering computation, Elsevier North-Holland, Inc., NewYork, NY, USA, pp. 81-94.
    .. [2] https://www.sfu.ca/~ssurjano/prpeak.html
    """

    def __init__(self, p, context=None):
        super(GenzProductPeak, self).__init__(p, context)

    def validate(self):
        pass

    def simulate(self, process_id=None):
        n = len(self.p.keys())

        # set constants
        u = 0.5 * np.ones(n)
        a = 5 * np.ones(n)

        # determine output
        y = np.ones(np.array(self.p[self.p.keys()[0]]).size)

        for i, key in enumerate(self.p.keys()):
            y *= 1 / (a[i] ** (-2) + (self.p[key] - u[i]) ** 2)

        y_out = y[:, np.newaxis]

        return y_out
