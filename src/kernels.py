import GPy
import numpy as np

GPyExponential = GPy.kern.Exponential
GPyMatern32 = GPy.kern.Matern32
GPyMatern52 = GPy.kern.Matern52
GPyRBF = GPy.kern.RBF


def rejection_sampling(pdf, size = (1,1)):
    """Pulled from QFF
    """
    n = size[0]
    d = size[1]
    output = np.zeros(shape =size)
    i = 0
    while i < n:
        Z = np.random.normal (size = (1,d))
        u = np.random.uniform()
        if pdf(Z) < u:
            output[i,:] = Z
            i=i+1

    return output


class RFFKernel(object):
    def sample(self, size):
        raise NotImplementedError

    @property
    def variance(self):
        raise NotImplementedError


class RFFMatern(RFFKernel):
    def __init__(self, lengthscale=0.1, nu=0.5, variance=0.1):
        self.nu = nu
        self.theta = np.array([lengthscale, variance])
        self.bounds = np.array([[1e-5, 1e5], [1e-5, 1e5]])

        self.pdf = lambda x: np.prod(2*(self.lengthscale)/(np.power((1. + self.lengthscale**2*x**2),self.nu) * np.pi),axis =1)

    @property
    def lengthscale(self):
        return self.theta[0]

    @property
    def variance(self):
        return self.theta[1]

    def sample(self, size):
        return rejection_sampling(self.pdf,size=size)


class RFFRBF(RFFKernel):
    def __init__(self, lengthscale=0.1, variance=0.1):
        self.theta = np.array([lengthscale, variance])
        self.bounds = np.array([[1e-5, 1e5], [1e-5, 1e5]])

    @property
    def lengthscale(self):
        return self.theta[0]

    @property
    def variance(self):
        return self.theta[1]

    def sample(self, size):
        # TODO: Samples do not depend on self.variance? (does not look like it...)
        return np.random.normal(size=size) * (1.0 / self.lengthscale)
