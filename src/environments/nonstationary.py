import numpy as np
import scipy.stats

from src.environments.core import BaseEnvironment


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


class Sin2DRotated(BaseEnvironment):
    bounds = np.array([[0,1], [0,1]])

    def _call(self, x):
        return np.sin( 2*np.pi*(x[...,0]+x[...,1]))[..., None]


__all__ = [
    'IncreasingOscillation',
    'IncreasingAmplitude',
    'IncreasingOscillationDecreasingAmplitude',
    'Sin2DRotated',
]
