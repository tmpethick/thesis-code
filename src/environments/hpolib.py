import numpy as np
from GPyOpt.objective_examples import experiments2d

from src.environments.core import BaseEnvironment


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


__all__ = [
    'GPyOptEnvironment',
    'Beale',
    'Branin',
    'Cosines',
    'Dropwave',
    'Eggholder',
    'Goldstein',
    'Mccormick',
    'Powers',
    'Rosenbrock',
    'Sixhumpcamel',
    'to_gpyopt_bounds',
]
