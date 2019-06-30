import numpy as np

from src.environments.core import BaseEnvironment
from src.models import Normalizer

# TODO: Compose environment (Transformation such as Rescale, Shift / embeddings)

class EnvironmentNormalizer(BaseEnvironment):
    """Wrap an function so it takes input from a normalized domain and produces a normalized output.
    """
    def __init__(self, env: BaseEnvironment, input_normalizer: Normalizer=None, output_normalizer: Normalizer=None):
        self._env = env
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer

    def __repr__(self):
        return "{}<{}>".format(type(self).__name__, self._env)

    @property
    def bounds(self):
        # Swap axes so we have (2, D)
        bounds = np.moveaxis(self._env.bounds, -1, 0)

        # Change them into bounds in the normalized space
        bounds = self.input_normalizer.normalize(bounds)

        # Swap back the axes
        return np.moveaxis(bounds, -1, 0)

    def _call(self, X):
        if self.input_normalizer is not None:
            X = self.input_normalizer.denormalize(X)

        Y = self._env(X)

        if self.output_normalizer is not None:
            Y = self.output_normalizer.normalize(Y)

        return Y


__all__ = ['EnvironmentNormalizer']
