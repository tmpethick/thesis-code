import numpy as np
from matplotlib import pyplot as plt

from src.experiment.config_helpers import ConfigMixin, construct_from_module
from src.models import ProbModel


class Transformer(object):
    @property
    def output_dim(self):
        raise NotImplementedError

    def fit(self, X, Y, Y_dir=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError


class ActiveSubspace(Transformer):
    """
    Requires feeding `M = α k log(m)` samples to `self.fit`
    where α=5..10, m is actually dim, and k<m.
    """

    def __init__(self, k=10, output_dim=None, threshold_factor=10):
        # How many eigenvalues are considered. We do not consider all
        # eigenvalues (i.e. k=m) as the samples required increases in k.
        self.k = k

        # Uses a fixed output dim if not zero
        self._output_dim = output_dim

        # Used to decide when a big change occurs (eig[i] > thresholds_factor * eig[i+1])
        self.threshold_factor = threshold_factor

        self.vals = None
        self.W = None

    @property
    def output_dim(self):
        if self._output_dim is not None:
            return self._output_dim
        elif self.W is not None:
            return self.W.shape[-1]
        else:
            raise Exception("No promises can be made about `output_dim` since it is dynamically determind.")

    def _get_active_subspace_index(self, vals):
        """ Given list of eigenvectors sorted in ascended order (e.g. `vals = [1,2,30,40,50]`) return the index `i` being the first occurrence of a big change in value (in the example `i=2`).
        """
        if self._output_dim is not None:
            return -self._output_dim

        # Only consider k largest
        vals = vals[-self.k:]

        for i in reversed(range(len(vals))):
            big = vals[i]
            small = vals[i - 1]
            if (big / self.threshold_factor > small):
                return i
        return 0

    def fit(self, X, Y, Y_dir):
        """[summary]

        Arguments:
            X {[type]} -- input
            Y {[type]} -- (unused) function evaluation
            Y_dir {[type]} -- function gradient
        """

        N = X.shape[0]
        CN = (Y_dir.T @ Y_dir) / N

        # find active subspace
        vals, vecs = np.linalg.eigh(CN)
        self.vals = vals

        i = self._get_active_subspace_index(vals)

        self.W = vecs[:, i:]

    def transform(self, X):
        assert self.W is not None, "Call `self.fit` first"
        # (W.T @ X.T).T
        return X @ self.W

    def plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(9, 5))
        x = np.arange(1, self.vals.shape[0] + 1)
        ax[0].plot(x, self.vals, ".", ms=15)
        ax[0].set_xlabel("Eigenvalues")
        ax[0].set_xticks(x)
        ax[0].set_ylabel("$\lambda$")

        ax[1].plot(x, np.linalg.norm(self.W, axis=1), ".", ms=15)
        ax[1].set_xlabel("Input dimension")
        ax[1].set_xticks(x)
        ax[1].set_ylabel("Magnitude of W")

        fig.tight_layout()
        return fig


class TransformerModel(ConfigMixin, ProbModel):
    """Proxy a ProbModel through a Transformer first.
    """

    def __init__(self, *, transformer: Transformer, prob_model: ProbModel):
        super(TransformerModel, self).__init__()
        self.transformer = transformer
        self.prob_model = prob_model

    @classmethod
    def from_config(cls, *, transformer=None, prob_model=None, **kwargs):
        import src.models as models_module
        return cls(
            transformer=construct_from_module(models_module, transformer),
            prob_model=construct_from_module(models_module, prob_model),
            **kwargs,
        )

    def __repr__(self):
        return "{}<{},{}>".format(type(self).__name__,
                                  type(self.transformer).__name__,
                                  type(self.prob_model).__name__)

    # TODO: proxy the rest of the interface to prob_model (incl self.X and self.Y)

    def init(self, X, Y, Y_dir=None, train=True):
        self.X = X
        self.Y = Y
        self.Y_dir = Y_dir

        self.transformer.fit(self.X, self.Y, Y_dir=self.Y_dir)
        X = self.transformer.transform(self.X)

        self.prob_model.init(X, Y, Y_dir=Y_dir, train=train)

    def add_observations(self, X_new, Y_new, Y_dir_new=None):
        self.X = np.concatenate([self.X, X_new])
        self.Y = np.concatenate([self.Y, Y_new])

        if self.Y_dir is not None:
            self.Y_dir = np.concatenate([self.Y, Y_dir_new])

        self.transformer.fit(self.X, self.Y, Y_dir=self.Y_dir)
        X = self.transformer.transform(self.X)

        # Necessary to call init again since we do not know if the transformation of previous observation stayed the same.
        self.prob_model.init(X, self.Y, Y_dir=self.Y_dir, train=True)
        # self.prob_model.add_observations(X_new, Y_new, Y_dir_new=Y_dir_new)

    def _get_statistics(self, X, full_cov=True):
        X = self.transformer.transform(X)
        mean, covar = self.prob_model.get_statistics(X, full_cov=full_cov)
        return mean, covar