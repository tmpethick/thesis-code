import numpy as np

from src.experiment.config_helpers import ConfigMixin, construct_from_module
from src.models import BaseModel


class Normalizer(object):
    def __init__(self, X_train):
        self.X, self._X_mean, self._X_std = zero_mean_unit_var_normalization(X_train)

    def get_transformed(self):
        return self.X

    def normalize(self, X):
        """Transform new X into rescaling constructed by X_train.
        """
        X, _, _ = zero_mean_unit_var_normalization(X, mean=self._X_mean, std=self._X_std)
        return X

    def denormalize(self, X):
        """Take normalized X and turn back into unnormalized version.
        (used for turning predicted output trained on normalized version into original domain.)
        """
        X = zero_mean_unit_var_unnormalization(X, self._X_mean, self._X_std)
        return X

    def denormalize_variance(self, X_var):
        return (X_var*(self._X_std**2))

    def denormalize_covariance(self, covariance):
        return (covariance[..., None]*(self._X_std**2))


def zero_mean_unit_var_normalization(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def zero_mean_unit_var_unnormalization(X_normalized, mean, std):
    return X_normalized * std + mean


class NormalizerModel(ConfigMixin, BaseModel):
        def __init__(self, model, normalize_input=True, normalize_output=True):
            super().__init__()
            self.model = model
            self.X_normalizer = None
            self.Y_normalizer = None
            self._normalize_input = normalize_input
            self._normalize_output = normalize_output

        @classmethod
        def process_config(cls, *, model=None, **kwargs):
            import src.models as models_module
            model = construct_from_module(models_module, model)
            return dict(
                model=model,
                **kwargs
            )

        def _normalize(self, X, Y):
            if self._normalize_input:
                self.X_normalizer = Normalizer(X)
                X = self.X_normalizer.get_transformed()

            if self._normalize_output:
                self.Y_normalizer = Normalizer(Y)
                Y = self.Y_normalizer.get_transformed()

            return X, Y

        def init(self, X, Y, Y_dir=None):
            # TODO: do not "copy/paste" behaviour from BaseModel.
            self._X = X
            self._Y = Y
            self.Y_dir = Y_dir

            X, Y = self._normalize(X, Y)
            self.model.init(X, Y, Y_dir)

        def add_observations(self, X_new, Y_new, Y_dir_new=None):
            # TODO: do not "copy/paste" behaviour from BaseModel.
            assert self._X is not None, "Call init first"

            # Update data
            X = np.concatenate([self._X, X_new])
            Y = np.concatenate([self._Y, Y_new])

            if self.Y_dir is not None:
                Y_dir = np.concatenate([self.Y_dir, Y_dir_new])

            self.init(X, Y, Y_dir)

        def get_statistics(self, X, full_cov=True):
            if self._normalize_input:
                X = self.X_normalizer.normalize(X)

            mean, covar = self.model.get_statistics(X, full_cov=full_cov)

            if self._normalize_output:
                mean = self.Y_normalizer.denormalize(mean)
                if full_cov:
                    covar = self.Y_normalizer.denormalize_covariance(covar)
                else:
                    covar = self.Y_normalizer.denormalize_variance(covar)
            return mean, covar
