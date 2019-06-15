from src import environments as environments_module, models as models_module, \
    acquisition_functions as acquisition_functions_module, algorithms as algorithm_module
from src.experiment.config_helpers import ConfigMixin, construct_from_module


class ExperimentContext(ConfigMixin):
    """An instance contains the top level context of an experiment.
    Using `from_config` it can translate a config into a context with objects.
    """
    def __init__(self,
        model=None,
        model2=None,
        obj_func=None,
        acquisition_function=None,
        bo=None,
        n_samples=None,
        gp_use_derivatives=None,
        tag=None,
        **kwargs
    ):
        self.model = model
        self.model2 = model2
        self.obj_func = obj_func
        self.acquisition_function = acquisition_function
        self.bo = bo
        self.n_samples = n_samples
        self.gp_use_derivatives = gp_use_derivatives
        self.tag = tag
        self.__dict__.update(kwargs)

        # Derivatives
        self.models = [model]
        if model2 is not None:
            self.models.append(model2)

    @classmethod
    def from_config(cls, *,
        model=None,
        model2=None,
        obj_func=None,
        acquisition_function=None,
        bo=None,
        **kwargs,
        ):

        obj_func = construct_from_module(environments_module, obj_func)
        model = construct_from_module(models_module, model)
        model2 = construct_from_module(models_module, model2)
        models = [model]
        if model2 is not None:
            models.append(model2)

        acquisition_function = construct_from_module(acquisition_functions_module, acquisition_function, dict(models=models))
        bo = construct_from_module(algorithm_module, bo, dict(
            f=obj_func,
            models=models,
            acquisition_function=acquisition_function,
        ))
        return cls(
            # TODO: pass models
            obj_func=obj_func,
            model=model,
            model2=model2,
            acquisition_function=acquisition_function,
            bo=bo,
            **kwargs,
        )