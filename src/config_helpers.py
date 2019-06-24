import functools

from src.models.lazy_constructor import LazyConstructor
from src import models as models_module
from src import acquisition_functions as acquisition_functions_module
from src import environments as environments_module
from src import kernels as kernels_module
from src import algorithms as algorithm_module


class ConfigMixin(object):
    @classmethod
    def from_config(cls, **kwargs):
        raise NotImplementedError


def from_module(module, x):
    return getattr(module, x)


def construct_from_module(module, config, overrides=None):
    if config is None:
        return None
    name = config['name']
    kwargs = config['kwargs']
    merged_kwargs = dict({}, kwargs)
    merged_kwargs.update(overrides or {})
    Class = from_module(module, name)
    return Class.from_config(**merged_kwargs)


def lazy_construct_from_module(module, config, overrides=None):
    if config is None:
        return None
    name = config['name']
    kwargs = config['kwargs']
    merged_kwargs = dict({}, kwargs)
    merged_kwargs.update(overrides or {})
    Class = from_module(module, name)
    return LazyConstructor(Class, **merged_kwargs)


class ExperimentContext(ConfigMixin, object):
    def __init__(self,
        model=None,
        model2=None,
        obj_func=None,
        acquisition_function=None,
        bo=None,
        n_samples=None,
        use_derivatives=None,
    ):
        self.model = model
        self.model2 = model2
        self.obj_func = obj_func
        self.acquisition_function = acquisition_function
        self.bo = bo
        self.n_samples = n_samples
        self.use_derivatives = use_derivatives

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
        models = [model, model2]

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

class Normalizer(object):
    @classmethod
    def from_config(cls, *, model=None, **kwargs):
        model = construct_from_module(models_module, **model)
        return cls(
            model=model, 
            **kwargs
        )

class GPModel(object):
    @classmethod
    def from_config(cls, *, kernel=None, **kwargs):
        kernel = lazy_construct_from_module(kernels_module, kernel)
        return cls(
            kernel = kernel,
            **kwargs,
        )



# TransformerModel
# Change kernel being passed input_dim
# Modify_config: name == 'DKLGPModel' => dklgpmodel_training_callback
