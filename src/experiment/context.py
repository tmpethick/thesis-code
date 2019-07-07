from src import environments as environments_module, models as models_module, \
    acquisition_functions as acquisition_functions_module, algorithms as algorithm_module
from src.experiment.config_helpers import ConfigMixin, construct_from_module, lazy_construct_from_module
from src.experiment import settings


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
                 gp_samples=None,
                 gp_test_samples=2500,
                 use_sample_grid=False,
                 gp_use_derivatives=None,
                 tag=None,
                 verbosity=None,
                 **kwargs
                 ):
        self.model = model
        self.model2 = model2
        self.obj_func = obj_func
        self.acquisition_function = acquisition_function
        self.bo = bo

        self.gp_samples = gp_samples
        self.gp_test_samples = gp_test_samples
        self.use_sample_grid = use_sample_grid
        self.gp_use_derivatives = gp_use_derivatives

        self.tag = tag
        self.verbosity = {
            'plot': settings.MODE is not settings.MODES.SERVER, # do not plot on server by default.
            'bo_show_iter': 30,
        }
        if isinstance(verbosity, dict):
            self.verbosity.update(verbosity)

        self.__dict__.update(kwargs)

        # Derivatives
        self.models = [model]
        if model2 is not None:
            self.models.append(model2)

    @classmethod
    def process_config(cls, *,
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

        acquisition_function_constructor = lazy_construct_from_module(acquisition_functions_module, acquisition_function)
        if acquisition_function_constructor is not None:
            acquisition_function = acquisition_function_constructor(*models)
        else:
            acquisition_function = None

        bo_constructor = lazy_construct_from_module(algorithm_module, bo)
        if bo_constructor is not None:
            bo = bo_constructor(
                f=obj_func,
                models=models,
                acquisition_function=acquisition_function,
            )
        else:
            bo = None

        return dict(
            # TODO: pass models
            obj_func=obj_func,
            model=model,
            model2=model2,
            acquisition_function=acquisition_function,
            bo=bo,
            **kwargs,
        )
