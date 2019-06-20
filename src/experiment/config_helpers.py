from typing import Generic, TypeVar

T = TypeVar('T')


class ConfigMixin(object):
    @classmethod
    def from_config(cls, kwargs, processed_kwargs=None):
        kwargs = cls.process_config(**kwargs)
        kwargs = dict({}, **kwargs)
        kwargs.update(processed_kwargs or {})
        return cls(**kwargs)

    @classmethod
    def process_config(cls, **kwargs):
        return kwargs

    #def store_defaults(self):
    #    pass

    #def get_config(self):
        # get defaults
        # merge with passed
        # for every ConfigMixin class do conversion back to config (even lazy).
    #    pass


class LazyConstructor(Generic[T]):
    def __init__(self, class_, **default_kwargs):
        self.class_ = class_
        self.default_kwargs = default_kwargs

        #self.from_config = False

    #def set_from_config(self, from_config):
    #    self.from_config = from_config

    def __call__(self, *args, **new_kwargs) -> T:
        kwargs = {}

        #if self.from_config:
        #    assert len(args) == 0, "args not supported for from_config."
        #    return self.class_.from_config(self.default_kwargs, overrides=new_kwargs)
        #else:
        kwargs.update(self.default_kwargs)
        kwargs.update(new_kwargs)
        return self.class_(*args, **kwargs)


def construct_from_module(module, config, overrides=None):
    if config is None:
        return None
    name = config['name']
    kwargs = config.get('kwargs', {})
    Class = getattr(module, name)
    return Class.from_config(kwargs, overrides)


def lazy_construct_from_module(module, config, overrides=None):
    """
    TODO: configs are currently not being processed but instead passed directly to __init__.
    TODO: currently this is what makes external classes such as GPyTorchKernels work...
    """
    if config is None:
        return None
    name = config['name']
    kwargs = config.get('kwargs', {})
    kwargs = dict({}, **kwargs)
    kwargs.update(overrides or {})
    Class = getattr(module, name)
    return LazyConstructor(Class, **kwargs)


# Storing configs
    # Make lazy run from_config
    # Fix kernel only working because of Lazy (wrap all GPyTorchKernels)
# Move default to Context (do not use __init__(**kwargs))
# Retrieve config merged with defaults from init.
# Fix kernel => kernel_constructor naming

def recursively_apply_to_dict(d, modifier=lambda k, v: None):
    def iterate(d_):
        for k, v in d_.items():
            if isinstance(v, dict):
                iterate(v)
            modifier(k, v)
    iterate(d)
