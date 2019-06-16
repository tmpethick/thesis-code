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


class LazyConstructor(Generic[T]):
    def __init__(self, class_, **default_kwargs):
        self.class_ = class_
        self.default_kwargs = default_kwargs

    def __call__(self, *args, **new_kwargs) -> T:
        kwargs = {}
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


# Make lazy run from_config

# Fix kernel only working because of Lazy
# Move default to Context (do not use __init__(**kwargs))
# Retrieve config merged with defaults from init.
# Fix kernel => kernel_constructor naming
