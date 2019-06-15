from src.experiment.lazy_constructor import LazyConstructor


class ConfigMixin(object):
    @classmethod
    def from_config(cls, **kwargs):
        return cls(**kwargs)


def from_module(module, x):
    return getattr(module, x)


def construct_from_module(module, config, overrides=None):
    if config is None:
        return None
    name = config['name']
    kwargs = config.get('kwargs', {})
    merged_kwargs = dict({}, **kwargs)
    merged_kwargs.update(overrides or {})
    Class = from_module(module, name)
    return Class.from_config(**merged_kwargs)


def lazy_construct_from_module(module, config, overrides=None):
    if config is None:
        return None
    name = config['name']
    kwargs = config.get('kwargs', {})
    merged_kwargs = dict({}, **kwargs)
    merged_kwargs.update(overrides or {})
    Class = from_module(module, name)
    return LazyConstructor(Class, **merged_kwargs)


# Test examples
# Fix kernel only working because of Lazy
# Move default to Context (do not use __init__(**kwargs))
# Retrieve config merged with defaults from init.

# Fix Acq and Algorithm not having from_config
# Fix kernel => kernel_constructor naming
