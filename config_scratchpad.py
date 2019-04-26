################################################ 
# If we are ever to standardise configs through
# class mixin this will come in handy.
################################################

# Can't do model config outside classes (not flexibel enough).
# Wouldn't be able to specify BO.GPModel.some_class

# Fully flexibile configuration would be too complex because of DAG dependency tree (models -> acq, bo).
# We hard code the relationship and instead create different variants.

from src import acquisition_functions
from src import models

# Incomplete config
config = ConfigBase({
    'acquisition_function': 'QuadratureAcquisition',
    'models': ConfigList([
        ConfigConstructor('DKLGPModel', {
            'n_iter': 10,
            'layers': (100, 500, 30, 2),
        }),
        ConfigConstructor('DKLGPModel')
    ]),
    'n_iter': 10
})


import functools

def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))

def compose(*fs):
    return functools.reduce(compose2, fs)

def string_required(x):
    if not isinstance(x, str):
        raise Exception("has to be string")
    return x

@functools.partial
def from_module(module, x):
    return getattr(module, x)

@functools.partial
def construct_from_module(module, x):
    name, kwargs = x
    Class = from_module(module, name)
    return Class.from_config(kwargs)

@functools.partial
def elementwise(func: callable, x):
    return list(map(func, x))

