from typing import TypeVar, Generic

T = TypeVar('T')

class LazyConstructor(Generic[T]):
    def __init__(self, class_, **default_kwargs):
        self.class_ = class_
        self.default_kwargs = default_kwargs
    
    def __call__(self, *args, **new_kwargs) -> T:
        kwargs = {}
        kwargs.update(self.default_kwargs)
        kwargs.update(new_kwargs)
        return self.class_(*args, **kwargs)
