# scatterfit/__init__.py

from .scatterfit import scatterfit as _scatterfit
def scatterfit(*args, **kwargs):
    return _scatterfit(*args, **kwargs)

__all__ = ['scatterfit']
__version__ = "0.1.0"