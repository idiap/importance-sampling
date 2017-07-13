#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""This module should provide some useful functions that I wish existed in the
functools standard package. But then again standard library developers know
better"""


placeholder = object()
x = placeholder
_ = placeholder
___ = placeholder


def partial(func, *args, **kwargs):
    """Partially define the argument list of func

    Example:
    >>> import functional as f
    >>> from operator import add
    >>> add2 = f.partial(add, f._, 2)
    >>> add2(5)
    7
    >>> add2(8)
    10
    >>> join = f.compose(' '.join, lambda *args: list(args))
    >>> create_sentence = f.partial(join, "The", f._, f._, "fox")
    >>> create_sentence("quick", "brown")
    'The quick brown fox'
    >>> import tempfile
    >>> fd, file_name = tempfile.mkstemp()
    >>> opener = f.partial(open, f._, "wb")
    >>> with opener(file_name) as out:
    ...     out.write("Hello World!")
    ...
    >>> with opener(file_name) as out:
    ...     out.write("Hello World!")
    ...
    >>> import os
    >>> os.remove(file_name)
    >>>
    """
    def inner(*args2, **kwargs2):
        # compute the kwargs to pass to the function
        fkwargs = kwargs.copy()
        fkwargs.update(kwargs2)

        # walk through the fargs list replacing each placeholder with an arg
        args2 = list(args2)
        fargs = list(args)
        for i in range(len(fargs)):
            if fargs[i] is placeholder:
                fargs[i] = args2.pop(0)
        fargs += args2

        # apply fargs and fkwargs to func
        return func(*fargs, **fkwargs)
    return inner


def compose(*functions):
    """Compose the functions with the following rules

    If a function returns a tuple the parameters are passed as positional
    arguments.

    Example:
    >>> import functional as f
    >>> add2 = f.compose(lambda x: x+1, lambda x: x+1)
    >>> add2(2)
    4
    >>> testf = f.compose(lambda x: x**2, lambda x: 2*x)
    >>> testf(1)
    4
    >>> testf = f.compose(lambda x: x**2, lambda a,b: a+b)
    >>> testf(1, 1)
    4
    >>>
    """
    def inner(*args, **kwargs):
        for f in reversed(functions):
            args = f(*args, **kwargs)
            if isinstance(args, dict):
                kwargs = args
                args = tuple()
            elif not isinstance(args, tuple):
                args = (args, )
            kwargs = {}
        if not isinstance(args, tuple) or len(args) > 1:
            return args
        else:
            return args[0]
    return inner


def call(f, *args, **kwargs):
    """Call f with args and kwargs"""
    return f(*args, **kwargs)


def attr(a):
    """Return a function that returns the attribute 'a'.

    It would be equal to partial(getattr(___, a)) but I define it separately
    for better documentation.
    """
    def inner(o):
        return getattr(o, a)
    return inner


def identity(x):
    return x
