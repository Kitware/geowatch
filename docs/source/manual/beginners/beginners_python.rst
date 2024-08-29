*****************
Beginner's Python
*****************

The GeoWATCH software is primarilly implemented in Python. To understand the codebase it will help to be familiar with Python basics.
This document provides a list of commonly used concepts and links to documentation.

Python datastructures like lists, sets, and dictionaries:
https://docs.python.org/3/tutorial/datastructures.html and the time complexity
of their associated operations: https://wiki.python.org/moin/TimeComplexity

Parameter unpacking with ``*args`` and ``**kwargs``: https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters

List comprehensions

An understanding of the terms::

    * expression
    * assignment
    * function
    * property
    * attribute
    * dunder
    * comprehension
    * generator
    * iterator
    * callable
    * iterable
    * dictionary
    * list
    * module
    * class
    * method
    * instance
    * type
    * namespace


Python Keywords
---------------

Official docs are here: https://docs.python.org/3/reference/lexical_analysis.html#keywords
It is imporant to be familiar with most of Python's keywords.

The majority of Python keywords are commonly used. It is imporatnt to understand ::

    # Values
    False None True

    # Definitions
    class def

    # Errors
    try except finally raise

    # Conditions
    if elif else

    # Contextual
    with

    # Imports
    import from as

    # Operators
    in is not or and

    # Loops
    for while continue break

    # Imperative
    assert return yield

Less common keywords that you should be aware of::

    # Async
    async await

    # Definitions
    lambda

    # Scoping
    nonlocal global

    # Imperative
    del pass

Dunder Methods
--------------

Duner refers to names that start with two (double) underscores: e.g. In
conversation it is easier to say "duner init" instead of ``__init__``.

Duner methods are part of Python's data model, which has many features:
https://docs.python.org/3/reference/datamodel.html but a subset of those are
commonly used and you should be familar with the following dunder methods you
can overload on a class::

    * __init__
    * __str__, * __repr__
    * __len__
    * __dir__
    * __call__
    * __iter__
    * __contains__
    * __getitem__, __setitem__
    * __getattr__, __setattr__
    * __add__, __sub__, __mul__, __truediv__
    * __and__, __or__, __xor__
    * __enter__, __exit__


You should also be aware of duner attributes like::

    * __dict__
    * __class__
    * __name__
    * __file__
    * __path__


Standard Library Modules
------------------------
Knowledge of the special duner methods:

The stdlib modules::

    argparse, itertools, functools, datetime, copy, collections, math, decimal,
    fractions, importlib, json, pathlib, pickle, platform, stat, re, random,
    shutil, textwrap, warnings

Pypi Modules
------------


Third party modules::

    numpy, scipy, torch, pandas, ubelt, xdoctest
    matplotlib, seaborn,
    shapely, rich, networkx,
    pytorch_lightning


Builtins
--------

.. code::

    # Debug
    breakpoint,

    # Async
    aiter,
    anext,

    # Bytes
    bytearray,
    bytes,

    # Strings
    format,
    hash,
    ascii,
    bin, hex, oct, ord,
    chr,
    str, repr,

    # Interpreter
    compile, eval, exec,

    # Introspection
    type,
    dir,
    isinstance, issubclass,
    callable,
    vars, globals, locals,
    id,
    len,

    # Attributes
    delattr, getattr, hasattr, setattr,

    # Math
    abs, sum, pow,
    divmod, round,
    max, min,
    sorted,
    all, any,

    # Constants
    False, True, None,
    Ellipsis,
    NotImplemented,

    # Low Level
    memoryview,

    # Numeric Structures
    bool,
    int,
    complex,
    float,

    # IO Functions
    input,
    print,
    open,

    # Data Structures
    tuple,
    dict,
    set,
    list,
    frozenset,

    # Iterators
    iter,
    reversed,
    range,
    enumerate,
    filter,
    map,
    zip,
    next,

    # Object Oriented
    property,
    object,
    super,
    staticmethod,
    classmethod,

    # Imperative
    slice,

    # Common Exceptions
    Exception,
    KeyboardInterrupt,
    AssertionError,
    AttributeError,
    MemoryError,
    ImportError,
    NameError,
    TypeError,
    ValueError,
    IndexError,
    IOError,
    KeyError,
    NotImplementedError,

    # Uncommon Exceptions
    BaseException, BaseExceptionGroup, GeneratorExit, SystemExit, ArithmeticError, BufferError, EOFError, LookupError,
    OSError, ReferenceError, RuntimeError, StopAsyncIteration, StopIteration, SyntaxError, SystemError,
    Warning, FloatingPointError, OverflowError, ZeroDivisionError, BytesWarning, DeprecationWarning, EncodingWarning,
    FutureWarning, ImportWarning, PendingDeprecationWarning, ResourceWarning, RuntimeWarning, SyntaxWarning, UnicodeWarning,
    UserWarning, BlockingIOError, ChildProcessError, ConnectionError,
    FileExistsError, FileNotFoundError, InterruptedError, IsADirectoryError, NotADirectoryError,
    PermissionError, ProcessLookupError, TimeoutError, IndentationError, ModuleNotFoundError,
    RecursionError, UnboundLocalError, UnicodeError, BrokenPipeError,
    ConnectionAbortedError, ConnectionRefusedError, ConnectionResetError,
    TabError, UnicodeDecodeError, UnicodeEncodeError, UnicodeTranslateError,
    ExceptionGroup, EnvironmentError,

    # Misc
    quit, exit,
    copyright, credits, license, help,

    # Dunder
    __name__, __doc__, __package__, __loader__, __spec__, __build_class__, __import__,
    __debug__,




Advanced Terms
--------------

Terms you may want to learn more about after getting the basics are::

    * metaclass
    * coroutine
    * duck-typing
    * docstring
    * GIL
    * hashable
    * mapping
    * immutable
    * method resolution order
    * virtual environment
    * type annotation
    * PEP

The official glossary is: https://docs.python.org/3/glossary.html


Python Package Structure / Importing
------------------------------------

I've written a stackoverflow answer that contains good information about python
packaging and how to handle importing external files.

https://stackoverflow.com/questions/74976153/what-is-the-best-practice-for-imports-when-developing-a-python-package/74976641#74976641

.. ... Preview:

   python -m instant_rst -f ~/code/geowatch/docs/source/manual/beginners/beginners_python.rst
