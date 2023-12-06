Installing Python
=================

On Linux we recommend using `pyenv to install Python <install_python_pyenv.rst>`_.
Alternatively, you can use `conda <install_python_conda.rst>`_, but note that
on Linux we only use ``pip install`` to install Python packages, and we never
use ``conda install``. This ensures we have access to all depenedencies while
minimizing the possibility of dependency conflicts.


To install `Python on windows <windows.rst>`_, we only support conda because we
don't know of any other easy way to obtain GDAL wheels.
