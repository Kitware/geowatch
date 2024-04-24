Running tests
-------------

The ``run_tests.py`` script provided here will run all tests in the ``tests``
directory and in docstrings and report coverage. This script is simply a
wrapper around the ``pytest`` command.

Alternatively doctests can be invoked specivially via ``xdoctest -m watch`` to
run all doctests, or ``xdoctest -m <path-to-file>`` to run all doctests in a
file.


Writing tests
-------------

GeoWATCH uses the ``pytest`` module for running unit tests. Unit tests
should be added into the ``tests`` directory and files should be
prefixed with ``test_``.

Additionally, code blocks in function docstrings will be interpreted as tests using
`xdoctest <https://xdoctest.readthedocs.io/en/latest/autoapi/xdoctest/index.html>`_
as part of the `Google docstring convention <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

For example here are what doctests look like for a class and for a function:

.. code:: python

    class GdalOpen:
        """
        A simple context manager for friendlier gdal use.

        Example:
            >>> # xdoctest: +REQUIRES(--network)
            >>> from watch.utils.util_raster import *
            >>> from watch.demo.landsat_demodata import grab_landsat_product
            >>> path = grab_landsat_product()['bands'][0]
            >>>
            >>> # standard use:
            >>> dataset = gdal.Open(path)
            >>> print(dataset.GetDescription())  # do stuff
            >>> del dataset  # or 'dataset = None'
            >>>
            >>> # equivalent:
            >>> with GdalOpen(path) as dataset:
            >>>     print(dataset.GetDescription())  # do stuff
        """
        # code goes here

    def my_cool_function(inputs):
        """
        The purpose of this function is to demonstrate how to write a doctest.

        Example:
            >>> # An example of how to use my cool function
            >>> # The xdoctest module will run this as a test
            >>> inputs = 'construct-demo-data'
            >>> my_cool_function(inputs)
        """
        import this
        print('You input: {}'.format(inputs))
