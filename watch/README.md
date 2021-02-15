# WATCH

Python module for WATCH

## Installation

The WATCH Python module can be installed with `pip` via the following command, where `/path/to/watch` is the absolute path to the directory containing this README.md file.

```
pip install -e /path/to/watch
```

## Running tests

We're using the `pytest` module for running unit tests.  Unit tests should be added into the `tests` directory and files should be prefixed with `test_`.

The `run_tests.py` script provided here will run all tests in the `tests` directory.

## Adding submodules

Library code can be added by creating a new subdirectory under the `watch` subdirectory.  You'll also want to create an empty `__init__.py` file in your new subdirectory (e.g. `touch watch/new_module/__init__.py`).

## Adding command line tools

New Python command line scripts can be added under the `watch/tools` directory.  To have the command line tool be installed with the module, an entry can be added to the `setup.py` setup call, under `entrypoints['console_scripts']`.
