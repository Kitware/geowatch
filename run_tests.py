#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    """

    cd ~/code/watch
    python run_tests.py watch tests scripts

    python run_tests.py watch/tasks/fusion --cov watch.tasks.fusion
    """
    import pytest
    import sys
    import ubelt as ub
    package_name = ub.argval('--cov', 'watch')
    pytest_args = [
        '--cov-config', '.coveragerc',
        '--cov-report', 'html',
        '--cov-report', 'term',
        '--cov=' + package_name,
    ]
    if not sys.argv[1:]:
        pytest_args += ['watch', 'tests', 'scripts']

    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
