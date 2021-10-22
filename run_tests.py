#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    """

    cd ~/code/watch
    python run_tests.py watch tests scripts

    python run_tests.py watch/tasks/fusion --cov watch.tasks.fusion --customdirs watch/tasks/fusion/fit.py
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
        '-s',
        '--durations=0',
    ]

    # if not sys.argv[1:]:
    if not ub.argflag('--customdirs'):
        # Default to these subdirs unless --custom-subdirs is specified
        pytest_args += ['watch', 'tests', 'scripts']

    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
