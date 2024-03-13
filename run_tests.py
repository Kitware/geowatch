#!/usr/bin/env python3
if __name__ == '__main__':
    """

    cd ~/code/geowatch
    python run_tests.py geowatch tests scripts

    python run_tests.py geowatch/tasks/fusion --cov geowatch.tasks.fusion --customdirs geowatch/tasks/fusion/fit.py
    """
    import pytest
    import sys
    import ubelt as ub
    package_name = ub.argval('--cov', 'geowatch')
    pytest_args = [
        '--cov-config', 'pyproject.toml',
        '--cov-report', 'html',
        '--cov-report', 'term',
        '--cov=' + package_name,
        # '-s',
        '--durations=100',
    ]

    # if not sys.argv[1:]:
    if not ub.argflag('--customdirs'):
        # Default to these subdirs unless --custom-subdirs is specified
        pytest_args += ['geowatch', 'tests']

    pytest_args = pytest_args + sys.argv[1:]
    sys.exit(pytest.main(pytest_args))
