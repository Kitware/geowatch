#!/usr/bin/env python
"""The setup script."""
from os.path import exists
from setuptools import setup, find_packages


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class VersionVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == '__version__':
                    self.version = node.value.s
    visitor = VersionVisitor()
    visitor.visit(pt)
    return visitor.version


def parse_requirements(fpath='requirements.txt', pinned='free'):
    """
    Args:
        pinned (str): can be
            free - remove all constraints
            loose - use the greater or equal (>=) in the req file
            strict - replace all greater equal with equals
    """
    # Note: different versions of pip might have different internals.
    # This may need to be fixed.
    from pip._internal.req import parse_requirements
    from pip._internal.network.session import PipSession
    requirements = []
    for req in parse_requirements(fpath, session=PipSession()):
        if pinned == 'free':
            req_name = req.requirement.split(' ')[0]
            requirements.append(req_name)
        elif pinned == 'loose':
            requirements.append(req.requirement)
        elif pinned == 'strict':
            requirements.append(req.requirement.replace('>=', '=='))
        else:
            raise KeyError(pinned)
    return requirements

VERSION = parse_version('watch/__init__.py')

try:
    with open('README.rst') as readme_file:
        README = readme_file.read()
except Exception:
    README = ''

try:
    REQUIREMENTS = (
        parse_requirements('requirements/production.txt')
    )
    EXTRAS_REQUIRES = {
        'all': parse_requirements('requirements.txt'),
        'development': parse_requirements('requirements/development.txt'),
        'problematic': parse_requirements('requirements/problematic.txt'),
        'optional': parse_requirements('requirements/optional.txt'),
        # Really annoying that this is the best we can do
        # The user *must* choose either headless or graphics
        # to get a complete working install.
        'headless': parse_requirements('requirements/headless.txt'),
        'graphics': parse_requirements('requirements/graphics.txt'),
    }
except Exception as ex:
    print('ex = {!r}'.format(ex))
    REQUIREMENTS = []
    EXTRAS_REQUIRES = {}

setup(
    author="WATCH developers",
    author_email='kitware@kitware.com',
    python_requires='>=3.8',
    # https://pypi.org/classifiers/
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="",
    entry_points={
        'console_scripts': [
            'watch-cli = watch.cli.__main__:main',
            'smartwatch= watch.cli.__main__:main',
            'smartwatch_dvc= watch.cli.find_dvc:_CLI.main',
        ],
    },
    install_requires=REQUIREMENTS,
    extras_require=EXTRAS_REQUIRES,
    long_description_content_type='text/x-rst',
    long_description=README,
    include_package_data=True,
    package_data={
        'watch.tasks.depth': [
            'config.json'
        ],
        'watch.rc': [
            'site-model.schema.json'
            'region-model.schema.json'
        ],
    },
    name='watch',
    packages=find_packages(include=['watch', 'watch.*']),
    url='https://gitlab.kitware.com/smart/watch.git',
    version=VERSION,
    zip_safe=False,
)
