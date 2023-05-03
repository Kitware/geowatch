#!/usr/bin/env python3
"""The setup script."""
import re
import sys
from os.path import exists, dirname, join
from setuptools import setup, find_packages


def parse_version(fpath):
    """
    Statically parse the version number from a python file
    """
    value = static_parse('__version__', fpath)
    return value


def static_parse(varname, fpath):
    """
    Statically parse the a constant variable from a python file
    """
    import ast
    if not exists(fpath):
        raise ValueError('fpath={!r} does not exist'.format(fpath))
    with open(fpath, 'r') as file_:
        sourcecode = file_.read()
    pt = ast.parse(sourcecode)
    class StaticVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            for target in node.targets:
                if getattr(target, 'id', None) == varname:
                    self.static_value = node.value.s
    visitor = StaticVisitor()
    visitor.visit(pt)
    try:
        value = visitor.static_value
    except AttributeError:
        import warnings
        value = 'Unknown {}'.format(varname)
        warnings.warn(value)
    return value


def parse_description():
    """
    Parse the description in the README file

    CommandLine:
        pandoc --from=markdown --to=rst --output=README.rst README.md
        python -c "import setup; print(setup.parse_description())"
    """
    from os.path import dirname, join, exists
    readme_fpath = join(dirname(__file__), 'README.rst')
    # This breaks on pip install, so check that it exists.
    if exists(readme_fpath):
        with open(readme_fpath, 'r') as f:
            text = f.read()
        return text
    return ''


def parse_requirements(fname='requirements.txt', versions='loose'):
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        versions (bool | str, default=False):
            If true include version specs.
            If strict, then pin to the minimum version.

    Returns:
        List[str]: list of requirements items
    """
    require_fpath = fname

    def parse_line(line, dpath=''):
        """
        Parse information from a line in a requirements text file

        line = 'git+https://a.com/somedep@sometag#egg=SomeDep'
        line = '-e git+https://a.com/somedep@sometag#egg=SomeDep'
        """
        # Remove inline comments
        comment_pos = line.find(' #')
        if comment_pos > -1:
            line = line[:comment_pos]

        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = join(dpath, line.split(' ')[1])
            for info in parse_require_file(target):
                yield info
        else:
            # See: https://www.python.org/dev/peps/pep-0508/
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                if '--find-links' in line:
                    # setuptools doesnt seem to handle find links
                    line = line.split('--find-links')[0]
                if ';' in line:
                    pkgpart, platpart = line.split(';')
                    # Handle platform specific dependencies
                    # setuptools.readthedocs.io/en/latest/setuptools.html
                    # #declaring-platform-specific-dependencies
                    plat_deps = platpart.strip()
                    info['platform_deps'] = plat_deps
                else:
                    pkgpart = line
                    platpart = None

                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, pkgpart, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        dpath = dirname(fpath)
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line, dpath=dpath):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if versions and 'version' in info:
                    if versions == 'strict':
                        # In strict mode, we pin to the minimum version
                        if info['version']:
                            # Only replace the first >= instance
                            verstr = ''.join(info['version']).replace('>=', '==', 1)
                            parts.append(verstr)
                    else:
                        parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    plat_deps = info.get('platform_deps')
                    if plat_deps is not None:
                        parts.append(';' + plat_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


# def parse_requirements_alt(fpath='requirements.txt', versions='loose'):
#     """
#     Args:
#         versions (str): can be
#             False or "free" - remove all constraints
#             True or "loose" - use the greater or equal (>=) in the req file
#             strict - replace all greater equal with equals
#     """
#     # Note: different versions of pip might have different internals.
#     # This may need to be fixed.
#     from pip._internal.req import parse_requirements
#     from pip._internal.network.session import PipSession
#     requirements = []
#     for req in parse_requirements(fpath, session=PipSession()):
#         if not versions or versions == 'free':
#             req_name = req.requirement.split(' ')[0]
#             requirements.append(req_name)
#         elif versions == 'loose' or versions is True:
#             requirements.append(req.requirement)
#         elif versions == 'strict':
#             part1, *rest = req.requirement.split(';')
#             strict_req = ';'.join([part1.replace('>=', '==')] + rest)
#             requirements.append(strict_req)
#         else:
#             raise KeyError(versions)
#     requirements = [r.replace(' ', '') for r in requirements]
#     return requirements


VERSION = parse_version('watch/__init__.py')

try:
    README = parse_description()
except Exception:
    README = ''

# try:
REQUIREMENTS = (
    parse_requirements('requirements/runtime.txt')
)
EXTRAS_REQUIRES = {
    'all': parse_requirements('requirements.txt'),
    'development': parse_requirements('requirements/development.txt'),
    'tests': parse_requirements('requirements/tests.txt'),
    'optional': parse_requirements('requirements/optional.txt'),
    'dvc': parse_requirements('requirements/dvc.txt'),
    'tensorflow': parse_requirements('requirements/tensorflow.txt'),
    # Really annoying that this is the best we can do
    # The user *must* choose either headless or graphics
    # to get a complete working install.
    'headless': parse_requirements('requirements/headless.txt'),
    'graphics': parse_requirements('requirements/graphics.txt'),
    'all-strict': parse_requirements('requirements.txt', versions='strict'),
    'runtime-strict': parse_requirements('requirements/runtime.txt', versions='strict'),
    'development-strict': parse_requirements('requirements/development.txt', versions='strict'),
    'tests-strict': parse_requirements('requirements/tests.txt', versions='strict'),
    'optional-strict': parse_requirements('requirements/optional.txt', versions='strict'),
    'dvc-strict': parse_requirements('requirements/dvc.txt', versions='strict'),
    'tensorflow-strict': parse_requirements('requirements/tensorflow.txt', versions='strict'),
    # Really annoying that this is the best we can do
    # The user *must* choose either headless or graphics
    # to get a complete working install.
    'headless-strict': parse_requirements('requirements/headless.txt', versions='strict'),
    'graphics-strict': parse_requirements('requirements/graphics.txt', versions='strict'),
}
# import ubelt as ub
# print('EXTRAS_REQUIRES = {}'.format(ub.urepr(EXTRAS_REQUIRES, nl=2)))


NAME = 'geowatch'

if __name__ == '__main__':
    setup(
        name=NAME,
        author="GEOWATCH developers",
        author_email='kitware@kitware.com',
        python_requires='>=3.9',
        # https://pypi.org/classifiers/
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
        ],
        description="",
        entry_points={
            'console_scripts': [
                'smartwatch= watch.cli.__main__:main',
                'smartwatch_dvc= watch.cli.find_dvc:_CLI.main',
                'geowatch= watch.cli.__main__:main',
                'geowatch_dvc= watch.cli.find_dvc:_CLI.main',
                # 'gwmlops= watch.mlops.__main__:main',
                # 'geowatch_dvc= watch.cli.find_dvc:__config__.main',
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
        packages=find_packages(include=[
            'watch', 'watch.*',
            # Alias the module while we transition to a new name.
            'geowatch', 'geowatch.*',
        ]),
        url='https://gitlab.kitware.com/computer-vision/geowatch.git',
        version=VERSION,
        zip_safe=False,
        license='Apache 2',
    )
