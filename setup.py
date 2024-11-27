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


VERSION = parse_version('geowatch/__init__.py')

try:
    README = parse_description()
except Exception:
    README = ''

# try:
REQUIREMENTS = (
    parse_requirements('requirements/runtime.txt')
)

__autogen__ = """

import ubelt as ub
dpath =ub.Path('requirements')
for p in dpath.ls():
    print(f'{p.stem!r},')

"""

nameable_requirements = [
    'aws',
    'cold',
    'development',
    'dvc',
    'gdal',
    'linting',
    'materials',
    'mmcv',
    'optional',
    'python_build_tools',
    'runtime',
    'tensorflow',
    'tests',

    'graphics',
    'headless',
    'compat',
]

EXTRAS_REQUIRES = {
    'all': parse_requirements('requirements.txt'),
}
for key in nameable_requirements:
    EXTRAS_REQUIRES[key] = parse_requirements(f'requirements/{key}.txt', versions='loose')
    EXTRAS_REQUIRES[key + '-strict'] = parse_requirements(f'requirements/{key}.txt', versions='strict')

NAME = 'geowatch'

if __name__ == '__main__':

    # Hacks for finding tpl packages

    included_tpl_dpaths = [
        'geowatch_tpl/submodules_static/torchview',
        'geowatch_tpl/submodules_static/segment-anything',
        'geowatch_tpl/submodules_static/scale-mae',
        'geowatch_tpl/submodules_static/jsonargparse',
        'geowatch_tpl/submodules_static/loss-of-plasticity',
        'geowatch_tpl/submodules_static/detectron2',
        'geowatch_tpl/modules',
    ]
    tpl_packages = []
    for tpl_dpath in included_tpl_dpaths:
        result = find_packages(tpl_dpath)
        tpl_packages.extend(
            [tpl_dpath.replace('/', '.') + '.' + s for s in result]
        )
    packages = find_packages(include=[
        'geowatch', 'geowatch.*',
        # TPL
        'geowatch_tpl',
        'geowatch_tpl.*',
        # Alias of the old module name to maintain backwards compatability
        # while we transition.
        # 'watch', 'watch.*',
    ]) + tpl_packages
    print(f'packages={packages}')

    setup(
        name=NAME,
        author="GeoWATCH developers",
        author_email='kitware@kitware.com',
        python_requires='>=3.8',
        # https://pypi.org/classifiers/
        classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Natural Language :: English',
            'Programming Language :: Python :: 3',
            # 'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
        ],
        description="",
        entry_points={
            'console_scripts': [
                'geowatch= geowatch.cli.__main__:main',
                # DEPRECATED: Use simple_dvc instead
                'geowatch_dvc= geowatch.cli.find_dvc:_CLI.main',
                # 'gwmlops= watch.mlops.__main__:main',
            ],
        },
        install_requires=REQUIREMENTS,
        extras_require=EXTRAS_REQUIRES,
        long_description_content_type='text/x-rst',
        long_description=README,
        include_package_data=True,
        package_data={
            'geowatch.tasks.depth': [
                'config.json'
            ],
            'geowatch.rc': [
                'site-model.schema.json',
                'region-model.schema.json',
                'job.schema.json',
                # 'dem.xml' do we want to include this?
                # 'egm96_15.gtx' do we want to include this?
            ],
            'geowatch.rc.requirements': [
                'aws.txt',
                'cold.txt',
                'development.txt',
                'docs.txt',
                'dvc.txt',
                'gdal-strict.txt',
                'gdal.txt',
                'graphics.txt',
                'headless.txt',
                'linting.txt',
                'materials.txt',
                'mmcv.txt',
                'optional.txt',
                'python_build_tools.txt',
                'runtime.txt',
                'tensorflow.txt',
                'tests.txt',
                'transitive_runtime.txt',
            ],
        },
        packages=packages,
        url='https://gitlab.kitware.com/computer-vision/geowatch.git',
        version=VERSION,
        zip_safe=False,
        license='Apache 2',
    )
