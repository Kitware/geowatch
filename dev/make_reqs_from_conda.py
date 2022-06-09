import ubelt as ub


def parse_requirement_file(fpath):
    # import requirements
    # with open('requirements.txt', 'r') as fd:
    #     for req in requirements.parse(fd):
    #         print(req.name, req.specs)
    from pip._internal.req import parse_requirements
    from pip._internal.network.session import PipSession
    declared_deps = []
    for req in parse_requirements(fpath, session=PipSession()):
        declared_deps.append(req.requirement)
    return declared_deps


def parse_conda_reqs(fpath, blocklist=set()):
    import yaml
    with open(fpath, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)
    context = []
    found = []
    for dep in data['dependencies']:
        if isinstance(dep, dict) and 'pip' in dep:
            for line in dep['pip']:
                if line.startswith('-r '):
                    req_file = line.split(' ')[-1].replace('file:', '')
                    found.extend(parse_requirement_file(req_file))
                elif line.startswith('--'):
                    context.append(line)
                else:
                    name = line.split(' ')[0]
                    if name not in blocklist:
                        found.append(' '.join([line] + context))
    return found


# def query_pypi_dependencies(package_name, version=None):
#     """
#     Args:
#         >>> package_name = 'sympy'
#         >>> version = None
#     """
#     import requests
#     if version is not None:
#         url = f'https://pypi.python.org/pypi/{package_name}/{version}/json'
#     else:
#         url = f'https://pypi.python.org/pypi/{package_name}/json'
#     resp = requests.get(url)
#     assert resp.status_code == 200
#     depinfo = resp.json()
#     print('depinfo = {}'.format(ub.repr2(depinfo, nl=-1)))


# def pip_explore_deps(package_name, version=None):
#     cache_dpath = ub.ensure_app_cache_dir('watch/pypi_dep_resolve')
#     cmd = f'pip download -r requirements.txt -d {cache_dpath}'
#     print(cmd)
#     pass


def make_upgrade_strict_line():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch/dev'))
    from make_reqs_from_conda import *  # NOQA
    """
    defined_req_lines = parse_conda_reqs('conda_env.yml')

    def normalize_name(name):
        return name.lower().replace('-', '_')

    name_to_conda_line = {
        normalize_name(line.split(' ')[0].split('>')[0]): line
        for line in defined_req_lines
    }
    declared_deps = list(name_to_conda_line.keys())
    deps = list(ub.oset(declared_deps) - {'gdal', 'opencv-python-headless'})

    print('pip install ' + ' '.join(deps) + ' -U')


def trace_all_deps(defined_req_lines):
    r"""
    TODO: make this work.

    The issue is that the packages dependencies need to be installed for this
    to correctly find the dependencies.

    pip install requirements-parser

    Ignore:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/dev'))
        from make_reqs_from_conda import *  # NOQA
        defined_req_lines = parse_conda_reqs('conda_env.yml')
        groups = trace_all_deps(defined_req_lines)

        print(r'\begin{multicols}{4}')
        print(r'\begin{itemize}')
        for line in groups['Defined']:
            nover = line.partition('>')[0].partition('=')[0].partition(' ')[0]
            print(r'    \item ' + nover.replace('_', '\_'))
        print(r'\end{itemize}')
        print(r'\end{multicols}')

        print('\begin{multicols}{4}')
        print('\begin{itemize}')
        for line in groups['Implied']:
            nover = line.partition('>')[0].partition('=')[0].partition(' ')[0]
            print('    \item ' + nover.replace('_', '\_'))
        print('\end{itemize}')
        print('\end{multicols}')

        for line in groups['Implied']:
            print('    \item ' + line)

        print('\n'.join(lines))


    """
    import pipdeptree
    from distutils.version import LooseVersion

    def normalize_name(name):
        return name.lower().replace('-', '_')

    name_to_conda_line = {
        normalize_name(line.split(' ')[0]): line
        for line in defined_req_lines
    }
    declared_deps = list(name_to_conda_line.keys())

    pkgs = pipdeptree.get_installed_distributions()
    tree = pipdeptree.PackageDAG.from_pkgs(pkgs)
    tree = tree.filter(declared_deps, [])

    # for pkg, deps in tree.items():
    #     pkg_label = '{0} >= {1}'.format(pkg.project_name, pkg.version)
    #     print(pkg_label)

    frozen = False
    list_all = True

    tree = tree.sort()
    nodes = tree.keys()
    branch_keys = set(r.key for r in pipdeptree.flatten(tree.values()))
    use_bullets = not frozen

    if not list_all:
        nodes = [p for p in nodes if p.key not in branch_keys]

    def aux(node, parent=None, indent=0, chain=None):
        chain = chain or []
        node_str = node.render(parent, frozen)
        if parent:
            prefix = ' ' * indent + ('- ' if use_bullets else '')
            node_str = prefix + node_str

        node_dict = node.as_dict()
        yield node_dict
        for c in tree.get_children(node.key):
            if c.project_name not in chain:
                yield from aux(c, node, indent=indent + 2, chain=chain + [c.project_name])

    items = list(pipdeptree.flatten([aux(p) for p in nodes]))
    item_variants = ub.group_items(items, key=lambda d: d['key'])

    blocklist = {
        'opencv-python',
    }

    resolved = []
    for pkg_name, variants in item_variants.items():
        variants = list(ub.unique(variants, key=ub.hash_data))
        if len(variants) == 1:
            variant = variants[0]
        else:
            variant = max(variants, key=lambda x: (LooseVersion(x['installed_version']), 'required_version' in x))
            # print('RESOLVE pkg_name = {!r}'.format(pkg_name))
            # for variant in variants:
            #     print('variant = {!r}'.format(variant))
        if variant['key'] not in blocklist:
            resolved.append(variant)

    key_to_newline = {}
    for item in resolved:
        key = normalize_name(item['package_name'])
        if key in defined_req_lines:
            line = defined_req_lines[key]
        else:
            line = key + '>=' + item['installed_version']
        key_to_newline[key] = line

    toplevel = list(name_to_conda_line.values())
    remain = list(ub.dict_diff(key_to_newline, ub.oset(name_to_conda_line)).values())

    groups = {}
    groups['Defined'] = toplevel
    groups['Implied'] = remain
    return groups


def main():
    """
    Export pip requirements in a conda env file to a requirements.txt file
    """
    fpath = 'conda_env.yml'

    # There are issues with netharn and imgaug due to opencv-python
    # So keep them out of the direct dependencies
    blocklist = {
        'netharn',
        'imgaug',
    }
    defined_req_lines = parse_conda_reqs(fpath, blocklist=blocklist)
    header1 = ub.codeblock(
        '''
        # This file is autogenerated from conda_env.yml and does not include
        # netharn or imgaug due to their forced dependency on opencv-python
        # See all-explicit.txt for how to install these safely
        ''')
    text = '\n'.join([header1] + defined_req_lines)
    with open('requirements/autogen/all-implicit.txt', 'w') as file:
        file.write(text)
    print(text)

    header2 = ub.codeblock(
        '''
        # This is semi-autogenerated such that a working environment should be
        # able to be installed via
        # `pip install -r requirements/autogen/all-explicit.txt --no-deps`
        # which might help in avoiding opencv issues
        ''')
    groups = trace_all_deps(defined_req_lines)
    new_lines = ['# Defined'] + groups['Defined'] + ['', '# Implied'] + groups['Implied']
    new_lines = [header2, ''] + new_lines
    new_text = ('\n'.join(new_lines))
    with open('requirements/autogen/all-explicit.txt', 'w') as file:
        file.write(new_text)
    print(new_text)


def compare_strict_versions():
    """
    Test what installed versions are different from the strict versions
    """
    from distutils.version import LooseVersion
    fpath = 'conda_env.yml'
    blocklist = {
        'netharn',
        'imgaug',
    }
    defined_req_lines = parse_conda_reqs(fpath, blocklist=blocklist)

    defined_versions = {}
    for line in defined_req_lines:
        if '>=' in line:
            name, ver = line.split('>=')
            name = name.strip().replace('-', '_')
            defined_versions[name] = LooseVersion(ver.strip().split(' ')[0])

    import pipdeptree
    pkgs = pipdeptree.get_installed_distributions()

    have_versions = {}
    for pkg in pkgs:
        name = pkg.project_name.replace('-', '_')
        have_versions[name] = LooseVersion(pkg.version)

    common = set(have_versions) & set(defined_versions)
    for key in common:
        have_v = have_versions[key]
        def_v = defined_versions[key]
        if have_v != def_v:
            print(f'Diff: {key:<30}, have={str(have_v):<10} != defined={str(def_v):<10}')


if __name__ == '__main__':
    """

    pip install pipdeptree
    python ~/code/watch/dev/make_reqs_from_conda.py


    pipdeptree -p ubelt

    pip install -r requirements.txt

    pip uninstall opencv-python opencv-python-headless  && pip install opencv-python-headless

    --no-deps
    # pip download -r requirements.txt -d ./tmp-deps --no-binary :all: -v
    """
    main()
