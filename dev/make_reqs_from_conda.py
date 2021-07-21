import ubelt as ub


__notes__ = """

# These requirements were pulled down at one time, need to verify
# if current logic is missing them or if they are not needed

  # List of other sub-deps not explicitly listed, but that will get
  # pulled down. Should these be added to the requirment spec
  Jinja2 >= 3.0.0
  MarkupSafe >= 2.0.0
  PyWavelets >= 1.1.1
  WTForms >= 2.3.3
  absl_py >= 0.12.0
  astunparse >= 1.6.3
  atomicwrites >= 1.4.0
  attrs >= 21.2.0
  bezier >= 2021.2.12
  cachetools >= 4.2.2
  chardet >= 4.0.0
  click >= 7.1.2
  click_plugins >= 1.1.1
  cligj >= 0.7.1
  configparser >= 5.0.2
  cycler >= 0.10.0
  decorator >= 4.4.2
  diskcache >= 5.2.1
  distro >= 1.5.0
  fasteners >= 0.16
  fiona >= 1.8.19
  flask >= 2.0.0
  flask_cors >= 3.0.10
  flask_wtf >= 0.14.3
  geomet >= 0.3.0
  girder_client >= 3.1.4
  google_auth >= 1.30.0
  google_auth_oauthlib >= 0.4.4
  grpcio >= 1.37.1
  idna >= 2.10
  imageio >= 2.9.0
  inflect >= 5.3.0
  iniconfig >= 1.1.1
  itsdangerous >= 2.0.0
  joblib >= 1.0.1
  jsonschema >= 3.2.0
  kiwisolver >= 1.3.1
  liberator >= 0.0.1
  markdown >= 3.3.4
  munch >= 2.5.0
  oauthlib >= 3.1.0
  ordered_set >= 4.0.2
  packaging >= 20.9
  pluggy >= 0.13.1
  py >= 1.10.0
  pyasn1 >= 0.4.8
  pyasn1_modules >= 0.2.8
  pyflakes >= 2.3.1
  pyparsing >= 2.4.7
  pyqtree >= 1.0.0
  pyrsistent >= 0.17.3
  pystac_client >= 0.1.1
  python_dateutil >= 2.8.1
  python_dotenv >= 0.17.1
  pytorch-ranger >= 0.1.1
  pytz >= 2021.1
  requests_oauthlib >= 1.3.0
  requests_toolbelt >= 0.9.1
  rsa >= 4.7.2
  six >= 1.16.0
  snuggs >= 1.4.7
  tabulate >= 0.8.9
  tensorboard_data_server >= 0.6.1
  tensorboard_plugin_wit >= 1.8.0
  threadpoolctl >= 2.1.0
  torchvision >= 0.9.1
  tqdm >= 4.60.0
  typing >= extensions >= 3.10.0.0
  uritools >= 3.0.2
  urllib3 >= 1.26.4
  werkzeug >= 2.0.0
  sortedcontainers >= 2.3.0
  toml >= 0.10.2
  pyyaml >= 5.4.1
"""


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
                if line.startswith('-r file:'):
                    req_file = line.split('file:')[-1]
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


def trace_all_deps(defined_req_lines):
    """
    TODO: make this work.

    The issue is that the packages dependencies need to be installed for this
    to correctly find the dependencies.

    pip install requirements-parser

    defined_req_lines = parse_conda_reqs('conda_env.yml')
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

    lines = ['# Defined'] + toplevel + ['', '# Implied'] + remain
    return lines


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
    new_lines = trace_all_deps(defined_req_lines)
    new_lines = [header2, ''] + new_lines
    new_text = ('\n'.join(new_lines))
    with open('requirements/autogen/all-explicit.txt', 'w') as file:
        file.write(new_text)
    print(new_text)


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
