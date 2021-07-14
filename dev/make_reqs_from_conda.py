def main():
    """
    Export pip requirements in a conda env file to a requirements.txt file
    """
    import yaml
    fpath = 'conda_env.yml'
    with open(fpath, 'r') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    context = []
    found = []
    for dep in data['dependencies']:
        if isinstance(dep, dict) and 'pip' in dep:
            for line in dep['pip']:
                if line.startswith('--'):
                    context.append(line)
                else:
                    found.append(' '.join([line] + context))

    text = '\n'.join(found)
    print(text)
    #with open('requirements.txt', 'w') as file:
    #    file.write(text)


def trace_all_deps():
    """
    TODO: make this work

    pip install requirements-parser

    """
    # import requirements
    # with open('requirements.txt', 'r') as fd:
    #     for req in requirements.parse(fd):
    #         print(req.name, req.specs)

    from pip._internal.req import parse_requirements
    from pip._internal.network.session import PipSession

    declared_deps = []
    for req in parse_requirements('requirements.txt', session=PipSession()):
        declared_deps.append(req.requirement.split(' ')[0])
    declared_deps = pkg_names

    import pipdeptree
    pkgs = pipdeptree.get_installed_distributions()
    tree = pipdeptree.PackageDAG.from_pkgs(pkgs)
    tree = tree.filter(declared_deps, [])

    # for pkg, deps in tree.items():
    #     pkg_label = '{0} >= {1}'.format(pkg.project_name, pkg.version)
    #     print(pkg_label)

    tree = tree.sort()
    nodes = tree.keys()
    branch_keys = set(r.key for r in pipdeptree.flatten(tree.values()))
    use_bullets = not frozen
    frozen = False
    list_all = True

    if not list_all:
        nodes = [p for p in nodes if p.key not in branch_keys]

    def aux(node, parent=None, indent=0, chain=None):
        chain = chain or []
        node_str = node.render(parent, frozen)
        if parent:
            prefix = ' '*indent + ('- ' if use_bullets else '')
            node_str = prefix + node_str

        node_dict = node.as_dict()
        yield node_dict
        for c in tree.get_children(node.key):
            if c.project_name not in chain:
                yield from aux(c, node, indent=indent+2, chain=chain+[c.project_name])

    items = list(pipdeptree.flatten([aux(p) for p in nodes]))
    unique_items = list(ub.oset([item['package_name'] for item in items]))

    declared_pkgs = sorted(set(declared_deps) | set(unique_items))
    declared_pkgs = sorted(set([p.replace('-', '_') for p in declared_pkgs]))
    print('declared_pkgs = {}'.format(ub.repr2(declared_pkgs, nl=1)))
    # print('unique_items = {}'.format(ub.repr2(unique_items, nl=1, sort=0)))
    # deptree = pipdeptree.render_json_tree(tree, indent=4)



if __name__ == '__main__':
    """
    python ~/code/watch/dev/make_reqs_from_conda.py > requirements.txt

    pip install pipdeptree

    pipdeptree -p ubelt

    pip install -r requirements.txt

    pip uninstall opencv-python opencv-python-headless  && pip install opencv-python-headless

    --no-deps
    # pip download -r requirements.txt -d ./tmp-deps --no-binary :all: -v
    """
    main()
