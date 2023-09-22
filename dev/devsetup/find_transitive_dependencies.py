# Quick and dirty way to grab potential transitive deps


def find_transitive_dependencies(repo_dpath):
    """
    import pathlib
    repo_dpath = "."
    """
    import setup
    import pathlib
    req_dpath = pathlib.Path(repo_dpath) / 'requirements'

    import ubelt as ub
    repo_dpath = ub.Path(repo_dpath)
    assert req_dpath.exists()

    req_fpaths = list(req_dpath.glob('*'))
    req_fpaths = [r for r in req_fpaths if 'transitive' not in r.stem]

    tmp_dpath = (repo_dpath / 'tmp_deps').ensuredir()

    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=16)
    for req_fpath in req_fpaths:
        req_name = req_fpath.stem
        dldir = (tmp_dpath / req_name).ensuredir()
        command = f'pip download --prefer-binary -r {req_fpath} --dest {dldir}'
        print(command)
        queue.submit(command)
    queue.print_commands()
    queue.run()

    import re
    pat = re.compile('[=><~]')
    req_to_names = {}
    req_to_closures = {}
    for req_fpath in req_fpaths:
        req_name = req_fpath.stem
        versions = setup.parse_requirements(req_fpath, versions='strict')
        existing = set([pat.split(v)[0].split('[')[0] for v in versions])
        req_to_names[req_name] = existing

        dldir = (tmp_dpath / req_name)

        closure_items = []
        pkgs = sorted(dldir.glob('*'))
        for fpath in (pkgs):
            parts = fpath.name.split('-')
            name = parts[0]
            version = parts[1]
            closure_items.append({
                'name': name,
                'version': version,
                'fname': fpath.name,
            })

        req_to_closures[req_name] = closure_items

    # print(chr(10).join(lines))

    import networkx as nx
    g = nx.DiGraph()

    for req_fpath in req_fpaths:
        req_name = req_fpath.stem
        g.add_node(req_name)

    for req_fpath in req_fpaths:
        req_name = req_fpath.stem
        if req_name != 'runtime':
            g.add_edge('runtime', req_name)

        g.add_edge(req_name, req_name + '_transitive')

    req_to_names = ub.udict(req_to_names)
    req_to_names = req_to_names.subdict(['runtime']) | req_to_names

    req_to_closures = ub.udict(req_to_closures)
    req_to_closures = req_to_closures.subdict(['runtime']) | req_to_closures

    for req_name, existing in req_to_names.items():
        print(f'req_name={req_name}')
        for pkgname in existing:
            if pkgname not in g.nodes:
                g.add_edge(req_name, pkgname)

    for req_name, rows in req_to_closures.items():
        for row in rows:
            if row['name'] not in g.nodes:
                g.add_node(row['name'])
                g.add_edge(f'{req_name}_transitive', row['name'])

    nx.write_network_text(g)
