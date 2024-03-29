#!/usr/bin/env python3
"""
Determine Transitive Dependencies from requirements.txt files
"""
import scriptconfig as scfg
import ubelt as ub


class FindTransitiveDependenciesCLI(scfg.DataConfig):
    requirements = scfg.Value([], nargs='+', help='one or more requirement files')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from find_transitive_dependencies import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict(requirements='requirements/*.txt')
            >>> cls = FindTransitiveDependenciesCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        import kwutil
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=2))
        req_fpaths = kwutil.util_path.coerce_patterned_paths(config.requirements)
        print(f'req_fpaths = {ub.urepr(req_fpaths, nl=1)}')
        find_transitive_dependencies(req_fpaths)


def find_transitive_dependencies(req_fpaths):
    """
    import pathlib
    repo_dpath = "."

    TODO:
        - [ ] allow strict versions
        - [ ] prevent duplicate downloads
        - [ ] reduce network usage if possible
    """
    import setup
    import ubelt as ub
    # repo_dpath = ub.Path(repo_dpath)

    # Exclude paths that have a "transitive" in the filename
    req_fpaths = [r for r in req_fpaths if 'transitive' not in r.stem]

    USE_TEMPFILE = 0
    if USE_TEMPFILE:
        import tempfile
        tempdir = tempfile.TemporaryDirectory()
        tmp_root = ub.Path(tempdir.name)
        tmp_dpath = (tmp_root / 'tmp_deps').ensuredir()
    else:
        tmp_dpath = ub.Path.appdir('geowatch/transitive_dependencies')
        tmp_dpath.ensuredir()

    root_dl_dpath = (tmp_dpath / 'packages').ensuredir()
    modified_req_dpath = (tmp_dpath / 'modified_requirements').ensuredir()

    # TODO: allow strict versions
    mode = 'strict'

    modified_infos = []
    # Actually download the requirement wheels
    import cmd_queue
    queue = cmd_queue.Queue.create(backend='tmux', size=16)
    for req_fpath in ub.oset(req_fpaths):
        req_text = req_fpath.read_text()
        if mode == 'strict':
            req_text = req_text.replace('>=', '==')
        hashid = ub.hash_data(req_text, hasher='sha256')
        new_stem = f'{req_fpath.stem}-{hashid}'
        modified_req_fpath = modified_req_dpath / (new_stem + '.txt')
        modified_req_fpath.write_text(req_text)
        dldir = (root_dl_dpath / new_stem).ensuredir()
        command = f'pip download --prefer-binary -r {modified_req_fpath} --dest {dldir}'

        modified_infos.append({
            'req_name': req_fpath.stem,
            'req_fpath': modified_req_fpath,
            'dldir': dldir,
        })
        print(command)
        queue.submit(command)
    queue.print_commands()
    queue.run()

    # Parse the output
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

    import sys
    major, minor = sys.version_info[0:2]
    pyver = f'cp{major}{minor}'
    pypart = f"python_version < '{major}.{minor + 1}'  and python_version >= '{major}.{minor}'"

    all_lines = []
    for fpath in req_fpaths:
        closures = req_to_closures[fpath.stem]
        new_lines = []
        for row in closures:
            line = row['name'] + '==' + row['version'] + '; ' + pypart
            new_lines.append(line)
        all_lines += new_lines
        new_text = '\n'.join(new_lines)
        transitive_fpath = fpath.augment(stemsuffix=f'-{pyver}-{mode}-transitive')
        transitive_fpath.write_text(new_text)

    all_fpath = ub.Path(f'all-{pyver}-{mode}-transitive.txt')
    all_text = '\n'.join(ub.unique(sorted(all_lines)))
    all_fpath.write_text(all_text)

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


__cli__ = FindTransitiveDependenciesCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/geowatch/dev/devsetup/find_transitive_dependencies.py
        python -m find_transitive_dependencies
    """
    main()
