def main():
    import ubelt as ub
    import watch
    watch_repo_dpath = ub.Path(watch.__file__).parent
    import networkx as nx
    import xdev
    dname_block_pattern = xdev.MultiPattern.coerce([
        # 'modules',
        '_*',
        '.*',
    ])

    # fname_block_pattern = xdev.MultiPattern.coerce([
    #     '_*',
    #     '.*',
    # ])

    from cmd_queue import util
    base = watch_repo_dpath
    path_to_info = ub.ddict(dict)
    g = nx.DiGraph()
    g.add_node(base, label=base.name)
    for root, dnames, fnames in base.walk():
        g.add_node(root, label=root.name)
        root_info = path_to_info[root]
        dnames[:] = [d for d in dnames if not dname_block_pattern.match(d)]
        # fnames[:] = [f for f in fnames if not fname_block_pattern.match(f)]
        root_info['num_good_files'] = len(fnames)
        root_info['num_good_dirs'] = len(dnames)
        root_info['ext_hist'] = ub.dict_hist([ub.Path(f).suffix for f in fnames])
        # root_info['ext_hist']['__init__.py'] = '__init__.py' in fnames
        root_info['has_init'] = '__init__.py' in fnames

        for d in dnames:
            dpath = root / d
            g.add_node(dpath, label=dpath.name)
            g.add_edge(root, dpath)

    for p in list(g.nodes):
        root_info = path_to_info[p]
        if not root_info.get('has_init', False):
            g.remove_node(p)

    for p in list(g.nodes):
        node_data = g.nodes[p]
        root_info = path_to_info[p]
        node_data['label'] = ub.color_text(node_data['label'], 'green') + ' ' + ub.repr2(root_info['ext_hist'], nl=0)

    print(util.graph_str(g, sources=[base]))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/dev/repo_structure_for_readme.py
    """
    main()
