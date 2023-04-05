"""
Mirror the watch package
"""
from watch import *  # NOQA

__autogen__ = """

    import networkx as nx
    import ubelt as ub
    import xdev

    import watch
    base = ub.Path(watch.__file__).parent
    repo_dpath = base.parent

    path_to_info = ub.ddict(dict)
    g = nx.DiGraph()
    g.add_node(base, label=base.name)
    for root, dnames, fnames in base.walk():
        g.add_node(root, label=root.name)
        root_info = path_to_info[root]
        # dnames[:] = [d for d in dnames if not dname_block_pattern.match(d)]
        root_info['has_init'] = '__init__.py' in fnames
        if not root_info['has_init']:
            dnames.clear()
            continue

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
        node_data['label'] = ub.color_text(node_data['label'], 'green')

    nx.write_network_text(g)

    for node in g.nodes():
        relpath = node.relative_to(base)
        if str(relpath) != '.':
            new_dpath = repo_dpath / 'geowatch' / relpath
            new_dpath.ensuredir()
            fpath = new_dpath / '__init__.py'
            modname = ub.modpath_to_modname(node)
            fpath.write_text(f'from {modname} import *  # NOQA')
"""
