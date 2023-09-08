def mwe_trace():
    import torchvision
    from torchview import draw_graph
    import torch
    import networkx as nx

    def model_layers(model):
        """ Extract named "leaf" layers from a module """
        stack = [('', '', model)]
        while stack:
            prefix, basename, item = stack.pop()
            name = '.'.join([p for p in [prefix, basename] if p])
            if isinstance(item, torch.nn.modules.conv._ConvNd):
                yield name, item
            elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
                yield name, item
            elif hasattr(item, 'reset_parameters'):
                yield name, item

            child_prefix = name
            for child_basename, child_item in list(
                    item.named_children())[::-1]:
                stack.append((child_prefix, child_basename, child_item))

    # Create example network
    net = torchvision.models.resnet18()
    model_graph = draw_graph(net, input_size=(2, 3, 224, 224), device='meta')

    # Remember the dotted layer name associated with each torch.Module
    # instance.  Usually a module will just have one name associated to an
    # instance, but it could have more than one.
    from collections import defaultdict
    named_layers = list(model_layers(net))
    id_to_names = defaultdict(list)
    for name, layer in named_layers:
        layer_id = id(layer)
        id_to_names[layer_id].append(name)

    def make_label(n, data):
        """ Create a nice printable label """
        n_id = id(n)
        n_id_str = str(n_id)
        parts = []
        if 'layer_name' in data:
            parts.append(data['layer_name'] + ':')
        parts.append(n.name)
        if n_id_str in model_graph.id_dict:
            idx = model_graph.id_dict[n_id_str]
            parts.append(f':{idx}')

        if n_id in id_to_names:
            parts.append(' ' + id_to_names[n_id])

        label = ''.join(parts)
        return label

    # Build a networkx version of the torchview model graph
    graph = nx.DiGraph()
    for node in model_graph.node_set:
        graph.add_node(node)

    for u, v in model_graph.edge_list:
        u_id = id(u)
        v_id = id(v)
        graph.add_edge(u_id, v_id)
        graph.nodes[u_id]['compute_node'] = u
        graph.nodes[v_id]['compute_node'] = v

    # Enrich each node with more info
    for n_id, data in graph.nodes(data=True):
        if 'compute_node' in data:
            n = data['compute_node']
            if hasattr(n, 'compute_unit_id'):
                if n.compute_unit_id in id_to_names:
                    layer_names = id_to_names[n.compute_unit_id]
                    if len(layer_names) == 1:
                        data['layer_name'] = layer_names[0]
                    else:
                        data['layer_names'] = layer_names[0]
            data['label'] = make_label(n, data)

    nx.write_network_text(graph, vertical_chains=1)
    # model_graph.visual_graph.view()

    # Now that we have a graph where a subset of nodes correspond to known
    # layers, we can postprocess it to only show effective connections between
    # the layers.

    # Determine which nodes have associated layer names
    remove_ids = []
    keep_ids = []
    for n_id, data in graph.nodes(data=True):
        if 'layer_name' in data:
            keep_ids.append(n_id)
        else:
            remove_ids.append(n_id)

    import ubelt as ub
    topo_order = ub.OrderedSet(nx.topological_sort(graph))
    keep_topo_order = (topo_order & keep_ids)

    # Find the nearest ancestor that we want to view and collapse the node we
    # dont care about into it. Do a final relabeling to keep the original node
    # ids where possible.
    collapseables = defaultdict(list)
    for n in remove_ids:
        valid_prev_nodes = keep_topo_order & set(nx.ancestors(graph, n))
        if valid_prev_nodes:
            p = valid_prev_nodes[-1]
            collapseables[p].append(n)
    from networkx.algorithms.connectivity.edge_augmentation import collapse
    grouped_nodes = []
    for p, vs in collapseables.items():
        grouped_nodes.append([p, *vs])
    g2 = collapse(graph, grouped_nodes)
    relabel = {n: n for n in g2.nodes}
    new_to_olds = ub.udict(g2.graph['mapping']).invert(unique_vals=0)
    for new, olds in new_to_olds.items():
        if len(olds) == 1:
            old = ub.peek(olds)
            relabel[new] = old
        else:
            keep_olds = keep_topo_order & olds
            old = ub.peek(keep_olds)
            relabel[new] = old
    g3 = nx.relabel_nodes(g2, relabel)

    def transfer_data(g_dst, g_src):
        for n in set(g_dst.nodes) & set(g_src.nodes):
            g_dst.nodes[n].update(g_src.nodes[n])

    # Show the collapsed graph
    transfer_data(g3, graph)
    nx.write_network_text(g3, vertical_chains=1)

    # Further reduce the graph to remove skip connection information
    g4 = nx.transitive_reduction(g3)
    transfer_data(g4, graph)
    nx.write_network_text(g4, vertical_chains=1)

    g2 = nx.transitive_closure(graph)
    g2 = nx.transitive_reduction(g2)
    transfer_data(g2, graph)

