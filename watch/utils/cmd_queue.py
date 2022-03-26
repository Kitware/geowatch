import ubelt as ub


class Job(ub.NiceRepr):
    """
    Base class for a job
    """
    def __init__(self, command=None, name=None, depends=None, **kwargs):
        if depends is not None and not ub.iterable(depends):
            depends = [depends]
        self.name = name
        self.command = command
        self.depends = depends
        self.kwargs = kwargs

    def __nice__(self):
        return self.name


class Queue(ub.NiceRepr):
    """
    Base class for a queue
    """

    def submit(self, command, depends=None, **kwargs) -> Job:
        job = Job(...)
        return job

    def sync(self):
        """
        Force all subsequent commands
        """
        pass

    @classmethod
    def create(cls, type='serial', **kwargs):
        from watch.utils import tmux_queue
        from watch.utils import serial_queue
        from watch.utils import slurm_queue
        if type == 'serial':
            kwargs.pop('size', None)
            self = serial_queue.SerialQueue(**kwargs)
        elif type == 'tmux':
            self = tmux_queue.TMUXMultiQueue(**kwargs)
        elif type == 'slurm':
            kwargs.pop('size', None)
            self = slurm_queue.SlurmQueue(**kwargs)
        else:
            raise KeyError
        return self

    def _dependency_graph(self):
        """
        Builds a networkx dependency graph for the current jobs

        Example:
            >>> from watch.utils.tmux_queue import *  # NOQA
            >>> self = TMUXMultiQueue(5, 'foo')
            >>> job1a = self.submit('echo hello && sleep 0.5')
            >>> job1b = self.submit('echo hello && sleep 0.5')
            >>> job2a = self.submit('echo hello && sleep 0.5', depends=[job1a])
            >>> job2b = self.submit('echo hello && sleep 0.5', depends=[job1b])
            >>> job3 = self.submit('echo hello && sleep 0.5', depends=[job2a, job2b])
            >>> jobX = self.submit('echo hello && sleep 0.5', depends=[])
            >>> jobY = self.submit('echo hello && sleep 0.5', depends=[jobX])
            >>> jobZ = self.submit('echo hello && sleep 0.5', depends=[jobY])
            >>> graph = self._dependency_graph()
        """
        import networkx as nx
        graph = nx.DiGraph()
        for index, job in enumerate(self.jobs):
            graph.add_node(job.name, job=job, index=index)
        for index, job in enumerate(self.jobs):
            if job.depends:
                for dep in job.depends:
                    graph.add_edge(dep.name, job.name)
        return graph


def graph_str(graph, with_labels=True, sources=None, write=None, ascii_only=False):
    """
    Attempt extension of forest_str

    Example:
        >>> from watch.utils.tmux_queue import *  # NOQA

        >>> import networkx as nx
        >>> graph = nx.DiGraph()
        >>> graph.add_nodes_from(['a', 'b', 'c', 'd', 'e'])
        >>> graph.add_edges_from([
        >>>     ('a', 'd'),
        >>>     ('b', 'd'),
        >>>     ('c', 'd'),
        >>>     ('d', 'e'),
        >>>     ('f1', 'g'),
        >>>     ('f2', 'g'),
        >>> ])
        >>> graph_str(graph, write=print)
        >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        >>> print('\nForest Str')
        >>> print(nx.forest_str(graph))
        >>> print('\nGraph Str (should be identical)')
        >>> print(graph_str(graph))
        >>> print('modified')
        >>> graph.add_edges_from([
        >>>     (7, 1)
        >>> ])
        >>> graph_str(graph, write=print)

        >>> print('\n\n next')
        >>> graph = nx.balanced_tree(r=2, h=3, create_using=nx.DiGraph)
        >>> graph.add_edges_from([
        >>>     (7, 1),
        >>>     (2, 4),
        >>> ])
        >>> graph_str(graph, write=print)

        >>> print('\n\n next')
        >>> graph = nx.erdos_renyi_graph(5, 1.0)
        >>> graph_str(graph, write=print)
    """
    import networkx as nx

    printbuf = []
    if write is None:
        _write = printbuf.append
    else:
        _write = write

    # Define glphys
    # Notes on available box and arrow characters
    # https://en.wikipedia.org/wiki/Box-drawing_character
    # https://stackoverflow.com/questions/2701192/triangle-arrow
    if ascii_only:
        glyph_empty = "+"
        glyph_newtree_last = "+-- "
        glyph_newtree_mid = "+-- "
        glyph_endof_forest = "    "
        glyph_within_forest = ":   "
        glyph_within_tree = "|   "

        glyph_directed_last = "L-> "
        glyph_directed_mid = "|-> "

        glyph_undirected_last = "L-- "
        glyph_undirected_mid = "|-- "
    else:
        glyph_empty = "╙"
        glyph_newtree_last = "╙── "
        glyph_newtree_mid = "╟── "
        glyph_endof_forest = "    "
        glyph_within_forest = "╎   "
        glyph_within_tree = "│   "

        glyph_directed_last = "└─╼ "
        glyph_directed_mid = "├─╼ "
        glyph_directed_backedge = '╾'

        glyph_undirected_last = "└── "
        glyph_undirected_mid = "├── "
        glyph_undirected_backedge = '─'

    print('graph = {!r}'.format(graph))
    if len(graph.nodes) == 0:
        _write(glyph_empty)
    else:
        is_directed = graph.is_directed()
        print('is_directed = {!r}'.format(is_directed))

        if is_directed:
            glyph_last = glyph_directed_last
            glyph_mid = glyph_directed_mid
            glyph_backedge = glyph_directed_backedge
            succ = graph.succ
            pred = graph.pred
        else:
            glyph_last = glyph_undirected_last
            glyph_mid = glyph_undirected_mid
            glyph_backedge = glyph_undirected_backedge
            succ = graph.adj
            pred = graph.adj

        if sources is None:
            if is_directed:
                # use real source nodes for directed trees
                sources = [n for n in graph.nodes if graph.in_degree[n] == 0]
            else:
                # use arbitrary sources for undirected trees
                sources = []

            if len(sources) == 0:
                # no clear source, choose something
                sources = [
                    min(cc, key=lambda n: graph.degree[n])
                    for cc in nx.connected_components(graph)
                ]

        print('sources = {!r}'.format(sources))
        # Populate the stack with each source node, empty indentation, and mark
        # the final node. Reverse the stack so sources are popped in the
        # correct order.
        last_idx = len(sources) - 1
        stack = [(None, node, "", (idx == last_idx))
                 for idx, node in enumerate(sources)][::-1]

        seen_nodes = set()
        seen_edges = set()
        implicit_seen = set()
        while stack:
            parent, node, indent, this_islast = stack.pop()
            edge = (parent, node)

            if node is not Ellipsis:
                if node in seen_nodes:
                    continue
                seen_nodes.add(node)
                seen_edges.add(edge)

            if not indent:
                # Top level items (i.e. trees in the forest) get different
                # glyphs to indicate they are not actually connected
                if this_islast:
                    this_prefix = indent + glyph_newtree_last
                    next_prefix = indent + glyph_endof_forest
                else:
                    this_prefix = indent + glyph_newtree_mid
                    next_prefix = indent + glyph_within_forest

            else:
                # For individual tree edges distinguish between directed and
                # undirected cases
                if this_islast:
                    this_prefix = indent + glyph_last
                    next_prefix = indent + glyph_endof_forest
                else:
                    this_prefix = indent + glyph_mid
                    next_prefix = indent + glyph_within_tree

            if node is Ellipsis:
                label = ' ...'
                children = []
            else:
                if with_labels:
                    label = graph.nodes[node].get("label", node)
                else:
                    label = node
                children = [child for child in succ[node] if child not in seen_nodes]
                neighbors = set(pred[node])
                others = (neighbors - set(children)) - {parent}
                implicit_seen.update(others)
                in_edges = [(node, other) for other in others]
                if in_edges:
                    in_nodes = ', '.join([str(uv[1]) for uv in in_edges])
                    suffix = ' '.join(['', glyph_backedge, in_nodes])
                else:
                    suffix = ''

            _write(this_prefix + str(label) + suffix)

            # Push children on the stack in reverse order so they are popped in
            # the original order.
            idx = 1
            for idx, child in enumerate(children[::-1], start=1):
                next_islast = idx <= 1
                try_frame = (node, child, next_prefix, next_islast)
                stack.append(try_frame)

            if node in implicit_seen:
                # if have used this node in any previous implicit edges, then
                # write an outgoing "implicit" connection.
                next_islast = idx <= 1
                try_frame = (node, Ellipsis, next_prefix, next_islast)
                stack.append(try_frame)

    if write is None:
        # Only return a string if the custom write function was not specified
        return "\n".join(printbuf)
