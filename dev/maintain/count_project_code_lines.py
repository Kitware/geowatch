def main():
    import networkx as nx
    import ubelt as ub

    module_name = 'watch'
    og = module_code_analysis(module_name)
    nx.write_network_text(og)
    if True:

        module = ub.import_module_from_name(module_name)
        module_dpath = ub.Path(module.__file__).parent

        nx.write_network_text(og, max_depth=2)

        nx.write_network_text(og, max_depth=2, sources={(module_dpath / 'tasks')})

        nx.write_network_text(og, max_depth=2, sources={(module_dpath / 'tasks' / 'fusion')})

        nx.write_network_text(og, max_depth=3, sources={(module_dpath / 'tasks' / 'depth_pcd')})

        nx.write_network_text(og, max_depth=3, sources={(module_dpath / 'tasks' / 'depth')})

        nx.write_network_text(og, max_depth=3, sources={(module_dpath / 'tasks' / 'rutgers_material_seg')})

        nx.write_network_text(og, max_depth=3, sources={(module_dpath / 'utils')})

        total = og.nodes[module_dpath]['stats']
        print('total = {}'.format(ub.urepr(total, nl=1)))

        notebooks = []
        for node in og.nodes:
            if '_notebook' in node.name:
                notebooks.append(node)
        tpl_list = [
            (module_dpath / 'tasks' / 'fusion' / 'experiments'),
            (module_dpath / 'tasks' / 'depth_pcd' / 'tpl'),
            (module_dpath / 'tasks' / 'depth' / 'modules'),
            (module_dpath / 'tasks' / 'sam' / 'tpl'),
            (module_dpath / 'tasks' / 'rutgers_material_seg' / 'scripts'),
        ]
        exclude = tpl_list + notebooks
        import pandas as pd
        extra_complexity = pd.DataFrame([v['stats'] for v in (ub.udict(og.nodes) & exclude).values()]).sum().to_dict()
        print('extra_complexity = {}'.format(ub.urepr(extra_complexity, nl=1)))

        """
        Based on Jack's question about lines of code we have, I wanted to take a
        deeper look.  I have code that already walks the module and builds a
        networkx graph of the directory structure, so it wasn't too much extra work
        to look at each file and count:

            * the total number of lines
            * the number of code lines (after empty lines, comments, and docstrings are removed)
            * the number of documentation lines (the number of lines in docstrings)

        Looking at this on a tree level also means we can get a sense of how much
        complexity each component is adding (e.g. dzyne includes a lot of third
        party libraries like lydorn utils that are mostly unused).

        To summarize...

        The entire module is:

            * 233,093 lines
            * 138,545 real code lines
            *  35,587 docstring lines

        The "extra complexity code" (which is things like my notebook files, the
        vendored tpl libraries, and misc stuff that should be cleaned up)
        contributes:

            *  38,503 lines
            *  23,567 code lines
            *   5,232 docstring lines
        """

    import networkx as nx
    import ubelt as ub

    supporting_modules = [
        'kwcoco',
        'kwimage',
        'kwarray',
        'kwutil',
        'kwplot',
        'scriptconfig',
        'cmd_queue',
        'delayed_image',
        'ndsampler',
    ]
    rows = []
    for module_name in supporting_modules:
        og = module_code_analysis(module_name)
        module = ub.import_module_from_name(module_name)
        module_dpath = ub.Path(module.__file__).parent
        nx.write_network_text(og, max_depth=1)
        row = og.nodes[module_dpath]['stats'].copy()
        row = {'name': module_name} | row
        rows.append(row)

    import pandas as pd
    df = pd.DataFrame(rows)
    print(df)
    print('')
    print(df.sum().drop('name'))

    """
    kwcoco: total_lines=32403,code_lines=16440,doc_lines=9800
    kwimage: total_lines=31736,code_lines=13022,doc_lines=14091
    kwarray: total_lines=9654,code_lines=3622,doc_lines=4553
    kwutil: total_lines=3939,code_lines=1796,doc_lines=1494
    kwplot: total_lines=5087,code_lines=2717,doc_lines=1310
    scriptconfig: total_lines=3923,code_lines=1682,doc_lines=1525
    cmd_queue: total_lines=6112,code_lines=2985,doc_lines=2081
    delayed_image: total_lines=8365,code_lines=3399,doc_lines=3677
    ndsampler: total_lines=8342,code_lines=4159,doc_lines=2608
    """


def module_code_analysis(module_name):
    import networkx as nx
    import ubelt as ub

    # old_name = 'watch'
    # module_name = 'kwcoco'
    module = ub.import_module_from_name(module_name)
    module_dpath = ub.Path(module.__file__).parent

    g = nx.DiGraph()
    g.add_node(module_dpath, label=module_dpath.name, type='dir')

    for root, dnames, fnames in module_dpath.walk():
        # dnames[:] = [d for d in dnames if not dname_block_pattern.match(d)]
        dnames[:] = [d for d in dnames if not d == '__pycache__']
        # if '__init__.py' not in fnames:
        #     dnames.clear()
        #     continue

        g.add_node(root, name=root.name, label=root.name, type='dir')
        if root != module_dpath:
            g.add_edge(root.parent, root)

        # for d in dnames:
        #     dpath = root / d
        #     g.add_node(dpath, label=dpath.name)
        #     g.add_edge(root, dpath)

        for f in fnames:
            if f.endswith('.py'):
                fpath = root / f
                g.add_node(fpath, name=fpath.name, label=fpath.name, type='file')
                g.add_edge(root, fpath)

    for p in list(g.nodes):
        node_data = g.nodes[p]
        ntype = node_data.get('type', None)
        if ntype == 'dir':
            node_data['label'] = ub.color_text(node_data['label'], 'blue')
        elif ntype == 'file':
            node_data['label'] = ub.color_text(node_data['label'], 'green')

    # nx.write_network_text(g)

    for fpath, node_data in g.nodes(data=True):
        if node_data['type'] == 'file':
            text = fpath.read_text()
            stats = parse_python_code_stats(text)
            node_data['stats'] = stats

    stat_keys = ['total_lines', 'code_lines', 'doc_lines']

    ### Iterate from leaf-to-root, and accumulate info in directories
    node_order = list(nx.topological_sort(g))[::-1]
    for node in node_order:
        children = g.succ[node]
        node_data = g.nodes[node]

        if node_data['type'] == 'dir':
            node_data['stats'] = accum_stats = {k: 0 for k in stat_keys}
            for child in children:
                child_data = g.nodes[child]
                child_stats = child_data.get('stats', {})
                for key in stat_keys:
                    accum_stats[key] += child_stats.get(key, 0)

        stats = node_data['stats']

        if ntype == 'dir':
            node_data['label'] = ub.color_text(node_data['name'], 'blue') + ': ' + ub.urepr(stats, nl=0, compact=1)
        elif ntype == 'file':
            node_data['label'] = ub.color_text(node_data['name'], 'green') + ': ' + ub.urepr(stats, nl=0, compact=1)

    ordered_nodes = dict(g.nodes(data=True))
    ordered_edges = []
    for node in node_order:
        # Sort children by total lines
        children = g.succ[node]
        children = ub.udict({c: g.nodes[c] for c in children})
        children = children.sorted_keys(lambda c: (g.nodes[c]['type'], g.nodes[c]['stats'].get('total_lines', 0)), reverse=True)
        for c, d in children.items():
            ordered_nodes.pop(c, None)
            ordered_nodes[c] = d
            ordered_edges.append((node, c))

        # ordered_nodes.update(children)

    assert not (set(g.edges) - set(ordered_edges))
    og = nx.DiGraph()
    og.add_nodes_from(ordered_nodes.items())
    og.add_edges_from(ordered_edges)
    return og


def parse_python_code_stats(text):
    raw_code = strip_comments_and_newlines(text)

    total_lines = text.count('\n')
    code_lines = raw_code.count('\n')

    # from xdoctest.core import package_calldefs
    from xdoctest.static_analysis import TopLevelVisitor
    self = TopLevelVisitor.parse(text)
    calldefs = self.calldefs

    total_doclines = 0
    for k, v in calldefs.items():
        if v.docstr is not None:
            total_doclines += v.docstr.count('\n')

    stats = {
        'total_lines': total_lines,
        'code_lines': code_lines,
        'doc_lines': total_doclines,
    }
    return stats


def strip_comments_and_newlines(source):
    """
    Removes hashtag comments from underlying source

    Args:
        source (str | List[str]):

    CommandLine:
        xdoctest -m xdoctest.static_analysis _strip_hashtag_comments_and_newlines

    TODO:
        would be better if this was some sort of configurable minify API

    Example:
        >>> from xdoctest.static_analysis import _strip_hashtag_comments_and_newlines
        >>> from xdoctest import utils
        >>> fmtkw = dict(sss=chr(39) * 3, ddd=chr(34) * 3)
        >>> source = utils.codeblock(
        >>>    '''
               # comment 1
               a = '# not a comment'  # comment 2

               multiline_string = {ddd}

               one

               {ddd}
               b = [
                   1,  # foo


                   # bar
                   3,
               ]
               c = 3
               ''').format(**fmtkw)
        >>> non_comments = _strip_hashtag_comments_and_newlines(source)
        >>> print(non_comments)
        >>> assert non_comments.count(chr(10)) == 10
        >>> assert non_comments.count('#') == 1
    """
    import tokenize
    if isinstance(source, str):
        import io
        f = io.StringIO(source)
        readline = f.readline
    else:
        readline = iter(source).__next__

    def strip_hashtag_comments(tokens):
        """
        Drop comment tokens from a `tokenize` stream.
        """
        return (t for t in tokens if t[0] != tokenize.COMMENT)

    def strip_consecutive_newlines(tokens):
        """
        Consecutive newlines are dropped and trailing whitespace

        Adapated from: https://github.com/mitogen-hq/mitogen/blob/master/mitogen/minify.py#L65
        """
        prev_typ = None
        prev_end_col = 0
        skipped_rows = 0
        for token_info in tokens:
            typ, tok, (start_row, start_col), (end_row, end_col), line = token_info
            if typ in (tokenize.NL, tokenize.NEWLINE):
                if prev_typ in (tokenize.NL, tokenize.NEWLINE, None):
                    skipped_rows += 1
                    continue
                else:
                    start_col = prev_end_col
                end_col = start_col + 1
            prev_typ = typ
            prev_end_col = end_col
            yield typ, tok, (start_row - skipped_rows, start_col), (end_row - skipped_rows, end_col), line

    tokens = tokenize.generate_tokens(readline)
    tokens = strip_hashtag_comments(tokens)
    tokens = strip_docstrings(tokens)
    tokens = strip_consecutive_newlines(tokens)
    new_source = tokenize.untokenize(tokens)
    return new_source


def strip_docstrings(tokens):
    """
    Replace docstring tokens with NL tokens in a `tokenize` stream.

    Any STRING token not part of an expression is deemed a docstring.
    Indented docstrings are not yet recognised.
    """
    import tokenize
    stack = []
    state = 'wait_string'
    for t in tokens:
        typ = t[0]
        if state == 'wait_string':
            if typ in (tokenize.NL, tokenize.COMMENT):
                yield t
            elif typ in (tokenize.DEDENT, tokenize.INDENT, tokenize.STRING):
                stack.append(t)
            elif typ == tokenize.NEWLINE:
                stack.append(t)
                start_line, end_line = stack[0][2][0], stack[-1][3][0] + 1
                for i in range(start_line, end_line):
                    yield tokenize.NL, '\n', (i, 0), (i, 1), '\n'
                for t in stack:
                    if t[0] in (tokenize.DEDENT, tokenize.INDENT):
                        yield t[0], t[1], (i + 1, t[2][1]), (i + 1, t[3][1]), t[4]
                del stack[:]
            else:
                stack.append(t)
                for t in stack:
                    yield t
                del stack[:]
                state = 'wait_newline'
        elif state == 'wait_newline':
            if typ == tokenize.NEWLINE:
                state = 'wait_string'
            yield t
