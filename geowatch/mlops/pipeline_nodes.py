"""
The core pipeline data structure for MLOps.

This module outlines the structure for a generic DAG of bash process nodes.  It
contains examples of generic test pipelines. For the SMART instantiation of
project-specific dags see: smart_pipeline.py

The basic idea is that each bash process knows about:

    * its filepath inputs
    * its filepath outputs
    * algorithm parameters
    * performance parameters
    * the command that invokes the job

Given a set of processes, a DAG is built by connecting process ouputs to
process inputs. This DAG can then be configured with customized input paths and
parameters. The resulting jobs can then be submitted to a cmd_queue.Queue for
actual execution.
"""
import functools
import networkx as nx
import os
import ubelt as ub
from functools import cached_property
from typing import Union, Dict, Set, List, Any, Optional
from geowatch.utils import util_dotdict

Collection = Optional[Union[Dict, Set, List]]
Configurable = Optional[Dict[str, Any]]


try:
    from xdev import profile  # NOQA
except ImportError:
    profile = ub.identity


class Pipeline:
    """
    A container for a group of nodes that have been connected.

    Allows these connected nodes to be jointly configured and submitted to a
    cmd-queue for execution. Adds extra bookkeeping jobs that write invoke.sh
    job_config.sh metadata as well as symlinks between node output directories.

    Example:
        >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
        >>> node_A1 = ProcessNode(name='node_A1', in_paths={'src'}, out_paths={'dst': 'dst.txt'}, executable='node_A1')
        >>> node_A2 = ProcessNode(name='node_A2', in_paths={'src'}, out_paths={'dst': 'dst.txt'}, executable='node_A2')
        >>> node_A3 = ProcessNode(name='node_A3', in_paths={'src'}, out_paths={'dst': 'dst.txt'}, executable='node_A3')
        >>> node_B1 = ProcessNode(name='node_B1', in_paths={'path1'}, out_paths={'path2': 'dst.txt'}, executable='node_B1')
        >>> node_B2 = ProcessNode(name='node_B2', in_paths={'path2'}, out_paths={'path3': 'dst.txt'}, executable='node_B2')
        >>> node_B3 = ProcessNode(name='node_B3', in_paths={'path3'}, out_paths={'path4': 'dst.txt'}, executable='node_B3')
        >>> node_C1 = ProcessNode(name='node_C1', in_paths={'src1', 'src2'}, out_paths={'dst1': 'dst.txt', 'dst2': 'dst.txt'}, executable='node_C1')
        >>> node_C2 = ProcessNode(name='node_C2', in_paths={'src1', 'src2'}, out_paths={'dst1': 'dst.txt', 'dst2': 'dst.txt'}, executable='node_C2')
        >>> # You can connect outputs -> inputs directly (RECOMMENDED)
        >>> node_A1.outputs['dst'].connect(node_A2.inputs['src'])
        >>> node_A2.outputs['dst'].connect(node_A3.inputs['src'])
        >>> # You can connect nodes to nodes that share input/output names (NOT RECOMMENDED)
        >>> node_B1.connect(node_B2)
        >>> node_B2.connect(node_B3)
        >>> #
        >>> # You can connect nodes to nodes that dont share input/output names
        >>> # If you specify the mapping (NOT RECOMMENDED)
        >>> node_A3.connect(node_B1, src_map={'dst': 'path1'})
        >>> #
        >>> # You can connect inputs to other inputs, which effectively
        >>> # forwards the input path to the destination
        >>> node_A1.inputs['src'].connect(node_C1.inputs['src1'])
        >>> # The pipeline is just a container for the nodes
        >>> nodes = [node_A1, node_A2, node_A3, node_B1, node_B2, node_B3, node_C1, node_C2]
        >>> self = Pipeline(nodes=nodes)
        >>> self.print_graphs()
    """

    def __init__(self, nodes=None, config=None, root_dpath=None):
        self.proc_graph = None
        self.io_graph = None
        if nodes is None:
            nodes = []
        self.nodes = nodes
        self.config = None

        self._dirty = True

        if self.nodes:
            self.build_nx_graphs()

        if config:
            self.configure(config, root_dpath=root_dpath)

    @classmethod
    def demo(cls):
        return demodata_pipeline()

    def _ensure_clean(self):
        if self._dirty:
            self.build_nx_graphs()

    def submit(self, executable, **kwargs):
        """
        Dynamically create a new unique process node and add it to the dag
        """
        task = ProcessNode(executable=executable, **kwargs)
        self.nodes.append(task)
        self._dirty = True
        return task

    @property
    def node_dict(self):
        if isinstance(self.nodes, dict):
            node_dict = self.nodes
        else:
            node_names = [node.name for node in self.nodes]
            if len(node_names) != len(set(node_names)):
                print('node_names = {}'.format(ub.urepr(node_names, nl=1)))
                raise AssertionError(f'{len(node_names)}, {len(set(node_names))}')
            node_dict = dict(zip(node_names, self.nodes))
        return node_dict

    @profile
    def build_nx_graphs(self):
        node_dict = self.node_dict

        self.proc_graph = nx.DiGraph()
        for name, node in node_dict.items():
            self.proc_graph.add_node(node.name, node=node)

            for s in node.successor_process_nodes():
                self.proc_graph.add_edge(node.name, s.name)

            for p in node.predecessor_process_nodes():
                self.proc_graph.add_edge(p.name, node.name)

        self.io_graph = nx.DiGraph()

        # Add nodes first
        for name, node in node_dict.items():
            self.io_graph.add_node(node.key, node=node, node_clsname=node.__class__.__name__)
            for iname, inode in node.inputs.items():
                self.io_graph.add_node(inode.key, node=inode, node_clsname=inode.__class__.__name__)
            for oname, onode in node.outputs.items():
                self.io_graph.add_node(onode.key, node=onode, node_clsname=onode.__class__.__name__)

        # Next add edges
        for name, node in node_dict.items():
            for iname, inode in node.inputs.items():
                self.io_graph.add_edge(inode.key, node.key)
                # Account for input/input connections
                for succ in inode.succ:
                    self.io_graph.add_edge(inode.key, succ.key)
            for oname, onode in node.outputs.items():
                self.io_graph.add_edge(node.key, onode.key)
                for oi_node in onode.succ:
                    self.io_graph.add_edge(onode.key, oi_node.key)
            # hack for nodes that dont have an io dependency
            # but still must run after one another
            for pred in node._pred_nodes_without_io_connection:
                self.io_graph.add_edge(pred.key, node.key)

        self._dirty = False

    @profile
    def inspect_configurables(self):
        """
        Show the user what config options should be specified.

        TODO:
            The idea is that we want to give the user a list of options that
            they could configure for this pipeline, as well as mark the one
            that are required / suggested / unnecessary. For now it gives a
            little bit of that information, but more work could be done to make
            it nicer.

        Example:
            >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
            >>> self = Pipeline.demo()
            >>> self.inspect_configurables()
        """
        import pandas as pd
        import rich
        from kwutil import util_yaml
        # Nodes don't always have full knowledge of their entire parameter
        # space, but they should at least have some knowledge of it.
        # Find required inputs
        # required_inputs = {
        #     n for n in self.io_graph if self.io_graph.in_degree[n] == 0
        # }

        rows = []
        for node in self.node_dict.values():
            # Build up information about each node option

            # TODO: determine if a source input node is required or not
            # by setting a required=False flags at the node leve.

            # Determine which inputs are connected vs unconnected
            for key, io_node in node.inputs.items():
                is_connected = self.io_graph.in_degree[io_node.key] > 0
                rows.append({
                    'node': node.name,
                    'key': key,
                    'connected': is_connected,
                    'type': 'in_path',
                    'maybe_required': not is_connected,
                })

            for key, io_node in node.outputs.items():
                is_connected = self.io_graph.out_degree[io_node.key] > 0
                rows.append({
                    'node': node.name,
                    'key': key,
                    'connected': is_connected,
                    'type': 'out_path',
                    'maybe_required': False,
                })

            for param in node.algo_params:
                rows.append({
                    'node': node.name,
                    'key': param,
                    'type': 'algo_param',
                    'maybe_required': True,
                })

            for param in node.perf_params:
                rows.append({
                    'node': node.name,
                    'key': param,
                    'type': 'perf_param',
                    'maybe_required': True,
                })

        df = pd.DataFrame(rows)
        df = df.sort_values(['maybe_required', 'type', 'node', 'key'], ascending=[False, True, True, True])
        rich.print(df.to_string())

        default = {}
        for _, row in df[df['maybe_required']].iterrows():
            default[row['node'] + '.' + row['key']] = None
        rich.print(util_yaml.Yaml.dumps(default))

    @profile
    def configure(self, config=None, root_dpath=None, cache=True):
        """
        Update the DAG configuration

        Example:
            >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
            >>> self = Pipeline.demo()
            >>> self.configure()
        """
        self._ensure_clean()

        if root_dpath is not None:
            root_dpath = ub.Path(root_dpath)
            self.root_dpath = root_dpath
            for node in self.node_dict.values():
                node.root_dpath = root_dpath
                node._configured_cache.clear()  # hack, make more elegant
        else:
            for node in self.node_dict.values():
                node._configured_cache.clear()  # hack, make more elegant

        if config is not None:
            self.config = config
            # print('CONFIGURE config = {}'.format(ub.urepr(config, nl=1)))

            # Set the configuration for each node in this pipeline.
            dotconfig = util_dotdict.DotDict(config)
            for node_name in nx.topological_sort(self.proc_graph):
                node = self.proc_graph.nodes[node_name]['node']
                node_config = dict(dotconfig.prefix_get(node.name, {}))
                node.configure(node_config, cache=cache)

    def print_graphs(self, shrink_labels=1, smart_colors=0):
        """
        Prints the Process and IO graph for the DAG.
        """
        self._ensure_clean()

        colors = ['bright_magenta', 'yellow', 'cyan']
        unused_colors = colors.copy()
        clsname_to_color = {
            'ProcessNode': 'yellow',
            'InputNode': 'bright_cyan',
            'OutputNode': 'bright_yellow',
        }

        def labelize_graph(graph, color_procs=0):
            # # self.io_graph.add_node(name + '.proc', node=node)
            all_names = []
            for _, data in graph.nodes(data=True):
                all_names.append(data['node'].name)

            if shrink_labels:
                ambiguous_names = list(ub.find_duplicates(all_names))

            for _, data in graph.nodes(data=True):

                if shrink_labels:
                    if data['node'].name in ambiguous_names:
                        data['label'] = data['node'].key
                    else:
                        data['label'] = data['node'].name
                else:
                    data['label'] = data['node'].key

                if color_procs:
                    clsname = data.get('node_clsname')
                    if clsname not in clsname_to_color:
                        if unused_colors:
                            color = unused_colors.pop()
                        else:
                            color = None
                        clsname_to_color[clsname] = color
                    color = clsname_to_color[clsname]
                    if color is not None:
                        label = data['label']
                        data['label'] = f'[{color}]{label}[/{color}]'

                elif smart_colors:
                    # SMART specific hack: remove later
                    if 'bas' in data['label']:
                        data['label'] = '[yellow]' + data['label']
                    elif 'sc' in data['label']:
                        data['label'] = '[cyan]' + data['label']
                    elif 'crop' in data['label']:
                        data['label'] = '[white]' + data['label']
                    elif 'building' in data['label']:
                        data['label'] = '[bright_magenta]' + data['label']
                    elif 'sv' in data['label']:
                        data['label'] = '[bright_magenta]' + data['label']
        labelize_graph(self.io_graph, color_procs=True)
        labelize_graph(self.proc_graph)

        import rich
        from cmd_queue.util import util_networkx
        print('')
        print('Process Graph')
        util_networkx.write_network_text(self.proc_graph, path=rich.print, end='', vertical_chains=True)

        print('')
        print('IO Graph')
        util_networkx.write_network_text(self.io_graph, path=rich.print, end='', vertical_chains=True)

    @profile
    def submit_jobs(self, queue=None, skip_existing=False, enable_links=True,
                    write_invocations=True, write_configs=True):
        """
        Submits the jobs to an existing command queue or creates a new one.

        Also takes care of adding special bookkeeping jobs that add helper
        files and symlinks to node output paths.
        """
        import cmd_queue
        import shlex
        import json
        import networkx as nx

        if queue is None:
            queue = {}

        if isinstance(queue, dict):
            # Create a simple serial queue if an existing one isn't given.
            default_queue_kw = {
                'backend': 'serial',
                'name': 'unnamed-mlops-pipeline',
                'size': 1,
                'gres': None,
            }
            queue_kw = ub.udict(default_queue_kw) | queue
            queue = cmd_queue.Queue.create(**queue_kw)

        node_order = list(nx.topological_sort(self.proc_graph))
        for node_name in node_order:
            node_data = self.proc_graph.nodes[node_name]
            try:
                node = node_data['node']
            except KeyError:
                import rich
                rich.print('[red]ERROR')
                print('node_name = {}'.format(ub.urepr(node_name, nl=1)))
                print('node_data = {}'.format(ub.urepr(node_data, nl=1)))
                raise
            node.will_exist = None

        summary = {
            'queue': queue,
            'node_status': {}
        }
        node_status = summary['node_status']

        for node_name in node_order:
            node = self.proc_graph.nodes[node_name]['node']
            # print('-----')
            # print(f'node_name={node_name}')
            # print(f'node.enabled={node.enabled}')
            if not node.enabled:
                node_status[node_name] = 'disabled'
                node.will_exist = node.does_exist
                continue

            pred_node_names = list(self.proc_graph.predecessors(node_name))
            pred_nodes = [
                self.proc_graph.nodes[n]['node']
                for n in pred_node_names
            ]

            ancestors_will_exist = all(n.will_exist for n in pred_nodes)
            if skip_existing and node.enabled != 'redo' and node.does_exist:
                node.enabled = False

            node.will_exist = ((node.enabled and ancestors_will_exist) or
                               node.does_exist)
            if 0:
                print(f'node.final_out_paths={node.final_out_paths}')
                print(f'Checking {node_name}, will_exist={node.will_exist}')

            skip_node = not (node.will_exist and node.enabled)

            if skip_node:
                node_status[node_name] = 'skipped'
            else:
                node_procid = node.process_id
                node_job = None

                # Another configuration may have submitted this job already
                if node_procid not in queue.named_jobs:
                    pred_node_procids = [n.process_id for n in pred_nodes
                                         if n.enabled]
                    # Submit a primary queue process
                    node_command = node.final_command()
                    node_job = queue.submit(command=node_command,
                                            depends=pred_node_procids,
                                            name=node_procid)
                    node_status[node_name] = 'new_submission'
                else:
                    # Some other config submitted this job, we can skip the
                    # rest of the work for this node.
                    node_status[node_name] = 'duplicate_submission'
                    continue

                # We might want to execute a few boilerplate instructions
                # before running each node.
                before_node_commands = []

                # Add symlink jobs that make the graph structure traversable in
                # the flat output directories.
                if enable_links:
                    # TODO: ability to bind jobs to be run in the same queue
                    # together

                    # TODO: should we filter the nodes where they are only linked
                    # via inputs?
                    for pred in node.predecessor_process_nodes():
                        link_path1 = pred.final_node_dpath / '.succ' / node.name / node.process_id
                        target_path1 = node.final_node_dpath
                        link_path2 = node.final_node_dpath / '.pred' / pred.name / pred.process_id
                        target_path2 = pred.final_node_dpath
                        target_path1 = os.path.relpath(target_path1.absolute(), link_path1.absolute().parent)
                        target_path2 = os.path.relpath(target_path2.absolute(), link_path2.absolute().parent)

                        parts = [
                            f'mkdir -p {link_path1.parent}',
                            f'mkdir -p {link_path2.parent}',
                            f'ln -sfT "{target_path1}" "{link_path1}"',
                            f'ln -sfT "{target_path2}" "{link_path2}"',
                        ]
                        # command = '(' + ' && '.join(parts) + ')'
                        before_node_commands.extend(parts)

                if write_invocations:
                    # Add a job that writes a file with the command used to
                    # execute this node.
                    # FIXME: this writes the file with a weird indentation.
                    # The effect is cosmetic, but not sure why its doing that.
                    invoke_fpath = node.final_node_dpath / 'invoke.sh'

                    invoke_lines = ['#!/bin/bash']
                    # TODO: can we topologically sort this?
                    depend_nodes = list(node.ancestor_process_nodes())
                    if depend_nodes:
                        invoke_lines.append('# See Also: ')
                        for depend_node in list(node.ancestor_process_nodes()):
                            invoke_lines.append('# ' + depend_node.final_node_dpath)
                    else:
                        invoke_lines.append('# Root node')
                    invoke_command = node._raw_command()
                    invoke_lines.append(invoke_command)
                    invoke_text = '\n'.join(invoke_lines)
                    command = '\n'.join([
                        f'mkdir -p {invoke_fpath.parent} && \\',
                        f'printf {shlex.quote(invoke_text)} \\',
                        f"> {invoke_fpath}",
                    ])
                    before_node_commands.append(command)

                if write_configs:
                    depends_config = node._depends_config()
                    # Add a job that writes a file with the command used to
                    # execute this node.
                    job_config_fpath = node.final_node_dpath / 'job_config.json'
                    json_text = json.dumps(depends_config)
                    if _has_jq():
                        command = '\n'.join([
                            f'mkdir -p {job_config_fpath.parent} && \\',
                            f"printf '{json_text}' | jq . > {job_config_fpath}",
                        ])
                    else:
                        command = '\n'.join([
                            f'mkdir -p {job_config_fpath.parent} && \\',
                            f"printf '{json_text}' > {job_config_fpath}",
                        ])
                    before_node_commands.append(command)

                if before_node_commands:
                    # TODO: nicer infastructure mechanisms (make the code
                    # prettier and easier to reason about)
                    before_command = ' && \\\n'.join(before_node_commands)
                    _procid = 'before_' + node_procid
                    if _procid not in queue.named_jobs:
                        _job = queue.submit(
                            command=before_command,
                            depends=pred_node_procids,
                            bookkeeper=1,
                            name=_procid,
                            tags=['boilerplate']
                        )
                        if node_job is not None:
                            node_job.depends.append(_job)

        # print(f'queue={queue}')
        return summary

    make_queue = submit_jobs


def glob_templated_path(template):
    """
    Given an unformated templated path, replace the format parts with "*" and
    return a glob.

    Args:
        template (str | PathLike): a path with a {} template pattern

    Example:
        template = '/foo{}/bar'
        glob_templated_path(template)
    """
    import parse
    from kwutil import util_pattern
    parser = parse.Parser(str(template))
    patterns = {n: '*' for n in parser.named_fields}
    pat = os.fspath(template).format(**patterns)
    mpat = util_pattern.Pattern.coerce(pat)
    fpaths = list(mpat.paths())
    return fpaths


@ub.memoize
def _has_jq():
    return ub.find_exe('jq')


class Node(ub.NiceRepr):
    """
    Abstract base class for a Process or IO Node.
    """

    __node_type__ = 'abstract'  # used to workaround IPython isinstance issues

    def __nice__(self):
        return f'{self.name!r}, pred={[n.name for n in self.pred]}, succ={[n.name for n in self.succ]}'

    def __init__(self, name: str):
        self.name = name
        self.pred = []
        self.succ = []

    def _connect_single(self, other, src_map, dst_map):
        """
        Handles connection rules between this node and another one.

        TODO: cleanup, these rules are too complex and confusing.
        There is a reasonable subset here; find and restrict to that.
        """
        # TODO: CLEANUP
        # print(f'Connect {type(self).__name__} {self.name} to '
        #       f'{type(other).__name__} {other.name}')
        if other not in self.succ:
            self.succ.append(other)
        if self not in other.pred:
            other.pred.append(self)

        self_is_proc = (self.__node_type__ == 'process')
        if self_is_proc:
            outputs = self.outputs
        else:
            assert self.__node_type__ == 'io'
            outputs = {self.name: self}

        other_is_proc = (other.__node_type__ == 'process')
        if other_is_proc:
            inputs = other.inputs
        else:
            assert other.__node_type__ == 'io'
            if not self_is_proc:
                # In this case we can make the name map implicit
                inputs = {self.name: other}
            else:
                inputs = {other.name: other}

        outmap = ub.udict({src_map.get(k, k): k for k in outputs.keys()})
        inmap = ub.udict({dst_map.get(k, k): k for k in inputs.keys()})

        common = outmap.keys() & inmap.keys()
        if len(common) == 0:
            print('inmap = {}'.format(ub.urepr(inmap, nl=1)))
            print('outmap = {}'.format(ub.urepr(outmap, nl=1)))
            raise Exception(f'Unknown io relationship {self.name=}, {other.name=}')

        if self_is_proc or other_is_proc:
            # print(f'Connect Process to Process {self.name=} to {other.name=}')
            self_output_keys = (outmap & common).values()
            other_input_keys = (inmap & common).values()

            for out_key, in_key in zip(self_output_keys, other_input_keys):
                out_node = outputs[out_key]
                in_node = inputs[in_key]
                # out_node._connect_single(in_node, src_map, dst_map)
                out_node._connect_single(in_node, {}, {})

    def connect(self, *others, param_mapping=None, src_map=None, dst_map=None):
        """
        Connect the outputs of ``self`` to the inputs of ``others``.

        Conceptually, this creates an edge between the two nodes.
        """
        # Connect these two nodes and return the original.
        if param_mapping is None:
            param_mapping = {}

        if src_map is None:
            src_map = param_mapping

        if dst_map is None:
            dst_map = param_mapping

        for other in others:
            self._connect_single(other, src_map, dst_map)

        return self

    @property
    def key(self):
        return self.name


class IONode(Node):
    __node_type__ = 'io'

    def __init__(self, name, parent):
        super().__init__(name)
        self.parent = parent
        self._final_value = None
        self._template_value = None

    @property
    def final_value(self):
        value = self._final_value
        if value is None:
            preds = list(self.pred)
            if preds:
                if len(preds) != 1:
                    # Handle Multi-Inputs
                    value = [p.final_value for p in preds]
                    # raise AssertionError(ub.paragraph(
                    #     f'''
                    #     Expected len(preds) == 1, but got {len(preds)}
                    #     {preds}
                    #     '''))
                else:
                    value = preds[0].final_value
        return value

    @final_value.setter
    def final_value(self, value):
        self._final_value = value

    @property
    def key(self):
        return self.parent.key + '.' + self.name


class InputNode(IONode):
    ...


class OutputNode(IONode):
    @property
    def final_value(self):
        # return self.parent._finalize_templates()['out_paths'][self.name]
        return self.parent.final_out_paths[self.name]

    @property
    def template_value(self):
        return self.parent.template_out_paths[self.name]

    @profile
    def matching_fpaths(self):
        """
        Find all paths for this node.
        """
        out_template = self.template_value
        return glob_templated_path(out_template)


def _classvar_init(self, args, fallbacks):
    """
    Helps initialize class instance variables from class variable defaults.

    Not sure what a good name for this is. The idea is that we will get a
    dictionary containing all init args, and anything that is None will be
    replaced by the class level attribute with the same name. Additionally, a
    dictionary of fallback defaults is used if the class variable is also None.

    Usage should look something like

    .. code:: python

        class MyClass:
            def __init__(self, a, b, c):
                args = locals()
                fallback = {
                    'a': [],
                }
                _classvar_init(self, args, fallbacks)

    """
    # Be careful of the magic '__class__' attribute that inherited classes will
    # get in the locals() of their `__init__` method. Workaround this by not
    # processing any '_'-prefixed name.
    cls = self.__class__
    for key, value in list(args.items()):
        if value is self or key.startswith('_'):
            continue
        if value is None:
            # Default to the class-level variable
            value = getattr(cls, key, value)
            if value is None:
                value = fallbacks.get(key, value)
            args[key] = value
        setattr(self, key, value)


class memoize_configured_method(object):
    """
    ubelt memoize_method but uses a special cache name
    """

    def __init__(self, func):
        self._func = func
        self._cache_name = '_cache__' + func.__name__
        # Mimic attributes of a bound method
        self.__func__ = func
        functools.update_wrapper(self, func)

    def __get__(self, instance, cls=None):
        """
        Descriptor get method. Called when the decorated method is accessed
        from an object instance.

        Args:
            instance (object): the instance of the class with the memoized method
            cls (type | None): the type of the instance
        """
        self._instance = instance
        return self

    def __call__(self, *args, **kwargs):
        """
        The wrapped function call
        """
        from ubelt.util_memoize import _make_signature_key
        func_cache = self._instance._configured_cache
        # func_cache = self._instance.__dict__
        cache = func_cache.setdefault(self._cache_name, {})
        key = _make_signature_key(args, kwargs)
        if key in cache:
            return cache[key]
        else:
            value = cache[key] = self._func(self._instance, *args, **kwargs)
            return value


def memoize_configured_property(fget):
    """
    ubelt memoize_property but uses a special cache name
    """
    # Unwrap any existing property decorator
    while hasattr(fget, 'fget'):
        fget = fget.fget

    attr_name = '_' + fget.__name__

    @functools.wraps(fget)
    def fget_memoized(self):
        cache = self._configured_cache
        if attr_name not in cache:
            cache[attr_name] = fget(self)
        return cache[attr_name]

    return property(fget_memoized)


# Uncomment for debugging
# memoize_configured_method = ub.identity
# memoize_configured_property = property


class ProcessNode(Node):
    """
    Represents a process in the pipeline.

    ProcessNodes are connected via their input / output nodes.

    You can create an instance of this directly, or inherit from it and set its
    class variables.

    CommandLine:
        xdoctest -m geowatch.mlops.pipeline_nodes ProcessNode

    Example:
        >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
        >>> from geowatch.mlops.pipeline_nodes import _classvar_init
        >>> dpath = ub.Path.appdir('geowatch/test/pipeline/TestProcessNode')
        >>> dpath.delete().ensuredir()
        >>> pycode = ub.codeblock(
        ...     '''
        ...     import ubelt as ub
        ...     src_fpath = ub.Path(ub.argval('--src'))
        ...     dst_fpath = ub.Path(ub.argval('--dst'))
        ...     foo = ub.argval('--foo')
        ...     bar = ub.argval('--bar')
        ...     new_text = foo + src_fpath.read_text() + bar
        ...     dst_fpath.write_text(new_text)
        ...     ''')
        >>> src_fpath = dpath / 'here.txt'
        >>> src_fpath.write_text('valid input')
        >>> dst_fpath = dpath / 'there.txt'
        >>> self = ProcessNode(
        >>>     name='proc1',
        >>>     config={
        >>>         'foo': 'baz',
        >>>         'bar': 'biz',
        >>>         'num_workers': 3,
        >>>         'src': src_fpath,
        >>>         'dst': dst_fpath
        >>>     },
        >>>     in_paths={'src'},
        >>>     out_paths={'dst': 'there.txt'},
        >>>     perf_params={'num_workers'},
        >>>     group_dname='predictions',
        >>>     #node_dname='proc1/{proc1_algo_id}/{proc1_id}',
        >>>     executable=f'python -c "{chr(10)}{pycode}{chr(10)}"',
        >>>     root_dpath=dpath,
        >>> )
        >>> self._finalize_templates()
        >>> print('self.command = {}'.format(ub.urepr(self.command, nl=1, sv=1)))
        >>> print(f'self.algo_id={self.algo_id}')
        >>> print(f'self.root_dpath={self.root_dpath}')
        >>> print(f'self.template_node_dpath={self.template_node_dpath}')
        >>> print('self.templates = {}'.format(ub.urepr(self.templates, nl=2)))
        >>> print('self.final = {}'.format(ub.urepr(self.final, nl=2)))
        >>> print('self.condensed = {}'.format(ub.urepr(self.condensed, nl=2)))

    Example:
        >>> # How to use a ProcessNode to handle an arbitrary process call
        >>> # First let's write a program to disk
        >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
        >>> import stat
        >>> dpath = ub.Path.appdir('geowatch/test/pipeline/TestProcessNode2')
        >>> dpath.delete().ensuredir()
        >>> pycode = ub.codeblock(
                '''
                #!/usr/bin/env python3
                import scriptconfig as scfg
                import ubelt as ub

                class MyCLI(scfg.DataConfig):
                    src = None
                    dst = None
                    foo = None
                    bar = None

                    @classmethod
                    def main(cls, cmdline=1, **kwargs):
                        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
                        print('config = ' + ub.urepr(config, nl=1))

                if __name__ == '__main__':
                    MyCLI.main()
        ...     ''')
        >>> fpath = dpath / 'mycli.py'
        >>> fpath.write_text(pycode)
        >>> fpath.chmod(fpath.stat().st_mode | stat.S_IXUSR)
        >>> # Now that we have a script that accepts some cli arguments
        >>> # Create a process node to represent it. We assume that
        >>> # everything is passed as key/val style params, which you *should*
        >>> # use for new programs, but this doesnt apply to a lot of programs
        >>> # out there, so we will show how to handle non key/val arguments
        >>> # later (todo).
        >>> mynode = ProcessNode(command=str(fpath))
        >>> # Get the invocation by runnning
        >>> command = mynode.final_command()
        >>> print(command)
        >>> # Use a dictionary to configure key/value pairs
        >>> mynode.configure({'src': 'a.txt', 'dst': 'b.txt'})
        >>> command = mynode.final_command()
        >>> # Note: currently because of backslash formatting
        >>> # we need to use shell=1 or system=1 with ub.cmd
        >>> # in the future we will fix this in ubelt (todo).
        >>> # Similarly this class should be able to provide the arglist
        >>> # style of invocation.
        >>> print(command)
        >>> ub.cmd(command, verbose=3, shell=1)

    """
    __node_type__ = 'process'

    name : Optional[str] = None

    # A path that will specified directly after the DAG root dpath.
    group_dname : Optional[str] = None

    # resources : Collection = None  # Unused?

    executable : Optional[str] = None

    algo_params : Collection = None  # algorithm parameters - impacts output

    perf_params : Collection = None  # performance parameters - no output impact

    # input paths
    # Should be specified as a set of names wrt the config or as dict mapping
    # from names to absolute paths.
    in_paths : Collection = None

    # output paths
    # Should be specified as templates
    out_paths : Collection = None

    def __init__(self,
                 *,  # TODO: allow positional arguments after we find a good order
                 name=None,
                 executable=None,
                 algo_params=None,
                 perf_params=None,
                 # resources=None,
                 in_paths=None,
                 out_paths=None,
                 group_dname=None,
                 root_dpath=None,
                 config=None,
                 node_dpath=None,  # overwrites configured node dapth
                 group_dpath=None,  # overwrites configured node dapth
                 _overwrite_node_dpath=None,  # overwrites the configured node dpath
                 _overwrite_group_dpath=None,  # overwrites the configured group dpath
                 _no_outarg=False,
                 _no_inarg=False,
                 **aliases):
        if aliases:
            if 'perf_config' in aliases:
                raise ValueError('You probably meant perf_params')
            if 'algo_config' in aliases:
                raise ValueError('You probably meant algo_params')

            if 'command' in aliases:
                executable = aliases['command']

        if node_dpath is not None:
            _overwrite_node_dpath = node_dpath
        if group_dpath is not None:
            _overwrite_group_dpath = group_dpath

        del node_dpath
        del group_dpath
        del aliases

        # if name is None and executable is None:
        #     name = f'unnamed_process_node_{id(self)}'
        if name is None :
            # Not sure exactly what's going on here, (i.e. why our smart nodes
            # are getting created without a name)
            if executable is not None or self.__class__.__name__ == 'ProcessNode':
                name = 'unnamed_process_node_' + str(id(self))  # ub.hash_data(executable)[0:8]

        args = locals()
        fallbacks = {
            # 'resources': {
            #     'cpus': 2,
            #     'gpus': 0,
            # },
            'config': {},
            'in_paths': {},
            'out_paths': {},
            'perf_params': {},
            'algo_params': {},
        }
        _classvar_init(self, args, fallbacks)
        super().__init__(args['name'])

        self._configured_cache = {}

        if self.group_dname is None:
            self.group_dname = '.'

        if self.root_dpath is None:
            self.root_dpath = '.'
        self.root_dpath = ub.Path(self.root_dpath)

        self.templates = None

        self.final_outdir = None
        self.final_opaths = None
        self.enabled = True
        self.cache = True
        self._no_outarg = _no_outarg
        self._no_inarg = _no_inarg

        # TODO: make specifying these overloads more natural
        # Basically: use templates unless the user gives these
        self._overwrite_node_dpath = _overwrite_node_dpath
        self._overwrite_group_dpath = _overwrite_group_dpath

        # TODO: need a better name for this.
        # This is just a list of nodes that must be run before us, but we don't
        # have an explicit connection between the inputs / outputs.
        # This is currently used as a workaround, but we should support it
        self._pred_nodes_without_io_connection = []

        self.configure(self.config)

    @profile
    def configure(self, config=None, cache=True, enabled=True):
        """
        Update the node configuration.

        This rebuilds the templates and formats them so the "final" variables
        take on directory names based on the given configuration. This a

        """
        self.cache = cache
        self._configured_cache.clear()  # Reset memoization caches
        if config is None:
            config = {}
        config = _fixup_config_serializability(config)
        self.enabled = config.pop('enabled', enabled)
        self.config = ub.udict(config)

        if isinstance(self.in_paths, dict):
            # In the case where the in paths is a dictionary, we can
            # prepopulate some of the config options.
            non_specified = ub.udict(self.in_paths) - self.config
            for key in non_specified:
                self.inputs[key].final_value = non_specified[key]

        # self.algo_params = set(self.config) - non_algo_keys
        in_path_keys = self.config & set(self.in_paths)
        for key in in_path_keys:
            self.inputs[key].final_value = self.config[key]

        self._build_templates()
        self._finalize_templates()

    @memoize_configured_property
    @profile
    def condensed(self):
        """
        This is the dictionary that supplies the templated strings with the
        values we will finalize them with. We may want to change the name.
        """
        condensed = {}
        for node in self.predecessor_process_nodes():
            condensed.update(node.condensed)
        condensed.update({
            self.name + '_algo_id': self.algo_id,
            self.name + '_id': self.process_id,
        })
        return condensed

    @memoize_configured_method
    @profile
    def _build_templates(self):
        templates = {}
        templates['root_dpath'] = str(self.template_root_dpath)
        templates['node_dpath'] = str(self.template_node_dpath)
        templates['out_paths'] = self.template_out_paths
        self.templates = templates
        return self.templates

    @memoize_configured_method
    @profile
    def _finalize_templates(self):
        templates = self.templates
        condensed = self.condensed
        final = {}
        try:
            final['root_dpath'] = self.final_root_dpath
            final['node_dpath'] = self.final_node_dpath
            final['out_paths'] = self.final_out_paths
            final['in_paths'] = self.final_in_paths
        except KeyError as ex:
            print('ERROR: {}'.format(ub.urepr(ex, nl=1)))
            print('condensed = {}'.format(ub.urepr(condensed, nl=1, sort=0)))
            print('templates = {}'.format(ub.urepr(templates, nl=1, sort=0)))
            raise
        self.final = final
        return self.final

    @memoize_configured_property
    def final_config(self):
        """
        This is not really "final" in the aggregate sense.
        It is more of a "finalized" requested config.
        """
        final_config = self.config.copy()
        if not self._no_inarg:
            final_config.update(self.final_in_paths)
        if not self._no_outarg:
            # Hacky option, improve the API to make this not necessary
            # when the full command is given and we dont need to
            # add the extra args
            final_config.update(self.final_out_paths)
        final_config.update(self.final_perf_config)
        final_config.update(self.final_algo_config)
        return final_config

    def _depends_config(self):
        """
        The dag config that specifies the parameters this node depends on.
        This is what we write to "job_config.json". Note: this output must be
        passed to dag.config, not node.config.
        """
        depends_config = {}
        for depend_node in list(self.ancestor_process_nodes()) + [self]:
            depends_config.update(_add_prefix(depend_node.name + '.', depend_node.config))
        return depends_config

    @memoize_configured_property
    def final_perf_config(self):
        final_perf_config = self.config & set(self.perf_params)
        if isinstance(self.perf_params, dict):
            for k, v in self.perf_params.items():
                if k not in final_perf_config:
                    final_perf_config[k] = v
        return final_perf_config

    @memoize_configured_property
    def final_algo_config(self):
        # TODO: Any node that does not have its inputs connected have to
        # include the configured input paths - or ideally the hash of their
        # contents - in the algo config.

        # Find keys that are not part of the algorithm config
        non_algo_sets = [self.out_paths, self.perf_params]
        non_algo_keys = (
            set.union(*[set(s) for s in non_algo_sets if s is not None])
            if non_algo_sets else set()
        )
        self.non_algo_keys = non_algo_keys

        # Find unconnected inputs, which point to data that is part of the
        # algorithm config
        unconnected_inputs = []
        for input_node in self.inputs.values():
            if not input_node.pred:
                unconnected_inputs.append(input_node.name)

        # OK... so previous design decisions have made things weird here.  The
        # question is: are the input paths included in the final algo config?
        # Currently the answer depends. If they are connected in as part of a
        # pipeline (i.e. the input paths are the output of some other step)
        # then they are not currently considered part of the algo config.
        # However, if they are unconnected (but maybe explicitly specified by
        # the user), then they are considered part of the algo config.  I'm not
        # sure how to fix this yet. Conceptually, perhaps the "algorithm
        # config", should not depend on the inputs, but I could see arguments
        # both ways. There should probably be a "final_input_config" (or maybe
        # just final_in_paths) that is separate from the "final_algo_config".
        # ...
        # ...
        # ... so the next question is, does anything currently depend on the
        # input paths being in the final algo config? If not we should
        # probably remove it.
        if self._no_inarg:
            unconnected_in_paths = ub.udict({})
        else:
            unconnected_in_paths = ub.udict(self.final_in_paths) & unconnected_inputs

        final_algo_config = (self.config - self.non_algo_keys) | unconnected_in_paths

        if isinstance(self.algo_params, dict):
            for k, v in self.algo_params.items():
                if k not in final_algo_config:
                    final_algo_config[k] = v
        return final_algo_config

    @memoize_configured_property
    def final_in_paths(self):
        final_in_paths = self.in_paths
        if final_in_paths is None:
            final_in_paths = {}
        elif isinstance(final_in_paths, dict):
            final_in_paths = final_in_paths.copy()
        else:
            final_in_paths = {k: None for k in final_in_paths}

        for key, input_node in self.inputs.items():
            final_in_paths[key] = input_node.final_value
        return final_in_paths

    @memoize_configured_property
    def template_out_paths(self):
        """
        Note: template out paths are not impacted by out path config overrides,
        but the final out paths are.

        SeeAlso:
            :func:`ProcessNode.final_out_paths`
        """
        if not isinstance(self.out_paths, dict):
            out_paths = self.config & self.out_paths
        else:
            out_paths = self.out_paths
        template_node_dpath = self.template_node_dpath
        template_out_paths = {
            k: str(template_node_dpath / v)
            for k, v in out_paths.items()
        }
        return template_out_paths

    @memoize_configured_property
    def final_out_paths(self):
        """
        These are the locations each output will actually be written to.

        This is based on :func:`ProcessNode.template_out_paths` as well as any
        manual overrides specified in ``self.config``.
        """
        condensed = self.condensed
        template_out_paths = self.template_out_paths
        final_out_paths = {
            k: ub.Path(v.format(**condensed))
            for k, v in template_out_paths.items()
        }
        # The use config is allowed to overload outpaths
        overloads = self.config & final_out_paths.keys()
        if overloads:
            final_out_paths.update(overloads)
        return final_out_paths

    @memoize_configured_property
    def final_node_dpath(self):
        """
        The configured directory where all outputs are relative to.
        """
        return ub.Path(str(self.template_node_dpath).format(**self.condensed))

    @memoize_configured_property
    def final_root_dpath(self):
        return ub.Path(str(self.template_root_dpath).format(**self.condensed))

    @property
    def template_group_dpath(self):
        """
        The template for the directory where the configured node dpath will be placed.
        """
        if self._overwrite_group_dpath is not None:
            return ub.Path(self._overwrite_group_dpath)
        if self.group_dname is None:
            return self.root_dpath / 'flat' / self.name
        else:
            return self.root_dpath / self.group_dname / 'flat' / self.name

    @memoize_configured_property
    def template_node_dpath(self):
        """
        The template for the configured directory where all outputs are relative to.
        """
        if self._overwrite_node_dpath is not None:
            return ub.Path(self._overwrite_node_dpath)
        key = self.name + '_id'
        return self.template_group_dpath / ('{' + key + '}')

    @memoize_configured_property
    def template_root_dpath(self):
        return self.root_dpath

    @memoize_configured_method
    @profile
    def predecessor_process_nodes(self):
        """
        Process nodes that this one depends on.
        """
        nodes = [pred.parent for k, v in self.inputs.items()
                 for pred in v.pred] + self._pred_nodes_without_io_connection
        return nodes

    @memoize_configured_method
    @profile
    def successor_process_nodes(self):
        """
        Process nodes that depend on this one.
        """
        nodes = [succ.parent for k, v in self.outputs.items()
                 for succ in v.succ]
        return nodes

    @memoize_configured_method
    @profile
    def ancestor_process_nodes(self):
        """
        Example:
            >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
            >>> pipe = Pipeline.demo()
            >>> self = pipe.node_dict['node_C1']
            >>> ancestors = self.ancestor_process_nodes()
            >>> print('ancestors = {}'.format(ub.urepr(ancestors, nl=1)))
        """
        # TODO: we need to ensure that this returns a consistent order
        seen = {}
        stack = [self]
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id not in seen:
                seen[node_id] = node
                nodes = node.predecessor_process_nodes()
                stack.extend(nodes)
        seen.pop(id(self))  # remove self
        ancestors = list(seen.values())
        return ancestors

    def _uncached_ancestor_process_nodes(self):
        # Not sure why the cached version of this is not working
        # in prepare-ta2-dataset. Hack around it for now.
        # TODO: we need to ensure that this returns a consistent order
        seen = {}
        stack = [self]
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id not in seen:
                seen[node_id] = node
                nodes = [
                    pred.parent for k, v in node.inputs.items()
                    for pred in v.pred
                ]
                # nodes = node.predecessor_process_nodes()
                stack.extend(nodes)
        seen.pop(id(self))  # remove self
        ancestors = list(seen.values())
        return ancestors

    @memoize_configured_property
    @profile
    def depends(self):
        """
        The mapping from ancestor and self node names to their algorithm ids
        Should probably rename.
        """
        ancestors = self.ancestor_process_nodes()
        # TODO:
        # We need to know what input paths have not been represented.  This
        # involves finding input paths that are not connected to the output of
        # a node involved in building this id.
        depends = {}
        for node in ancestors:
            depends[node.name] = node.algo_id
        depends[self.name] = self.algo_id
        depends = ub.udict(sorted(depends.items()))
        return depends

    @memoize_configured_property
    @profile
    def algo_id(self) -> str:
        """
        A unique id to represent the output of a deterministic process.

        This does NOT have a dependency on the larger the DAG.
        """
        from geowatch.utils.reverse_hashid import condense_config
        algo_id = condense_config(
            self.final_algo_config, self.name + '_algo_id', register=False)
        return algo_id

    @memoize_configured_property
    @profile
    def process_id(self) -> str:
        """
        A unique id to represent the output of a deterministic process in a
        pipeline. This id combines the hashes of all ancestors in the DAG with
        its own hashed id.

        This DOES have a dependency on the larger DAG.
        """
        from geowatch.utils.reverse_hashid import condense_config
        depends = self.depends
        proc_id = condense_config(
            depends, self.name + '_id', register=False)
        return proc_id

    @staticmethod
    @profile
    def _make_argstr(config):
        # parts = [f'    --{k}="{v}" \\' for k, v in config.items()]
        parts = []
        import shlex
        for k, v in config.items():
            if isinstance(v, list):
                # Handle variable-args params
                quoted_varargs = [shlex.quote(str(x)) for x in v]
                preped_varargs = ['        ' + x + ' \\' for x in quoted_varargs]
                parts.append(f'    --{k} \\')
                parts.extend(preped_varargs)
            else:
                if isinstance(v, dict):
                    # This relies on the underlying program being able to
                    # interpret YAML specified on the commandline.
                    from kwutil.util_yaml import Yaml
                    vstr = Yaml.dumps(v)
                    vstr = shlex.quote(vstr)
                    if '\n' in vstr and vstr[0] == "'":
                        # hack to prevent yaml indent errors
                        vstr = "'\n" + vstr[1:]
                    parts.append(f'    --{k}={vstr} \\')
                else:
                    import shlex
                    vstr = shlex.quote(str(v))
                    parts.append(f'    --{k}={vstr} \\')

        return '\n'.join(parts).lstrip().rstrip('\\')

    @cached_property
    @profile
    def inputs(self):
        """
        Input nodes representing specific input locations.

        The output nodes of other processes can be connected to these.
        Also input nodes for one process can connect to input nodes of another
        process representing that they share the same input data.

        Returns:
            Dict[str, InputNode]
        """
        inputs = {k: InputNode(name=k, parent=self) for k in self.in_paths}
        return inputs

    @cached_property
    @profile
    def outputs(self):
        """
        Output nodes representing specific output locations. These can be
        connected to the input nodes of other processes.

        Returns:
            Dict[str, OutputNode]
        """
        outputs = {k: OutputNode(name=k, parent=self) for k in self.out_paths}
        return outputs

    @property
    @profile
    def command(self) -> str:
        """
        Returns the string shell command that will execute the process.

        Basic version of command, can be overwritten
        """
        argstr = self._make_argstr(self.final_config)
        if argstr:
            command = self.executable + ' \\\n    ' + argstr
        else:
            command = self.executable
        return command

    @profile
    def test_is_computed_command(self):
        r"""
        Generate a bash command that will test if all output paths exist

        Example:
            >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
            >>> self = ProcessNode(out_paths={
            >>>     'foo': 'foo.txt',
            >>>     'bar': 'bar.txt',
            >>>     'baz': 'baz.txt',
            >>>     'biz': 'biz.txt',
            >>> }, node_dpath='.')
            >>> test_cmd = self.test_is_computed_command()
            >>> print(test_cmd)
            test -e foo.txt -a \
                 -e bar.txt -a \
                 -e baz.txt -a \
                 -e biz.txt
            >>> self = ProcessNode(out_paths={
            >>>     'foo': 'foo.txt',
            >>>     'bar': 'bar.txt',
            >>> }, node_dpath='.')
            >>> test_cmd = self.test_is_computed_command()
            >>> print(test_cmd)
            test -e foo.txt -a \
                 -e bar.txt
            >>> self = ProcessNode(out_paths={
            >>>     'foo': 'foo.txt',
            >>> }, node_dpath='.')
            >>> test_cmd = self.test_is_computed_command()
            >>> print(test_cmd)
            test -e foo.txt
            >>> self = ProcessNode(out_paths={}, node_dpath='.')
            >>> test_cmd = self.test_is_computed_command()
            >>> print(test_cmd)
            None
        """
        if not self.final_out_paths:
            return None
        import shlex
        quoted_paths = [shlex.quote(str(p))
                        for p in self.final_out_paths.values()]
        # Make the command look nicer
        tmp_paths = [f'-e {p}' for p in quoted_paths]
        tmp_paths = [p + ' -a' for p in tmp_paths[:-1]] + tmp_paths[-1:]
        *tmp_first, tmp_last = tmp_paths
        tmp_paths = [p + ' \\' for p in tmp_first] + [tmp_last]
        test_expr = '\n     '.join(tmp_paths)
        test_cmd = 'test ' +  test_expr

        # test_expr = ' -a '.join(
        #     [f'-e "{p}"' for p in self.final_out_paths.values()])
        # test_cmd = 'test ' +  test_expr
        return test_cmd

    @memoize_configured_property
    @profile
    def does_exist(self) -> bool:
        """
        Check if all of the output paths that would be written by this node
        already exists.
        """
        if len(self.final_out_paths) == 0:
            # Can only cache if we know what output paths are
            return False
        # return all(self.out_paths.map_values(lambda p: p.exists()).values())
        return all(ub.Path(p).expand().exists() for p in self.final_out_paths.values())

    @memoize_configured_property
    @profile
    def outputs_exist(self) -> bool:
        """
        Alias for does_exist

        Check if all of the output paths that would be written by this node
        already exists.
        """
        return self.does_exist

    def _raw_command(self):
        command = self.command
        if not isinstance(command, str):
            assert callable(command)
            command = command()
        return command

    @profile
    def final_command(self):
        """
        Wraps ``self.command`` with optional checks to prevent the command from
        executing if its outputs already exist.
        """
        command = self._raw_command()

        # Cleanup the command
        base_command = command.rstrip().rstrip('\\').rstrip()
        lines = base_command.split('\n')
        base_command = '\n'.join([line for line in lines if line.strip() != '\\'])

        if self.cache or (not self.enabled and self.enabled != 'redo'):
            test_cmd = self.test_is_computed_command()
            if test_cmd is None:
                return base_command
            else:
                return test_cmd + ' || \\\n' + base_command
        else:
            return base_command

    def find_template_outputs(self, workers=8):
        """
        Look in the DAG root path for output paths that are complete or
        unfinished
        """
        template = self.template_node_dpath
        existing_dpaths = list(glob_templated_path(template))
        # Figure out which ones are finished / unfinished

        json_jobs = ub.Executor(mode='thread', max_workers=workers)

        rows = []
        for dpath in ub.ProgIter(existing_dpaths, desc='parsing templates'):

            out_fpaths = {}
            for out_key, out_fname in self.out_paths.items():
                out_fpath = dpath / out_fname
                out_fpaths[out_key] = out_fpath

            is_finished = all(p.exists() for p in out_fpaths.values())
            config_fpath = (dpath / 'job_config.json')
            has_config = config_fpath.exists()
            if has_config:
                job = json_jobs.submit(_load_json, config_fpath)
            else:
                job = None
                request_config = {}

            rows.append({
                'dpath': dpath,
                'is_finished': is_finished,
                'has_config': has_config,
                'job': job,
            })

        for row in ub.ProgIter(rows, desc='finalize templates'):
            job = row.pop('job')
            if job is not None:
                request_config = job.result()
                request_config = util_dotdict.DotDict(request_config).add_prefix('request')
                row.update(request_config)

        num_configured = sum([r['has_config'] for r in rows])
        num_finished = sum([r['has_config'] for r in rows])
        num_started = len(rows)
        print(f'num_configured={num_configured}')
        print(f'num_finished={num_finished}')
        print(f'num_started={num_started}')
        return rows


def _load_json(fpath):
    import json
    with open(fpath, 'r') as file:
        return json.load(file)


def _add_prefix(prefix, dict_):
    return {prefix + k: v for k, v in dict_.items()}


def _fixup_config_serializability(config):
    # Do minor chanes to make the config json serializable.
    fixed_config = {}
    for k, v in config.items():
        if isinstance(v, os.PathLike):
            fixed_config[k] = os.fspath(v)
        else:
            fixed_config[k] = v
    return fixed_config


def demodata_pipeline():
    """
    A simple test pipeline.

    Example:
        >>> # Self test
        >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
        >>> demodata_pipeline()
    """
    dpath = ub.Path.appdir('geowatch/tests/mlops/pipeline').ensuredir()
    dpath.delete().ensuredir()
    script_dpath = (dpath / 'src').ensuredir()
    inputs_dpath = (dpath / 'inputs').ensuredir()
    runs_dpath = (dpath / 'runs').ensuredir()

    # Make simple scripts to stand in for the more complex processes that we
    # will orchestrate. The important thing is they have CLI input and output
    # paths / arguments.
    fpath1 = script_dpath / 'demo_script1.py'
    fpath2 = script_dpath / 'demo_script2.py'
    fpath3 = script_dpath / 'demo_script3.py'
    fpath1.write_text(ub.codeblock(
        '''
        import ubelt as ub
        src = ub.Path(ub.argval('--src'))
        dst = ub.Path(ub.argval('--dst'))
        dst.parent.ensuredir()
        algo_param1 = ub.argval('--algo_param1', default='')
        perf_param1 = ub.argval('--perf_param1', default='')
        dst.write_text(src.read_text() + algo_param1)
        '''))
    fpath2.write_text(ub.codeblock(
        '''
        import ubelt as ub
        src1 = ub.Path(ub.argval('--src1'))
        src2 = ub.Path(ub.argval('--src2'))
        dst1 = ub.Path(ub.argval('--dst1'))
        dst2 = ub.Path(ub.argval('--dst2'))
        dst1.parent.ensuredir()
        dst2.parent.ensuredir()
        algo_param2 = ub.argval('--algo_param2', default='')
        perf_param2 = ub.argval('--perf_param2', default='')
        dst1.write_text(src1.read_text() + algo_param2)
        dst2.write_text(src2.read_text() + algo_param2)
        '''))
    fpath3.write_text(ub.codeblock(
        '''
        import ubelt as ub
        src1 = ub.Path(ub.argval('--src1'))
        src2 = ub.Path(ub.argval('--src2'))
        dst = ub.Path(ub.argval('--dst'))
        dst.parent.ensuredir()
        algo_param3 = ub.argval('--algo_param3', default='')
        perf_param3 = ub.argval('--perf_param3', default='')
        dst.write_text(src1.read_text() + algo_param3 + src2.read_text())
        '''))
    executable1 = f'python {fpath1}'
    executable2 = f'python {fpath2}'
    executable3 = f'python {fpath3}'

    # Now that we have executables we need to create a ProcessNode that
    # describes how each process might be run. This can be done via inheritence
    # or specifying constructor variables.
    node_A1 = ProcessNode(
        name='node_A1',
        in_paths={
            'src',
        },
        algo_params={
            'algo_param1': '',
        },
        perf_params={
            'perf_param1': '',
        },
        out_paths={
            'dst': 'out.txt'
        },
        executable=executable1
    )
    node_A2 = ProcessNode(
        name='node_A2',
        in_paths={
            'src',
        },
        algo_params={
            'algo_param1': '',
        },
        perf_params={
            'perf_param1': '',
        },
        out_paths={
            'dst': 'out.txt'
        },
        executable=executable1
    )
    node_B1 = ProcessNode(
        name='node_B1',
        in_paths={
            'src1',
            'src2'
        },
        algo_params={
            'algo_param2': '',
        },
        perf_params={
            'perf_param2': '',
        },
        out_paths={
            'dst1': 'out1.txt',
            'dst2': 'out2.txt'
        },
        executable=executable2
    )
    node_C1 = ProcessNode(
        name='node_C1',
        in_paths={
            'src1',
            'src2'
        },
        algo_params={
            'algo_param3': '',
        },
        perf_params={
            'perf_param3': '',
        },
        out_paths={
            'dst': 'out.txt'
        },
        executable=executable3
    )

    # Given the process nodes we need to connect their inputs / outputs for
    # form a pipeline.
    node_A1.outputs['dst'].connect(node_B1.inputs['src1'])
    node_A2.outputs['dst'].connect(node_B1.inputs['src2'])
    node_A2.inputs['src'].connect(node_C1.inputs['src1'])
    node_B1.outputs['dst1'].connect(node_C1.inputs['src2'])

    # The pipeline is just a container for the nodes
    nodes = [node_A1, node_A2, node_B1, node_C1]
    dag = Pipeline(nodes=nodes)

    # Given a dag, there will often be top level input parameters that must be
    # configured along with any other algorithm or performance parameters

    # Create the inputs and configure the graph
    input1_fpath = inputs_dpath / 'input1.txt'
    input2_fpath = inputs_dpath / 'input2.txt'
    input1_fpath.write_text('spam')
    input2_fpath.write_text('eggs')

    dag.configure({
        'node_A1.src': str(input1_fpath),
        'node_A2.src': str(input2_fpath),
        'node_A2.dst': dpath / 'DST_OVERRIDE',
        'node_C1.perf_param3': 'GOFAST',
    }, root_dpath=runs_dpath, cache=False)

    return dag


def demo_pipeline_run():
    """
    A simple test pipeline.

    Example:
        >>> # Self test
        >>> from geowatch.mlops.pipeline_nodes import *  # NOQA
        >>> demo_pipeline_run()
    """
    dag = Pipeline.demo()

    dag.print_graphs()
    dag.inspect_configurables()

    # The jobs can now be submitted to a command queue which can be
    # executed or inspected at your leasure.
    status = dag.submit_jobs(queue=ub.udict({
        'backend': 'serial',
    }))
    queue = status['queue']
    queue.print_commands(exclude_tags='boilerplate', with_locks=False)
    queue.run()


# Backwards compat
PipelineDAG = Pipeline
