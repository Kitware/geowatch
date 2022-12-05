import ubelt as ub
# from dataclasses import dataclass
from typing import Union, Dict, Set, List, Any, Optional
from watch.utils import util_param_grid  # NOQA
import networkx as nx
from functools import cached_property
from cmd_queue.util import util_networkx  # NOQA

Collection = Optional[Union[Dict, Set, List]]
Configurable = Optional[Dict[str, Any]]


class PipelineDAG:
    def __init__(self, nodes=[], config=None, root_dpath=None):
        self.proc_graph = None
        self.io_graph = None
        self.nodes = nodes
        self.config = None

        if config:
            self.configure(config, root_dpath=root_dpath)

    def configure(self, config, root_dpath=None):
        self.config = config

        if root_dpath is not None:
            root_dpath = ub.Path(root_dpath)
            self.root_dpath = root_dpath

        if isinstance(self.nodes, dict):
            node_dict = self.nodes
        else:
            node_names = [node.name for node in self.nodes]
            assert len(node_names) == len(set(node_names))
            node_dict = dict(zip(node_names, self.nodes))

        # if __debug__:
        #     for name, node in node_dict.values():
        #         assert node.name == name, (
        #             'node instances require unique consistent names')

        self.proc_graph = nx.DiGraph()
        for name, node in node_dict.items():
            self.proc_graph.add_node(node.name, node=node)

            for s in node.succ:
                self.proc_graph.add_edge(node.name, s.name)

            for p in node.pred:
                self.proc_graph.add_edge(p.name, node.name)

        # util_networkx.write_network_text(self.proc_graph)

        self.io_graph = nx.DiGraph()
        for name, node in node_dict.items():
            self.io_graph.add_node(node.key, node=node)

            for iname, inode in node.inputs.items():
                self.io_graph.add_node(inode.key, node=inode)
                self.io_graph.add_edge(inode.key, node.key)

            for oname, onode in node.outputs.items():
                self.io_graph.add_node(onode.key, node=onode)
                self.io_graph.add_edge(node.key, onode.key)

                for oi_node in onode.succ:
                    self.io_graph.add_edge(onode.key, oi_node.key)

        # Set the configuration for each node in this pipeline.
        dotconfig = util_param_grid.DotDict(config)
        for node_name in nx.topological_sort(self.proc_graph):
            node = self.proc_graph.nodes[node_name]['node']
            if root_dpath is not None:
                node.root_dpath = root_dpath
            node_config = dict(dotconfig.prefix_get(node.name, {}))
            node.configure(node_config)

    def print_graphs(self):

        def labelize_graph(graph):
            # # self.io_graph.add_node(name + '.proc', node=node)
            all_names = []
            for _, data in graph.nodes(data=True):
                all_names.append(data['node'].name)

            ambiguous_names = list(ub.find_duplicates(all_names))
            for _, data in graph.nodes(data=True):

                if data['node'].name in ambiguous_names:
                    data['label'] = data['node'].key
                else:
                    data['label'] = data['node'].name

                if 'bas' in data['label']:
                    data['label'] = '[yellow]' + data['label']
                elif 'sc' in data['label']:
                    data['label'] = '[cyan]' + data['label']
                elif 'crop' in data['label']:
                    data['label'] = '[white]' + data['label']
        labelize_graph(self.io_graph)
        labelize_graph(self.proc_graph)

        import rich
        print('')
        print('Process Graph')
        util_networkx.write_network_text(self.proc_graph, path=rich.print, end='')

        print('')
        print('IO Graph')
        util_networkx.write_network_text(self.io_graph, path=rich.print, end='')

    def make_cmd_queue(self):
        import cmd_queue

        config = {
            # 'backend': 'tmux'
            'backend': 'serial'
        }
        queue = cmd_queue.Queue.create(
            backend=config['backend'], name='prep-ta2-dataset',
            size=1, gres=None,
            # environ=environ
        )

        import networkx as nx
        for node_name in list(nx.topological_sort(self.proc_graph)):
            pred_nodes = list(self.proc_graph.predecessors(node_name))
            node = self.proc_graph.nodes[node_name]['node']
            queue.submit(command=node.resolved_command(), depends=pred_nodes,
                         name=node_name)

        # for node in self.nodes.values():
        #     print('# --- ')
        #     print('node.config = {}'.format(ub.repr2(node.config, nl=1)))
        #     print(node.resolved_command())
        return queue


class Node(ub.NiceRepr):

    def __nice__(self):
        return f'{self.name!r}, p={[n.name for n in self.pred]}, s={[n.name for n in self.succ]}'

    def __init__(self, name: str):
        self.name = name
        self.pred = []
        self.succ = []

    def _connect_single(self, other, src_map, dst_map):
        # TODO: CLEANUP
        print(f'Connect {type(self).__name__} {self.name} to {type(other).__name__} {other.name}')
        if other not in self.succ:
            self.succ.append(other)
        if self not in other.pred:
            other.pred.append(self)

        self_is_proc = isinstance(self, ProcessNode)
        if self_is_proc:
            outputs = self.outputs
        else:
            assert isinstance(self, IONode)
            outputs = {self.name: self}

        other_is_proc = isinstance(other, ProcessNode)
        if other_is_proc:
            inputs = other.inputs
        else:
            assert isinstance(other, IONode)
            if not self_is_proc:
                # In this case we can make the name map implicit
                inputs = {self.name: other}
            else:
                inputs = {other.name: other}

        outmap = ub.udict({src_map.get(k, k): k for k in outputs.keys()})
        inmap = ub.udict({dst_map.get(k, k): k for k in inputs.keys()})

        common = outmap.keys() & inmap.keys()
        if len(common) == 0:
            print('inmap = {}'.format(ub.repr2(inmap, nl=1)))
            print('outmap = {}'.format(ub.repr2(outmap, nl=1)))
            raise Exception(f'Unknown io relationship {self.name} - {other.name}')

        if self_is_proc or other_is_proc:
            print(f'Connect Process to Process {self.name=} to {other.name=}')
            self_output_keys = (outmap & common).values()
            other_input_keys = (inmap & common).values()

            for out_key, in_key in zip(self_output_keys, other_input_keys):
                out_node = outputs[out_key]
                in_node = inputs[in_key]
                out_node._connect_single(in_node, src_map, dst_map)

        # elif self_is_proc and not other_is_proc:
        #     print(f'Connect Process to Output {self.name=} to {other.name=}')
        #     outputs
        #     raise NotImplementedError
        # elif not self_is_proc and other_is_proc:
        #     print(f'Connect Input to Process {self.name=} to {other.name=}')
        #     inputs
        #     raise NotImplementedError
        # else:
        #     print(f'Connect IOProcess {self.name=} to {other.name=}')

    def connect(self, *others, param_mapping=None, src_map=None, dst_map=None):
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
    def __init__(self, name, parent):
        super().__init__(name)
        self.parent = parent
        self._resolved_value = None
        self._template_value = None

    @property
    def resolved_value(self):
        value = self._resolved_value
        if value is None:
            preds = list(self.pred)
            if preds:
                assert len(preds) == 1
                value = preds[0].resolved_value
        return value

    @resolved_value.setter
    def resolved_value(self, value):
        self._resolved_value = value

    @property
    def key(self):
        return self.parent.key + '.' + self.name


class InputNode(IONode):
    ...


class OutputNode(IONode):
    @property
    def resolved_value(self):
        if self._resolved_value is None:
            return self.parent._resolve_templates()['out_paths'][self.name]
        return self._resolved_value


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


# @dataclass(kw_only=True)  # makes things harder
class ProcessNode(Node):
    """
    Represents a process in the pipeline.

    ProcessNodes are connected via their input / output nodes.

    May want to rename to a "bash" node.

    Example:
        >>> from watch.mlops.pipeline_nodes import *  # NOQA
        >>> from watch.mlops.pipeline_nodes import _classvar_init
        >>> dpath = ub.Path.appdir('watch/test/pipeline/TestProcessNode')
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
        >>>     node_dname='proc1/{proc1_algo_id}/{proc1_id}',
        >>>     executable=f'python -c "{chr(10)}{pycode}{chr(10)}"',
        >>>     root_dpath=dpath,
        >>> )
        >>> self._resolve_templates()
        >>> print('self.command = {}'.format(ub.repr2(self.command, nl=1, sv=1)))
        >>> print(f'self.algo_id={self.algo_id}')
        >>> print(f'self.root_dpath={self.root_dpath}')
        >>> print(f'self.template_node_dpath={self.template_node_dpath}')
        >>> print('self.templates = {}'.format(ub.repr2(self.templates, nl=2)))
        >>> print('self.resolved = {}'.format(ub.repr2(self.resolved, nl=2)))
        >>> print('self.condensed = {}'.format(ub.repr2(self.condensed, nl=2)))
    """
    name : Optional[str] = None

    # A path that will specified directly after the DAG root dpath.
    group_dname : Optional[str] = None

    # A path relative to a prefix used to construct an output directory.
    node_dname : Optional[str] = None

    resources : Collection = None

    executable : Optional[str] = None

    # algo_params : Collection = None  # algorithm parameters - impacts output

    perf_params : Collection = None  # performance parameters - no output impact

    # input paths
    # Should be specified as a set of names wrt the config or as dict mapping
    # from names to absolute paths.
    in_paths : Collection = None

    # output paths
    # Should be specified as templates
    out_paths : Collection = None

    def __init__(self,
                 name=None,
                 executable=None,
                 # algo_params=None,
                 perf_params=None,
                 resources=None,
                 in_paths=None,
                 out_paths=None,
                 group_dname=None,
                 node_dname=None,
                 root_dpath=None,
                 config=None):
        args = locals()
        fallbacks = {
            'resources': {
                'cpus': 2,
                'gpus': 0,
            },
            'config': {},
            'in_paths': {},
            'out_paths': {},
            'perf_params': {},
        }
        _classvar_init(self, args, fallbacks)
        super().__init__(args['name'])

        if self.node_dname is None:
            self.node_dname = '.'
        self.node_dname = ub.Path(self.node_dname)

        if self.group_dname is None:
            self.group_dname = '.'

        if self.root_dpath is None:
            self.root_dpath = '.'
        self.root_dpath = ub.Path(self.root_dpath)

        # self._dirtype = 'tree'
        self._dirtype = 'flat'

        self.templates = None

        self.template_outdir = None
        self.template_opaths = None

        self.resolved_outdir = None
        self.resolved_opaths = None

        self.configure(self.config)

    def configure(self, config):
        if config is None:
            config = {}
        self.config = ub.udict(config)
        if True:  # self.algo_params is None:
            # non_algo_sets = [self.in_paths, self.out_paths, self.perf_params]
            non_algo_sets = [self.out_paths, self.perf_params]
            non_algo_keys = (
                set.union(*[set(s) for s in non_algo_sets if s is not None])
                if non_algo_sets else set()
            )
            self.algo_params = set(self.config) - non_algo_keys

        in_path_keys = self.config & set(self.in_paths)
        for key in in_path_keys:
            self.inputs[key].resolved_value = self.config[key]

        self._build_templates()
        self._resolve_templates()

    @property
    def condensed(self):
        condensed = {}
        for node in self.predecessor_process_nodes():
            condensed.update(node.condensed)
        condensed.update({
            self.name + '_algo_id': self.algo_id,
            self.name + '_id': self.process_id,
        })
        return condensed

    def _build_templates(self):
        templates = {}
        templates['root_dpath'] = str(self.template_root_dpath)
        templates['node_dpath'] = str(self.template_node_dpath)
        templates['out_paths'] = self.template_out_paths
        self.templates = templates
        return self.templates

    def _resolve_templates(self):
        templates = self.templates
        condensed = self.condensed
        resolved = {}
        try:
            resolved['root_dpath'] = self.resolved_root_dpath
            resolved['node_dpath'] = self.resolved_node_dpath
            resolved['out_paths'] = self.resolved_out_paths
            resolved['in_paths'] = self.resolved_in_paths
        except KeyError as ex:
            print('ERROR: {}'.format(ub.repr2(ex, nl=1)))
            print('condensed = {}'.format(ub.repr2(condensed, nl=1, sort=0)))
            print('templates = {}'.format(ub.repr2(templates, nl=1, sort=0)))
            raise
        self.resolved = resolved
        return self.resolved

    @property
    def resolved_config(self):
        resolved_config = self.config.copy()
        resolved_config.update(self.resolved_in_paths)
        resolved_config.update(self.resolved_out_paths)
        if isinstance(self.perf_params, dict):
            for k, v in self.perf_params.items():
                if k not in resolved_config:
                    resolved_config[k] = v
        return resolved_config

    @property
    def resolved_in_paths(self):
        resolved_in_paths = self.in_paths
        if resolved_in_paths is None:
            resolved_in_paths = {}
        elif isinstance(resolved_in_paths, dict):
            resolved_in_paths = resolved_in_paths.copy()
        else:
            resolved_in_paths = {k: None for k in resolved_in_paths}

        for key, input_node in self.inputs.items():
            resolved_in_paths[key] = input_node.resolved_value
            # preds = list(input_node.pred)
            # if preds:
            #     assert len(preds) == 1
            #     pred = preds[0]
            #     value = pred.resolved_value
            #     # parent._resolve_templates()['out_paths'][pred.name]
            #     resolved_in_paths[key] = value
            # else:
            #     resolved_in_paths[key] = self.config.get(key, None)
        return resolved_in_paths

    @property
    def template_out_paths(self):
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

    @property
    def resolved_out_paths(self):
        condensed = self.condensed
        template_out_paths = self.template_out_paths
        resolved_out_paths = {
            k: ub.Path(v.format(**condensed))
            for k, v in template_out_paths.items()
        }
        return resolved_out_paths

    @property
    def template_dag_dname(self):
        return self.template_depends_dname / self.node_dname

    @property
    def resolved_node_dpath(self):
        return ub.Path(str(self.template_node_dpath).format(**self.condensed))

    @property
    def resolved_root_dpath(self):
        return ub.Path(str(self.template_root_dpath).format(**self.condensed))

    @property
    def template_root_dpath(self):
        return self.root_dpath

    @property
    def template_node_dpath(self):
        if self._dirtype == 'flat':
            return self.root_dpath / self.group_dname / 'flat' / self.name / ('{' + self.name + '_id}')
        else:
            return self.root_dpath / self.group_dname / self.template_dag_dname

    @property
    def algo_config(self):
        return self.config & self.algo_params

    @property
    def depends_config(self):
        # Any manually specified inputs need to be inserted into this
        # dictionary. Derived inputs can be ignored.
        depends_config = self.algo_config.copy()
        return depends_config
        # return self.config & self.algo_params

    @property
    def algo_id(self):
        """
        A unique id to represent the output of a deterministic process.

        This does NOT have a dependency on the larger the DAG.
        """
        from watch.utils.reverse_hashid import condense_config
        algo_id = condense_config(self.algo_config, self.name + '_algo_id')
        return algo_id

    @property
    def template_depends_dname(self):
        """
        Predecessor part of the output path.
        """
        pred_nodes = self.predecessor_process_nodes()
        if not pred_nodes:
            return ub.Path('.')
        elif len(pred_nodes) == 1:
            return pred_nodes[0].template_dag_dname
        else:
            return ub.Path('.')
            # return ub.Path('multi' + str(pred_nodes))

    def predecessor_process_nodes(self):
        """
        Predecessor process nodes
        """
        nodes = []
        for k, v in self.inputs.items():
            for pred in v.pred:
                # assert isinstance(pred, OutputNode)
                proc = pred.parent
                assert isinstance(proc, ProcessNode)
                nodes.append(proc)
        return nodes

    def ancestor_process_nodes(self):
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

    @property
    def depends(self):
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

    @property
    def node_info(self):
        ancestors = self.ancestor_process_nodes()
        # TODO:
        # We need to know what input paths have not been represented.  This
        # involves finding input paths that are not connected to the output of
        # a node involved in building this id.
        info = {
            'node': self.name,
            'process_id': self.process_id,
            'algo_id': self.algo_id,
            'depends': self.depends,
            'config': self.config,
            'ancestors': []
        }
        for node in ancestors[::-1]:
            info['ancestors'].append({
                'node': node.name,
                'process_id': node.process_id,
                'algo_id': node.algo_id,
                'config': node.config,
                'depends': node.depends,
            })
        return info

    @property
    def process_id(self):
        """
        A unique id to represent the output of a deterministic process in a
        pipeline. This id combines the hashes of all ancestors in the DAG with
        its own hashed id.

        This DOES have a dependency on the larger DAG.
        """
        from watch.utils.reverse_hashid import condense_config
        depends = self.depends
        proc_id = condense_config(depends, self.name + '_id')
        return proc_id

    @staticmethod
    def _make_argstr(config):
        parts = [f'    --{k}={v} \\' for k, v in config.items()]
        return chr(10).join(parts).lstrip().rstrip('\\')

    @cached_property
    def inputs(self):
        # inputs = {k: InputNode(name=self.name + '.' + k) for k in self.in_paths}
        inputs = {k: InputNode(name=k, parent=self) for k in self.in_paths}
        # for v in inputs.values():
        #     v.connect(self)
        return inputs

    @cached_property
    def outputs(self):
        # outputs = {k: OutputNode(name=self.name + '.' + k) for k in self.out_paths}
        outputs = {k: OutputNode(name=k, parent=self) for k in self.out_paths}
        # for v in outputs.values():
        #     self.connect(v)
        return outputs

    @property
    def command(self):
        """
        Basic version of command, can be overwritten
        """
        argstr = self._make_argstr(self.config)
        command = self.executable + ' ' + argstr
        return command

    def test_is_computed_command(step):
        test_expr = ' -a '.join(
            [f'-e "{p}"' for p in step.resolved_out_paths.values()])
        test_cmd = 'test ' +  test_expr
        return test_cmd

    @ub.memoize_property
    def does_exist(self):
        # return all(self.out_paths.map_values(lambda p: p.exists()).values())
        return all(ub.Path(p).expand().exists() for p in self.resolved_out_paths.values())

    def resolved_command(self):
        command = self.command()
        self.otf_cache = True
        self.enabled = True
        base_command = command.rstrip().rstrip('\\').rstrip()

        if self.otf_cache and self.enabled != 'redo':
            return self.test_is_computed_command() + ' || ' + base_command
        else:
            return base_command
