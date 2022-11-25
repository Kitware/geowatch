"""
Define the individual nodes that can be composed in a SMART pipeline.

The topology of the pipeline will define the resulting filesystem structure
used to store results.


CommandLine:
    xdoctest -m /home/joncrall/code/watch/watch/mlops/smart_pipeline_nodes.py __doc__

Example:
    >>> from watch.mlops.smart_pipeline_nodes import *  # NOQA
    >>> from cmd_queue.util import util_networkx
    >>> #
    >>> config = {
    >>>     'bas_pxl.package_fpath': None,
    >>>     'bas_pxl.workers': 2,
    >>>     'bas_pxl.data.test_dataset': 'foobar.kwcoco.json',
    >>> #
    >>>     'bas_poly.thresh': 0.1,
    >>>     'bas_poly.moving_window_size': 0.1,
    >>> #
    >>>     'sc_poly.use_viterbi': 0,
    >>> }
    >>> #
    >>> nodes = joint_bas_sc_nodes()
    >>> #nodes = bas_nodes()
    >>> #nodes = sc_nodes()
    >>> print('nodes = {}'.format(ub.repr2(nodes, nl=1, si=1)))
    >>> dag = PipelineDAG(nodes, config)
    >>> #
    >>> for node in dag.nodes.values():
    >>>     print(f'{node.name=}')
    >>>     print(f'{node.in_paths=}')
    >>>     print(f'{node.out_paths=}')
    >>>     print(f'{node.resources=}')
    >>> dag.configure(config)
    >>> #util_networkx.write_network_text(dag.proc_graph)
    >>> #util_networkx.write_network_text(dag.io_graph)
"""
import ubelt as ub
# from dataclasses import dataclass
from typing import Union, Dict, Set, List, Any, Optional
from watch.utils import util_param_grid  # NOQA
import networkx as nx
from functools import cached_property
from cmd_queue.util import util_networkx  # NOQA
# from

Collection = Optional[Union[Dict, Set, List]]
Configurable = Optional[Dict[str, Any]]


class PipelineDAG:
    def __init__(self, nodes, config=None):
        self.proc_graph = None
        self.io_graph = None
        self.nodes = nodes
        self.config = None

        if config:
            self.configure(config)

    def configure(self, config):
        # nested = util_param_grid.dotdict_to_nested(config)
        ...

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

    @property
    def key(self):
        return self.parent.key + '.' + self.name


class InputNode(IONode):
    ...


class OutputNode(IONode):
    ...


# @dataclass(kw_only=True)  # makes things harder
class ProcessNode(Node):
    executable : Optional[str] = None

    path_params : Collection = None

    algo_params : Collection = None

    perf_params : Collection = None

    resources : Collection = None

    in_paths : Collection = None

    out_paths : Collection = None

    def __init__(self, paths=None, params=None, resources=None, name=None):
        if name is None:
            name = self.__class__.name
        super().__init__(name)
        if resources is None:
            resources = {
                'cpus': 2,
                'gpus': 0,
            }
        self.paths = ub.udict({} if paths is None else paths)
        self.params = ub.udict({} if params is None else params)
        self.resources = ub.udict({} if resources is None else resources)

    @staticmethod
    def _make_argstr(params):
        parts = [f'    --{k}={v} \\' for k, v in params.items()]
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


class FeatureComputation(ProcessNode):
    executable = 'python -m watch.cli.run_metrics_framework'
    in_paths = {'src'}
    out_paths = {'dst'}

    def command(self):
        # paths = ub.udict(paths)
        # viz_pred_trk_poly_kw = paths.copy()
        # viz_pred_trk_poly_kw['extra_header'] = f"\\n{condensed['trk_pxl_cfg']}-{condensed['trk_poly_cfg']}"
        # viz_pred_trk_poly_kw['viz_channels'] = "red|green|blue,salient"
        command = ub.codeblock(
            r'''
            smartwatch teamfeat invariant
            ''')
        return command


class FeatureUnion(ProcessNode):
    name = 'featunion'
    executable = 'smartwatch feature_union'
    in_paths = {'src'}
    out_paths = {'dst'}

    def command(self):
        command = ub.codeblock(
            r'''
            kwcoco union todo
            ''')
        return command


class HeatmapPrediction(ProcessNode):
    executable = 'python -m watch.tasks.fusion.predict'

    resources = {
        'cpus': 2,
        'gpus': 1,
    }

    perf_params = {
        'num_workers': 2,
        'devices': '0,',
        'accelerator': 'gpu',
        'batch_size': 1,
    }

    in_paths = {
        'package_fpath',
        'test_dataset',
    }

    out_paths = {
        'pred_pxl_fpath',
    }

    def command(self):
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.predict \
                --package_fpath={package_fpath} \
                --test_dataset={test_dataset} \
                --pred_dataset={pred_pxl_fpath} \
                {params_argstr} \
                {perf_argstr}
            ''')
        return command


class PolygonPrediction(ProcessNode):
    executable = 'python -m watch.cli.run_tracker'
    default_track_fn = NotImplemented

    in_paths = {
        'pred_pxl_fpath'
    }

    out_paths = {
        'site_summaries_fpath',
        'site_summaries_dpath',
        'sites_fpath',
        'sites_dpath',
        'poly_kwcoco_fpath',
    }

    def command(self):
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_tracker \
                "{pred_pxl_fpath}" \
                --default_track_fn {self.default_track_fn} \
                --track_kwargs {kwargs_str} \
                --site_summary '{site_summary}' \
                --out_site_summaries_fpath "{site_summaries_fpath}" \
                --out_site_summaries_dir "{site_summaries_dpath}" \
                --out_sites_fpath "{sites_fpath}" \
                --out_sites_dir "{sites_dpath}" \
                --out_kwcoco "{poly_kwcoco_fpath}"
            ''')
        return command


class PolygonEvaluation(ProcessNode):
    executable = 'python -m watch.cli.run_metrics_framework'

    in_paths = {
        'true_region_dpath',
        'true_site_dpath',
        'sites_fpath',
    }

    out_paths = {
        'eval_dpath',
        'eval_fpath',
    }

    def command(self):
        self.tmp_dpath = self.paths['eval_dpath'] / 'tmp'
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_metrics_framework \
                --merge=True \
                --name "{name_suffix}" \
                --true_site_dpath "{true_site_dpath}" \
                --true_region_dpath "{true_region_dpath}" \
                --pred_sites "{sites_fpath}" \
                --tmp_dir "{self.tmp_dpath}" \
                --out_dir "{eval_dpath}" \
                --merge_fpath "{eval_fpath}"
            ''')
        return command


class HeatmapEvaluation(ProcessNode):
    executable = 'python -m watch.tasks.fusion.evaluate'

    in_paths = {
        'true_dataset',
        'pred_pxl_fpath',
    }

    out_paths = {
        'eval_pxl_dpath',
        'eval_pxl_fpath',
    }

    def command(self):
        # TODO: better score space
        extra_opts = {
            'draw_curves': True,
            'draw_heatmaps': True,
            'viz_thresh': 0.2,
            'workers': 2,
            'score_space': 'video',
        }
        extra_argstr = self._make_argstr(extra_opts)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.evaluate \
                --true_dataset={true_dataset} \
                --pred_dataset={pred_pxl_fpath} \
                --eval_dpath={eval_pxl_dpath} \
                --eval_fpath={eval_pxl_fpath} \
                {extra_argstr}
            ''')
        # .format(**eval_act_pxl_kw).strip().rstrip('\\')
        return command


class KWCocoVisualization(ProcessNode):
    executable = 'python -m watch.cli.coco_visualize_videos'

    resources = {
        'cpus': 2,
    }

    in_paths = {
        'poly_kwcoco_fpath',
    }

    out_paths = {
        'viz_stamp_fpath',
    }

    def command(self):
        # paths = ub.udict(paths)
        # viz_pred_trk_poly_kw = paths.copy()
        # viz_pred_trk_poly_kw['extra_header'] = f"\\n{condensed['trk_pxl_cfg']}-{condensed['trk_poly_cfg']}"
        # viz_pred_trk_poly_kw['viz_channels'] = "red|green|blue,salient"
        command = ub.codeblock(
            r'''
            smartwatch visualize \
                "{poly_kwcoco_fpath}" \
                --channels="auto" \
                --stack=only \
                --workers=2 \
                --extra_header="{extra_header}" \
                --animate=True && touch {viz_stamp_fpath}
            ''')
        return command


###
# Team Feature Nodes
###


class InvariantFeatureComputation(FeatureComputation):
    name = 'invar_feat'
    ...


class MaterialFeatureComputation(FeatureComputation):
    name = 'mat_feat'


class LandcoverFeatureComputation(FeatureComputation):
    name = 'land_feat'


###
# BAS / SC Nodes
###

# ---

class BAS_HeatmapPrediction(HeatmapPrediction):
    name = 'bas_pxl'
    prefix = 'bas_pxl/{bas_model}/{bas_test_dset}/{bas_pxl_cfg}/{bas_pxl_id}'


class SC_HeatmapPrediction(HeatmapPrediction):
    name = 'sc_pxl'
    prefix = 'sc_pxl/{sc_model}/{sc_test_dset}/{sc_pxl_cfg}/{sc_pxl_id}'

# ---


class BAS_PolygonPrediction(PolygonPrediction):
    name = 'bas_poly'
    prefix = 'bas_poly/{bas_poly_cfg}/{bas_poly_id}'
    default_track_fn = 'saliency_heatmaps'


class SC_PolygonPrediction(PolygonPrediction):
    name = 'sc_poly'
    prefix = 'sc_poly/{sc_poly_cfg}/{sc_poly_id}'
    default_track_fn = 'class_heatmaps'

# ---


class BAS_HeatmapEvaluation(HeatmapEvaluation):
    name = 'bas_pxl_eval'
    prefix = 'bas_pxl_eval'


class SC_HeatmapEvaluation(HeatmapEvaluation):
    name = 'sc_pxl_eval'
    prefix = 'sc_pxl_eval'


# ---

class BAS_PolygonEvaluation(PolygonEvaluation):
    name = 'bas_poly_eval'
    prefix = 'bas_poly_eval'


class SC_PolygonEvaluation(PolygonEvaluation):
    name = 'sc_poly_eval'
    prefix = 'sc_poly_eval'

# ---


class BAS_Visualization(KWCocoVisualization):
    name = 'bas_viz'
    prefix = 'bas_viz'


class SC_Visualization(KWCocoVisualization):
    name = 'sc_viz'
    prefix = 'sc_viz'


# ---


class SiteCropping(ProcessNode):
    name = 'crop'
    prefix = 'crop/{src_dset}/{regions_id}/{crop_cfg}/{crop_id}'

    in_paths = {
        'crop_src_fpath'
    }
    out_paths = {
        'crop_dst_fpath'
    }

    def command(self):
        # paths = ub.udict(paths)
        # from watch.cli import coco_align
        # confobj = coco_align.__config__
        # known_args = set(confobj.default.keys())
        # assert not len(ub.udict(crop_params) - known_args), 'unknown args'
        crop_params = {
            'geo_preprop': 'auto',
            'keep': 'img',
            'force_nodata': -9999,
            'rpc_align_method': 'orthorectify',
            'target_gsd': 4,
            'site_summary': True,
        }
        # } | ub.udict(crop_params)

        # The best setting of this depends on if the data is remote or not.
        # When networking, around 20+ workers is a good idea, but that's a very
        # bad idea for local images or if the images are too big.
        # Parametarizing would be best.
        CROP_IMAGE_WORKERS = 16
        CROP_AUX_WORKERS = 8

        perf_options = {
            'verbose': 1,
            'workers': CROP_IMAGE_WORKERS,
            'aux_workers': CROP_AUX_WORKERS,
            'debug_valid_regions': False,
            'visualize': False,
        }
        # crop_kwargs = { **paths }
        crop_kwargs = { }
        crop_kwargs['crop_params_argstr'] = self._make_argstr(crop_params)
        crop_kwargs['crop_perf_argstr'] = self._make_argstr(perf_options)

        command = ub.codeblock(
            r'''
            python -m watch.cli.coco_align \
                --src "{crop_src_fpath}" \
                --dst "{crop_dst_fpath}" \
                {crop_params_argstr} \
                {crop_perf_argstr} \
            ''').format(**crop_kwargs).strip().rstrip('\\')

        # FIXME: parametarize and only if we need secrets
        # secret_fpath = ub.Path('$HOME/code/watch/secrets/secrets').expand()
        # # if ub.Path.home().name.startswith('jon'):
        #     # if secret_fpath.exists():
        #     #     secret_fpath
        #         # command = f'source {secret_fpath} && ' + command
        command = 'AWS_DEFAULT_PROFILE=iarpa GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR ' + command
        return command
        # name = 'crop'
        # step = Step(name, command,
        #             in_paths=paths & {'crop_regions', 'crop_src_fpath'},
        #             out_paths=paths & {'crop_dst_fpath', 'crop_dpath'},
        #             resources={'cpus': 2})
        # return step


def bas_nodes():
    nodes = {}

    nodes['bas_pxl'] = BAS_HeatmapPrediction()
    nodes['bas_poly'] = BAS_PolygonPrediction()
    nodes['bas_pxl_eval'] = BAS_HeatmapEvaluation()
    nodes['bas_poly_eval'] = BAS_PolygonEvaluation()
    nodes['bas_poly_viz'] = BAS_Visualization()

    nodes['bas_pxl'].connect(
        nodes['bas_pxl_eval'],
        nodes['bas_poly'],
    )
    nodes['bas_poly'].connect(
        nodes['bas_poly_eval'],
        nodes['bas_poly_viz'],
    )

    if 1:
        nodes['bas_invariants'] = InvariantFeatureComputation()
        nodes['bas_land'] = LandcoverFeatureComputation()
        nodes['bas_materials'] = MaterialFeatureComputation()
        nodes['bas_featunion'] = FeatureUnion()

        nodes['bas_invariants'].connect(
            nodes['bas_featunion'],
            src_map={'dst': '_'},
            dst_map={'src': '_'},
        )
        nodes['bas_land'].connect(
            nodes['bas_featunion'],
            src_map={'dst': '_'},
            dst_map={'src': '_'},
        )
        nodes['bas_materials'].connect(
            nodes['bas_featunion'],
            src_map={'dst': '_'},
            dst_map={'src': '_'},
        )

        nodes['bas_featunion'].connect(nodes['bas_pxl'], param_mapping={
            'dst': 'test_dataset'
        })

        # nodes['bas_featunion'].connect(nodes['bas_pxl'], param_mapping={
        #     ''
        # })
        # nodes['bas_land'].connect(nodes['bas_pxl'])
        # nodes['bas_materials'].connect(nodes['bas_pxl'])
    return nodes


def sc_nodes():
    nodes = {}
    nodes['sc_pxl'] = SC_HeatmapPrediction()
    nodes['sc_poly'] = SC_PolygonPrediction()
    nodes['sc_pxl_eval'] = SC_HeatmapEvaluation()
    nodes['sc_poly_eval'] = SC_PolygonEvaluation()
    nodes['sc_poly_viz'] = SC_Visualization()

    nodes['sc_pxl'].connect(
        nodes['sc_pxl_eval'],
        nodes['sc_poly'].connect(
            nodes['sc_poly_eval'],
            nodes['sc_poly_viz'],
        ),
    )
    return nodes


def joint_bas_sc_nodes():
    nodes = {}
    nodes.update(bas_nodes())
    nodes['crop'] = SiteCropping()
    nodes.update(sc_nodes())

    # outputs['site_summaries_fpath'].connect(
    nodes['bas_poly'].connect(
        nodes['crop'],
        param_mapping={
            'site_summaries_fpath': 'crop_src_fpath',
        }
    )

    nodes['crop'].connect(
        # outputs['crop_fpath'].connect(
        nodes['sc_pxl'],
        param_mapping={
            'crop_dst_fpath': 'test_dataset',
        }
    )
    return nodes
