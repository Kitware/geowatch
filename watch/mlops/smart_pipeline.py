"""
Define the individual nodes that can be composed in a SMART pipeline.

The topology of the pipeline will define the resulting filesystem structure
used to store results.


CommandLine:
    xdoctest -m watch.mlops.smart_pipeline __doc__:0
    WATCH_DEVCHECK=1 xdoctest -m watch.mlops.smart_pipeline __doc__:1

Example:
    >>> # xdoctest: +SKIP
    >>> from watch.mlops.smart_pipeline import *  # NOQA
    >>> from cmd_queue.util import util_networkx
    >>> #
    >>> config = {
    >>>     'bas_pxl.package_fpath': '/global/models/bas_model2.pt',
    >>>     'bas_pxl.num_workers': 3,
    >>>     'bas_pxl.tta_time': 1,
    >>>     'bas_pxl.test_dataset': '/global/datasets/foobar.kwcoco.zip',
    >>> #
    >>>     'bas_poly.thresh': 0.1,
    >>>     'bas_poly.moving_window_size': 0.1,
    >>> #
    >>>     'sc_pxl.package_fpath': '/global/models/sc_model2.pt',
    >>>     'sc_pxl.tta_fliprot': 8,
    >>>     'sc_poly.use_viterbi': 0,
    >>> }
    >>> #
    >>> dag = make_smart_pipeline('joint_bas_sc')
    >>> # dag = make_smart_pipeline('bas')
    >>> # dag = make_smart_pipeline('sc')
    >>> dag.configure(config, root_dpath='/dag-root/dag-id')
    >>> #
    >>> for node in dag.nodes.values():
    >>>     print('---')
    >>>     print(f'node={type(node)}')
    >>>     print(f'{node.name=}')
    >>>     print(f'{node.config=}')
    >>>     print(f'{node.in_paths=}')
    >>>     print(f'{node.out_paths=}')
    >>>     print(f'{node.resources=}')
    >>>     print(f'{node.algo_params=}')
    >>>     print('node.depends = {}'.format(ub.repr2(node.depends, nl=1, sort=0)))
    >>>     final = node._finalize_templates()
    >>>     print('final = {}'.format(ub.repr2(final, nl=2)))
    >>>     print('---')
    >>> dag.print_graphs()
    >>> print('dag.config = {}'.format(ub.repr2(dag.config, nl=1)))
    >>> dag_templates = {}
    >>> dag_paths = {}
    >>> for node in dag.nodes.values():
    >>>     dag_templates[node.name] = node._build_templates()['node_dpath']
    >>>     dag_paths[node.name] = node._finalize_templates()['node_dpath']
    >>>     print(node.command())
    >>> import rich
    >>> rich.print('dag_templates = {}'.format(
    >>>     ub.repr2(dag_templates, nl=1, sv=1, align=':', sort=0)))
    >>> rich.print('dag_paths = {}'.format(
    >>>     ub.repr2(dag_paths, nl=1, sv=1, align=':', sort=0)))


Example:
    >>> # xdoctest: +REQUIRES(env:WATCH_DEVCHECK)
    >>> from watch.mlops.smart_pipeline import *  # NOQA
    >>> import watch
    >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    >>> #
    >>> config = {}
    >>> config['bas_pxl.test_dataset'] = data_dvc_dpath / 'Drop4-BAS/KR_R001.kwcoco.zip'
    >>> #config['bas_pxl.test_dataset'] = data_dvc_dpath / 'Drop4-BAS/KR_R002.kwcoco.zip'
    >>> #config['bas_pxl.test_dataset'] = data_dvc_dpath / 'Drop4-BAS/BR_R001.kwcoco.zip'
    >>> config['bas_pxl.package_fpath'] = expt_dvc_dpath / 'models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt'
    >>> config['bas_pxl.num_workers'] = 6
    >>> #config['bas_pxl.chip_dims'] = "512,512"
    >>> #config['bas_pxl.time_span'] = "1m"
    >>> #config['bas_pxl.time_sampling'] = "hardish2"
    >>> #config['bas_pxl.use_cloudmask'] = 0
    >>> #config['bas_pxl.set_cover_algo'] = 'approx'
    >>> #config['bas_pxl.resample_invalid_frames'] = 0
    >>> config['bas_poly.thresh'] = 0.1
    >>> #config['sc_pxl.chip_dims'] = "256,256"
    >>> #config['sc_pxl.use_cloudmask'] = 0
    >>> #config['sc_pxl.set_cover_algo'] = 'approx'
    >>> #config['sc_pxl.resample_invalid_frames'] = 0
    >>> config['sc_pxl.num_workers'] = 6
    >>> config['sc_pxl.package_fpath'] = expt_dvc_dpath / 'models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt'
    >>> #
    >>> root_dpath = data_dvc_dpath / '_testdag'
    >>> #
    >>> nodes = joint_bas_sc_nodes()
    >>> #nodes = bas_nodes()
    >>> from watch.mlops.pipeline_nodes import PipelineDAG
    >>> self = dag = PipelineDAG(nodes)
    >>> dag.configure(config=config, root_dpath=root_dpath)
    >>> dag.print_graphs()
    >>> cmd_queue = dag.submit_jobs()
    >>> cmd_queue.write_network_text()
    >>> cmd_queue.rprint()
    >>> #cmd_queue.run()

"""
import ubelt as ub
import shlex
import json
from watch.mlops.pipeline_nodes import ProcessNode


try:
    from xdev import profile  # NOQA
except ImportError:
    profile = ub.identity

PREDICT_NAME  = 'pred'
EVALUATE_NAME = 'eval'


class FeatureComputation(ProcessNode):
    executable = 'python -m watch.cli.run_metrics_framework'
    group_dname = PREDICT_NAME

    node_dname = 'feats/{src_dset}'

    in_paths = {'src'}

    @profile
    def command(self):
        command = ub.codeblock(
            r'''
            smartwatch teamfeat invariant # TODO
            ''')
        return command

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['src_dset'] = 'todo'
        return condensed


class FeatureUnion(ProcessNode):
    name = 'featunion'
    executable = 'smartwatch feature_union'
    group_dname = PREDICT_NAME
    in_paths = {'src'}
    out_paths = {
        'dst': 'combo_{featunion_id}.kwcoco.zip'
    }

    @profile
    def command(self):
        command = ub.codeblock(
            r'''
            kwcoco union todo
            ''')
        return command


class HeatmapPrediction(ProcessNode):
    executable = 'python -m watch.tasks.fusion.predict'
    group_dname = PREDICT_NAME

    resources = {
        'cpus': 2,
        'gpus': 1,
    }

    perf_params = {
        'num_workers': 2,
        'devices': '0,',
        #'accelerator': 'gpu',
        'batch_size': 1,
        'with_saliency': 'auto',
        'with_class': 'auto',
        'with_change': 'auto',
    }

    in_paths = {
        'package_fpath',
        'test_dataset',
    }

    algo_params = {
        'drop_unused_frames': True,
    }

    out_paths = {
        'pred_pxl_fpath' : 'pred.kwcoco.zip',
    }

    @profile
    def command(self):
        fmtkw = self.final_config.copy()
        perf_config = self.final_perf_config
        algo_config = self.final_algo_config - {
            'package_fpath', 'test_dataset', 'pred_dataset'}
        fmtkw['params_argstr'] = self._make_argstr(algo_config)
        fmtkw['perf_argstr'] = self._make_argstr(perf_config)
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.predict \
                --package_fpath={package_fpath} \
                --test_dataset={test_dataset} \
                --pred_dataset={pred_pxl_fpath} \
                {params_argstr} \
                {perf_argstr}
            ''').format(**fmtkw).rstrip().rstrip('\\').rstrip()
        return command


class PolygonPrediction(ProcessNode):
    executable = 'python -m watch.cli.run_tracker'
    group_dname = PREDICT_NAME
    default_track_fn = NotImplemented

    in_paths = {
        'pred_pxl_fpath',
        'site_summary',
    }

    # also
    # algo_params = {
    #     'resolution': None,
    # }

    out_paths = {
        'site_summaries_fpath': 'site_summaries_manifest.json',
        'site_summaries_dpath': 'site_summaries',
        'sites_fpath': 'sites_manifest.json',
        'sites_dpath': 'sites',
        'poly_kwcoco_fpath': 'poly.kwcoco.zip'
    }

    @profile
    def command(self):
        fmtkw = self.final_config.copy()
        fmtkw['default_track_fn'] = self.default_track_fn
        track_kwargs = self.final_algo_config.copy() - {'site_summary'}
        fmtkw['kwargs_str'] = shlex.quote(json.dumps(track_kwargs))
        command = ub.codeblock(
            r'''
            python -m watch.cli.run_tracker \
                "{pred_pxl_fpath}" \
                --default_track_fn {default_track_fn} \
                --track_kwargs {kwargs_str} \
                --clear_annots \
                --site_summary '{site_summary}' \
                --out_site_summaries_fpath "{site_summaries_fpath}" \
                --out_site_summaries_dir "{site_summaries_dpath}" \
                --out_sites_fpath "{sites_fpath}" \
                --out_sites_dir "{sites_dpath}" \
                --out_kwcoco "{poly_kwcoco_fpath}"
            ''').format(**fmtkw)
        return command


class PolygonEvaluation(ProcessNode):
    executable = 'python -m watch.cli.run_metrics_framework'
    group_dname = EVALUATE_NAME

    in_paths = {
        'true_region_dpath',
        'true_site_dpath',
        'sites_fpath',
    }

    out_paths = {
        'eval_dpath': '.',
        'eval_fpath': 'poly_eval.json',
    }

    @profile
    def command(self):
        # self.tmp_dpath = self.paths['eval_dpath'] / 'tmp'
        # self.tmp_dpath = self.paths['eval_dpath'] / 'tmp'
        fmtkw = self.final_config.copy()
        fmtkw['params_argstr'] = self._make_argstr(self.final_algo_config)
        fmtkw['perf_argstr'] = self._make_argstr(self.final_perf_config)
        fmtkw['tmp_dpath'] = self.final_node_dpath / 'tmp'

        # Hack:
        if fmtkw['true_site_dpath'] is None:
            dvc_dpath = _phase2_dvc_data_dpath()
            fmtkw['true_site_dpath'] = dvc_dpath / 'annotations/site_models'
            fmtkw['true_region_dpath'] = dvc_dpath / 'annotations/region_models'

        name_parts = {
            k: v for k, v in sorted(self.condensed.items())
            if 'eval' not in k and (('algo_id' in k) or ('id' not in v))
        }
        fmtkw['name_suffix'] = '-'.join(name_parts.values())

        command = ub.codeblock(
            r'''
            python -m watch.cli.run_metrics_framework \
                --merge=True \
                --name "{name_suffix}" \
                --true_site_dpath "{true_site_dpath}" \
                --true_region_dpath "{true_region_dpath}" \
                --pred_sites "{sites_fpath}" \
                --tmp_dir "{tmp_dpath}" \
                --out_dir "{eval_dpath}" \
                --merge_fpath "{eval_fpath}"
            ''').format(**fmtkw)
        return command


@ub.memoize
def _phase2_dvc_data_dpath():
    import watch
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    return dvc_dpath


class HeatmapEvaluation(ProcessNode):
    executable = 'python -m watch.tasks.fusion.evaluate'
    group_dname = EVALUATE_NAME

    in_paths = {
        'true_dataset',
        'pred_pxl_fpath',
    }

    out_paths = {
        'eval_pxl_dpath': '.',
        'eval_pxl_fpath': 'pxl_eval.json',
    }

    @profile
    def command(self):
        # TODO: better score space
        fmtkw = self.final_config.copy()
        extra_opts = {
            'draw_curves': True,
            'draw_heatmaps': False,
            'viz_thresh': 0.2,
            'workers': 2,
            'score_space': 'video',
        }
        fmtkw['extra_argstr'] = self._make_argstr(extra_opts)  # NOQA
        command = ub.codeblock(
            r'''
            python -m watch.tasks.fusion.evaluate \
                --true_dataset={true_dataset} \
                --pred_dataset={pred_pxl_fpath} \
                --eval_dpath={eval_pxl_dpath} \
                --eval_fpath={eval_pxl_fpath} \
                {extra_argstr}
            ''').format(**fmtkw)
        # .format(**eval_act_pxl_kw).strip().rstrip('\\')
        return command


class KWCocoVisualization(ProcessNode):
    executable = 'python -m watch.cli.coco_visualize_videos'
    group_dname = PREDICT_NAME

    resources = {
        'cpus': 2,
    }

    in_paths = {
        'poly_kwcoco_fpath',
    }

    out_paths = {
        'viz_dpath': '.',
        'viz_stamp_fpath': '_viz.stamp'
    }

    @profile
    def command(self):
        fmtkw = self.final_config.copy()
        name_parts = {
            k: v for k, v in sorted(self.condensed.items())
            if 'eval' not in k and (('algo_id' in k) or ('id' not in v))
        }
        fmtkw['name_suffix'] = '-'.join(name_parts.values())
        # paths = ub.udict(paths)
        # viz_pred_trk_poly_kw = paths.copy()
        # fmtkw['extra_header'] = f"\\n{condensed['trk_pxl_algo_id']}-{condensed['trk_poly_algo_id']}"
        # viz_pred_trk_poly_kw['viz_channels'] = "red|green|blue,salient"
        command = ub.codeblock(
            r'''
            smartwatch visualize \
                "{poly_kwcoco_fpath}" \
                --viz_dpath={viz_dpath} \
                --channels="auto" \
                --stack=only \
                --workers=2 \
                --extra_header="{name_suffix}" \
                --animate=True && touch {viz_stamp_fpath}
            ''').format(**fmtkw)
        return command


###
# Team Feature Nodes
###


class InvariantFeatureComputation(FeatureComputation):
    name = 'invar_feat'

    out_paths = {
        'dst': 'feat_I_{invar_feat_id}.kwcoco.zip'
    }


class MaterialFeatureComputation(FeatureComputation):
    name = 'mat_feat'

    out_paths = {
        'dst': 'feat_M_{mat_feat_id}.kwcoco.zip'
    }


class LandcoverFeatureComputation(FeatureComputation):
    name = 'land_feat'

    out_paths = {
        'dst': 'feat_L_{land_feat_id}.kwcoco.zip'
    }


###
# BAS / SC Nodes
###

# ---

class BAS_HeatmapPrediction(HeatmapPrediction):
    """

    CommandLine:
        xdoctest -m watch.mlops.smart_pipeline BAS_HeatmapPrediction

    Example:
        >>> from watch.mlops.smart_pipeline import *  # NOQA
        >>> node = BAS_HeatmapPrediction()
        >>> node.configure({
        >>>     'tta_time': 2,
        >>>     'package_fpath': 'foo.pt',
        >>>     'test_dataset': 'bar.json',
        >>> })
        >>> command = node.command()
        >>> assert 'tta_time="2"' in command
        >>> print(command)
    """
    name = 'bas_pxl'
    # node_dname = 'bas_pxl/{bas_model}/{bas_test_dset}/{bas_pxl_algo_id}/{bas_pxl_id}'
    node_dname = 'bas_pxl/{bas_pxl_algo_id}/{bas_pxl_id}'

    perf_params = ub.udict(HeatmapPrediction.perf_params) | {
        'with_saliency': True,
        'with_class': False,
        'with_change': False,
    }

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['bas_model'] = 'todo'
        condensed['bas_test_dset'] = 'todo'
        return condensed


class SC_HeatmapPrediction(HeatmapPrediction):
    name = 'sc_pxl'
    # node_dname = 'sc_pxl/{sc_model}/{sc_test_dset}/{sc_pxl_algo_id}/{sc_pxl_id}'
    node_dname = 'sc_pxl/{sc_pxl_algo_id}/{sc_pxl_id}'

    perf_params = ub.udict(HeatmapPrediction.perf_params) | {
        'with_saliency': False,
        'with_class': True,
        'with_change': False,
    }

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['sc_model'] = 'todo'
        condensed['sc_test_dset'] = 'todo'
        return condensed

# ---


class BAS_PolygonPrediction(PolygonPrediction):
    name = 'bas_poly'
    node_dname = 'bas_poly/{bas_poly_algo_id}/{bas_poly_id}'
    default_track_fn = 'saliency_heatmaps'

    @property
    def final_algo_config(self):
        return ub.udict({
            # 'boundaries_as': 'polys'
            "agg_fn": "probs",
        }) | super().final_algo_config


class SC_PolygonPrediction(PolygonPrediction):
    name = 'sc_poly'
    node_dname = 'sc_poly/{sc_poly_algo_id}/{sc_poly_id}'
    default_track_fn = 'class_heatmaps'

    @property
    def final_algo_config(self):
        return ub.udict({
            'boundaries_as': 'polys'
        }) | super().final_algo_config

# ---


class BAS_HeatmapEvaluation(HeatmapEvaluation):
    name = 'bas_pxl_eval'
    node_dname = 'bas_pxl_eval'


class SC_HeatmapEvaluation(HeatmapEvaluation):
    name = 'sc_pxl_eval'
    node_dname = 'sc_pxl_eval'


# ---

class BAS_PolygonEvaluation(PolygonEvaluation):
    name = 'bas_poly_eval'
    node_dname = 'bas_poly_eval'


class SC_PolygonEvaluation(PolygonEvaluation):
    name = 'sc_poly_eval'
    node_dname = 'sc_poly_eval'

# ---


class BAS_Visualization(KWCocoVisualization):
    name = 'bas_poly_viz'
    node_dname = 'bas_poly_viz'


class SC_Visualization(KWCocoVisualization):
    name = 'sc_poly_viz'
    node_dname = 'sc_poly_viz'


# ---


class SiteCropping(ProcessNode):
    name = 'sitecrop'
    node_dname = 'sitecrop/{src_dset}/{regions_id}/{sitecrop_algo_id}/{sitecrop_id}'
    group_dname = PREDICT_NAME

    in_paths = {
        'crop_src_fpath',
        'regions',
    }
    out_paths = {
        'crop_dst_fpath': 'sitecrop.kwcoco.zip'
    }

    algo_params = {
        # 'include_channels': 'red|green|blue|cloudmask',  # fixme: not a good default
        'include_channels': None,
        'exclude_sensors': 'L8',  # fixme: not a good default
        'target_gsd': 4,
        # 'context_factor': 2.0,
        'context_factor': 1.5,
        'force_nodata': -9999,
        'rpc_align_method': 'orthorectify',
    }

    # The best setting of this depends on if the data is remote or not.  When
    # networking, around 20+ workers is a good idea, but that's a very bad idea
    # for local images or if the images are too big.
    perf_params = {
        'verbose': 1,
        # 'workers': 8,
        # 'aux_workers': 16,
        'workers': 32,
        'aux_workers': 4,
        'debug_valid_regions': False,
        'visualize': False,
        'keep': 'img',
        'geo_preprop': 'auto',
    }

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['regions_id'] = 'todo'
        condensed['src_dset'] = 'todo'
        return condensed

    @profile
    def command(self):
        fmtkw = self.final_config.copy()
        algo_config = self.final_algo_config - {'crop_src_fpath'}
        fmtkw['crop_algo_argstr'] = self._make_argstr(algo_config)
        fmtkw['crop_perf_argstr'] = self._make_argstr(self.final_perf_config)
        fmtkw.update(self.final_in_paths)
        fmtkw.update(self.final_out_paths)

        command = ub.codeblock(
            r'''
            python -m watch.cli.coco_align \
                --src "{crop_src_fpath}" \
                --dst "{crop_dst_fpath}" \
                --regions="{regions}" \
                --site_summary=True \
                {crop_perf_argstr} \
                {crop_algo_argstr}
            ''').format(**fmtkw)

        # FIXME: parametarize and only if we need secrets
        # secret_fpath = ub.Path('$HOME/code/watch/secrets/secrets').expand()
        # # if ub.Path.home().name.startswith('jon'):
        #     # if secret_fpath.exists():
        #     #     secret_fpath
        #         # command = f'source {secret_fpath} && ' + command
        command = 'AWS_DEFAULT_PROFILE=iarpa GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR ' + command
        return command


def bas_nodes():
    nodes = {}

    nodes['bas_pxl'] = BAS_HeatmapPrediction()
    nodes['bas_poly'] = BAS_PolygonPrediction()
    nodes['bas_pxl_eval'] = BAS_HeatmapEvaluation()
    nodes['bas_poly_eval'] = BAS_PolygonEvaluation()
    nodes['bas_poly_viz'] = BAS_Visualization()

    nodes['bas_pxl'].inputs['test_dataset'].connect(
        nodes['bas_pxl_eval'].inputs['true_dataset']
    )

    nodes['bas_pxl'].connect(
        nodes['bas_pxl_eval'],
        nodes['bas_poly'],
    )
    nodes['bas_poly'].connect(
        nodes['bas_poly_eval'],
        nodes['bas_poly_viz'],
    )

    if 0:
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

    nodes['sc_pxl'].inputs['test_dataset'].connect(
        nodes['sc_pxl_eval'].inputs['true_dataset']
    )

    nodes['sc_pxl'].connect(
        nodes['sc_pxl_eval'],
        nodes['sc_poly'].connect(
            nodes['sc_poly_eval'],
            nodes['sc_poly_viz'],
        ),
    )
    return nodes


def joint_bas_sc_nodes(with_bas=True, crops=True):
    nodes = {}

    if with_bas:
        nodes.update(bas_nodes())

    if crops:
        nodes['sitecrop'] = SiteCropping()

        if with_bas:
            # outputs['site_summaries_fpath'].connect(
            nodes['bas_poly'].connect(
                nodes['sitecrop'],
                param_mapping={
                    'site_summaries_fpath': 'regions',
                }
            )

            nodes['bas_pxl'].inputs['test_dataset'].connect(
                nodes['sitecrop'].inputs['crop_src_fpath']
            )

    if 1:
        nodes.update(sc_nodes())

        if crops:
            nodes['sitecrop'].connect(
                nodes['sc_pxl'],
                param_mapping={
                    'crop_dst_fpath': 'test_dataset',
                }
            )
        else:
            if with_bas:
                nodes['bas_pxl'].inputs['test_dataset'].connect(
                    nodes['sc_pxl'].inputs['test_dataset']
                )

        if with_bas:
            nodes['bas_poly'].outputs['site_summaries_fpath'].connect(
                nodes['sc_poly'].inputs['site_summary']
            )
    return nodes


def make_smart_pipeline(name):
    """
    Get an unconfigured instance of the SMART pipeline
    """
    from watch.mlops.pipeline_nodes import PipelineDAG
    from functools import partial
    node_makers = {
        'joint_bas_sc': partial(joint_bas_sc_nodes, crops=True),
        'joint_bas_sc_nocrop': partial(joint_bas_sc_nodes, crops=False),
        'crop_sc': partial(joint_bas_sc_nodes, with_bas=False, crops=True),
        'sc': sc_nodes,
        'bas': bas_nodes,
    }
    make_nodes = node_makers[name]
    nodes = make_nodes()
    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()
    return dag


# from xdev import profile  # NOQA
# profile.add_module()
