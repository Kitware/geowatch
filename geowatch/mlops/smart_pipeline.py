"""
Define the individual nodes that can be composed in a SMART pipeline.

The topology of the pipeline will define the resulting filesystem structure
used to store results.


CommandLine:
    xdoctest -m geowatch.mlops.smart_pipeline __doc__:0
    WATCH_DEVCHECK=1 xdoctest -m geowatch.mlops.smart_pipeline __doc__:1

Example:
    >>> # xdoctest: +SKIP
    >>> from geowatch.mlops.smart_pipeline import *  # NOQA
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
    >>>     #print(f'{node.resources=}')
    >>>     print(f'{node.algo_params=}')
    >>>     print('node.depends = {}'.format(ub.urepr(node.depends, nl=1, sort=0)))
    >>>     final = node._finalize_templates()
    >>>     print('final = {}'.format(ub.urepr(final, nl=2)))
    >>>     print('---')
    >>> dag.print_graphs()
    >>> print('dag.config = {}'.format(ub.urepr(dag.config, nl=1)))
    >>> dag_templates = {}
    >>> dag_paths = {}
    >>> for node in dag.nodes.values():
    >>>     dag_templates[node.name] = node._build_templates()['node_dpath']
    >>>     dag_paths[node.name] = node._finalize_templates()['node_dpath']
    >>>     print(node.command())
    >>> import rich
    >>> rich.print('dag_templates = {}'.format(
    >>>     ub.urepr(dag_templates, nl=1, sv=1, align=':', sort=0)))
    >>> rich.print('dag_paths = {}'.format(
    >>>     ub.urepr(dag_paths, nl=1, sv=1, align=':', sort=0)))


Example:
    >>> # xdoctest: +REQUIRES(env:WATCH_DEVCHECK)
    >>> from geowatch.mlops.smart_pipeline import *  # NOQA
    >>> import geowatch
    >>> expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
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
    >>> nodes = make_smart_pipeline_nodes()
    >>> #nodes = bas_nodes()
    >>> from geowatch.mlops.pipeline_nodes import PipelineDAG
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
from geowatch.mlops.pipeline_nodes import ProcessNode


try:
    from xdev import profile  # NOQA
except ImportError:
    profile = ub.identity

PREDICT_NAME  = 'pred'
EVALUATE_NAME = 'eval'


class FeatureComputation(ProcessNode):
    executable = 'python -m geowatch.cli.run_metrics_framework'
    group_dname = PREDICT_NAME

    # node_dname = 'feats/{src_dset}'

    in_paths = {'src'}

    @profile
    def command(self):
        command = ub.codeblock(
            r'''
            geowatch teamfeat invariant # TODO
            ''')
        return command

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['src_dset'] = 'todo'
        return condensed


class FeatureUnion(ProcessNode):
    name = 'featunion'
    executable = 'geowatch feature_union'
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
    executable = 'python -m geowatch.tasks.fusion.predict'
    group_dname = PREDICT_NAME

    # resources = {
    #     'cpus': 2,
    #     'gpus': 1,
    # }

    perf_params = {
        'num_workers': 2,
        'devices': '0,',
        #'accelerator': 'gpu',
        'batch_size': 1,
    }

    in_paths = {
        'package_fpath',
        'test_dataset',
    }

    algo_params = {
        'drop_unused_frames': True,
        'with_saliency': 'auto',
        'with_class': 'auto',
        'with_change': 'auto',
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
            python -m geowatch.tasks.fusion.predict \
                --package_fpath={package_fpath} \
                --test_dataset={test_dataset} \
                --pred_dataset={pred_pxl_fpath} \
                {params_argstr} \
                {perf_argstr}
            ''').format(**fmtkw).rstrip().rstrip('\\').rstrip()
        return command


class PolygonPrediction(ProcessNode):
    executable = 'python -m geowatch.cli.run_tracker'
    group_dname = PREDICT_NAME
    default_track_fn = NotImplemented

    in_paths = {
        'pred_pxl_fpath',
        'site_summary',
        'boundary_region',
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
        external_args = {
            'site_summary', 'boundary_region', 'site_score_thresh',
            'smoothing', 'append_mode',
            'time_pad_before',
            'time_pad_after',
        }
        track_kwargs = self.final_algo_config.copy() - external_args
        track_kwargs = track_kwargs - {'pred_pxl_fpath'}  # not sure why this is needed
        fmtkw['kwargs_str'] = shlex.quote(json.dumps(track_kwargs))
        fmtkw['external_argstr'] = self._make_argstr(self.final_config & external_args)
        # --site_summary '{site_summary}' \
        # --boundary_region '{boundary_region}' \
        command = ub.codeblock(
            r'''
            python -m geowatch.cli.run_tracker \
                --input_kwcoco "{pred_pxl_fpath}" \
                --default_track_fn {default_track_fn} \
                --track_kwargs {kwargs_str} \
                --clear_annots=True \
                --out_site_summaries_fpath "{site_summaries_fpath}" \
                --out_site_summaries_dir "{site_summaries_dpath}" \
                --out_sites_fpath "{sites_fpath}" \
                --out_sites_dir "{sites_dpath}" \
                --out_kwcoco "{poly_kwcoco_fpath}" \
                {external_argstr}
            ''').format(**fmtkw)
        return command


class PolygonEvaluation(ProcessNode):
    executable = 'python -m geowatch.cli.run_metrics_framework'
    group_dname = EVALUATE_NAME

    in_paths = {
        'true_region_dpath',
        'true_site_dpath',
        'sites_fpath',  # bad name, this is site-coercable
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

        handled = {
            'name', 'true_site_dpath', 'merge', 'true_region_dpath',
            'true_region_dpath', 'pred_sites', 'tmp_dir', 'out_dir',
            'merge_fpath',
        }

        fmtkw['params_argstr'] = self._make_argstr(self.final_algo_config - handled)
        fmtkw['perf_argstr'] = self._make_argstr(self.final_perf_config - handled)

        fmtkw['tmp_dpath'] = self.final_node_dpath / 'tmp'

        # Hack:
        if fmtkw['true_site_dpath'] is None:

            raise Exception(f'You must specify true_site_dpath and true_region_dpath fornode= {self.name}')

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
            python -m geowatch.cli.run_metrics_framework \
                --merge=True \
                --name "{name_suffix}" \
                --true_site_dpath "{true_site_dpath}" \
                --true_region_dpath "{true_region_dpath}" \
                --pred_sites "{sites_fpath}" \
                --tmp_dir "{tmp_dpath}" \
                --out_dir "{eval_dpath}" \
                --merge_fpath "{eval_fpath}" \
                --enable_viz=False \
                {params_argstr} \
                {perf_argstr}
            ''').format(**fmtkw)
        command = command.rstrip().rstrip('\\').rstrip()
        return command


@ub.memoize
def _phase2_dvc_data_dpath():
    import geowatch
    dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    return dvc_dpath


class HeatmapEvaluation(ProcessNode):
    executable = 'python -m geowatch.tasks.fusion.evaluate'
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
            'viz_thresh': 'auto',
            'workers': 2,
            'score_space': 'video',
        }
        fmtkw['extra_argstr'] = self._make_argstr(extra_opts)  # NOQA
        command = ub.codeblock(
            r'''
            python -m geowatch.tasks.fusion.evaluate \
                --true_dataset={true_dataset} \
                --pred_dataset={pred_pxl_fpath} \
                --eval_dpath={eval_pxl_dpath} \
                --eval_fpath={eval_pxl_fpath} \
                {extra_argstr}
            ''').format(**fmtkw)
        # .format(**eval_act_pxl_kw).strip().rstrip('\\')
        return command


class KWCocoVisualization(ProcessNode):
    executable = 'python -m geowatch.cli.coco_visualize_videos'
    group_dname = PREDICT_NAME

    # resources = {
    #     'cpus': 2,
    # }

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
            geowatch visualize \
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
        xdoctest -m geowatch.mlops.smart_pipeline BAS_HeatmapPrediction

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> node = BAS_HeatmapPrediction()
        >>> node.configure({
        >>>     'tta_time': 2,
        >>>     'package_fpath': 'foo.pt',
        >>>     'test_dataset': 'bar.json',
        >>> })
        >>> command = node.command()
        >>> assert 'tta_time=2' in command
        >>> print(command)
    """
    name = 'bas_pxl'
    # # node_dname = 'bas_pxl/{bas_model}/{bas_test_dset}/{bas_pxl_algo_id}/{bas_pxl_id}'
    # node_dname = 'bas_pxl/{bas_pxl_algo_id}/{bas_pxl_id}'

    algo_params = ub.udict(HeatmapPrediction.algo_params) | {
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
    # # node_dname = 'sc_pxl/{sc_model}/{sc_test_dset}/{sc_pxl_algo_id}/{sc_pxl_id}'
    # node_dname = 'sc_pxl/{sc_pxl_algo_id}/{sc_pxl_id}'

    algo_params = ub.udict(HeatmapPrediction.algo_params) | {
        'saliency_chan_code': 'ac_salient',
        'with_saliency': True,
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
    # node_dname = 'bas_poly/{bas_poly_algo_id}/{bas_poly_id}'
    default_track_fn = 'saliency_heatmaps'

    @property
    def final_algo_config(self):
        return ub.udict({
            # 'boundaries_as': 'polys'
            "agg_fn": "probs",
        }) | super().final_algo_config


class SC_PolygonPrediction(PolygonPrediction):
    name = 'sc_poly'
    # node_dname = 'sc_poly/{sc_poly_algo_id}/{sc_poly_id}'
    default_track_fn = 'class_heatmaps'

    @property
    def final_algo_config(self):
        return ub.udict({
            'boundaries_as': 'polys'
        }) | super().final_algo_config

# ---


class BAS_HeatmapEvaluation(HeatmapEvaluation):
    name = 'bas_pxl_eval'
    # node_dname = 'bas_pxl_eval'


class SC_HeatmapEvaluation(HeatmapEvaluation):
    name = 'sc_pxl_eval'
    # node_dname = 'sc_pxl_eval'


# ---

class BAS_PolygonEvaluation(PolygonEvaluation):
    name = 'bas_poly_eval'
    # node_dname = 'bas_poly_eval'


class SC_PolygonEvaluation(PolygonEvaluation):
    name = 'sc_poly_eval'
    # node_dname = 'sc_poly_eval'

# ---


class BAS_Visualization(KWCocoVisualization):
    name = 'bas_poly_viz'
    # node_dname = 'bas_poly_viz'


class SC_Visualization(KWCocoVisualization):
    name = 'sc_poly_viz'
    # node_dname = 'sc_poly_viz'


# ---


class Cropping(ProcessNode):
    """
    Used for both site cropping and validation-cropping
    """
    executable = 'python -m geowatch.cli.align'
    group_dname = PREDICT_NAME

    algo_params = {
        # 'include_channels': 'red|green|blue|cloudmask',  # fixme: not a good default
        'include_channels': None,
        'exclude_sensors': 'L8',  # fixme: not a good default
        'target_gsd': 4,
        # 'context_factor': 2.0,
        'context_factor': 1.5,
        'force_nodata': -9999,
        'rpc_align_method': 'orthorectify',
        'convexify_regions': True,
        'force_min_gsd': None,
        'minimum_size': None,
    }

    # The best setting of this depends on if the data is remote or not.  When
    # networking, around 20+ workers is a good idea, but that's a very bad idea
    # for local images or if the images are too big.
    perf_params = {
        'verbose': 1,
        # 'workers': 8,
        # 'aux_workers': 16,
        'img_workers': 32,
        'aux_workers': 4,
        'debug_valid_regions': False,
        'visualize': False,
        'keep': 'img',
        'geo_preprop': 'auto',
    }

    in_paths = {
        'crop_src_fpath',
        'regions',
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

        # NOTE: We are potentially double-specifying regions here at the
        # moment, due to an unresolved design decision in the
        # final_algo_config.

        command = ub.codeblock(
            r'''
            python -m geowatch.cli.coco_align \
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
        # command = 'AWS_DEFAULT_PROFILE=iarpa GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR ' + command
        return command


class SiteClustering(ProcessNode):
    """
    Crop to each image of every site.

    CommandLine:
        xdoctest -m geowatch.mlops.smart_pipeline SiteClustering

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> node = SiteClustering()
        >>> command = node.final_command()
        >>> print(command)
    """
    executable = 'python -m geowatch.cli.cluster_sites'
    group_dname = PREDICT_NAME

    name = 'cluster_sites'
    group_dname = 'crops'

    algo_params = {
        'minimum_size': '128x128@2GSD',
        'maximum_size': '1024x1024@2GSD',
        'crop_time': True,
        'context_factor': 1.5,
    }
    perf_params = {
        'io_workers': 4,
        'draw_clusters': False,
    }

    in_paths = {
        'src',
    }

    out_paths = {
        'dst_dpath': 'clustered',
        'dst_region_fpath': 'clustered.geojson',
    }


class SC_Cropping(Cropping):
    """
    Crop to each image of every site.

    CommandLine:
        xdoctest -m geowatch.mlops.smart_pipeline SC_Cropping

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> node = SC_Cropping()
        >>> command = node.command()
        >>> print(command)
        >>> assert '--regions' in command
    """
    name = 'sc_crop'
    group_dname = 'crops'
    # node_dname = 'sitecrop/{src_dset}/{regions_id}/{sitecrop_algo_id}/{sitecrop_id}'

    algo_params = {
        # 'include_channels': 'red|green|blue|cloudmask',  # fixme: not a good default
        'include_channels': None,
        'exclude_sensors': 'L8',  # fixme: not a good default
        'target_gsd': 4,
        # 'context_factor': 2.0,
        'context_factor': 1.0,
        'force_nodata': -9999,
        'rpc_align_method': 'orthorectify',
        'convexify_regions': True,
        'minimum_size': '128x128@10GSD',
        'force_min_gsd': 2,
        # 'unsigned_nodata': 256, # todo: uncomment
        # 'site_summary': True # todo uncomment
    }

    out_paths = {
        'crop_dst_fpath': 'sitecrop.kwcoco.zip'
    }


class SV_Cropping(Cropping):
    """
    Crop to high res images as the start / end of a sequence

    CommandLine:
        xdoctest -m geowatch.mlops.smart_pipeline SV_Cropping

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> node = SV_Cropping()
        >>> command = node.command()
        >>> print(command)
        >>> assert '--regions' in command
    """
    name = 'sv_crop'
    # node_dname = 'sv_crop/{src_dset}/{regions_id}/{valicrop_algoid}/{sitecrop_id}'

    algo_params = {
        # 'include_channels': 'red|green|blue|cloudmask',  # fixme: not a good default
        'include_sensors': 'WV',
        # None,
        # 'exclude_sensors': 'L8',  # fixme: not a good default
        # 'context_factor': 2.0,
        'context_factor': 1.3,
        'force_nodata': -9999,
        'rpc_align_method': 'orthorectify',
        'minimum_size': '128x128@2GSD',
        'target_gsd': 2,
        'force_min_gsd': 2,
        'convexify_regions': True,
        'num_end_frames': 3,
        'num_start_frames': 3,
    }

    out_paths = {
        'crop_dst_fpath': 'sv_crop.kwcoco.zip'
    }


class SV_DepthPredict(ProcessNode):
    """
    Node for DZYNEs high res depth-based parallel change detector.

    This takes in a kwcoco file with images, and geojson annotations, projects
    those annotations onto the videos, scores each video / track, and then
    writes a modified kwcoco file.

    Example:
        >>> from geowatch.mlops import smart_pipeline
        >>> self = node = smart_pipeline.SV_DepthPredict(root_dpath='/ROOT/DPATH/')
        >>> node.configure({
        >>>     'input_kwcoco': 'my_highres.kwcoco.zip',
        >>>     'input_region': 'myregion.geojson',
        >>>     'input_sites': 'myinput_sites',
        >>>     'model_fpath': 'models/depth_pcd/basicModel2.h5',
        >>> })
        >>> print(node.command())
    """
    name = 'sv_depth_score'
    executable = 'python -m geowatch.tasks.depth_pcd.score_tracks'
    group_dname = PREDICT_NAME

    in_paths = {
        'input_kwcoco',
        'input_region',
    }
    out_paths = {
        'out_kwcoco': 'pred_depth_scores.kwcoco.zip',
    }

    algo_params = {
        'model_fpath': None,
    }

    @profile
    def command(self):
        config = (ub.udict(self.final_config) | self.final_algo_config) | self.final_perf_config
        config_argstr = self._make_argstr(config)
        command = ub.codeblock(
            r'''
            {executable} \
                {config_argstr}
            ''').format(
                executable=self.executable,
                config_argstr=config_argstr,
            )
        return command


class SV_DepthFilter(ProcessNode):
    """
    Node for DZYNEs high res depth-based parallel change detector.

    Takes in a scored kwcoco file from SV_DepthPredict and geojson annotations
    and then filters the annotations based on the scores in the kwcoco file.

    Example:
        >>> from geowatch.mlops import smart_pipeline
        >>> self = node = smart_pipeline.SV_DepthFilter(node_dpath='/MY/OUPUT/DIR/')
        >>> node.configure({
        >>>     'input_kwcoco': 'myscored.kwcoco.zip',
        >>>     'input_region': 'myregion.geojson',
        >>>     'input_sites': 'mysites/*.geojson',
        >>> })
        >>> print(node.command())

    Example:
        >>> from geowatch.mlops import smart_pipeline
        >>> self = node = smart_pipeline.SV_DepthFilter(root_dpath='/ROOT/DPATH/')
        >>> node.configure({
        >>>     'input_kwcoco': 'foo.kwcoco',
        >>>     'input_region': 'region.geojson',
        >>>     'input_sites': 'input_sites',
        >>>     #'output_sites_dpath': 'I_WANT_OUT_SITES_HERE',
        >>>     'output_region_fpath': 'I_WANT_OUT_REGIONS_HERE',
        >>>     'output_site_manifest_fpath': 'I_WANT_SITE_MANIFESTS_HERE',
        >>> })
        >>> print('self.template_out_paths = {}'.format(ub.urepr(self.template_out_paths, nl=1)))
        >>> print('self.final_out_paths = {}'.format(ub.urepr(self.final_out_paths, nl=1)))
        >>> print(node.command())
    """
    name = 'sv_depth_filter'
    executable = 'python -m geowatch.tasks.depth_pcd.filter_tracks'
    group_dname = PREDICT_NAME

    in_paths = {
        'input_kwcoco',
        'input_region',
        'input_sites',
    }
    out_paths = {
        'output_region_fpath': 'sv_depth_out_region.geojson',
        'output_sites_dpath': 'sv_depth_out_sites',
        'output_site_manifest_fpath': 'sv_depth_out_site_manifest.json',
    }

    algo_params = {
        'threshold': 0.4,
    }

    @profile
    def command(self):
        # Not sure why final-config doesn't have everything
        config = (ub.udict(self.final_config) | self.final_algo_config) | self.final_perf_config
        config_argstr = self._make_argstr(config)
        command = ub.codeblock(
            r'''
            {executable} \
                {config_argstr}
            ''').format(
                executable=self.executable,
                config_argstr=config_argstr,
            )
        return command

# from geowatch.tasks.dino_detector import predict as dino_predict
# ub.udict(dino_predict.BuildingDetectorConfig.__default__)


class DinoBoxDetector(ProcessNode):
    """
    Used for both site cropping and validation-cropping

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> node = DinoBoxDetector(root_dpath='/root/dpath/')
        >>> node.configure({
        >>>     'coco_fpath': 'foo.kwcoco',
        >>>     'package_fpath': 'model.pt',
        >>>     'data_workers': 2,
        >>> })
        >>> print(node.command())
        >>> algo_id1 = node.algo_id
        >>> print(f'node.algo_id={node.algo_id}')
        >>> print(f'node.process_id={node.process_id}')
        >>> node.configure({
        >>>     'coco_fpath': 'foo.kwcoco',
        >>>     'package_fpath': 'model.pt',
        >>>     'data_workers': 10,
        >>> })
        >>> algo_id2 = node.algo_id
        >>> node.configure({
        >>>     'fixed_resolution': "10GSD",
        >>> })
        >>> algo_id3 = node.algo_id
        >>> assert algo_id1 == algo_id2, 'perf params dont change hash'
        >>> assert algo_id1 != algo_id3, 'algo params do change hash'
    """
    name = 'sv_dino_boxes'
    executable = 'python -m geowatch.tasks.dino_detector.predict'
    group_dname = PREDICT_NAME

    in_paths = {
        'coco_fpath',
    }
    out_paths = {
        'out_coco_fpath': 'pred_boxes.kwcoco.zip'
    }

    algo_params = {
        'fixed_resolution': "3GSD",
        'window_dims': 256,
        'window_overlap': 0.5,
        'batch_size': 1,
        'package_fpath': None
    }

    # The best setting of this depends on if the data is remote or not.  When
    # networking, around 20+ workers is a good idea, but that's a very bad idea
    # for local images or if the images are too big.
    perf_params = {
        'device': 0,
        'data_workers': 2,
    }

    # @property
    # def condensed(self):
    #     condensed = super().condensed
    #     condensed['regions_id'] = 'todo'
    #     condensed['src_dset'] = 'todo'
    #     return condensed

    @profile
    def command(self):
        fmtkw = {}
        # Not sure why final-config doesn't have everything
        config = (ub.udict(self.final_config) | self.final_algo_config) | self.final_perf_config
        if config['package_fpath'] is None:
            raise ValueError(f'{self.__class__.__name__} / {self.name} requires package_fpath as path to a model')
        fmtkw['config_argstr'] = self._make_argstr(config)
        command = ub.codeblock(
            r'''
            python -m geowatch.tasks.dino_detector.predict \
                {config_argstr}
            ''').format(**fmtkw)
        return command


# from geowatch.tasks.dino_detector import building_validator  # NOQA
# ub.udict(building_validator.BuildingValidatorConfig.__default__)

class SV_DinoFilter(ProcessNode):
    """
    Used for both site cropping and validation-cropping

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> self = node = SV_DinoFilter(root_dpath='/ROOT/DPATH/')
        >>> node.configure({
        >>>     'input_kwcoco': 'foo.kwcoco',
        >>>     'input_region': 'region.geojson',
        >>>     'input_sites': 'input_sites',
        >>>     #'output_sites_dpath': 'I_WANT_OUT_SITES_HERE',
        >>>     'output_region_fpath': 'I_WANT_OUT_REGIONS_HERE',
        >>>     'output_site_manifest_fpath': 'I_WANT_SITE_MANIFESTS_HERE',
        >>> })
        >>> print('self.template_out_paths = {}'.format(ub.urepr(self.template_out_paths, nl=1)))
        >>> print('self.final_out_paths = {}'.format(ub.urepr(self.final_out_paths, nl=1)))
        >>> print(node.command())
    """
    name = 'sv_dino_filter'
    executable = 'python -m geowatch.tasks.dino_detector.building_validator'
    group_dname = PREDICT_NAME

    in_paths = {
        'input_kwcoco',
        'input_region',
        'input_sites',
    }
    out_paths = {
        'output_region_fpath': 'out_region.geojson',
        'output_sites_dpath': 'out_sites',
        'output_site_manifest_fpath': 'out_site_manifest.json',
    }

    algo_params = {
        'box_isect_threshold': 0.1,
        'box_score_threshold': 0.1,
        'start_max_score': 1.0,
        'end_min_score': 0.1,
    }

    @profile
    def command(self):
        # Not sure why final-config doesn't have everything
        config = (ub.udict(self.final_config) | self.final_algo_config) | self.final_perf_config
        config_argstr = self._make_argstr(config)
        command = ub.codeblock(
            r'''
            {executable} \
                {config_argstr}
            ''').format(
                executable=self.executable,
                config_argstr=config_argstr,
            )
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


def make_smart_pipeline_nodes(with_bas=True, building_validation=False,
                              depth_validation=False, site_crops=True,
                              with_acsc=True, site_cluster=False):
    nodes = {}

    sc_input_region = None
    sc_input_kwcoco = None
    bas_input_kwcoco = None

    true_regions = None
    true_sites = None

    if with_bas:
        nodes.update(bas_nodes())
        bas_output_region = nodes['bas_poly'].outputs['site_summaries_fpath']
        bas_output_sites = nodes['bas_poly'].outputs['sites_fpath']

        bas_input_kwcoco = nodes['bas_pxl'].inputs['test_dataset']
        sc_input_kwcoco = bas_input_kwcoco

        # By default the input regions to SC are BAS outputs
        sc_input_region = bas_output_region

        # By default SC will use the same input as BAS
        sc_input_kwcoco = sc_input_kwcoco

        true_regions = nodes['bas_poly_eval'].inputs['true_region_dpath']
        true_sites = nodes['bas_poly_eval'].inputs['true_site_dpath']

    with_sv = building_validation or depth_validation

    if with_sv:
        nodes['sv_crop'] = SV_Cropping()
        if bas_input_kwcoco is not None:
            bas_output_region.connect(nodes['sv_crop'].inputs['regions'])
            bas_input_kwcoco.connect(nodes['sv_crop'].inputs['crop_src_fpath'])
        sv_output_kwcoco = nodes['sv_crop'].outputs['crop_dst_fpath']
        sv_region = bas_output_region
        sv_sites = bas_output_sites

        if building_validation:
            nodes['sv_dino_boxes'] = DinoBoxDetector()
            nodes['sv_dino_filter'] = SV_DinoFilter()
            sv_output_kwcoco.connect(nodes['sv_dino_boxes'].inputs['coco_fpath'])
            nodes['sv_dino_boxes'].outputs['out_coco_fpath'].connect(nodes['sv_dino_filter'].inputs['input_kwcoco'])
            sv_region.connect(nodes['sv_dino_filter'].inputs['input_region'])
            sv_sites.connect(nodes['sv_dino_filter'].inputs['input_sites'])
            sv_region = nodes['sv_dino_filter'].outputs['output_region_fpath']
            sv_sites = nodes['sv_dino_filter'].outputs['output_sites_dpath']

        if depth_validation:
            nodes['sv_depth_score'] = SV_DepthPredict()
            nodes['sv_depth_filter'] = SV_DepthFilter()

            sv_output_kwcoco.connect(nodes['sv_depth_score'].inputs['input_kwcoco'])
            sv_region.connect(nodes['sv_depth_score'].inputs['input_region'])
            # sv_sites.connect(nodes['sv_depth_score'].inputs['input_sites'])
            scored_kwcoco = nodes['sv_depth_score'].outputs['out_kwcoco']

            scored_kwcoco.connect(nodes['sv_depth_filter'].inputs['input_kwcoco'])
            sv_region.connect(nodes['sv_depth_filter'].inputs['input_region'])
            sv_sites.connect(nodes['sv_depth_filter'].inputs['input_sites'])

            sv_region = nodes['sv_depth_filter'].outputs['output_region_fpath']
            sv_sites = nodes['sv_depth_filter'].outputs['output_sites_dpath']

        # Add an evaluation step after bas validation
        nodes['sv_poly_eval'] = PolygonEvaluation(name='sv_poly_eval')
        sv_sites.connect(nodes['sv_poly_eval'].inputs['sites_fpath'])
        # If we have a validation step, use those as inputs to SC
        sc_input_region = sv_region

        if true_regions is not None:
            true_regions.connect(nodes['sv_poly_eval'].inputs['true_region_dpath'])
            true_sites.connect(nodes['sv_poly_eval'].inputs['true_site_dpath'])

    if site_crops:

        if site_cluster:
            nodes['site_cluster'] = SiteClustering()

        nodes['sc_crop'] = SC_Cropping()

        if site_cluster:
            # If we cluster, the AC/SC
            assert sc_input_region is not None
            if sc_input_region is not None:
                sc_input_region.connect(nodes['site_cluster'].inputs['src'])
                nodes['site_cluster'].outputs['dst_region_fpath'].connect(nodes['sc_crop'].inputs['regions'])
        else:
            # If we predicted regions to go to SC, crop to those.
            if sc_input_region is not None:
                sc_input_region.connect(nodes['sc_crop'].inputs['regions'])
                # nodes['bas_pxl'].inputs['test_dataset'].connect(sc_input_kwcoco)

        # If we are site cropping, then use its outputs as SC input
        sc_input_kwcoco = nodes['sc_crop'].outputs['crop_dst_fpath']

    if with_acsc:
        nodes.update(sc_nodes())

        if sc_input_kwcoco is not None:
            sc_input_kwcoco.connect(nodes['sc_pxl'].inputs['test_dataset'])

        if sc_input_region is not None:
            sc_input_region.connect(nodes['sc_poly'].inputs['site_summary'])

        if true_regions is not None:
            true_regions.connect(nodes['sc_poly_eval'].inputs['true_region_dpath'])
            true_sites.connect(nodes['sc_poly_eval'].inputs['true_site_dpath'])
    return nodes


def make_smart_pipeline(name):
    """
    Get an unconfigured instance of the SMART pipeline

    CommandLine:
        xdoctest -m geowatch.mlops.smart_pipeline make_smart_pipeline

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> dag = make_smart_pipeline('sc')
        >>> dag.print_graphs()
        >>> dag.inspect_configurables()

        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> dag = make_smart_pipeline('joint_bas_sc')
        >>> dag.print_graphs()
        >>> dag.inspect_configurables()

        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> dag = make_smart_pipeline('joint_bas_sv_sc')
        >>> dag.print_graphs()
        >>> dag.inspect_configurables()

        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> dag = make_smart_pipeline('full')
        >>> dag.print_graphs()
        >>> dag.inspect_configurables()

    Ignore:
        from geowatch.mlops.smart_pipeline import *  # NOQA
        dag = make_smart_pipeline('joint_bas_sv_sc')
        dag.print_graphs()
        dag.inspect_configurables()
        # Make a graphviz illustration of the DAG
        from kwutil import util_yaml
        # Change the labels a bit

        proc_graph = dag.proc_graph.copy()
        proc_graph.remove_nodes_from([n for n in proc_graph.nodes if n.endswith(('_viz', '_eval'))])

        import kwimage
        relabel = {}
        for node in proc_graph.nodes:
            if node.startswith('bas_'):
                proc_graph.nodes[node]['color'] = kwimage.Color.coerce('kitware_yellow').ashex()
            if node.startswith('sv_'):
                proc_graph.nodes[node]['color'] = kwimage.Color.coerce('kitware_blue').ashex()
            if node.startswith(('sc_', 'ac_')):
                relabel[node] = node.replace('sc_', 'ac_')
                proc_graph.nodes[node]['label'] = node.replace('sc_', 'ac_')
                proc_graph.nodes[node]['color'] = kwimage.Color.coerce('kitware_green').ashex()
        proc_graph = nx.relabel_nodes(proc_graph, relabel)
        nx.write_network_text(proc_graph)


        for node, data in proc_graph.nodes(data=True):
            data['label'] = node
        from graphid import util
        util.util_graphviz.dump_nx_ondisk(proc_graph, 'proc_graph.png')
        import xdev
        xdev.startfile('proc_graph.png')
        # for node, data in dag.io_graph.nodes(data=True):
        #     data['label'] = node
        # util.util_graphviz.dump_nx_ondisk(dag.io_graph, 'io_graph.png')

    Example:
        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> dag = make_smart_pipeline('bas_depth_vali')
        >>> dag.print_graphs()
        >>> dag.inspect_configurables()

        >>> from geowatch.mlops.smart_pipeline import *  # NOQA
        >>> dag = make_smart_pipeline('dzyne_sv_only')
        >>> dag.print_graphs()
        >>> dag.inspect_configurables()
    """
    from geowatch.mlops.pipeline_nodes import PipelineDAG
    from functools import partial
    node_makers = {
        'full': partial(make_smart_pipeline_nodes, site_crops=True,
                        building_validation=True,
                        depth_validation=True, site_cluster=True),


        'joint_bas_sc': partial(make_smart_pipeline_nodes, site_crops=True),

        'joint_bas_sv_sc': partial(make_smart_pipeline_nodes, site_crops=True,
                                   building_validation=True,
                                   depth_validation=True),

        'joint_bas_sc_nocrop': partial(make_smart_pipeline_nodes, site_crops=False),
        'crop_sc': partial(make_smart_pipeline_nodes, with_bas=False, site_crops=True),
        'sc': sc_nodes,
        'bas': bas_nodes,
        'bas_building_vali': partial(make_smart_pipeline_nodes, with_bas=True,
                                     building_validation=True,
                                     site_crops=False, with_acsc=False),
        'bas_depth_vali': partial(make_smart_pipeline_nodes, with_bas=True,
                                  depth_validation=True,
                                  site_crops=False, with_acsc=False),

        'bas_building_and_depth_vali': partial(make_smart_pipeline_nodes, with_bas=True,
                                               building_validation=True,
                                               depth_validation=True,
                                               site_crops=False, with_acsc=False),
    }

    if name == 'dzyne_sv_only':
        nodes = dzyne_sv_only_pipeline()
    else:
        make_nodes = node_makers[name]
        nodes = make_nodes()
    dag = PipelineDAG(nodes)
    dag.build_nx_graphs()
    return dag


def dzyne_sv_only_pipeline():
    r"""

    Demo Schedule Evaluate Inovcation:

        HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
        DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

        python -m geowatch.mlops.schedule_evaluation --params="
            pipeline: dzyne_sv_only
            matrix:
                sv_depth_score.input_region:
                    $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002.geojson
                    # $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CN_C500.geojson
                    # $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CO_C501.geojson
                    # $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KW_C501.geojson
                sv_depth_score.model_fpath:
                    - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
                    # - $DVC_EXPT_DPATH/models/depth_pcd/model3.h5
                sv_depth_filter.threshold:
                    - 0.1
                    # - 0.2
                sv_poly_eval.true_region_dpath:  $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models
                sv_poly_eval.true_site_dpath:  $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_truth/site_models
                pre_poly_eval.true_region_dpath:  $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models
                pre_poly_eval.true_site_dpath:  $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_truth/site_models

            submatrices:
                # For each region, pair it with the appropriate input kwcoco

                - sv_depth_score.input_region: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002.geojson
                  sv_depth_filter.input_sites: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002
                  pre_poly_eval.sites_fpath: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002
                  sv_depth_score.input_kwcoco: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/KR_R002/imgonly-KR_R002-rawbands-small.kwcoco.zip

                - sv_depth_score.input_region: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CN_C500.geojson
                  sv_depth_filter.input_sites: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CN_C500
                  pre_poly_eval.sites_fpath: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CN_C500
                  sv_depth_score.input_kwcoco: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/CN_C000/imgonly-CN_C000-rawbands-small.kwcoco.zip

                - sv_depth_score.input_region: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CO_C501.geojson
                  sv_depth_filter.input_sites: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CO_C501
                  pre_poly_eval.sites_fpath: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/CO_C501
                  sv_depth_score.input_kwcoco: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/CO_C001/imgonly-CO_C001-rawbands-small.kwcoco.zip

                - sv_depth_score.input_region: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KW_C501.geojson
                  sv_depth_filter.input_sites: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KW_C501
                  pre_poly_eval.sites_fpath: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KW_C501
                  sv_depth_score.input_kwcoco: $HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip

            " \
            --root_dpath="$DVC_EXPT_DPATH/_test_dzyne_sv_only" \
            --devices="0,1" --tmux_workers=8 \
            --backend=tmux --queue_name "_test_dzyne_sv_only" \
            --skip_existing=0 \
            --run=0
    """
    nodes = {}

    # Three nodes: crop, score, filter, evaluate

    score_node = nodes['sv_depth_score'] = SV_DepthPredict()
    filter_node = nodes['sv_depth_filter'] = SV_DepthFilter()
    eval_node = nodes['sv_poly_eval'] = PolygonEvaluation(name='sv_poly_eval')

    # Connect the scored output of the predictor to the input of the filter
    score_node.outputs['out_kwcoco'].connect(filter_node.inputs['input_kwcoco'])

    # Ensure the same regions passed to scoring are also passed to filtering
    score_node.inputs['input_region'].connect(filter_node.inputs['input_region'])

    # Add an evaluation step after bas validation
    filter_node.outputs['output_sites_dpath'].connect(eval_node.inputs['sites_fpath'])

    # This adds a dummy node for pre-evaluation. Connections are not working
    # correctly, so they are hacked in the submatrix.
    nodes['pre_poly_eval'] = PolygonEvaluation(name='pre_poly_eval')

    return nodes


# from xdev import profile  # NOQA
# profile.add_module()
