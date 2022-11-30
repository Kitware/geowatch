"""
Define the individual nodes that can be composed in a SMART pipeline.

The topology of the pipeline will define the resulting filesystem structure
used to store results.


CommandLine:
    xdoctest -m watch.mlops.smart_pipeline_nodes __doc__

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
    >>> from watch.mlops.pipeline_nodes import PipelineDAG
    >>> dag = PipelineDAG(nodes, config)
    >>> #
    >>> for node in dag.nodes.values():
    >>>     print(f'node={type(node)}')
    >>>     print(f'{node.name=}')
    >>>     print(f'{node.in_paths=}')
    >>>     print(f'{node.out_paths=}')
    >>>     print(f'{node.resources=}')
    >>>     resolved = node.resolve_templates()
    >>>     print('resolved = {}'.format(ub.repr2(resolved, nl=1)))
    >>> dag.configure(config)
    >>> dag_templates = {}
    >>> for node in dag.nodes.values():
    >>>     dag_templates[node.name] = str(node.output_dpath)
    >>> import rich
    >>> rich.print('dag_templates = {}'.format(ub.repr2(dag_templates, nl=1, sv=1, align=':')))
    >>> #util_networkx.write_network_text(dag.proc_graph)
    >>> #util_networkx.write_network_text(dag.io_graph)
"""
import ubelt as ub
from watch.mlops.pipeline_nodes import ProcessNode


class FeatureComputation(ProcessNode):
    executable = 'python -m watch.cli.run_metrics_framework'
    in_paths = {'src'}
    out_paths = {'dst'}

    def command(self):
        command = ub.codeblock(
            r'''
            smartwatch teamfeat invariant # TODO
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
        'pred_pxl_fpath' : 'pred.kwcoco.json',
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
        'site_summaries_fpath': 'site_summaries_manifest.json',
        'site_summaries_dpath': 'site_summaries',
        'sites_fpath': 'sites_manifest.json',
        'sites_dpath': 'sites',
        'poly_kwcoco_fpath': 'poly.kwcoco.json'
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
        'eval_dpath': '.',
        'eval_fpath': 'poly_eval.json',
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
        'eval_pxl_dpath': '.',
        'eval_pxl_fpath': 'pxl_eval.json',
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
        extra_argstr = self._make_argstr(extra_opts)  # NOQA
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
        'viz_stamp_fpath': '_viz.stamp'
    }

    def command(self):
        # paths = ub.udict(paths)
        # viz_pred_trk_poly_kw = paths.copy()
        # viz_pred_trk_poly_kw['extra_header'] = f"\\n{condensed['trk_pxl_algo_id']}-{condensed['trk_poly_algo_id']}"
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
    # output_dname = 'bas_pxl/{bas_model}/{bas_test_dset}/{bas_pxl_algo_id}/{bas_pxl_id}'
    output_dname = 'bas_pxl//{bas_pxl_algo_id}/{bas_pxl_id}'

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['bas_model'] = 'todo'
        condensed['bas_test_dset'] = 'todo'
        return condensed


class SC_HeatmapPrediction(HeatmapPrediction):
    name = 'sc_pxl'
    # output_dname = 'sc_pxl/{sc_model}/{sc_test_dset}/{sc_pxl_algo_id}/{sc_pxl_id}'
    output_dname = 'sc_pxl/{sc_pxl_algo_id}/{sc_pxl_id}'

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['sc_model'] = 'todo'
        condensed['sc_test_dset'] = 'todo'
        return condensed

# ---


class BAS_PolygonPrediction(PolygonPrediction):
    name = 'bas_poly'
    output_dname = 'bas_poly/{bas_poly_algo_id}/{bas_poly_id}'
    default_track_fn = 'saliency_heatmaps'


class SC_PolygonPrediction(PolygonPrediction):
    name = 'sc_poly'
    output_dname = 'sc_poly/{sc_poly_algo_id}/{sc_poly_id}'
    default_track_fn = 'class_heatmaps'

# ---


class BAS_HeatmapEvaluation(HeatmapEvaluation):
    name = 'bas_pxl_eval'
    output_dname = 'bas_pxl_eval'


class SC_HeatmapEvaluation(HeatmapEvaluation):
    name = 'sc_pxl_eval'
    output_dname = 'sc_pxl_eval'


# ---

class BAS_PolygonEvaluation(PolygonEvaluation):
    name = 'bas_poly_eval'
    output_dname = 'bas_poly_eval'


class SC_PolygonEvaluation(PolygonEvaluation):
    name = 'sc_poly_eval'
    output_dname = 'sc_poly_eval'

# ---


class BAS_Visualization(KWCocoVisualization):
    name = 'bas_viz'
    output_dname = 'bas_viz'


class SC_Visualization(KWCocoVisualization):
    name = 'sc_viz'
    output_dname = 'sc_viz'


# ---


class SiteCropping(ProcessNode):
    name = 'crop'
    output_dname = 'crop/{src_dset}/{regions_id}/{crop_algo_id}/{crop_id}'

    in_paths = {
        'crop_src_fpath'
    }
    out_paths = {
        'crop_dst_fpath'
    }

    perf_params = {
        'verbose': 1,
        'workers': 8,
        'aux_workers': 16,
        'debug_valid_regions': False,
        'visualize': False,
    }

    @property
    def condensed(self):
        condensed = super().condensed
        condensed['regions_id'] = 'todo'
        condensed['src_dset'] = 'todo'
        return condensed

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
        # crop_kwargs = { **paths }
        crop_kwargs = { }
        crop_kwargs['crop_params_argstr'] = self._make_argstr(crop_params)
        crop_kwargs['crop_perf_argstr'] = self._make_argstr(self.perf_params)

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
