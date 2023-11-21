#!/usr/bin/env python3
r"""
This is a SMART-specific analysis of TP/FP/TN/FN site cases with
visualizations.

#### LORES

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python -m geowatch.mlops.confusor_analysis \
    --metrics_node_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_drop7_nowinter_baseline_joint_bas_sc/eval/flat/bas_poly_eval/bas_poly_eval_id_ec937017/ \
    --out_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_drop7_nowinter_baseline_joint_bas_sc/eval/flat/bas_poly_eval/bas_poly_eval_id_ec937017/lores-confusion \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop7/region_models \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop7/site_models \
    --viz_sites=True --reload=0


#### TEST WITH HIGHRES KWCOCO

# ON KR_R002

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python -m geowatch.mlops.confusor_analysis \
    --metrics_node_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_drop7_nowinter_baseline_joint_bas_sc/eval/flat/bas_poly_eval/bas_poly_eval_id_ec937017/ \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop7/region_models \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop7/site_models \
    --src_kwcoco=$DVC_DATA_DPATH/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip \
    --viz_sites=True --reload=0


#### TEST WITH AC KWCOCO

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python -m geowatch.mlops.confusor_analysis \
    --metrics_node_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_demo_ac_eval/eval/flat/sc_poly_eval/sc_poly_eval_id_f689ba48 \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop7/region_models \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop7/site_models \
    --viz_sites=True --reload=1
"""
import scriptconfig as scfg
import ubelt as ub
import math


class ConfusorAnalysisConfig(scfg.DataConfig):
    """
    Requires that IARPA metrics are computed
    """

    metrics_node_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        A path to an IARPA metrics MLops output directory node.

        Use this in the special case that you have an mlops or smartflow output
        directory. This is only used to infer other values.  Not needed if
        other values are specified.
        '''))

    detections_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        truth assignments
        usually detections_tau=0.2_rho=0.5_min_area=0.csv

        Only required if bas_metric_dpath is not given.
        '''))
    proposals_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        detection assignments.
        usually proposals_tau=0.2_rho=0.5_min_area=0.csv

        This does not need to be specified if bas_metric_dpath is given to an
        mlops output path.
        '''))

    src_kwcoco = scfg.Value(None, help='the input kwcoco file to project onto')
    dst_kwcoco = scfg.Value(None, help='the reprojected output kwcoco file to write. Will default based on out_dpath')

    bas_kwcoco = scfg.Value(None, help='path to kwcoco files containing bas heatmap predictions')
    ac_kwcoco = scfg.Value(None, help='path to kwcoco files containing AC heatmap predictions')

    bas_metric_dpath = scfg.Value(None, help='A path to bas metrics if det/prop paths are not specified')

    pred_sites = scfg.Value(None, help='the path to the predicted sites manifest / directory / globstr')

    stage_to_sites = scfg.Value(None, help='A YAML mapping from stages in a pipeline to intermediate site output')
    stage_to_metrics = scfg.Value(None, help='A YAML mapping from stages in a pipeline to intermediate metrics')

    region_id = scfg.Value(None, help='the id for the region')
    true_site_dpath = scfg.Value(None, help='input')
    true_region_dpath = scfg.Value(None, help='input')

    performer_id = scfg.Value('kit', help='the performer id')

    viz_sites = scfg.Value(False, isflag=True, help='if True writes case visualizations')
    viz_summary = scfg.Value(False, isflag=True, help='too slow')

    out_dpath = scfg.Value(None, help='where to write results')

    reload = scfg.Value(False, isflag=True, help='if True, reload previously dumped confusion cases.')

    embed = scfg.Value(False, isflag=True, help='if True, embed to interact')

    strict = scfg.Value(False, isflag=True, help='if True dont ignore errors')

    def __post_init__(self):
        if self.bas_metric_dpath is not None:
            self.bas_metric_dpath = ub.Path(self.bas_metric_dpath)

        if self.true_region_dpath is not None:
            self.true_region_dpath = ub.Path(self.true_region_dpath)

        if self.true_site_dpath is not None:
            self.true_site_dpath = ub.Path(self.true_site_dpath)

        if self.out_dpath is not None:
            self.out_dpath = ub.Path(self.out_dpath)

        from kwutil.util_yaml import Yaml
        self.stage_to_sites = Yaml.coerce(self.stage_to_sites)
        self.stage_to_metrics = Yaml.coerce(self.stage_to_metrics)

    def _infer_from_mlops_node(self):
        import json
        if self.metrics_node_dpath is not None:
            # Infer things using assumptions about mlops directory structures
            self.metrics_node_dpath = ub.Path(self.metrics_node_dpath)

            if self.pred_sites is None:
                pred_sites_cands = list(self.metrics_node_dpath.glob('.pred/*/*/sites'))
                if len(pred_sites_cands) == 0:
                    pred_sites_cands = list(self.metrics_node_dpath.glob('.pred/*/*/sv_depth_out_sites'))
                assert len(pred_sites_cands) == 1, 'mlops assumption violated ' + str(len(pred_sites_cands))
                self.pred_sites = pred_sites_cands[0]

            if self.src_kwcoco is None:
                src_kwcoco_cands = list(self.metrics_node_dpath.glob('.pred/*/*/poly.kwcoco.zip'))
                if len(src_kwcoco_cands) == 0:
                    src_kwcoco_cands = list(self.metrics_node_dpath.glob('.pred/bas_poly_eval/*/.pred/bas_poly/*/poly.kwcoco.zip'))
                assert len(src_kwcoco_cands) == 1, 'mlops assumption violated'
                self.src_kwcoco = src_kwcoco_cands[0]

            if self.bas_kwcoco is None:

                # Hack for AC node on full pipeline
                # Need to have a nicer way to get a reference to the BAS coco
                # file for AC nodes.
                bas_kwcoco_cands = list(self.metrics_node_dpath.glob('.pred/sc_poly/*/.pred/sc_pxl/*/.pred/sc_crop/*/.pred/cluster_sites/*/.pred/sv_depth_filter/*/.pred/sv_dino_filter/*/.pred/sv_dino_boxes/*/.pred/sv_crop/*/.pred/bas_poly/*/poly.kwcoco.zip'))

                if len(bas_kwcoco_cands) == 0:
                    # hack: not robust
                    bas_kwcoco_cands = list(self.metrics_node_dpath.glob('.pred/*/*/poly.kwcoco.zip'))
                if len(bas_kwcoco_cands) == 0:
                    bas_kwcoco_cands = list(self.metrics_node_dpath.glob('.pred/bas_poly_eval/*/.pred/bas_poly/*/poly.kwcoco.zip'))
                assert len(bas_kwcoco_cands) == 1, 'mlops assumption violated'
                self.bas_kwcoco = bas_kwcoco_cands[0]

            overall_cands = list(self.metrics_node_dpath.glob('*/overall'))
            assert len(overall_cands) == 1, 'mlops assumption violated'
            overall_dpath = overall_cands[0]

            self.bas_metric_dpath = overall_dpath / 'bas'
            self.region_id = overall_dpath.parent.name

            @ub.memoize
            def get_job_config():
                job_config_fpath = self.metrics_node_dpath / 'job_config.json'
                job_config = json.loads(job_config_fpath.read_text())
                return job_config

            if self.true_region_dpath is None:
                # TODO: also read in things like model names and hashids if
                # possible.
                job_config = get_job_config()
                self.true_region_dpath = job_config['bas_poly_eval.true_region_dpath']

            if self.true_site_dpath is None:
                job_config = get_job_config()
                self.true_site_dpath = job_config['bas_poly_eval.true_site_dpath']

            if self.out_dpath is None:
                self.out_dpath = (self.metrics_node_dpath / 'confusion_analysis')

        if self.dst_kwcoco is None and self.out_dpath is not None:
            self.dst_kwcoco = self.out_dpath / 'confusion_kwcoco' / 'confusion.kwcoco.zip'

        if self.bas_metric_dpath is not None:
            if self.detections_fpath is None:
                self.detections_fpath = self.bas_metric_dpath / 'detections_tau=0.2_rho=0.5_min_area=0.csv'
            if self.proposals_fpath is None:
                self.proposals_fpath = self.bas_metric_dpath / 'proposals_tau=0.2_rho=0.5_min_area=0.csv'

        self.__post_init__()


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/watch/geowatch/mlops/confusor_analysis.py main
        HAS_DVC=1 xdoctest -m geowatch.mlops.confusor_analysis main:0

    Example:
        >>> # xdoctest: +REQUIRES(env:HAS_DVC)
        >>> from geowatch.mlops.confusor_analysis import *  # NOQA
        >>> import geowatch
        >>> data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> region_id = 'NZ_R001'
        >>> true_site_dpath = data_dvc_dpath / 'annotations/drop6/site_models'
        >>> true_region_dpath = data_dvc_dpath / 'annotations/drop6/region_models'
        >>> src_kwcoco = data_dvc_dpath / f'Drop6-MeanYear10GSD/imgonly-{region_id}.kwcoco.zip'
        >>> dst_kwcoco = data_dvc_dpath / f'Drop6-MeanYear10GSD/confusor-{region_id}.kwcoco.zip'
        >>> dag_dpath = ub.Path('/data/joncrall/dvc-repos/smart_expt_dvc/_airflow/ta2_preeval10_pyenv_t33_post3')
        >>> dpath = dag_dpath / region_id
        >>> bas_metric_dpath = dpath / 'metrics/overall/bas'
        >>> #bas_metric_dpath = dpath / 'local_metrics' / region_id / 'overall/bas'
        >>> out_dpath = dpath / 'local_metrics'
        >>> cmdline = 0
        >>> kwargs = dict(
        >>>     bas_metric_dpath=bas_metric_dpath,
        >>>     pred_sites=(dpath / 'sc-fusion/sc_out_site_models'),
        >>>     true_site_dpath=true_site_dpath,
        >>>     true_region_dpath=true_region_dpath,
        >>>     out_dpath=out_dpath,
        >>>     dst_kwcoco=dst_kwcoco,
        >>>     src_kwcoco=src_kwcoco,
        >>>     region_id=region_id,
        >>> )
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = ConfusorAnalysisConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    config._infer_from_mlops_node()
    rich.print('config = ' + ub.urepr(config, nl=1, align=':'))

    self = ConfusionAnalysis(config)
    rich.print(f'Will Output Confusion Analysis In: [link={self.out_dpath}]{self.out_dpath}[/link]')
    rich.print('\n')

    if config.reload:
        try:
            self.reload()
        except Exception:
            if config.reload != 'auto':
                raise
            config.reload = False

    if not config.reload:
        self.load_confusion_assignment()
        self.load_geojson_models()
        self.add_confusion_to_geojson_models()
        self.build_hard_cases()

    if config.embed:
        # Using embed this way because our linter disallows the regular debuggy
        # way.
        from xdev import embed
        embed()

    if not config.reload:
        self.dump_confusion_geojson()
        self.dump_hardneg_geojson()
        if config.src_kwcoco is not None:
            self.dump_confusion_kwcoco()
            self.dump_hardneg_kwcoco()
        rich.print(f'Dumped Confusion Analysis: [link={self.out_dpath}]{self.out_dpath}[/link]')

    if config.viz_sites:
        self.dump_site_case_viz()
    if config.viz_summary:
        self.dump_summary_viz()


class ConfusionAnalysis:
    """
    Note: this class is a refactoring of a large mono-function so its functions
    need to be called in a particular order.
    """

    def __init__(self, config):
        self.config = config

        self.region_id = config.region_id

        self.true_sites = None
        self.pred_sites = None
        self.id_to_pred_site = None
        self.true_region_model = None
        self.id_to_true_site = None
        self.id_to_pred_site = None
        self.type_to_sites = None
        self.type_to_summary = None

        self.true_confusion_rows = None
        self.pred_confusion_rows = None

        self.new_sites = None
        self.new_region = None

        self.cfsn_coco = None

        self.out_dpath = config.out_dpath

        self.enriched_dpath = self.out_dpath / 'enriched_annots'
        self.enriched_sites_dpath = (self.enriched_dpath / 'site_models')
        self.enriched_region_dpath = (self.enriched_dpath / 'region_models')
        self.cfsn_group_dpath = self.out_dpath / 'confusion_groups'

    # @classmethod
    # def from_config(cls, config):
    #     ...

    def reload(self):
        """
        Reloads data we assume is previously written

        FIXME:
            be robust to the case of anyone putting bad files in these dirs
        """
        import kwcoco
        from geowatch.geoannots.geomodels import SiteModel
        from geowatch.geoannots.geomodels import RegionModel
        coco_dset = kwcoco.CocoDataset(self.config.dst_kwcoco)

        region_paths = []
        site_dpaths = []
        for p in self.cfsn_group_dpath.ls():
            if p.endswith('.geojson'):
                region_paths.append(p)
            else:
                site_dpaths.append(p)

        type_to_summary = ub.udict({p.stem: RegionModel.coerce(p) for p in region_paths})
        type_to_summary.map_values(lambda x: len(x['features']))

        type_to_sites = ub.udict({p.name: list(SiteModel.coerce_multiple(p)) for p in site_dpaths})
        type_to_sites.map_values(len)

        self.type_to_sites = type_to_sites
        self.type_to_summary = type_to_summary
        self.cfsn_coco = coco_dset

    def load_geojson_models(self):
        """
        Loads the true and predicted site models
        """
        from geowatch.geoannots.geomodels import SiteModel
        from geowatch.geoannots.geomodels import RegionModel
        from geowatch.geoannots.geomodels import SiteModelCollection
        from geowatch.utils import util_gis
        import rich
        import itertools as it

        config = self.config
        region_id = config.region_id

        true_site_dpath = config.true_site_dpath
        true_region_dpath = config.true_region_dpath

        pred_site_fpaths = list(util_gis.coerce_geojson_paths(config.pred_sites))
        rm_files = list(true_region_dpath.glob(region_id + '*.geojson'))
        gt_files = list(true_site_dpath.glob(region_id + '*.geojson'))
        sm_files = pred_site_fpaths

        true_sites = SiteModelCollection(list(SiteModel.coerce_multiple(gt_files)))
        pred_sites = SiteModelCollection(list(SiteModel.coerce_multiple(sm_files)))

        # for site in true_sites:
        #     if site.start_date is None and site.end_date is None:
        #         raise AssertionError
        orig_regions = list(RegionModel.coerce_multiple(rm_files))
        if len(orig_regions) != 1:
            msg = ub.paragraph(
                f'''
                Unable to load groundtruth region files. Got {orig_regions=}.
                Set the value of ``true_region_dpath`` correctly. The current
                value is {config.true_region_dpath}
                ''')
            rich.print(f'[yellow]WARNING {msg}')
            true_region_model = None
        else:
            true_region_model = orig_regions[0]
            true_region_model.fixup()
            true_region_model.header['properties'].setdefault('cache', {})

        # Ensure all site data has misc-info
        # Ensure all data cast to site models
        for site in it.chain(pred_sites, true_sites):
            for feat in site.features:
                feat['properties'].setdefault('cache', {})

        id_to_true_site = {s.site_id: s for s in true_sites}
        id_to_pred_site = {s.site_id: s for s in pred_sites}

        self.true_sites = true_sites
        self.pred_sites = pred_sites
        self.id_to_true_site = id_to_true_site
        self.id_to_pred_site = id_to_pred_site

        self.true_region_model = true_region_model

    def load_confusion_assignment(self):
        """
        Load the association between true and predicted site models computed by
        the metrics framework.

        Note:
            The possible confusion codes and the corresponding confusion_color
            are assigned in :py:obj:`geowatch.heuristics.IARPA_CONFUSION_COLORS`
        """
        config = self.config

        import rich
        import pandas as pd
        from geowatch import heuristics

        performer_id = config.performer_id
        region_id = config.region_id

        config.detections_fpath = ub.Path(config.detections_fpath)
        config.proposals_fpath = ub.Path(config.proposals_fpath)

        confusion_unavailable = False
        if not config.proposals_fpath.exists():
            confusion_unavailable = True
            rich.print('[yellow]WARNING: no proposal path, cannot compute confusion')
        if not config.detections_fpath.exists():
            rich.print('[yellow]WARNING: no detection path, cannot compute confusion')
            confusion_unavailable = True

        if confusion_unavailable:
            self.true_confusion_rows = []
            self.pred_confusion_rows = []
            return

        true_assign = pd.read_csv(config.detections_fpath)
        pred_assign = pd.read_csv(config.proposals_fpath)

        rich.print(' --- Loaded True Assignment From : {config.detections_fpath} ---')
        rich.print(true_assign)
        rich.print(' --- Loaded Pred Assignment From : {config.proposals_fpath} ---')
        rich.print(pred_assign)
        rich.print(f'{len(true_assign)=}')
        rich.print(f'{len(pred_assign)=}')

        needs_recompute = any('_seq_' in m or m.startswith('seq_') for m in pred_assign['site model'] if m)
        assert not needs_recompute

        ### Assign a confusion label to each truth and predicted annotation
        true_confusion_rows = []
        pred_confusion_rows = []
        site_to_status = {}

        score_cols = [c for c in true_assign.columns if 'score' in c]
        assert len(score_cols) == 1
        score_col = score_cols[0]

        true_te_cols = [
            'spatial overlap',
            'temporal iot',
            'temporal iop',
            'site count',
            'association status',
            'associated',
            'color code',
            'site area',
        ]
        for row in true_assign.to_dict('records'):
            true_site_id = row['truth site']

            true_site_id = fix_site_id(true_site_id, region_id, performer_id)
            pred_site_ids = []
            truth_status = row['site type']
            site_to_status[true_site_id] = truth_status
            if isinstance(row['matched site models'], str):
                for name in row['matched site models'].split(','):
                    pred_site_id = name
                    pred_site_id = fix_site_id(pred_site_id, region_id, performer_id)
                    pred_site_ids.append(pred_site_id)
            has_positive_match = len(pred_site_ids)
            true_cfsn = heuristics.iarpa_assign_truth_confusion(truth_status, has_positive_match)

            if true_cfsn is None:
                print('truth_status = {}'.format(ub.urepr(truth_status, nl=1)))
                print('has_positive_match = {}'.format(ub.urepr(has_positive_match, nl=1)))
                raise AssertionError('no true cfsn')

            cfsn_row = {
                'true_site_id': true_site_id,
                'pred_site_ids': pred_site_ids,
                'type': true_cfsn,
            }
            for te_key in true_te_cols:
                our_key = 'te_' + te_key.replace(' ', '_')
                cfsn_row[our_key] = nan_to_null(row[te_key])
            cfsn_row['te_score'] = row[score_col]
            true_confusion_rows.append(cfsn_row)

        pred_te_cols = [
            'site count',
            'association status',
            'associated',
            'color code',
            'site area',
        ]
        for row in pred_assign.to_dict('records'):
            pred_site_id = row['site model']
            pred_site_id = fix_site_id(pred_site_id, region_id, performer_id)
            true_site_ids = []
            truth_match_statuses = []
            if isinstance(row['matched truth sites'], str):
                for name in row['matched truth sites'].split(','):
                    true_site_id = name
                    true_site_id = fix_site_id(true_site_id, region_id, performer_id)
                    true_status = site_to_status[true_site_id]
                    truth_match_statuses.append(true_status)
                    true_site_ids.append(true_site_id)

            pred_cfsn = heuristics.iarpa_assign_pred_confusion(truth_match_statuses)
            if pred_cfsn is None:
                print('row = {}'.format(ub.urepr(row, nl=1)))
                print('truth_match_statuses = {}'.format(ub.urepr(truth_match_statuses, nl=1)))
                raise AssertionError('no pred cfsn')

            cfsn_row = {
                'pred_site_id': pred_site_id,
                'true_site_ids': true_site_ids,
                'type': pred_cfsn,
                'te_associated': nan_to_null(row['associated']),
                'te_color_code': nan_to_null(row['color code']),
                'te_association_status': nan_to_null(row['association status']),
                'te_site_count': nan_to_null(row['site count']),
            }
            for te_key in pred_te_cols:
                our_key = 'te_' + te_key.replace(' ', '_')
                cfsn_row[our_key] = nan_to_null(row[te_key])
            pred_confusion_rows.append(cfsn_row)

        for row in true_confusion_rows + pred_confusion_rows:
            row['color'] = heuristics.IARPA_CONFUSION_COLORS.get(row['type'])

        self.true_confusion_rows = true_confusion_rows
        self.pred_confusion_rows = pred_confusion_rows

        if 1:
            true_df = pd.DataFrame(self.true_confusion_rows)
            pred_df = pd.DataFrame(self.pred_confusion_rows)
            print(pred_df[['type', 'te_color_code', 'te_association_status', 'te_associated', 'te_site_count', 'color']].value_counts())
            print(true_df[['type', 'te_color_code', 'te_association_status', 'te_associated', 'te_site_count', 'color']].value_counts())

    def load_new_stage_stuff(self):
        """
        We should redo confusion stuff at each stage of the pipeline and
        determine when mistakes and good decisions are made.
        """
        from geowatch.mlops.smart_result_parser import load_iarpa_evaluation
        from geowatch.geoannots.geomodels import SiteModel
        # from geowatch.geoannots.geomodels import RegionModel
        from geowatch.geoannots.geomodels import SiteModelCollection
        import pandas as pd
        import rich

        # New stuff with stages
        stage_preds = {}
        for stage, sites_dpath in self.config.stage_to_sites.items():
            sites = SiteModelCollection(list(SiteModel.coerce_multiple(sites_dpath)))
            stage_preds[stage] = sites

        def fff(c):
            return [s.header for s in c]

        stage_order = ['bas', 'dzyne-depth-sv', 'dino-sv', 'acsc']
        stage_to_df = ub.udict(stage_preds).map_values(lambda x: x.as_region_model().pandas_summaries())
        stage_to_df = stage_to_df.subdict(stage_order)
        stage_to_df.map_values(len)
        stage_to_df = stage_to_df.map_values(lambda d: d.set_index('site_id', drop=False))

        df1 = stage_to_df['bas']
        df2 = stage_to_df['dzyne-depth-sv']
        df3 = stage_to_df['dino-sv']
        df4 = stage_to_df['acsc']

        did_depth_filter = (df1.loc[df1.site_id]['status'] != df2.loc[df1.site_id]['status'])
        depth_filtered = did_depth_filter[did_depth_filter]  # NOQA

        did_dino_filter = (df3.loc[df2.site_id]['status'] != df2.loc[df2.site_id]['status'])
        dino_filtered = did_dino_filter[did_dino_filter].index  # NOQA

        did_ac_filter = (df3.loc[df4.site_id]['status'] != df4.loc[df4.site_id]['status'])
        acsc_filtered = did_ac_filter[did_ac_filter]  # NOQA

        ub.oset(df1['site_id']) - ub.oset(df2['site_id'])
        ub.oset(df2['site_id']) - ub.oset(df3['site_id'])
        ub.oset(df3['site_id']) - ub.oset(df4['site_id'])

        # New stuff with stages
        rows = []
        for stage, metrics_fpath in self.config.stage_to_metrics.items():
            iarpa_result = load_iarpa_evaluation(ub.Path(metrics_fpath))
            row = iarpa_result['metrics']
            row['stage'] = stage
            rows.append(row)
        datacols = ['stage', 'bas_f1', 'sc_macro_f1', 'macro_f1_active', 'macro_f1_siteprep', 'bas_tp', 'bas_fp', 'bas_fn', 'bas_ffpa', 'bas_ppv', 'bas_tpr']
        table = pd.DataFrame(rows)
        rich.print(table[datacols])

    def add_confusion_to_geojson_models(self):
        """
        Modify region / site models with a confusion info in their cache.

        Add properties to each site model (and their associated site summaries)
        indicating the type of confusion they are causing based on the
        following "confusion specs".

        .. code::

            True Confusion Spec
            -------------------

            "cache":  {
                "confusion": {
                    "true_site_id": str,          # redundant site id information,
                    "pred_site_ids": List[str],   # the matching predicted site ids,
                    "type": str,                  # the type of true confusion assigned by T&E
                    "color": str,                 # a named color coercable via kwimage.Color.coerce
                }
            }

            Predicted Confusion Spec
            -------------------

            "cache":  {
                "confusion": {
                    "pred_site_id": str,          # redundant site id information,
                    "true_site_ids": List[str],   # the matching predicted site ids,
                    "type": str,        # the type of predicted confusion assigned by T&E
                    "color": str,       # a named color coercable via kwimage.Color.coerce
                }
            }
        """
        # Add the confusion info as misc data in new site files
        # We will later reproject them onto the truth for visualization.
        from geowatch.geoannots.geomodels import SiteModelCollection

        true_region_model = self.true_region_model
        id_to_true_site = self.id_to_true_site
        id_to_pred_site = self.id_to_pred_site
        true_sites = self.true_sites
        pred_sites = self.pred_sites
        performer_id = self.config.performer_id

        true_confusion_rows = self.true_confusion_rows
        pred_confusion_rows = self.pred_confusion_rows

        # Add confusion metadata to predicted and truth models
        # https://gis.stackexchange.com/questions/346518/opening-geojson-style-properties-in-qgis
        for row in true_confusion_rows:
            site = id_to_true_site[row['true_site_id']]
            site.header['properties']['cache']['confusion'] = row

        for row in pred_confusion_rows:
            try:
                pred_site_id = row['pred_site_id']
                site = id_to_pred_site[pred_site_id]
                site.header['properties']['cache']['confusion'] = row
            except Exception:
                print('warning: unexpected key error')

        # TODO: need to figure out if it was correctly or incorrectly
        # rejected, and what stage rejected it.
        for site in id_to_pred_site.values():
            if 'confusion' not in site.header['properties']['cache']:
                site.header['properties']['cache']['confusion'] = {
                    'type': 'unhandled_cfsn_' + site.header['properties']['status'],
                    'color': 'pink',
                }
        for site in id_to_true_site.values():
            if 'confusion' not in site.header['properties']['cache']:
                site.header['properties']['cache']['confusion'] = {
                    'type': 'unhandled_cfsn_' + site.header['properties']['status'],
                    'color': 'pink',
                }

        VALIDATE = 0
        if VALIDATE:
            all_models = SiteModelCollection(pred_sites + true_sites)
            all_models.fixup()
            all_models.validate(stop_on_failure=False, strict=False)
            # all_models.validate(workers=0)

        # Group by confusion type
        true_type_to_sites = ub.ddict(list)
        for true_site_id, true_site in id_to_true_site.items():
            confusion_type = true_site.header['properties']['cache']['confusion']['type']
            true_type_to_sites[confusion_type].append(true_site)

        pred_type_to_sites = ub.ddict(list)
        for pred_site_id, pred_site in id_to_pred_site.items():
            confusion_type = pred_site.header['properties']['cache']['confusion']['type']
            pred_type_to_sites[confusion_type].append(pred_site)

        assert set(true_type_to_sites).isdisjoint(set(pred_type_to_sites))

        type_to_sites = ub.udict(pred_type_to_sites) | true_type_to_sites
        type_to_sites['pred'] = list(id_to_pred_site.values())
        type_to_sites['true'] = list(id_to_true_site.values())

        # Create site summaries for each type of confusion
        type_to_summary = {}
        for group_type, sites in type_to_sites.items():
            sites = SiteModelCollection(sites)
            if true_region_model is not None:
                cfsn_summary = sites.as_region_model(
                    true_region_model.header,
                    region_id=self.config.region_id
                )
            else:
                cfsn_summary = sites.as_region_model(
                    region_id=self.config.region_id, strict=False)

            if group_type not in {'true', 'pred'}:
                if 'cache' not in cfsn_summary.header['properties']:
                    cfsn_summary.header['properties']['cache'] = {}
                cfsn_summary.header['properties']['cache']['confusion_type'] = group_type
                cfsn_summary.header['properties']['cache']['originator'] = performer_id
            type_to_summary[group_type] = cfsn_summary

        for group_type, summary in type_to_summary.items():
            for feat in summary.features:
                if 'cache' not in feat['properties']:
                    feat['properties']['cache'] = {}
                assert 'cache' in feat['properties']

        for group_type, sites in type_to_sites.items():
            for site in sites:
                for feat in site.features:
                    if 'cache' not in feat['properties']:
                        feat['properties']['cache'] = {}
                    assert 'cache' in feat['properties']

        self.type_to_sites = type_to_sites
        self.type_to_summary = type_to_summary

    def build_hard_cases(self):
        """
        Build feedback to retrain on. Does to things:

        1. Find the false positive cases that do not overlap with any truth and
        add them as negative examples to a new set of "hard annotations".

        2. Finds the false negative examples and increases their weight.
        """
        from geowatch.utils import util_gis
        import rich

        true_sites = self.true_sites
        pred_sites = self.pred_sites
        id_to_pred_site = self.id_to_pred_site
        true_region_model = self.true_region_model
        region_id = self.region_id
        id_to_pred_site = self.id_to_pred_site

        if true_region_model is None:
            rich.print('[yellow] WARNING: Cannot build hard cases without region truth')
            self.new_sites = None
            self.new_region = None
            return

        pred_region_model = pred_sites.as_region_model(true_region_model.header)
        pred_df = pred_region_model.pandas_summaries()
        true_df = true_region_model.pandas_summaries()
        idx1_to_idx2 = util_gis.geopandas_pairwise_overlaps(pred_df, true_df)

        # Find predicted annotations that have no truth overlap
        non_intersecting_idxs = [idx1 for idx1, idx2s in idx1_to_idx2.items() if not len(idx2s)]
        cand_df = pred_df.iloc[non_intersecting_idxs]

        hard_positive_site_ids = []
        for true_site in true_sites:
            misc = true_site.header['properties']['cache']
            if misc['confusion']['type'] in 'gt_false_neg':
                hard_positive_site_ids.append(true_site.header['properties']['site_id'])

        hard_negative_sites = []
        for site_id in cand_df['site_id']:
            pred_site = id_to_pred_site[site_id]
            misc = pred_site.header['properties']['cache']
            if misc['confusion']['type'] in 'sm_completely_wrong':
                hard_negative_sites.append(pred_site.deepcopy())

        orig_site_num = int(true_df['site_id'].max().split('_')[-1])
        # hack to have true / pred together and comply with the strict schema
        base_site_num = max(700, orig_site_num)
        for num, hard_neg in enumerate(hard_negative_sites, start=base_site_num):
            header_prop = hard_neg.header['properties']
            header_prop['site_id'] = region_id + f'_{num:04d}'
            header_prop['status'] = 'negative'
            header_prop['model_content'] = 'annotation'
            header_prop['cache'].pop('confusion', None)
            header_prop['cache']['kwcoco'] = {'weight': 1.5}
            header_prop['comments'] = 'hard_negative'
            for obs in hard_neg.observations():
                props = obs['properties']
                props['current_phase'] = None
                props['is_site_boundary'] = None
                props.pop('predicted_phase_transition', None)
                props.pop('predicted_phase_transition_date', None)

        # Need to build site summaries from site models.
        hard_neg_summaries = [s.as_summary() for s in hard_negative_sites]
        new_region = true_region_model.deepcopy()
        new_region['features'] += hard_neg_summaries
        new_true_sites = [s.deepcopy() for s in true_sites]

        # Upweight hard positive true sites
        print('hard_positive_site_ids = {}'.format(ub.urepr(hard_positive_site_ids, nl=1)))
        new_true_props = [n.header['properties'] for n in new_true_sites] + [s['properties'] for s in new_region.site_summaries()]
        for prop in new_true_props:
            if 'cache' not in prop:
                prop['cache'] = {}
            prop['cache'].pop('confusion', None)
            if prop['site_id'] in hard_positive_site_ids:
                prop['cache']['kwcoco'] = {'weight': 2.0}
                if 'comments' not in prop or not prop['comments']:
                    prop['comments'] = 'hard_positive'
                else:
                    prop['comments'] += ';hard_positive'

        # Add in hard negatives
        new_sites = new_true_sites + hard_negative_sites
        self.new_sites = new_sites
        self.new_region = new_region

    def dump_confusion_geojson(self):
        """
        Write confusion geojson files for visualization and analysis
        """
        import json
        import kwcoco
        config = self.config
        type_to_sites = self.type_to_sites
        type_to_summary = self.type_to_summary

        # Dump confusion categorized site models to disk
        cfsn_group_dpath = self.cfsn_group_dpath
        print(ub.urepr(type_to_sites.map_values(len)))

        for group_type, sites in type_to_sites.items():
            import xdev
            with xdev.embed_on_exception_context:
                cfsn_summary = type_to_summary[group_type]
                group_site_dpath = (cfsn_group_dpath / group_type).ensuredir()
                group_region_fpath = (cfsn_group_dpath / (group_type + '.geojson'))
                text = cfsn_summary.dumps(indent='    ')
                group_region_fpath.write_text(text)
            for site in sites:
                site_fpath = group_site_dpath / (site.site_id + '.geojson')
                text = json.dumps(site, indent='    ')
                site_fpath.write_text(text)

        USE_KML = 1
        if USE_KML:
            try:
                import simplekml  # NOQA
            except ImportError:
                print('Warning: simplekml is not installed, cannot write kml files')
            else:
                cfsn_kml_dpath = (config.out_dpath / 'confusion_kml').ensuredir()
                for group_type, sites in type_to_sites.items():
                    cfsn_summary = type_to_summary[group_type]
                    try:
                        # data = cfsn_summary
                        kml = to_styled_kml(cfsn_summary)
                        kml_fpath = cfsn_kml_dpath / (group_type + '.kml')
                        kml.save(kml_fpath)
                    except Exception:
                        if cfsn_summary.header['geometry'] is not None:
                            raise
                        else:
                            print('skip kml with empty geom')

                if 0:
                    # TODO: write nice images that can be used with QGIS
                    src_dset = kwcoco.CocoDataset(config.src_kwcoco)
                    coco_img = src_dset.images().coco_images[0]

                    fpath = coco_img.primary_image_filepath()
                    img_lpath = cfsn_kml_dpath / 'img.tiff'
                    ub.symlink(fpath, img_lpath)

                    salient_asset = coco_img.find_asset('salient')
                    if salient_asset is not None:
                        fpath = ['file_name']
                        img_lpath = cfsn_kml_dpath / 'salient_heatmap.tiff'
                        ub.symlink(fpath, img_lpath)

        # TIME_OVERLAP_SUMMARY = 0
        # if TIME_OVERLAP_SUMMARY:
        #     visualize_time_overlap(type_to_summary, type_to_sites)

    def dump_hardneg_geojson(self):
        """
        Write new annotation file that can be fed back to the system.
        """
        import json
        import rich
        region_id = self.region_id

        new_sites = self.new_sites
        new_region = self.new_region

        if new_sites is None:
            rich.print('[yellow] WARNING: Cannot dump hardneg without truth')
            return

        enriched_dpath = self.enriched_dpath
        enriched_region_dpath = self.enriched_region_dpath.ensuredir()
        enriched_sites_dpath = self.enriched_sites_dpath.ensuredir()

        rich.print(f'enriched_dpath: [link={enriched_dpath}]{enriched_dpath}[/link]')
        new_region_fpath = enriched_region_dpath / (region_id + '.geojson')
        new_region_fpath.write_text(json.dumps(new_region, indent='    '))
        for new_site in new_sites:
            fpath = enriched_sites_dpath / (new_site.site_id + '.geojson')
            text = json.dumps(new_site, indent='    ')
            fpath.write_text(text)

    def dump_hardneg_kwcoco(self):
        """
        Write kwcoco files for potential system feedback (not used atm)
        """
        from geowatch.cli import reproject_annotations
        config = self.config

        if self.new_sites is None:
            import rich
            rich.print('[yellow] WARNING: Cannot dump_hardneg_kwcoco')
            return

        region_id = self.region_id
        enriched_dpath = self.enriched_dpath
        new_coco_fpath = enriched_dpath / f'hardneg-{region_id}.kwcoco.zip'
        common_kwargs = ub.udict(
            src=config.src_kwcoco,
            dst=new_coco_fpath,
            site_models=self.enriched_sites_dpath,
            region_models=self.enriched_region_dpath,
            workers=2,
            # ignore_system_rejected=False,
        )
        reproject_annotations.main(cmdline=0, **common_kwargs)

    def dump_confusion_kwcoco(self):
        """
        Write confusion kwcoco files for visualization and analysis
        """
        import rich
        from geowatch.cli import reproject_annotations
        import kwcoco
        config = self.config

        true_sites = self.true_sites
        pred_sites = self.pred_sites

        # Write a new "enriched truth" file that reweights false negatives add
        # false positive as hard negatives.
        rich.print(f'Confusion Analysis: [link={self.out_dpath}]{self.out_dpath}[/link]')

        # Project confusion site models onto kwcoco for visualization
        src_dset = kwcoco.CocoDataset(config.src_kwcoco)
        dst_dset = src_dset.copy()
        # dst_dset._update_fpath(config.dst_kwcoco)
        dst_dset.reroot(absolute=True)
        dst_dset.fpath = config.dst_kwcoco

        dst_dset.clear_annotations()

        true_site_infos2 = [s.pandas() for s in true_sites]
        pred_site_infos2 = [s.pandas() for s in pred_sites]

        # Differentiate true and predicted site-ids when projecting onto a
        # single file.
        for site_df in true_site_infos2:
            site_id = site_df.iloc[0]['site_id']
            new_site_id = differentiate_site_id(site_id, 'te')
            site_df.loc[site_df.index[0], 'site_id'] = new_site_id

        for site_df in pred_site_infos2:
            site_id = site_df.iloc[0]['site_id']
            new_site_id = differentiate_site_id(site_id, config.performer_id)
            site_df.loc[site_df.index[0], 'site_id'] = new_site_id

        # for site_df in true_site_infos2:
        #     reproject_annotations.validate_site_dataframe(site_df)

        dst_dset.clear_annotations()
        common_kwargs = ub.udict(
            clear_existing=False,
            dst='return',
            workers=2,
        )
        true_kwargs = common_kwargs | ub.udict(
            role='true_confusion',
            site_models=true_site_infos2,
            ignore_system_rejected=False,
        )
        # kwargs = true_kwargs
        pred_kwargs = common_kwargs | ub.udict(
            role='pred_confusion',
            site_models=pred_site_infos2,
            ignore_system_rejected=False,
        )

        # I don't know why this isn't in-place. Maybe it is a scriptconfig thing?
        repr1 = str(dst_dset.annots())
        print(f'repr1={repr1}')
        dst_dset = reproject_annotations.main(cmdline=0, src=dst_dset, **true_kwargs)
        repr2 = str(dst_dset.annots())
        print(f'repr1={repr1}')
        print(f'repr2={repr2}')

        set(dst_dset.index.trackid_to_aids)

        dst_dset = reproject_annotations.main(cmdline=0, src=dst_dset, **pred_kwargs)
        # repr3 = str(dst_dset.annots())
        # print(f'repr1={repr1}')
        # print(f'repr2={repr2}')
        # print(f'repr3={repr3}')

        self.bas_dset = None

        if config.dst_kwcoco is not None:

            if config.bas_kwcoco and config.src_kwcoco != config.bas_kwcoco:
                # Let the AC coco files know about bas heatmaps
                from geowatch import heuristics
                from geowatch.tasks.cold import transfer_features
                bas_dset = kwcoco.CocoDataset(config.bas_kwcoco)
                heuristics.normalize_sensors(bas_dset)
                heuristics.normalize_sensors(dst_dset)
                # tf_fpath = dst_dset.fpath.augment(stemsuffix="-tf", multidot=1)
                transfer_config = {
                    'coco_fpath': bas_dset,
                    'combine_fpath': dst_dset,
                    'new_coco_fpath': 'return',
                    'channels_to_transfer': ['salient'],
                    'max_propogate': None,
                    'allow_affine_approx': True,
                }
                new = transfer_features.transfer_features_main(cmdline=0, **transfer_config)
                dst_dset = new

                bas_dset.clear_annotations()
                bas_dset = reproject_annotations.main(cmdline=0, src=bas_dset, **true_kwargs)
                bas_dset = reproject_annotations.main(cmdline=0, src=bas_dset, **pred_kwargs)
                self.bas_dset = bas_dset
            else:
                ub.Path(dst_dset.fpath).parent.ensuredir()
                print(f'dump to dst_dset.fpath={dst_dset.fpath}')
                dst_dset.dump()

        self.cfsn_coco = dst_dset

        # set(dst_dset.annots().lookup('role', None))
        # set([x.get('role', None) for x in dst_dset.annots().lookup('cache', None)])

    def dump_summary_viz(self):
        """
        Too slow
        """
        # dst_dset.annots().take([0, 1, 2])
        dst_dset = self.cfsn_coco
        if self.config.summary_visualization:
            viz_dpath = (self.config.out_dpath / 'summary_viz').ensuredir()
            make_summary_visualization(dst_dset, viz_dpath)

    def dump_site_case_viz(self):
        """
        Per-site visualization for analysis and presentations.
        """
        import kwimage
        import rich
        type_to_sites = self.type_to_sites
        coco_dset = self.cfsn_coco
        cases = self.build_site_confusion_cases()
        viz_dpath = self.out_dpath / 'site_viz'

        # from geowatch.utils.kwcoco_extensions import covered_video_geo_regions
        # if self.bas_dset is not None:
        #     # If we can't visualize the site with the AC dataset,
        #     # we probably can with the BAS dataset.
        #     bas_covered_gdf = covered_video_geo_regions(self.bas_dset)
        #     main_covered_gdf = covered_video_geo_regions(coco_dset)

        if 1:
            import pandas as pd
            case_df = pd.DataFrame(cases)
            try:
                print(case_df[['te_association_status', 'te_associated', 'te_color_code', 'type']].value_counts())
            except KeyError:
                ...

        true_id_to_site = {s.site_id: s for s in type_to_sites['true']}
        pred_id_to_site = {s.site_id: s for s in type_to_sites['pred']}

        rich.print(f'Dumping Cases to: [link={viz_dpath}]{viz_dpath}[/link]')

        errors = []
        from kwutil import util_progress
        pman = util_progress.ProgressManager()

        bytrueid_dpath = (viz_dpath / '_by_true_id')
        bypredid_dpath = (viz_dpath / '_by_pred_id')

        with pman:
            total = 0
            for case in pman.progiter(cases, desc='dump cases', verbose=3):
                print(case['name'])
                # if 'KW_C501_0393' in case['name']:
                #     import xdev
                #     xdev.snapshot()
                # if 'CH_R001_0076' in case['name']:
                #     raise Exception
                #     import xdev
                #     xdev.embed()
                # else:
                #     continue
                total += 1
                dpath = (viz_dpath / case['type'])
                fname = (case['name'] + '.jpg')

                # if '0032' not in case['name'] and '0031' not in case['name']:
                #     continue

                fpath = dpath / fname

                try:
                    try:
                        canvas = visualize_case(
                            coco_dset, case, true_id_to_site, pred_id_to_site)
                    except Exception:
                        if self.bas_dset is not None:
                            # Fallback on using the bas dataset if neeeded
                            canvas = visualize_case(
                                self.bas_dset, case, true_id_to_site, pred_id_to_site)
                        else:
                            raise
                except Exception as ex:
                    errors.append(f'Failed to plot {case["name"]} due to {ex!r}')
                    rich.print('ex = {}'.format(ub.urepr(ex, nl=1)))
                    rich.print(f'[red] ERRORS {len(errors)} / {total}')
                    # import xdev
                    # xdev.embed()
                    if self.config.strict:
                        raise
                    continue
                dpath.ensuredir()
                canvas = kwimage.ensure_uint255(canvas)
                # import xdev
                # xdev.embed()
                # print(f'fpath={fpath}')
                # print(f'canvas.shape={canvas.shape}')
                kwimage.imwrite(fpath, canvas)

                bytrue_link_fpath = bytrueid_dpath / (case['bytrue_name'] + '.jpg')
                bypred_link_fpath = bypredid_dpath / (case['bypred_name'] + '.jpg')
                bytrue_link_fpath.parent.ensuredir()
                bypred_link_fpath.parent.ensuredir()
                ub.symlink(real_path=fpath, link_path=bytrue_link_fpath)
                ub.symlink(real_path=fpath, link_path=bypred_link_fpath)

        if errors:
            rich.print(f'[red]There were {len(errors)} errors in viz')
            print('errors = {}'.format(ub.urepr(errors, nl=1)))
            rich.print(f'[red]There were {len(errors)} errors in viz')

        rich.print(f'Dumped Cases to: [link={viz_dpath}]{viz_dpath}[/link]')

    def build_site_confusion_cases(self):
        """
        Build a set of cases that inspect the predictions of a single site.
        """
        from geowatch.utils import util_gis
        import numpy as np
        print('Building confusion cases')

        performer_id = self.config.performer_id
        type_to_summary = self.type_to_summary
        type_to_sites = self.type_to_sites

        # Ensure data structures have consistent ordering so we can used indexes
        for key in type_to_summary.keys():
            summary = type_to_summary[key]
            sites = type_to_sites[key]
            summary_gdf = summary.pandas_summaries()
            assert not ub.find_duplicates([s.site_id for s in sites])
            id_to_site = ub.udict({s.site_id: s for s in sites})
            if len(summary_gdf):
                new_sites = list(id_to_site.take(summary_gdf['site_id']))
            else:
                new_sites = []
            assert len(new_sites) == len(sites)
            type_to_sites[key] = new_sites

        # Double check ordering worked
        for key in type_to_summary.keys():
            summary = type_to_summary[key]
            sites = type_to_sites[key]
            summary_gdf = summary.pandas_summaries()
            if len(summary_gdf):
                assert summary_gdf['site_id'].values.tolist() == [s.site_id for s in sites]

        # Time analysis of false positives that overlap with something.
        true_sites_all = type_to_sites['true']

        true_region = type_to_summary['true']
        true_gdf = true_region.pandas_summaries()
        if len(true_gdf):
            true_utm_gdf = util_gis.project_gdf_to_local_utm(true_gdf, mode=1)
        else:
            true_utm_gdf = true_gdf

        region_start_date = true_region.start_date
        region_end_date = true_region.end_date

        pred_summary = type_to_summary['pred']
        pred_sites_all = type_to_sites['pred']
        pred_gdf = pred_summary.pandas_summaries()
        pred_utm_gdf = util_gis.project_gdf_to_local_utm(pred_gdf, mode=1)

        all_idx1s = np.arange(len(pred_utm_gdf))
        # all_idx2s = np.arange(len(true_utm_gdf))

        # For each incorrect prediction check if it spatially overlaps any truth
        idx2_to_idxs1 = util_gis.geopandas_pairwise_overlaps(true_utm_gdf, pred_utm_gdf)
        idx1_to_idxs1 = util_gis.geopandas_pairwise_overlaps(pred_utm_gdf, pred_utm_gdf)
        idx1_to_idxs2 = {idx1: [] for idx1 in all_idx1s}
        for idx2, idxs1 in idx2_to_idxs1.items():
            for idx1 in idxs1:
                idx1_to_idxs2[idx1].append(idx2)

        # idx2_to_idxs1 = util_gis.geopandas_pairwise_overlaps(true_utm_gdf, pred_utm_gdf)
        cases = []
        for idx1 in all_idx1s:
            pred_site = pred_sites_all[idx1]
            # pred_geom = pred_utm_gdf.iloc[idx1].geometry

            assert pred_utm_gdf.iloc[idx1]['site_id'] == pred_site.site_id
            pred_confusion = pred_site.header['properties']['cache']['confusion']
            confusion_type = pred_confusion['type']

            pred_idxs = idx1_to_idxs1[idx1]
            true_idxs = idx1_to_idxs2[idx1]

            pred_sites = list(ub.take(pred_sites_all, pred_idxs))
            true_sites = list(ub.take(true_sites_all, true_idxs))

            if len(true_idxs):
                matched_status = '_'.join(sorted({s.status for s in true_sites}))
                cfsn_status = '_overlaps_' + matched_status
            else:
                cfsn_status = '_nomatch'
                true_sites = []

            true_utm_geoms = true_utm_gdf.iloc[true_idxs].geometry
            pred_utm_geoms = pred_utm_gdf.iloc[pred_idxs].geometry
            main_pred_idx = np.where(pred_idxs == idx1)[0][0]

            case = make_case(
                pred_sites,
                true_sites,
                true_utm_geoms,
                pred_utm_geoms,
                main_pred_idx,
                region_start_date,
                region_end_date,
                performer_id,
                confusion_type + cfsn_status,
            )
            cases.append(case)

        # Handle truth sites that didn't match anything.
        if len(true_utm_gdf):
            _true_utm_gdf = true_utm_gdf.set_index('site_id')
            for true_site in type_to_sites.get('gt_false_neg', []):
                confusion_type = true_site.header['properties']['cache']['confusion']['type']
                # true_geom = true_site.geometry
                cfsn_status = '_falseneg'

                true_utm_geoms = _true_utm_gdf.loc[[true_site.site_id]].geometry

                case = make_case(
                    [],
                    [true_site],
                    true_utm_geoms,
                    None,
                    None,
                    region_start_date,
                    region_end_date,
                    performer_id,
                    confusion_type + cfsn_status,
                )
                cases.append(case)

        print(f'Found {len(cases)} cases')
        return cases


def make_case(pred_sites,
              true_sites,
              true_utm_geoms,
              pred_utm_geoms,
              main_pred_idx,
              region_start_date,
              region_end_date,
              performer_id,
              type_):
    """
    Build a dict with enough into to make a plot
    """
    import pandas as pd
    from kwutil import util_time

    # pred_site_ids = [s.site_id for s in pred_sites]
    # true_site_ids = [s.site_id for s in true_sites]

    has_pred = len(pred_sites) > 0
    has_true = len(true_sites) > 0

    case = {}

    if has_pred:
        main_pred_site = pred_sites[main_pred_idx]
        main_pred_geom = pred_utm_geoms.iloc[main_pred_idx]

        pred_obs = main_pred_site.pandas_observations()
        pred_dates = pred_obs['observation_date'].values
        pred_dates = list(map(util_time.coerce_datetime, pred_dates))

        pred_duration = pred_dates[-1] - pred_dates[0]

        pred_coco_site_id = differentiate_site_id(main_pred_site.site_id, performer_id)

        pred_confusion = ub.udict(main_pred_site.header['properties']['cache']['confusion'])
        pred_confusion &= {k for k in pred_confusion if k.startswith('te_')}
        pred_area = main_pred_geom.area
        case.update({
            'main_pred_site': main_pred_site,
            'pred_sites': pred_sites,
            'pred_dates': pred_dates,
            'pred_site_id': main_pred_site.site_id,
            'pred_coco_site_id': pred_coco_site_id,
            'pred_area': pred_area,
            **pred_confusion,
        })

    if has_pred and has_true:
        # main_pred_cache = main_pred_site.header['properties']['cache']
        # if 'confusion' in main_pred_cache:
        #     main_true_site_ids = main_pred_cache['confusion'].get('true_site_ids', [])
        #     main_true_idxs = [true_site_ids.index(_) for _ in main_true_site_ids]
        # else:
        isect_areas = true_utm_geoms.intersection(main_pred_geom).area
        union_areas = true_utm_geoms.union(main_pred_geom).area
        ious = isect_areas / union_areas
        main_true_idx = ious.argmax()
    else:
        main_true_idx = None

    if has_true:
        # TODO generalize
        if main_true_idx is None:
            main_true_idx = 0
        # main_true_idx = main_true_idxs[0]
        main_true_geom = true_utm_geoms.iloc[main_true_idx]
        main_true_site = true_sites[main_true_idx]

        true_area = main_true_geom.area

        site_start_date = main_true_site.start_date or region_start_date
        site_end_date = main_true_site.end_date or region_end_date

        true_obs = main_true_site.pandas_observations()
        true_dates = true_obs['observation_date']
        true_dates = list(true_dates[~pd.isnull(true_dates)])
        true_dates = [site_start_date] + true_dates + [site_end_date]
        true_dates = list(map(util_time.coerce_datetime, true_dates))

        true_duration = true_dates[-1] - true_dates[0]
        true_coco_site_id = differentiate_site_id(main_true_site.site_id, 'te')

        true_confusion = ub.udict(main_true_site.header['properties']['cache']['confusion'])
        true_confusion &= {k for k in true_confusion if k.startswith('te_')}

        case.update({
            'main_true_site': main_true_site,
            'true_sites': true_sites,
            'true_site_id': main_true_site.site_id,
            'true_coco_site_id': true_coco_site_id,
            'true_dates': true_dates,
            'true_area': true_area,
            **true_confusion,
        })

    if has_pred and has_true:
        isect_start = max(true_dates[0], pred_dates[0])
        union_start = min(true_dates[0], pred_dates[0])
        isect_end = min(true_dates[-1], pred_dates[-1])
        union_end = max(true_dates[-1], pred_dates[-1])

        isect_area = main_true_geom.intersection(main_pred_geom).area
        union_area = main_true_geom.union(main_pred_geom).area
        space_iou = isect_area / union_area
        space_iot = isect_area / true_area
        space_iop = isect_area / pred_area

        isect_duration = max((isect_end - isect_start), util_time.coerce_timedelta(0))
        union_duration = max((union_end - union_start), util_time.coerce_timedelta(0))

        time_iou = safediv(isect_duration, union_duration)
        time_iot = safediv(isect_duration, true_duration)
        time_iop = safediv(isect_duration, pred_duration)

        true_duration = true_dates[-1] - true_dates[0]
        pred_duration = pred_dates[-1] - pred_dates[0]

        main_true_name = main_true_site.site_id
        main_pred_name = main_pred_site.site_id

        case.update({
            'space_iou': space_iou,
            'space_iot': space_iot,
            'space_iop': space_iop,

            'time_iou': time_iou,
            'time_iot': time_iot,
            'time_iop': time_iop,
        })
    else:
        if has_pred:
            main_pred_name = main_pred_site.site_id
            main_true_name = 'null'
        elif has_true:
            main_true_name = main_true_site.site_id
            main_pred_name = 'null'
        else:
            raise AssertionError('no pred or true')

    case['name'] = f'{main_pred_name}-vs-{main_true_name}'

    # Additional names for symlinks
    case['bytrue_name'] = f'bytrue-{main_true_name}-vs-{main_pred_name}'
    case['bypred_name'] = f'bypred-{main_pred_name}-vs-{main_true_name}'
    case['type'] = type_

    case.update({
        # 'type': type_,
        'region_start_date': region_start_date,
        'region_end_date': region_end_date,
    })
    return case


def visualize_case(coco_dset, case, true_id_to_site, pred_id_to_site):
    """
    Creates a visualization for a confusion case.

    cases = sorted(cases, key=lambda x: x['time_iou'])[::-1]
    case = cases[1]
    """
    from kwutil import util_time
    from shapely.ops import unary_union
    from geowatch import heuristics
    from geowatch.geoannots.geomodels import RegionModel
    from geowatch.utils import util_gis
    from geowatch.utils import util_kwimage
    import geopandas as gpd
    import kwarray
    import kwcoco
    import kwimage
    import numpy as np

    # from geowatch.utils.kwcoco_extensions import covered_video_geo_regions
    # gdf = covered_video_geo_regions(coco_dset)

    all_aids = set()
    all_sites = []
    main_pred_aids = set()
    main_true_aids = set()

    main_trackids = []

    errors = []

    for site_type in ['pred_sites', 'true_sites']:
        if site_type == 'pred_sites':
            tag = 'kit'
        else:
            tag = 'te'
        for site in case.get(site_type, []):
            all_sites.append(site)
            coco_site_id = differentiate_site_id(site.site_id, tag)
            site.header['properties']['cache']['coco_site_id'] = coco_site_id

            if coco_site_id in getattr(coco_dset.index, 'name_to_track', set()):
                raise NotImplementedError
                # tracks = coco_dset.tracks(names=[coco_site_id])
                # track = tracks.objs[0]
                # tid = track['id']
                # annots = tracks.annots[0]
            else:
                try:
                    site_aids = list(coco_dset.index.trackid_to_aids[coco_site_id])
                except KeyError:
                    errors.append(f'Failed to lookup {coco_site_id} in {site_type}')
                else:
                    all_aids.update(site_aids)
                    annots = coco_dset.annots(site_aids)
                    site_trackid = annots[0:1].lookup('track_id')[0]
                    main_trackids.append(site_trackid)
                    # main_pred_aids.update(pred_aids)
                    if site_type == 'pred_sites':
                        main_pred_aids.update(site_aids)
                    else:
                        main_true_aids.update(site_aids)

    # case['main_pred_site']

    all_aids = sorted(all_aids)
    _all_annots = coco_dset.annots(all_aids)
    _all_gids = list(set(_all_annots.images))
    _all_images = coco_dset.images(_all_gids)

    unique_vidids = sorted(set(_all_images.lookup('video_id')))
    if len(unique_vidids) > 1:
        # Need to pick a single video to inspect if a track is in more than 1.
        unique_videos = coco_dset.videos(unique_vidids)

        main_summaries = RegionModel(features=[s.as_summary() for s in all_sites ])
        summary_gdf = main_summaries.pandas_summaries()
        summary_utm_gdf = util_gis.project_gdf_to_local_utm(summary_gdf, mode=1)

        video_geoms = [kwimage.MultiPolygon.coerce(g).to_shapely()
                       for g in unique_videos.lookup('valid_region_geos')]
        video_gdf = gpd.GeoDataFrame(geometry=video_geoms, crs=util_gis.get_crs84())
        video_utm_gdf = video_gdf.to_crs(summary_utm_gdf.crs)

        summary_areas = summary_utm_gdf.area
        idx_to_score = {}
        for idx in range(len(video_utm_gdf)):
            single_video_utm_gdf = video_utm_gdf.iloc[idx: idx + 1]
            isect = summary_utm_gdf.intersection(single_video_utm_gdf.geometry.iloc[0])
            isect_area = isect.area
            ioos = isect_area / summary_areas
            score = ioos.mean()
            idx_to_score[idx] = score
        chosen_idx = ub.argmax(idx_to_score)
        chosen_vidid = unique_vidids[chosen_idx]

        flags = np.array(_all_images.lookup('video_id')) == chosen_vidid
        _all_images = _all_images.compress(flags)

    _sortx = ub.argsort(_all_images.lookup('frame_index'))
    all_images = _all_images.take(_sortx)

    unique_vidids = sorted(set(_all_images.lookup('video_id')))
    if len(unique_vidids) == 0:
        raise AssertionError('no video')

    if len(unique_vidids) > 1:
        errors.append('Matched multiple videos')

    video_id = unique_vidids[0]

    # In most cases try to only use the "lo" number of images, but allow us to
    # choose up to "hi" to get as many infinite weight examples as possible.
    # but still limit the total number of shown examples.
    MAX_IMAGES_LO = 8
    MAX_IMAGES_HI = 21
    gid_to_weight = ub.ddict(lambda: 0)
    if MAX_IMAGES_LO is not None:
        rng = kwarray.ensure_rng(0)

        for tid in main_trackids:

            track_annots = coco_dset.annots(track_id=tid)
            flags = np.array(track_annots.images.lookup('video_id')) == video_id
            track_annots = track_annots.compress(flags)

            catnames = track_annots.category_names

            # Assing a weight for how much we want to show each frame, because we
            # can only show so many.
            # Randomly take images by default
            # track_annot_weights = np.ones(len(track_annots))

            track_images = track_annots.images
            sensors = track_annots.images.lookup('sensor_coarse')

            channel_weights = [20 * ('salient' in c.channels) + 0.1 for c in track_images.coco_images]

            sensor_weights = np.array([
                heuristics.SENSOR_TRACK_PRIORITY.get(heuristics.TE_SENSOR_NAMES.get(s, s), 1)
                for s in sensors], dtype=float)

            random_weights = rng.rand(len(track_annots))

            track_annot_weights = (
                sensor_weights *
                random_weights *
                channel_weights
            )

            for catname, idxs in zip(*kwarray.group_indices(catnames)):
                consec_groups = kwarray.group_consecutive(idxs)
                for consec_group in consec_groups:
                    idx = consec_group[0]
                    # Must show these frames where transition happens.
                    track_annot_weights[idx] = np.inf
                    if idx > 0:
                        track_annot_weights[idx - 1] = np.inf
                    if idx < len(track_annot_weights) - 1:
                        track_annot_weights[idx + 1] = np.inf

            for gid, weights in ub.group_items(track_annot_weights, track_annots.image_id).items():
                gid_to_weight[gid] += sum(weights)

        gid_to_weight = ub.udict(gid_to_weight).sorted_values(reverse=True)
        _gids = list(gid_to_weight.keys())
        _weights = np.array(list(gid_to_weight.values()))
        mustinclude_idxs = np.where(_weights >= np.inf)[0]
        if len(mustinclude_idxs):
            really_want_idx = max(mustinclude_idxs)
            really_want_idx = min(really_want_idx, MAX_IMAGES_HI)
        else:
            really_want_idx = MAX_IMAGES_LO

        MAX_IMAGES = max(MAX_IMAGES_LO, really_want_idx)

        final_gids = _gids[0:MAX_IMAGES]
        all_images = coco_dset.images(ub.oset(all_images) & set(final_gids))

    if len(all_images) > 0:

        have_frame_indexes = list(all_images.lookup('frame_index'))
        min_frame_index = min(have_frame_indexes)
        max_frame_index = max(have_frame_indexes)

        # Add in context images if possible
        vidid = all_images.objs[0]['video_id']
        videos = coco_dset.videos([vidid])
        video_images = videos.images[0]

        vid_frame_idxs = np.array(video_images.lookup('frame_index'))
        before_idxs = np.where(vid_frame_idxs < min_frame_index)[0]
        after_idxs = np.where(vid_frame_idxs > max_frame_index)[0]

        chosen_before_idxs = ub.oset(before_idxs[0:1]) | ub.oset(before_idxs[-2:])
        chosen_after_idxs = ub.oset(after_idxs[0:2]) | ub.oset(after_idxs[-1:])
        before_gids = video_images.take(chosen_before_idxs)._ids
        after_gids = video_images.take(chosen_after_idxs)._ids

        new_all_gids = before_gids + list(all_images) + after_gids
        all_images = coco_dset.images(new_all_gids)

    # Get context before / after images
    all_annots = coco_dset.annots(all_aids)
    gid_to_aids = ub.group_items(all_annots, all_annots.images)

    # if true_site_id is not None and pred_site_id is not None:
    #     assert set(all_annots.lookup('track_id')) == {true_site_id, pred_site_id}

    tci_channel_priority = [
        'red|green|blue',
        'pan',
    ]

    resolution = '2GSD'

    gid_to_dets = {}
    # Get the relevant annotations in each image
    for coco_img in ub.ProgIter(all_images.coco_images, desc='building case', enabled=False):
        gid = coco_img['id']
        aids = gid_to_aids[gid]
        dets = coco_img._detections_for_resolution(aids=aids, space='video', resolution=resolution)
        gid_to_dets[gid] = dets

    all_vidspace_polys = [
        p.to_shapely() for dets in gid_to_dets.values()
        for p in dets.data['segmentations']]
    vidspace_hull = unary_union(all_vidspace_polys).convex_hull

    vidspace_poly = kwimage.MultiPolygon.from_shapely(vidspace_hull)
    scale_factor = 1.5
    vidspace_box = vidspace_poly.box()

    BE_A_SQUARE = 0
    if BE_A_SQUARE:
        target_sidelen = (np.sqrt(vidspace_box.area) * scale_factor)
        # target_area = target_sidelen ** 2
        min_dim, max_dim = sorted([vidspace_box.width, vidspace_box.height])
        sf1 = target_sidelen / max_dim
        sf2 = target_sidelen / min_dim
        if vidspace_box.width > vidspace_box.height:
            sf1, sf2 = sf2, sf1
        ar_scale_factor = (sf1, sf2)
        scaled_box = vidspace_box.scale(ar_scale_factor, about='centroid')
    else:
        scaled_box = vidspace_box.scale(scale_factor, about='centroid')

    vidspace_bound = scaled_box.quantize()

    BAS_CHANNELS = kwcoco.FusedChannelSpec.coerce('salient')
    AC_CHANNELS = kwcoco.FusedChannelSpec.coerce('Site Preparation|Active Construction|Post Construction|No Activity')
    AC_SALIENT_CHANNELS = kwcoco.FusedChannelSpec.coerce('ac_salient')

    cells = []
    from geowatch.utils import kwcoco_extensions
    for coco_img in ub.ProgIter(all_images.coco_images, desc='building case', enabled=False):
        gid = coco_img['id']
        dets = gid_to_dets[gid]

        colors = []
        for obj in coco_dset.annots(dets.data['aids']).objs:
            color = obj['cache']['confusion']['color']
            colors.append(color)

        channels = kwcoco_extensions.pick_channels(coco_img, tci_channel_priority)

        tci_delayed = coco_img.imdelay(channels=channels, resolution=resolution, nodata_method='float')
        tci_imcrop = tci_delayed.crop(vidspace_bound.to_slice(), wrap=False, clip=False)

        tostack = []

        tci_canvas = tci_imcrop.finalize()
        tci_canvas = kwarray.robust_normalize(tci_canvas)
        tci_canvas = kwimage.fill_nans_with_checkers(tci_canvas)
        rel_dets = dets.translate((-vidspace_bound.tl_x, -vidspace_bound.tl_y))

        if main_true_aids:
            is_main_true = [a in main_true_aids for a in dets.data['aids']]
            true_dets = rel_dets.compress(is_main_true)
            det_true_canvas = np.ones_like(tci_canvas)
            det_true_canvas = true_dets.draw_on(det_true_canvas, alpha=0.9, color='classes')
            det_true_canvas = kwimage.draw_text_on_image(
                det_true_canvas, 'true', (1, 2), valign='top', color='kitware_green')
            tostack.append(det_true_canvas)

        if main_pred_aids:
            # For some reason this isn't the real "main" prediction.
            is_main_pred = [a in main_pred_aids for a in dets.data['aids']]
            pred_dets = rel_dets.compress(is_main_pred)
            det_pred_canvas = np.ones_like(tci_canvas)
            det_pred_canvas = pred_dets.draw_on(det_pred_canvas, alpha=0.9, color='classes')

            # Highlight the the actual singular main case:
            if case.get('main_pred_site', None) is not None:
                actual_main_aid = None
                annots = coco_dset.annots(dets.data['aids'])
                for aid, tid in zip(annots, annots.lookup('track_id')):
                    # FIXME: this will break when tids become integers
                    if tid.replace('_kit', '') == case['main_pred_site'].site_id:
                        actual_main_aid = aid

                if actual_main_aid is not None:
                    is_actual_main_pred = [a == actual_main_aid for a in dets.data['aids']]
                    actual_main_pred_dets = rel_dets.compress(is_actual_main_pred)
                    det_pred_canvas = actual_main_pred_dets.data['boxes'].draw_on(det_pred_canvas, alpha=0.9, color='kitware_orange')
                    det_pred_canvas = kwimage.draw_text_on_image(
                        det_pred_canvas, 'pred', (1, 2), valign='top', color='kitware_blue')

            det_pred_canvas = kwimage.draw_text_on_image(
                det_pred_canvas, 'pred', (1, 2), valign='top', color='kitware_blue')
            tostack.append(det_pred_canvas)

        if 0:
            det_both_canvas = np.ones_like(tci_canvas)
            det_both_canvas = rel_dets.draw_on(det_both_canvas.copy(), color=colors, alpha=0.5)
            tostack.append(det_both_canvas)

        det_tci_canvas = rel_dets.draw_on(tci_canvas, color=colors, alpha=0.5)

        tostack.append(det_tci_canvas)
        tostack.append(tci_canvas)

        if (coco_img.channels & AC_CHANNELS).numel():
            heatmap_delayed = coco_img.imdelay(channels=AC_CHANNELS, resolution=resolution, nodata_method='float')
            heatmap_imcrop = heatmap_delayed.crop(vidspace_bound.to_slice(), wrap=False, clip=False)
            heatmap = heatmap_imcrop.finalize()
            cat_to_color = ub.udict({cat['name']: cat['color'] for cat in heuristics.CATEGORIES})
            channel_colors = list(cat_to_color.take(AC_CHANNELS.to_list()))
            heatmap_canvas = util_kwimage.perchannel_colorize(heatmap, channel_colors=channel_colors)
            heatmap_canvas = kwimage.nodata_checkerboard(heatmap_canvas, on_value=0.3)
            if main_pred_aids:
                # Draw main polgon in the heatmap
                heatmap_canvas = pred_dets.data['segmentations'].draw_on(
                    heatmap_canvas, alpha=0.3, edgecolor='kitware_lightgray',
                    fill=False)
            tostack.append(heatmap_canvas)

        if 0:
            # DEBUG
            import kwplot
            kwplot.autompl()
            heatmap_delayed1 = coco_img.imdelay(channels=AC_SALIENT_CHANNELS, resolution=resolution, nodata_method='float')
            heatmap_delayed2 = coco_img.imdelay(channels=AC_CHANNELS, resolution=resolution, nodata_method='float')

            im1 = heatmap_delayed1.finalize()
            im2 = heatmap_delayed2.finalize()

            canvas1 = kwimage.make_heatmask(im1[:, :, 0], cmap='viridis')
            canvas2 = util_kwimage.perchannel_colorize(im2, channel_colors=channel_colors)
            canvas1 = kwimage.nodata_checkerboard(canvas1, on_value=0.3)
            canvas2 = kwimage.nodata_checkerboard(canvas2, on_value=0.3)
            kwplot.imshow(canvas1, fnum=1)
            kwplot.imshow(canvas2, fnum=2)

            heatmap_imcrop1 = heatmap_delayed1.crop(vidspace_bound.to_slice(), wrap=False, clip=False)
            heatmap_imcrop2 = heatmap_delayed2.crop(vidspace_bound.to_slice(), wrap=False, clip=False)
            im1 = heatmap_imcrop1.finalize()[:, :, 0]
            im2 = heatmap_imcrop2.finalize()
            canvas1 = kwimage.make_heatmask(im1, cmap='viridis')
            canvas2 = util_kwimage.perchannel_colorize(im2, channel_colors=channel_colors)
            canvas1 = kwimage.nodata_checkerboard(canvas1, on_value=0.3)
            canvas2 = kwimage.nodata_checkerboard(canvas2, on_value=0.3)
            kwplot.imshow(canvas1, fnum=3)
            kwplot.imshow(canvas2, fnum=4)

            im1 = heatmap_imcrop1.finalize(optimize=False)[:, :, 0]
            im2 = heatmap_imcrop2.finalize(optimize=False)
            canvas1 = kwimage.make_heatmask(im1, cmap='viridis')
            canvas2 = util_kwimage.perchannel_colorize(im2, channel_colors=channel_colors)
            canvas1 = kwimage.nodata_checkerboard(canvas1, on_value=0.3)
            canvas2 = kwimage.nodata_checkerboard(canvas2, on_value=0.3)
            kwplot.imshow(canvas1, fnum=5)
            kwplot.imshow(canvas2, fnum=6)

            print('AC-SALIENT FULL')
            heatmap_delayed1.print_graph()

            heatmap_delayed2.print_graph()

            print('AC-SALIENT CROP RAW')
            heatmap_imcrop1.print_graph(fields=1)
            print('AC-CLASS CROP RAW')
            heatmap_imcrop2.print_graph(fields=1)

            print('AC-SALIENT CROP OPT')
            heatmap_imcrop1.optimize().print_graph(fields=1)
            print('AC-CLASS CROP OPT')
            heatmap_imcrop2.optimize().print_graph(fields=1)

        if (coco_img.channels & AC_SALIENT_CHANNELS).numel():
            heatmap_delayed = coco_img.imdelay(channels=AC_SALIENT_CHANNELS, resolution=resolution, nodata_method='float')
            heatmap_imcrop = heatmap_delayed.crop(vidspace_bound.to_slice(), wrap=False, clip=False)
            heatmap = heatmap_imcrop.finalize()[:, :, 0]
            heatmap_canvas = kwimage.make_heatmask(heatmap, cmap='viridis')
            heatmap_canvas = kwimage.nodata_checkerboard(heatmap_canvas, on_value=0.3)
            if main_pred_aids:
                # Draw main polgon in the heatmap
                heatmap_canvas = pred_dets.data['segmentations'].draw_on(
                    heatmap_canvas, alpha=0.3, edgecolor='kitware_lightgray',
                    fill=False)

            # if case['name'] == 'KW_C501_0304-vs-KW_C501_0139':
            #     import xdev
            #     xdev.embed()
            tostack.append(heatmap_canvas)

        if (coco_img.channels & BAS_CHANNELS).numel():
            heatmap_delayed = coco_img.imdelay(channels=BAS_CHANNELS, resolution=resolution, nodata_method='float')
            heatmap_imcrop = heatmap_delayed.crop(vidspace_bound.to_slice(), wrap=False, clip=False)
            heatmap = heatmap_imcrop.finalize()[:, :, 0]
            heatmap_canvas = kwimage.make_heatmask(heatmap, cmap='viridis')
            heatmap_canvas = kwimage.nodata_checkerboard(heatmap_canvas, on_value=0.3)
            if main_pred_aids:
                # Draw main polgon in the heatmap
                heatmap_canvas = pred_dets.data['segmentations'].draw_on(
                    heatmap_canvas, alpha=0.3, edgecolor='kitware_lightgray',
                    fill=False)
            tostack.append(heatmap_canvas)

        cell_canvas = kwimage.stack_images(tostack, axis=0, pad=5)[..., 0:3]

        header_lines = [
            coco_img.img.get('sensor_coarse'),
            util_time.coerce_datetime(coco_img.img.get('date_captured')).date().isoformat(),
        ]

        header = kwimage.draw_text_on_image(None, text='\n'.join(header_lines), halign='center')
        header = kwimage.ensure_float01(header)
        if 1:
            cell_canvas = kwimage.imresize(cell_canvas, min_dim=header.shape[1]).clip(0, 1)
            cell_canvas = kwimage.stack_images([header, cell_canvas], axis=0)
            cell_canvas = kwimage.ensure_uint255(cell_canvas)
            # cell_canvas = kwimage.draw_header_text(cell_canvas, '\n'.join(header_lines), fit='grow')
        cells.append(cell_canvas)

    case = case.copy()

    if video_id is not None:
        case['video_name'] = coco_dset.index.videos[video_id]['name']

    toshow = ub.udict(case) & {
        'name',
        'video_name',
        'pred_coco_site_id',
        'pred_area',
        'te_associated',
        'te_color_code',
        'te_association_status',
        'te_site_count',
        'te_site_area',
        'true_coco_site_id',
        'true_area',
        'te_spatial_overlap',
        'te_temporal_iot',
        'te_temporal_iop',
        'te_score',
        'name',
        'space_iou',
        'space_iot',
        'space_iop',
        'time_iou',
        'time_iot',
        'time_iop',
        'type',
    }

    case['img_dates'] = all_images.lookup('date_captured')

    parts = []
    if 1:
        grid_canvas = kwimage.stack_images(cells, axis=1, pad=10)
        grid_canvas = kwimage.ensure_uint255(grid_canvas)

        if 1:
            text = ub.urepr(toshow, nobr=1, precision=2)
            grid_canvas = kwimage.draw_header_text(grid_canvas, text=text, halign='left', color='kitware_blue')

        if errors:
            import rich
            rich.print(f'[yellow]There were {len(errors)} recoverable errors in {case["name"]}')
            text = ub.urepr(errors, nobr=1, precision=2)
            grid_canvas = kwimage.draw_header_text(grid_canvas, text=text, halign='left', color='kitware_red')

        parts.append(grid_canvas)

    if 1:
        timeline_canvas = make_case_timeline(case)
        timeline_canvas = kwimage.ensure_float01(timeline_canvas)
        timeline_canvas = kwimage.imresize(timeline_canvas, dsize=(grid_canvas.shape[1], None)).clip(0, 1)
        timeline_canvas = kwimage.ensure_uint255(timeline_canvas)
        parts.append(timeline_canvas)

        main_pred_site = case.get('main_pred_site', None)
        if main_pred_site is not None:
            score_canvas = make_pred_score_timeline(main_pred_site)
            score_canvas = kwimage.ensure_float01(score_canvas)
            score_canvas = kwimage.imresize(score_canvas, dsize=(grid_canvas.shape[1], None)).clip(0, 1)
            score_canvas = kwimage.ensure_uint255(score_canvas)
            parts.append(score_canvas)

    final = kwimage.stack_images(parts, axis=0)
    return final

    # kwplot.imshow(final, fnum=1)


def make_pred_score_timeline(main_pred_site):
    from kwutil import util_time
    observations = list(main_pred_site.observations())
    longform = []
    for obs in observations:
        class_to_scores = obs['properties']['cache']['raw_multi_scores']
        class_to_scores = class_to_scores[0]  # hack

        date = obs['properties']['observation_date']
        for catname, score in class_to_scores.items():
            longform.append({
                'date': util_time.coerce_datetime(date),
                'class': catname,
                'score': score
            })

    import pandas as pd
    df = pd.DataFrame(longform)
    import kwplot
    sns = kwplot.autosns()
    fig = kwplot.figure(fnum=1321322)
    ax = fig.gca()
    ax.cla()

    from geowatch import heuristics
    import kwimage
    name_to_color = {d['name']: d['color'] for d in heuristics.CATEGORIES}
    name_to_color['ac_salient'] = 'pink'
    name_to_color['salient'] = 'black'
    palette = {k: kwimage.Color.coerce(v).as01() for k, v in name_to_color.items()}

    fig.set_size_inches([10, 3])
    fig.subplots_adjust(left=.1, bottom=0.3, top=.7, right=0.9)

    ax = sns.lineplot(data=df, x='date', y='score', hue='class', ax=ax,
                      palette=palette, legend=False)
    ax.set_ylim(0, 1)
    imdata = kwplot.render_figure_to_image(ax.figure)
    return imdata
    # kwimage.imwrite('foo.png', imdata)


def make_case_timeline(case):
    """
    executor = ub.Executor('process', max_workers=1)
    future = executor.submit(make_case_timeline, case)
    future.result()

    Ignore:
        from geowatch.mlops.confusor_analysis import *  # NOQA
        import kwplot
        kwplot.autompl()
        from kwutil import util_time
        pred_dates = sorted([
            util_time.datetime.random(start='2010-01-01', end='2020-01-01')
            for _ in range(10)
        ])
        true_dates = sorted([
            util_time.datetime.random(start='2010-01-01', end='2020-01-01')
            for _ in range(10)
        ])
        case = {}
        case['true_dates'] = true_dates
        case['pred_dates'] = pred_dates

        img_dates = sorted([
            util_time.datetime.random(start='2010-01-01', end='2020-01-01')
            for _ in range(10)
        ])
        case['img_dates'] = img_dates
        canvas = make_case_timeline(case)
        kwplot.imshow(canvas, fnum=1)
        ...
    """
    import kwplot
    from geowatch.utils import util_kwplot
    from kwutil import util_time
    # plt = kwplot.plt
    import matplotlib.dates as mdates
    fig = kwplot.figure(fnum=1321321)
    ax = fig.gca()
    ax.cla()

    # case['main_true_site']
    main_true_sites = case.get('true_sites', [])
    if 'main_pred_site' in case:
        main_pred_sites = [case['main_pred_site']]
    else:
        main_pred_sites = []

    artman = util_kwplot.ArtistManager()

    ylabel_map = {0: ''}

    from geowatch import heuristics
    import numpy as np
    import itertools as it
    import kwimage
    name_to_color = {d['name']: d['color'] for d in heuristics.CATEGORIES}
    # {d['status']: d['color'] heuristics.HUERISTIC_STATUS_DATA}

    yloc = 1
    for site_type, site in it.chain((('pred', s) for s in main_pred_sites),
                                    (('true', s) for s in main_true_sites)):
        start_date = util_time.coerce_datetime(site.start_date) or case['region_start_date']
        end_date = util_time.coerce_datetime(site.end_date) or case['region_end_date']
        ylabel_map[yloc] = site_type

        status = site.header['properties']['status']
        status_color = heuristics.IARPA_STATUS_TO_INFO[status]['color']

        obs_gdf = site.pandas_observations()
        obs_gdf['current_phase']

        obs_xs = np.array(util_kwplot.fix_matplotlib_dates(obs_gdf['observation_date']))
        obs_colors = obs_gdf['current_phase'].apply(name_to_color.get).values

        prev_x = None
        prev_c = None
        for c, idxs in it.groupby(range(len(obs_colors)), key=obs_colors.__getitem__):
            if c is None:
                c = kwimage.Color.coerce(status_color)
            else:
                c = kwimage.Color.coerce(c)
            idxs = list(idxs)
            xs = obs_xs[idxs]
            if prev_x is not None:
                # c2 = prev_c.interpolate(c, ispace='hsv', ospace='rgb')
                c2 = prev_c
                artman.plot([prev_x, xs[0]], yloc, color=c2)
            artman.plot(xs, yloc, color=c)
            artman.add_circle_marker((xs[0], yloc), 2, color=c, zorder=2)
            prev_x = xs[-1]
            prev_c = c

        if site_type == 'pred':
            bg_color = 'kitware_blue'
        else:
            bg_color = 'kitware_green'

        start_x, end_x = util_kwplot.fix_matplotlib_dates([start_date, end_date])

        artman.add_circle_marker((start_x, yloc), 5, color=bg_color, zorder=1)
        artman.add_circle_marker((end_x, yloc), 5, color=bg_color, zorder=1)
        # artman.plot((start_x, end_x), yloc, color=bg_color, zorder=-10)
        yloc += 1

    ylabel_map[yloc] = ''

    # Add vertical lines to indicate where shown images lie on the timeline.
    if 'img_dates' in case:
        img_xs = util_kwplot.fix_matplotlib_dates(case['img_dates'])
        for img_x in img_xs:
            artman.plot([img_x, img_x], [0, yloc], color='kitware_gray', zorder=-2, linewidth=2)

        img_xs = util_kwplot.fix_matplotlib_dates([case['region_start_date'], case['region_end_date']])
        for img_x in img_xs:
            artman.plot([img_x, img_x], [0, yloc], color='black', zorder=-1, linewidth=4)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=720))
    artman.add_to_axes(ax=ax)
    artman.setlims(ax=ax)

    ax.set_ylim(0, yloc)

    fig.set_size_inches([10, 3])
    fig.subplots_adjust(left=.1, bottom=0.3, top=.7, right=0.9)
    # kwplot.plt.locator_params(axis='x', nbins=4)
    # kwplot.plt.locator_params(axis='x', nbins=4)
    # true_annots.images.coco_images
    # pred_annots.images.coco_images

    labelmod = util_kwplot.LabelModifier()
    # labelmod.add_mapping({1.0: 'pred', 2.0: 'true'})
    labelmod.add_mapping(ylabel_map)
    labelmod.relabel_yticks(ax=ax)

    canvas = kwplot.render_figure_to_image(fig)
    return canvas


def visualize_all_timelines(cases, coco_dset, type_to_sites, type_to_summary):
    # from geowatch.geoannots.geomodels import SiteSummary
    # from kwutil import util_time

    true_id_to_site = {s.site_id: s for s in type_to_sites['true']}
    pred_id_to_site = {s.site_id: s for s in type_to_sites['pred']}
    # true_id_to_summary = {ss.site_id: ss for ss in map(SiteSummary.coerce, type_to_summary['true'].site_summaries())}
    # pred_id_to_summary = {ss.site_id: ss for ss in map(SiteSummary.coerce, type_to_summary['pred'].site_summaries())}

    cases = sorted(cases, key=lambda x: x['time_iou'])[::-1]
    # coco_upgrade_track_ids(coco_dset)
    case = cases[4]

    import kwplot
    kwplot.autosns()

    import kwplot
    kwplot.autosns()
    fig = kwplot.figure(fnum=1)
    fig.clf()

    from geowatch.utils import util_kwplot
    artman = util_kwplot.ArtistManager()
    yloc = 1

    # min_date = min([min(case['pred_dates'] + case['true_dates']) for case in cases])
    # min_x = util_kwplot.fix_matplotlib_dates([min_date])[0]
    # plt = kwplot.plt

    for case in cases[:]:
        pred_xs = util_kwplot.fix_matplotlib_dates(case['pred_dates'])
        true_xs = util_kwplot.fix_matplotlib_dates(case['true_dates'])
        artman.plot(pred_xs, yloc, color='kitware_blue')
        yloc += 1
        artman.plot(true_xs, yloc, color='kitware_green')
        yloc += 1

        true_site = true_id_to_site[case['true_site_id']]  # NOQA
        pred_site = pred_id_to_site[case['pred_site_id']]  # NOQA

        pred_site_id = pred_site.site_id
        true_site_id = true_site.site_id

        show = ub.udict(case) & {'space_iou', 'time_iou', 'pred_area', 'true_area'}
        show['pred'] = pred_site_id
        show['true'] = true_site_id
        # text = ub.urepr(show, precision=2, nl=0)
        # med_x = (max_x + min_x) / 2
        # plt.annotate(text, (min_x, yloc))

        yloc += 20

    artman.add_to_axes()
    ax = fig.gca()
    # TODO: make this formatter fixup work better.
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y\n%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=720))
    artman.setlims()


def differentiate_site_id(site_id, tag):
    assert site_id.count('_') == 2
    a, b = site_id.rsplit('_', 1)
    new_site_id = f'{a}_{tag}_{b}'
    return new_site_id


def fix_site_id(site_id, region_id, performer_id):
    site_id = site_id.strip()
    splitters = ['_te_', '_iMERIT_', f'_{performer_id}_']
    for marker in splitters:
        site_id = site_id.split(marker)[0]
    # Hack because idk why the metrics code does this.
    if site_id.startswith('_'):
        site_id = region_id + site_id
    return site_id


def coco_upgrade_track_ids(coco_dset):
    # coco_dset = kwcoco.CocoDataset(coco_fpath)
    for tid, aids in list(coco_dset.index.trackid_to_aids.items()):
        ...
        if tid not in coco_dset.index.tracks:
            if isinstance(tid, str):
                name = tid
            else:
                name = f'track_{tid:03d}'
            assert name not in coco_dset.index.name_to_track
            new_tid = coco_dset.add_track(name=name)

            for aid in aids:
                coco_dset.index.anns[aid]['track_id'] = new_tid
            coco_dset.index.trackid_to_aids[new_tid] = aids
            coco_dset.index.trackid_to_aids.pop(tid)


def make_summary_visualization(dst_dset, viz_dpath):
    import kwplot
    import numpy as np

    resolution = '10GSD'

    from kwutil import util_progress
    from kwutil import util_time
    # from geowatch.utils import util_kwimage
    import kwarray
    import kwimage
    from shapely.ops import unary_union

    pman = util_progress.ProgressManager()
    with pman:
        # video_to_tracking_heatmap = {}
        for video in pman(dst_dset.videos().objs, desc='make video summary'):
            # Simulate the tracking heatmap (todo: get what the data really was)
            images = dst_dset.images(video_id=video['id'])
            det_accum = []
            for coco_img in pman(images.coco_images, desc='load dets'):
                dets = coco_img._detections_for_resolution(resolution=resolution)
                det_accum.append(dets)

            running = kwarray.RunningStats()
            jobs = ub.JobPool('process', max_workers=2, transient=True)

            chan_priority = [
                'red|green|blue',
                'pan',
            ]

            def acceptable_channels(channels, chan_priority):
                import kwcoco
                for want in chan_priority:
                    want = kwcoco.FusedChannelSpec.coerce(want)
                    cand = (channels & want)
                    if cand.numel() == want.numel():
                        return cand
                return None

            if 1:
                for coco_img in pman(images.coco_images, desc='submit delay jobs'):
                    chanels = acceptable_channels(coco_img.channels, chan_priority)
                    delayed = coco_img.imdelay(chanels, resolution=resolution, nodata_method='float')
                    job = jobs.submit(delayed.finalize)

                    time = util_time.coerce_datetime(coco_img['timestamp'])
                    job.time = time
                    job.coco_img = coco_img

                oldest_time = None
                newest_time = None
                # oldest_img = None
                # newest_img = None

                job_loader = pman(jobs.as_completed(), total=len(jobs), desc='averaging heatmaps')
                for job in job_loader:
                    im = job.result()
                    sensor = job.coco_img['sensor_coarse']

                    if sensor in {'S2', 'Sentinel-2'}:
                        if oldest_time is None or oldest_time > job.time:
                            oldest_time = job.time
                            # oldest_img = im

                        if newest_time is None or newest_time < job.time:
                            newest_time = job.time
                            # newest_img = im

                    im = kwimage.atleast_3channels(im)
                    running.update(im)
                    # track_ids = dst_dset.annots(dets.data['aids']).lookup('track_id')

            all_dets = kwimage.Detections.concatenate(det_accum)
            all_annots = dst_dset.annots(all_dets.data['aids'])
            all_dets.data['frame_index'] = np.array(all_annots.images.lookup('frame_index'))
            all_dets.data['track_id'] = np.array(all_annots.lookup('track_id'))
            all_dets.data['role'] = np.array(all_annots.lookup('role'))
            all_dets.data['cache'] = np.array(all_annots.lookup('cache'))

            groupers = list(zip(all_dets.data['role'], all_dets.data['track_id']))
            unique_tids, groupxs = kwarray.group_indices(groupers)

            track_summaries = []
            for (role, tid), groupx in zip(unique_tids, groupxs):
                track_dets = all_dets.take(groupx)
                cache = track_dets.data['cache'][0]
                row = cache.copy()
                row['role'] = role
                row['confusion_color'] = row['confusion']['color']
                # assert row['role'] == role
                sh_poly = unary_union([p.to_shapely() for p in track_dets.data['segmentations']])
                kw_poly = kwimage.MultiPolygon.from_shapely(sh_poly)
                row['poly'] = kw_poly
                track_summaries.append(row)

            role_to_summary = ub.udict(ub.group_items(track_summaries, key=lambda x: x.get('role', None)))
            print(ub.udict(role_to_summary).map_values(len))

            # canvas = kwplot.make_heatmask(util_kwimage.exactly_1channel(mean_heatmap), cmap='magma')[:, :, 0:3]
            # old_canvas = kwimage.normalize_intensity(oldest_img, axis=2)
            # new_canvas = kwimage.normalize_intensity(newest_img, axis=2)

            current = running.current()
            mean_heatmap = current['mean']
            # min_heatmap = current['min']
            # max_heatmap = current['max']
            canvas = kwimage.normalize_intensity(mean_heatmap)
            # canvas = kwplot.make_heatmask(util_kwimage.exactly_1channel(mean_heatmap), cmap='viridis')[:, :, 0:3]
            # canvas_raw = canvas.copy()
            # canvas_true = canvas.copy()
            # canvas_pred = canvas.copy()
            # canvas_cfsn = canvas.copy()

            canvases = {
                # 'new': {
                #     'true': new_canvas.copy(),
                #     'pred': new_canvas.copy(),
                #     'cfsn': new_canvas.copy(),
                # },
                # 'old': {
                #     'true': old_canvas.copy(),
                #     'pred': old_canvas.copy(),
                #     'cfsn': old_canvas.copy(),
                # },
                'avg': {
                    'true': canvas.copy(),
                    'pred': canvas.copy(),
                    'cfsn': canvas.copy(),
                },
            }

            alpha = 0.6

            for row in ub.ProgIter(role_to_summary.get('true_confusion', []), desc='true cfsn'):
                for k1, v1 in canvases.items():
                    v1['true'] = row['poly'].draw_on(v1['true'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                    v1['cfsn'] = row['poly'].draw_on(v1['cfsn'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                # row['poly'].draw_on(canvas_true, fill=False, edgecolor=row['confusion_color'])
                # row['poly'].draw_on(canvas_cfsn, fill=False, edgecolor=row['confusion_color'])

            for row in ub.ProgIter(role_to_summary.get('pred_confusion', []), desc='pred cfsn'):
                for k1, v1 in canvases.items():
                    v1['pred'] = row['poly'].draw_on(v1['pred'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                    v1['cfsn'] = row['poly'].draw_on(v1['cfsn'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                # row['poly'].draw_on(canvas_pred, fill=False, edgecolor=row['confusion_color'])
                # row['poly'].draw_on(canvas_cfsn, fill=False, edgecolor=row['confusion_color'])

            for k1, v1 in canvases.items():
                v1['true'] = kwimage.draw_header_text(v1['true'], 'true confusion')
                v1['pred'] = kwimage.draw_header_text(v1['pred'], 'pred confusion')
                v1['cfsn'] = kwimage.draw_header_text(v1['cfsn'], 'both')

            row_keys = ub.oset(['avg', 'old', 'new']) & set(canvases.keys())
            rows = []
            for rk in row_keys:
                _row = kwimage.stack_images(list(canvases[rk].values()), axis=1, pad=10)
                _row = kwimage.ensure_uint255(_row)
                rows.append(_row)
            final_canvas = kwimage.stack_images(rows, axis=0, pad=5)

            fpath = viz_dpath / f'confusion_{video["name"]}.jpg'
            kwimage.imwrite(fpath, final_canvas)

    from geowatch import heuristics
    legend_img = kwplot.make_legend_img(heuristics.IARPA_CONFUSION_COLORS)
    kwimage.imwrite(viz_dpath / 'confusion_legend.png', legend_img)
    import rich
    rich.print(f'Viz Dpath: [link={viz_dpath}]{viz_dpath}[/link]')


def to_styled_kml(data):
    """
    Make a kml version of the geojson that works nice with QGIS
    """
    import kwimage
    import simplekml
    kml = simplekml.Kml()
    for feat in data['features']:
        if feat['geometry']['type'] == 'Polygon':

            if 'site_id' in feat['properties']:
                name = feat['properties']['site_id']
            else:
                name = feat['properties']['region_id']

            poly = kml.newpolygon(name=name,
                                  description='test',
                                  outerboundaryis=feat['geometry']['coordinates'][0])

            cache = feat['properties']['cache']
            if 'confusion' in cache:
                hexcol = (kwimage.Color.coerce(cache['confusion']['color']).ashex()[1:] + 'ff')
                kmlcol = ''.join(ub.flatten(list(ub.chunks(hexcol, chunksize=2))[::-1]))
                linecol = simplekml.Color.changealphaint(100, kmlcol)
                facecol = simplekml.Color.changealphaint(50, kmlcol)
                poly.style.linestyle.color = linecol
                poly.style.linestyle.width = 5
                poly.style.polystyle.color = facecol
            else:
                if feat['properties']['type'] == 'region':
                    hexcol = (kwimage.Color.coerce('white').ashex()[1:] + 'ff')
                    kmlcol = ''.join(ub.flatten(list(ub.chunks(hexcol, chunksize=2))[::-1]))
                    linecol = simplekml.Color.changealphaint(100, hexcol)
                    facecol = simplekml.Color.changealphaint(1, hexcol)
                    poly.style.linestyle.color = linecol
                    poly.style.linestyle.width = 5
                    poly.style.polystyle.color = facecol
                else:
                    hexcol = (kwimage.Color.coerce('kitware_darkgray').ashex()[1:] + 'ff')
                    kmlcol = ''.join(ub.flatten(list(ub.chunks(hexcol, chunksize=2))[::-1]))
                    linecol = simplekml.Color.changealphaint(100, kmlcol)
                    facecol = simplekml.Color.changealphaint(50, kmlcol)
                    poly.style.linestyle.color = linecol
                    poly.style.linestyle.width = 5
                    poly.style.polystyle.color = facecol

        elif feat['geometry']['type'] == 'LineString':
            kml.newlinestring(name=name,
                              description='test',
                              coords=feat['geometry']['coordinates'])
        elif feat['geometry']['type'] == 'Point':
            kml.newpoint(name=name,
                         description='test',
                         coords=[feat['geometry']['coordinates']])
    return kml


def nan_to_null(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    else:
        return x


def safediv(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/geowatch/mlops/confusor_analysis.py
        python -m geowatch.mlops.confusor_analysis
    """
    main()
