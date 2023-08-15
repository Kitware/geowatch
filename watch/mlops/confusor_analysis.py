#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class ConfusorAnalysisConfig(scfg.DataConfig):
    """
    Requires that IARPA metrics are computed
    """

    metrics_node_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        A path to an IARPA metrics MLops output directory node.

        Use this in the special case that you have an mlops or smartflow output
        directory.
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
    dst_kwcoco = scfg.Value(None, help='the reprojected output kwcoco file to write')

    bas_metric_dpath = scfg.Value(None, help='A path to bas metrics if det/prop paths are not specified')

    pred_sites = scfg.Value(None, help='the path to the predicted sites manifest / directory / globstr')

    region_id = scfg.Value(None, help='the id for the region')
    true_site_dpath = scfg.Value(None, help='input')
    true_region_dpath = scfg.Value(None, help='input')

    performer_id = scfg.Value('kit', help='the performer id')

    summary_visualization = scfg.Value(False, isflag=True)

    out_dpath = scfg.Value(None, help='where to write results')

    def __post_init__(self):
        if self.bas_metric_dpath is not None:
            self.bas_metric_dpath = ub.Path(self.bas_metric_dpath)

        if self.true_region_dpath is not None:
            self.true_region_dpath = ub.Path(self.true_region_dpath)

        if self.true_site_dpath is not None:
            self.true_site_dpath = ub.Path(self.true_site_dpath)

        if self.out_dpath is not None:
            self.out_dpath = ub.Path(self.out_dpath)

    def _infer_from_mlops_node(self):
        import json
        if self.metrics_node_dpath is not None:
            # Infer things using assumptions about mlops directory structures
            self.metrics_node_dpath = ub.Path(self.metrics_node_dpath)

            overall_cands = list(self.metrics_node_dpath.glob('*/overall'))
            sites_cands = list(self.metrics_node_dpath.glob('.pred/*/*/sites'))
            src_kwcoco_cands = list(self.metrics_node_dpath.glob('.pred/*/*/poly.kwcoco.zip'))
            assert len(overall_cands) == 1, 'mlops assumption violated'
            assert len(sites_cands) == 1, 'mlops assumption violated'
            assert len(src_kwcoco_cands) == 1, 'mlops assumption violated'

            self.src_kwcoco = src_kwcoco_cands[0]
            overall_dpath = overall_cands[0]
            self.pred_sites = sites_cands[0]

            self.bas_metric_dpath = overall_dpath / 'bas'
            self.region_id = overall_dpath.parent.name
            job_config_fpath = self.metrics_node_dpath / 'job_config.json'
            job_config = json.loads(job_config_fpath.read_text())

            if self.true_region_dpath is None:
                self.true_region_dpath = job_config['bas_poly_eval.true_region_dpath']

            if self.true_site_dpath is None:
                self.true_site_dpath = job_config['bas_poly_eval.true_site_dpath']

            if self.out_dpath is None:
                self.out_dpath = (self.metrics_node_dpath / 'confusion_analysis')
            self.dst_kwcoco = self.out_dpath / 'confusion_kwcoco' / 'confusion.kwcoco.zip'

        if self.bas_metric_dpath is not None:
            if self.detections_fpath is None:
                self.detections_fpath = self.bas_metric_dpath / 'detections_tau=0.2_rho=0.5_min_area=0.csv'
            if self.proposals_fpath is None:
                self.proposals_fpath = self.bas_metric_dpath / 'proposals_tau=0.2_rho=0.5_min_area=0.csv'

        self.__post_init__()


class ConfusionAnalysis:
    def __init__(self, config):
        self.config = config

        self.true_sites = None
        self.id_to_pred_site = None
        self.true_region_model = None
        self.region_id = None
        self.id_to_true_site = None
        self.id_to_pred_site = None

        self.new_sites = None
        self.new_region = None
        self.type_to_sites = None

        self.enriched_dpath = None
        self.enriched_sites_dpath = None
        self.enriched_region_dpath = None

    def load_assignment(self):
        """
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

        # The possible confusion codes and the corresponding confusion_color they
        # will be assigned is defined in heuristics.IARPA_CONFUSION_COLORS
        """
        config = self.config

        import rich
        import pandas as pd
        import itertools as it
        from watch import heuristics
        from watch.utils import util_gis
        from watch.geoannots.geomodels import SiteModel
        from watch.geoannots.geomodels import RegionModel
        from watch.geoannots.geomodels import SiteModelCollection

        performer_id = config.performer_id
        true_site_dpath = config.true_site_dpath
        true_region_dpath = config.true_region_dpath
        pred_site_fpaths = list(util_gis.coerce_geojson_paths(config.pred_sites))
        region_id = config.region_id

        config.detections_fpath = ub.Path(config.detections_fpath)
        config.proposals_fpath = ub.Path(config.proposals_fpath)
        assert config.proposals_fpath.exists()
        assert config.detections_fpath.exists()

        assign1 = pd.read_csv(config.detections_fpath)
        assign2 = pd.read_csv(config.proposals_fpath)

        rich.print(assign1)
        rich.print(assign2)
        rich.print(f'{len(assign1)=}')
        rich.print(f'{len(assign2)=}')

        needs_recompute = any('_seq_' in m or m.startswith('seq_') for m in assign2['site model'] if m)
        assert not needs_recompute

        def fix_site_id(site_id):
            site_id = site_id.strip()
            splitters = ['_te_', '_iMERIT_', f'_{performer_id}_']
            for marker in splitters:
                site_id = site_id.split(marker)[0]
            # Hack because idk why the metrics code does this.
            if site_id.startswith('_'):
                site_id = region_id + site_id
            return site_id

        ### Assign a confusion label to each truth and predicted annotation
        true_confusion_rows = []
        pred_confusion_rows = []
        site_to_status = {}
        for row in assign1.to_dict('records'):
            true_site_id = row['truth site']
            true_site_id = fix_site_id(true_site_id)
            pred_site_ids = []
            truth_status = row['site type']
            site_to_status[true_site_id] = truth_status
            if isinstance(row['matched site models'], str):
                for name in row['matched site models'].split(','):
                    pred_site_id = name
                    pred_site_id = fix_site_id(pred_site_id)
                    pred_site_ids.append(pred_site_id)
            has_positive_match = len(pred_site_ids)
            true_cfsn = heuristics.iarpa_assign_truth_confusion(truth_status, has_positive_match)

            if true_cfsn is None:
                print('truth_status = {}'.format(ub.urepr(truth_status, nl=1)))
                print('has_positive_match = {}'.format(ub.urepr(has_positive_match, nl=1)))
                raise AssertionError

            true_confusion_rows.append({
                'true_site_id': true_site_id,
                'pred_site_ids': pred_site_ids,
                'type': true_cfsn,
            })

        for row in assign2.to_dict('records'):
            pred_site_id = row['site model']
            pred_site_id = fix_site_id(pred_site_id)
            true_site_ids = []
            truth_match_statuses = []
            if isinstance(row['matched truth sites'], str):
                for name in row['matched truth sites'].split(','):
                    true_site_id = name
                    true_site_id = fix_site_id(true_site_id)
                    truth_match_statuses.append(site_to_status[true_site_id])
                    true_site_ids.append(true_site_id)
            pred_cfsn = heuristics.iarpa_assign_pred_confusion(truth_match_statuses)
            if pred_cfsn is None:
                print('row = {}'.format(ub.urepr(row, nl=1)))
                print('truth_match_statuses = {}'.format(ub.urepr(truth_match_statuses, nl=1)))
                raise AssertionError

            pred_confusion_rows.append({
                'pred_site_id': pred_site_id,
                'true_site_ids': true_site_ids,
                'type': pred_cfsn,
            })

        for row in true_confusion_rows + pred_confusion_rows:
            row['color'] = heuristics.IARPA_CONFUSION_COLORS.get(row['type'])

        # Add the confusion info as misc data in new site files and reproject them
        # onto the truth for visualization.
        # pred_sites_fpath = poly_pred_dpath / 'sites_manifest.json'
        # assert pred_sites_fpath.exists()
        # pred_site_fpaths = list(util_gis.coerce_geojson_paths(pred_sites_fpath))

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
            raise AssertionError(f'Got {orig_regions=}')

        # Ensure all site data has misc-info
        # Ensure all data cast to site models

        true_region_model = orig_regions[0]
        true_region_model.fixup()

        for site in it.chain(pred_sites, true_sites):
            site.header['properties'].setdefault('cache', {})

        id_to_true_site = {s.site_id: s for s in true_sites}
        id_to_pred_site = {s.site_id: s for s in pred_sites}

        # Add confusion metadata to predicted and truth models
        # https://gis.stackexchange.com/questions/346518/opening-geojson-style-properties-in-qgis
        for row in true_confusion_rows:
            site = id_to_true_site[row['true_site_id']]
            site.header['properties']['cache']['confusion'] = row
        for row in pred_confusion_rows:
            site = id_to_pred_site[row['pred_site_id']]
            site.header['properties']['cache']['confusion'] = row

        VALIDATE = 1
        if VALIDATE:
            all_models = SiteModelCollection(pred_sites + true_sites)
            all_models.fixup()
            all_models.validate(workers=0)

        self.true_sites = true_sites
        self.pred_sites = pred_sites
        self.id_to_pred_site = id_to_pred_site
        self.true_region_model = true_region_model
        self.region_id = region_id
        self.id_to_true_site = id_to_true_site
        self.id_to_pred_site = id_to_pred_site

    def build_hard_cases(self):
        from watch.utils import util_gis

        true_sites = self.true_sites
        pred_sites = self.pred_sites
        id_to_pred_site = self.id_to_pred_site
        true_region_model = self.true_region_model
        region_id = self.region_id
        id_to_true_site = self.id_to_true_site
        id_to_pred_site = self.id_to_pred_site

        pred_region_model = pred_sites.as_region_model(region=true_region_model.header)
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

        # Add in hard negatives
        new_sites = new_true_sites + hard_negative_sites
        self.new_sites = new_sites
        self.new_region = new_region
        self.type_to_sites = type_to_sites

    def dump_confusion_geojson(self):
        import json
        import rich
        import kwcoco
        from watch.geoannots.geomodels import SiteModelCollection
        config = self.config
        region_id = self.region_id
        performer_id = config.performer_id
        type_to_sites = self.type_to_sites

        true_region_model = self.true_region_model
        new_sites = self.new_sites
        new_region = self.new_region

        enriched_dpath = config.out_dpath / 'enriched_annots'
        enriched_sites_dpath = enriched_dpath / 'site_models'
        enriched_region_dpath = enriched_dpath / 'region_models'

        enriched_sites_dpath.ensuredir()
        enriched_region_dpath.ensuredir()

        rich.print(f'enriched_dpath: [link={enriched_dpath}]{enriched_dpath}[/link]')
        new_region_fpath = enriched_region_dpath / (region_id + '.geojson')
        new_region_fpath.write_text(json.dumps(new_region, indent='    '))
        for new_site in new_sites:
            fpath = enriched_sites_dpath / (new_site.site_id + '.geojson')
            text = json.dumps(new_site, indent='    ')
            fpath.write_text(text)

        # Dump confusion categorized site models to disk
        cfsn_group_dpath = config.out_dpath / 'confusion_groups'
        print(ub.urepr(type_to_sites.map_values(len)))

        # Create site summaries for each type of confusion
        type_to_summary = {}
        for group_type, sites in type_to_sites.items():
            sites = SiteModelCollection(sites)
            cfsn_summary = sites.as_region_model(true_region_model.header)
            if group_type not in {'true', 'pred'}:
                cfsn_summary.header['properties']['cache']['confusion_type'] = group_type
                cfsn_summary.header['properties']['cache']['originator'] = performer_id
            type_to_summary[group_type] = cfsn_summary

        for group_type, sites in type_to_sites.items():
            cfsn_summary = type_to_summary[group_type]
            group_site_dpath = (cfsn_group_dpath / group_type).ensuredir()
            group_region_fpath = (cfsn_group_dpath / (group_type + '.geojson'))
            text = cfsn_summary.dumps(indent='    ')
            group_region_fpath.write_text(text)
            for site in sites:
                site_fpath = group_site_dpath / (site.site_id + '.geojson')
                text = json.dumps(site, indent='    ')
                site_fpath.write_text(text)

        TIME_OVERLAP_SUMMARY = 0
        if TIME_OVERLAP_SUMMARY:
            visualize_time_overlap(type_to_summary, type_to_sites)

        USE_KML = 1
        if USE_KML:
            cfsn_kml_dpath = (config.out_dpath / 'confusion_kml').ensuredir()
            for group_type, sites in type_to_sites.items():
                cfsn_summary = type_to_summary[group_type]
                # data = cfsn_summary
                kml = to_styled_kml(cfsn_summary)
                kml_fpath = cfsn_kml_dpath / (group_type + '.kml')
                kml.save(kml_fpath)

        if USE_KML and 1:
            # TODO: write nice images that can be used with QGIS
            src_dset = kwcoco.CocoDataset(config.src_kwcoco)
            coco_img = src_dset.images().coco_images[0]
            fpath = coco_img.find_asset('salient')['file_name']
            img_lpath = cfsn_kml_dpath / 'heatmap.tiff'
            ub.symlink(fpath, img_lpath)
            fpath = coco_img.primary_image_filepath()
            img_lpath = cfsn_kml_dpath / 'img.tiff'
            ub.symlink(fpath, img_lpath)


        self.enriched_dpath = enriched_dpath
        self.enriched_sites_dpath = enriched_sites_dpath
        self.enriched_region_dpath = enriched_region_dpath

    def dump_confusion_kwcoco(self):
        import rich

        region_id = self.region_id
        enriched_dpath = self.enriched_dpath
        enriched_sites_dpath = self.enriched_sites_dpath
        enriched_region_dpath = self.enriched_region_dpath
        id_to_true_site = self.id_to_true_site
        id_to_pred_site = self.id_to_pred_site

        config = self.config
        # Write a new "enriched truth" file that reweights false negatives add
        # false positive as hard negatives.
        rich.print(f'Confusion Analysis: [link={config.out_dpath}]{config.out_dpath}[/link]')

        REPROJECT = config.src_kwcoco is not None
        if REPROJECT:
            from watch.cli import reproject_annotations
            new_coco_fpath = enriched_dpath / f'hardneg-{region_id}.kwcoco.zip'
            common_kwargs = ub.udict(
                src=config.src_kwcoco,
                dst=new_coco_fpath,
                site_models=enriched_sites_dpath,
                region_models=enriched_region_dpath,
                workers=2,
            )
            reproject_annotations.main(cmdline=0, **common_kwargs)

            if 1:
                # Project confusion site models onto kwcoco for visualization
                import kwcoco
                from watch.cli import reproject_annotations
                src_dset = kwcoco.CocoDataset(config.src_kwcoco)
                dst_dset = src_dset.copy()
                dst_dset.fpath = config.dst_kwcoco
                dst_dset.clear_annotations()
                true_site_infos2 = [s.pandas() for s in id_to_true_site.values()]
                pred_site_infos2 = [s.pandas() for s in id_to_pred_site.values()]

                # Differentiate true and predicted site-ids when projecting onto a
                # single file.
                for site_df in true_site_infos2:
                    site_id = site_df.iloc[0]['site_id']
                    a, b = site_id.rsplit('_', 1)
                    new_site_id = f'{a}_te_{b}'
                    site_df.loc[site_df.index[0], 'site_id'] = new_site_id

                for site_df in pred_site_infos2:
                    site_id = site_df.iloc[0]['site_id']
                    a, b = site_id.rsplit('_', 1)
                    new_site_id = f'{a}_{config.performer_id}_{b}'
                    site_df.loc[site_df.index[0], 'site_id'] = new_site_id

                for site_df in true_site_infos2:
                    reproject_annotations.validate_site_dataframe(site_df)

                dst_dset.clear_annotations()
                common_kwargs = ub.udict(
                    clear_existing=False,
                    src=dst_dset,
                    dst='return',
                    workers=2,
                )
                true_kwargs = common_kwargs | ub.udict(
                    role='true_confusion',
                    # propogate_strategy=False,
                    # propogate_strategy=False,
                    site_models=true_site_infos2,
                    # viz_dpath=(out_dpath / '_true_projection'),
                )
                # kwargs = true_kwargs
                pred_kwargs = common_kwargs | ub.udict(
                    role='pred_confusion',
                    site_models=pred_site_infos2,
                    # viz_dpath=(out_dpath / '_pred_projection'),
                )

                # I don't know why this isn't in-place. Maybe it is a scriptconfig thing?
                repr1 = str(dst_dset.annots())
                print(f'repr1={repr1}')
                dst_dset = reproject_annotations.main(cmdline=0, **true_kwargs)
                repr2 = str(dst_dset.annots())
                print(f'repr1={repr1}')
                print(f'repr2={repr2}')
                pred_kwargs['src'] = dst_dset
                dst_dset = reproject_annotations.main(cmdline=0, **pred_kwargs)
                repr3 = str(dst_dset.annots())
                print(f'repr1={repr1}')
                print(f'repr2={repr2}')
                print(f'repr3={repr3}')

                if config.dst_kwcoco is not None:
                    ub.Path(dst_dset.fpath).parent.ensuredir()
                    print(f'dump to dst_dset.fpath={dst_dset.fpath}')
                    dst_dset.dump()

                # set(dst_dset.annots().lookup('role', None))
                # set([x.get('role', None) for x in dst_dset.annots().lookup('cache', None)])

                # dst_dset.annots().take([0, 1, 2])
                viz_dpath = (config.out_dpath / 'summary_viz').ensuredir()
                if config.summary_visualization:
                    make_summary_visualization(dst_dset, viz_dpath)


def main(cmdline=1, **kwargs):
    """
    CommandLine:
        xdoctest -m /home/joncrall/code/watch/watch/mlops/confusor_analysis.py main
        HAS_DVC=1 xdoctest -m watch.mlops.confusor_analysis main:0

    Example:
        >>> # xdoctest: +REQUIRES(env:HAS_DVC)
        >>> from watch.mlops.confusor_analysis import *  # NOQA
        >>> import watch
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
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
    self.load_assignment()
    self.build_hard_cases()
    self.dump_confusion_geojson()
    self.dump_confusion_kwcoco()


def make_summary_visualization(dst_dset, viz_dpath):
    import kwplot
    import numpy as np

    resolution = '10GSD'

    from kwutil import util_progress
    from kwutil import util_time
    # from watch.utils import util_kwimage
    import kwarray
    import kwimage

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
            from shapely.ops import unary_union
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

    from watch import heuristics
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


def visualize_time_overlap(type_to_summary, type_to_sites, coco_dset=None):
    """
    dpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_test/_imeritbas/eval/flat/bas_poly_eval/bas_poly_eval_id_fd88699a/')
    group_dpath = (dpath / 'confusion_analysis/confusion_groups')

    import kwcoco
    coco_fpath = (dpath / 'confusion_analysis/confusion_kwcoco/confusion.kwcoco.zip')
    coco_dset = kwcoco.CocoDataset(coco_fpath)

    from watch.geoannots.geomodels import SiteModel
    from watch.geoannots.geomodels import RegionModel

    region_paths = []
    site_dpaths = []
    for p in group_dpath.ls():
        if p.endswith('.geojson'):
            region_paths.append(p)
        else:
            site_dpaths.append(p)

    type_to_summary = ub.udict({p.stem: RegionModel.coerce(p) for p in region_paths})
    type_to_summary.map_values(lambda x: len(x['features']))

    type_to_sites = ub.udict({p.name: list(SiteModel.coerce_multiple(p)) for p in site_dpaths})
    type_to_sites.map_values(len)
    """
    import pandas as pd
    from kwutil import util_time
    from watch.utils import util_gis
    from watch.geoannots.geomodels import SiteSummary

    # Ensure data structures have consistent ordering
    for key in type_to_summary.keys():
        summary = type_to_summary[key]
        sites = type_to_sites[key]
        summary_gdf = summary.pandas_summaries()
        assert not ub.find_duplicates([s.site_id for s in sites])
        id_to_site = ub.udict({s.site_id: s for s in sites})
        new_sites = list(id_to_site.take(summary_gdf['site_id']))
        assert len(new_sites) == len(sites)
        type_to_sites[key] = new_sites

    # Double check ordering worked
    for key in type_to_summary.keys():
        summary = type_to_summary[key]
        sites = type_to_sites[key]
        summary_gdf = summary.pandas_summaries()
        assert summary_gdf['site_id'].values.tolist() == [s.site_id for s in sites]

    # set(type_to_summary['true'].pandas_summaries()['site_id'])
    # set(type_to_summary['pred'].pandas_summaries()['site_id'])

    true_id_to_site = {s.site_id: s for s in type_to_sites['true']}
    pred_id_to_site = {s.site_id: s for s in type_to_sites['pred']}
    true_id_to_summary = {ss.site_id: ss for ss in map(SiteSummary.coerce, type_to_summary['true'].site_summaries())}
    pred_id_to_summary = {ss.site_id: ss for ss in map(SiteSummary.coerce, type_to_summary['pred'].site_summaries())}

    # Time analysis of false positives that overlap with something.
    true_summary = type_to_summary['true']
    wrong_summary = type_to_summary['sm_completely_wrong']

    region_start_date = true_summary.start_date
    region_end_date = true_summary.end_date

    true_gdf = true_summary.pandas_summaries()
    wrong_gdf = wrong_summary.pandas_summaries()

    # Work in UTM
    true_gdf = util_gis.project_gdf_to_local_utm(true_gdf, mode=1)
    wrong_gdf = util_gis.project_gdf_to_local_utm(wrong_gdf, mode=1)

    # Ensure the ordering is the same as the summaries
    true_sites = type_to_sites['true']
    wrong_sites = type_to_sites['sm_completely_wrong']

    SANITY_CHECKS = 1
    if SANITY_CHECKS:
        annots = coco_dset.annots()
        tid_to_aids = ub.udict(ub.group_items(annots, annots.lookup('track_id')))
        tid_to_annots = tid_to_aids.map_values(coco_dset.annots)
        tid_to_dups = tid_to_annots.map_values(lambda x: ub.find_duplicates(x.lookup('image_id')))
        assert not any(map(any, tid_to_dups.values()))

    # For each incorrect prediction check if it spatially overlaps any truth
    idx1_to_idxs2 = util_gis.geopandas_pairwise_overlaps(wrong_gdf, true_gdf)
    inspect_cases = []
    for idx1, idxs2 in idx1_to_idxs2.items():
        if len(idxs2):
            inspect_cases.append((idx1, idxs2))

    def _hack_site_id(site_id, tag):
        assert site_id.count('_') == 2
        a, b = site_id.rsplit('_', 1)
        new_site_id = f'{a}_{tag}_{b}'
        return new_site_id

    cases = []
    for idx1, idxs2 in inspect_cases:
        pred_site = wrong_sites[idx1]
        pred_obs = pred_site.pandas_observations()

        pred_dates = pred_obs['observation_date'].values
        pred_dates = list(map(util_time.coerce_datetime, pred_dates))

        assert wrong_gdf.iloc[idx1]['site_id'] == pred_site.site_id
        pred_geom = wrong_gdf.iloc[idx1].geometry
        pred_area = pred_geom.area

        for idx2 in idxs2:
            true_site = true_sites[idx2]
            true_obs = true_site.pandas_observations()

            true_geom = true_gdf.iloc[idx2].geometry
            true_area = true_geom.area

            assert pred_site.geometry.intersection(true_site.geometry).area > 0

            isect_area = true_geom.intersection(pred_geom).area
            union_area = true_geom.union(pred_geom).area
            space_iou = isect_area / union_area
            space_iot = isect_area / true_area
            space_iop = isect_area / pred_area

            assert space_iou > 0

            site_start_date = true_site.start_date or region_start_date
            site_end_date = true_site.end_date or region_end_date

            # start_x = util_kwplot.fix_matplotlib_dates([site_start_date])
            # end_x = util_kwplot.fix_matplotlib_dates([site_end_date])
            true_dates = true_obs['observation_date']
            true_dates = list(true_dates[~pd.isnull(true_dates)])
            true_dates = [site_start_date] + true_dates + [site_end_date]
            true_dates = list(map(util_time.coerce_datetime, true_dates))

            true_duration = true_dates[-1] - true_dates[0]
            pred_duration = pred_dates[-1] - pred_dates[0]

            isect_start = max(true_dates[0], pred_dates[0])
            union_start = min(true_dates[0], pred_dates[0])
            isect_end = min(true_dates[-1], pred_dates[-1])
            union_end = max(true_dates[-1], pred_dates[-1])

            isect_duration = max((isect_end - isect_start), util_time.coerce_timedelta(0))
            union_duration = max((union_end - union_start), util_time.coerce_timedelta(0))

            time_iou = isect_duration / union_duration
            time_iot = isect_duration / true_duration
            time_iop = isect_duration / pred_duration

            true_duration = true_dates[-1] - true_dates[0]
            pred_duration = pred_dates[-1] - pred_dates[0]

            true_coco_site_id = _hack_site_id(true_site.site_id, 'te')
            pred_coco_site_id = _hack_site_id(pred_site.site_id, 'kit')

            case = {
                'true_site_id': true_site.site_id,
                'pred_site_id': pred_site.site_id,

                'true_coco_site_id': true_coco_site_id,
                'pred_coco_site_id': pred_coco_site_id,

                'pred_idx': idx1,
                'true_idx': idx2,

                'pred_area': pred_area,
                'true_area': true_area,

                'space_iou': space_iou,
                'space_iot': space_iot,
                'space_iop': space_iop,

                'time_iou': time_iou,
                'time_iot': time_iot,
                'time_iop': time_iop,

                'pred_dates': pred_dates,
                'true_dates': true_dates,
            }
            cases.append(case)

    cases = sorted(cases, key=lambda x: x['time_iou'])[::-1]
    # coco_upgrade_track_ids(coco_dset)
    case = cases[4]

    import kwplot
    kwplot.autosns()

    if coco_dset is not None:
        true_site_id = case['true_coco_site_id']
        pred_site_id = case['pred_coco_site_id']

        true_site = true_id_to_site[case['true_site_id']]  # NOQA
        pred_site = pred_id_to_site[case['pred_site_id']]  # NOQA
        true_summary = true_id_to_summary[case['true_site_id']]  # NOQA
        pred_summary = pred_id_to_summary[case['pred_site_id']]  # NOQA

        if true_site_id in getattr(coco_dset.index, 'name_to_track', set()):
            raise NotImplementedError
            # true_tracks = coco_dset.tracks(names=[case['true_site_id']])
            # pred_tracks = coco_dset.tracks(names=[case['pred_site_id']])
            # true_track = true_tracks.objs[0]
            # pred_track = pred_tracks.objs[0]
            # pred_tid = pred_track['id']
            # true_tid = true_track['id']
            # true_annots = true_tracks.annots[0]
            # pred_annots = pred_tracks.annots[0]
        else:
            true_aids = list(coco_dset.index.trackid_to_aids[true_site_id])
            pred_aids = list(coco_dset.index.trackid_to_aids[pred_site_id])
            true_annots = coco_dset.annots(true_aids)
            pred_annots = coco_dset.annots(pred_aids)

        pred_annots.images.lookup('date_captured')
        true_annots.images.lookup('date_captured')

        all_aids = list(set(true_annots) | set(pred_annots))
        all_gids = list(set(true_annots.images) | set(pred_annots.images))
        all_images = coco_dset.images(all_gids)
        sortx = ub.argsort(all_images.lookup('frame_index'))
        all_images = all_images.take(sortx)

        all_annots = coco_dset.annots(all_aids)
        gid_to_aids = ub.group_items(all_annots, all_annots.images)

        assert set(all_annots.lookup('track_id')) == {true_site_id, pred_site_id}

        tci_channel_priority = [
            'red|green|blue',
            'pan',
        ]

        from shapely.ops import unary_union
        import kwimage
        import kwarray

        gid_to_dets = {}
        # Get the relevant annotations in each image
        for coco_img in ub.ProgIter(all_images.coco_images, desc='building case'):
            gid = coco_img['id']
            aids = gid_to_aids[gid]
            dets = coco_img._detections_for_resolution(aids=aids, space='video')
            gid_to_dets[gid] = dets

        all_vidspace_polys = [
            p.to_shapely() for dets in gid_to_dets.values()
            for p in dets.data['segmentations']]
        vidspace_hull = unary_union(all_vidspace_polys).convex_hull
        vidspace_poly = kwimage.Polygon.from_shapely(vidspace_hull)
        vidspace_bound = vidspace_poly.box().scale(2.0, about='centroid').quantize()

        cells = []
        for coco_img in ub.ProgIter(all_images.coco_images, desc='building case'):
            gid = coco_img['id']
            dets = gid_to_dets[gid]

            colors = []
            for obj in coco_dset.annots(dets.data['aids']).objs:
                color = obj['cache']['confusion']['color']
                colors.append(color)

            channels = find_visual_channels(coco_img, tci_channel_priority)

            tci_delayed = coco_img.imdelay(channels=channels)
            tci_imcrop = tci_delayed.crop(vidspace_bound.to_slice(), wrap=False, clip=False)

            heatmap_delayed = coco_img.imdelay(channels='salient')
            heatmap_imcrop = heatmap_delayed.crop(vidspace_bound.to_slice(), wrap=False, clip=False)

            heatmap_canvas = heatmap_imcrop.finalize()

            tci_canvas = tci_imcrop.finalize()
            tci_canvas = kwarray.robust_normalize(tci_canvas)
            tci_canvas = kwimage.fill_nans_with_checkers(tci_canvas)
            rel_dets = dets.translate((-vidspace_bound.tl_x, -vidspace_bound.tl_y))

            det_canvas = rel_dets.draw_on(tci_canvas, color=colors, alpha=0.5)
            cell_canvas = kwimage.stack_images([det_canvas, tci_canvas, heatmap_canvas], axis=0, pad=5)[..., 0:3]
            header_lines = [
                coco_img.img.get('sensor_coarse'),
                util_time.coerce_datetime(coco_img.img.get('date_captured')).date().isoformat(),
            ]

            header = kwimage.draw_text_on_image(None, text='\n'.join(header_lines))
            header = kwimage.ensure_float01(header)
            cell_canvas = kwimage.imresize(cell_canvas, dsize=(header.shape[1], None)).clip(0, 1)
            cell_canvas = kwimage.stack_images([header, cell_canvas], axis=0)
            # cell_canvas = kwimage.draw_header_text(cell_canvas, '\n'.join(header_lines), fit='grow')
            cells.append(cell_canvas)

        canvas = kwimage.stack_images(cells, axis=1, pad=10)
        kwplot.imshow(canvas, fnum=1)

        if 1:
            from watch.utils import util_kwplot
            plt = kwplot.plt
            import matplotlib.dates as mdates
            fig = kwplot.figure(fnum=1321321)
            ax = fig.gca()
            ax.cla()

            pred_xs = util_kwplot.fix_matplotlib_dates(case['pred_dates'])
            true_xs = util_kwplot.fix_matplotlib_dates(case['true_dates'])

            lineman = util_kwplot.LineManager()
            lineman.plot(pred_xs, 1, color='kitware_blue')
            lineman.plot(true_xs, 2, color='kitware_green')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=360))
            lineman.add_to_axes(ax=ax)
            lineman.setlims(ax=ax)

            ax.set_ylim(0, 3)

            fig.set_size_inches([10, 3])
            fig.subplots_adjust(left=0, bottom=0.1, top=.9, right=1)

            true_annots.images.coco_images
            pred_annots.images.coco_images

        # import ndsampler
        # TODO: use ndsampler to make this a little easier.
        # sampler = ndsampler.CocoSampler(coco_dset)
        # if 0:
        #     gid = coco_img['id']
        #     target = {
        #         'gids': [gid],
        #         'space_slice': bound.to_slice(),
        #         'channels': channels,
        #         'aids': aids
        #     }
        #     sample = sampler.load_sample(target, with_annots=True, annot_ids=aids)
        #     imdata = sample['im']
        #     dets = sample['annots']['frame_dets'][0]

    # max_date = max([max(case['pred_dates'] + case['true_dates']) for case in cases])
    # max_x = util_kwplot.fix_matplotlib_dates([max_date])[0]

    import kwplot
    kwplot.autosns()
    fig = kwplot.figure(fnum=1)
    fig.clf()

    from watch.utils import util_kwplot
    lineman = util_kwplot.LineManager()
    yloc = 1

    min_date = min([min(case['pred_dates'] + case['true_dates']) for case in cases])
    min_x = util_kwplot.fix_matplotlib_dates([min_date])[0]

    for case in cases[:]:
        pred_xs = util_kwplot.fix_matplotlib_dates(case['pred_dates'])
        true_xs = util_kwplot.fix_matplotlib_dates(case['true_dates'])
        lineman.plot(pred_xs, yloc, color='kitware_blue')
        yloc += 1
        lineman.plot(true_xs, yloc, color='kitware_green')
        yloc += 1

        idx1 = case['pred_idx']
        idx2 = case['true_idx']

        pred_site = wrong_sites[idx1]
        true_site = true_sites[idx2]

        pred_site_id = pred_site.site_id
        true_site_id = true_site.site_id

        plt = kwplot.plt

        show = ub.udict(case) & {'space_iou', 'time_iou', 'pred_area', 'true_area'}
        show['pred'] = pred_site_id
        show['true'] = true_site_id
        text = ub.urepr(show, precision=2, nl=0)
        # med_x = (max_x + min_x) / 2
        plt.annotate(text, (min_x, yloc))

        yloc += 20

    lineman.add_to_axes()
    ax = fig.gca()
    # TODO: make this formatter fixup work better.
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=360))
    lineman.setlims()


def find_visual_channels(coco_img, channel_priority):
    import kwcoco
    have_chans = coco_img.channels
    for p in channel_priority:
        p = kwcoco.FusedChannelSpec.coerce(p)
        common = have_chans & p
        if common.numel() == p.numel():
            return p


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


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/watch/mlops/confusor_analysis.py
        python -m watch.mlops.confusor_analysis
    """
    main()
