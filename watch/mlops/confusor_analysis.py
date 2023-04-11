#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class ConfusorAnalysisConfig(scfg.DataConfig):
    """
    Requires that IARPA metrics are computed
    """
    detections_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        truth assignments
        usually detections_tau=0.2_rho=0.5_min_area=0.csv
        '''))
    proposals_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        detection assignments
        usually proposals_tau=0.2_rho=0.5_min_area=0.csv
        '''))

    src_kwcoco = scfg.Value(None, help='the input kwcoco file to project onto')
    dst_kwcoco = scfg.Value(None, help='the reprojected output kwcoco file to write')

    pred_sites = scfg.Value(None, help='the path to the predicted sites manifest / directory / globstr')
    bas_metric_dpath = scfg.Value(None, help='A path to bas metrics if det/prop paths are not specified')

    region_id = scfg.Value(None, help='the id for the region')
    true_site_dpath = scfg.Value(None, help='input')
    true_region_dpath = scfg.Value(None, help='input')

    performer_id = scfg.Value('kit', help='the performer id')

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
    import pandas as pd
    import itertools as it
    from watch import heuristics
    from watch.utils import util_gis
    rich.print('config = ' + ub.urepr(config, nl=1, align=':'))
    region_id = config.region_id

    performer_id = config.performer_id
    true_site_dpath = config.true_site_dpath
    true_region_dpath = config.true_region_dpath
    pred_site_fpaths = list(util_gis.coerce_geojson_paths(config.pred_sites))

    if config.bas_metric_dpath is not None:
        if config.detections_fpath is None:
            config.detections_fpath = config.bas_metric_dpath / 'detections_tau=0.2_rho=0.5_min_area=0.csv'
        if config.proposals_fpath is None:
            config.proposals_fpath = config.bas_metric_dpath / 'proposals_tau=0.2_rho=0.5_min_area=0.csv'

    config.detections_fpath = ub.Path(config.detections_fpath)
    config.proposals_fpath = ub.Path(config.proposals_fpath)
    assert config.proposals_fpath.exists()
    assert config.detections_fpath.exists()

    assign1 = pd.read_csv(config.detections_fpath)
    assign2 = pd.read_csv(config.proposals_fpath)

    needs_recompute = any('_seq_' in m or m.startswith('seq_') for m in assign2['site model'] if m)
    assert not needs_recompute

    def fix_site_id(site_id):
        # Hack because idk why the metrics code does this.
        if site_id.startswith('_'):
            site_id = region_id + site_id
        return site_id

    ### Assign a confusion label to each truth and predicted annotation
    true_confusion_rows = []
    pred_confusion_rows = []
    site_to_status = {}
    for row in assign1.to_dict('records'):
        true_site_id = row['truth site'].split('_te_')[0]
        true_site_id = fix_site_id(true_site_id)
        pred_site_ids = []
        truth_status = row['site type']
        site_to_status[true_site_id] = truth_status
        if isinstance(row['matched site models'], str):
            for name in row['matched site models'].split(','):
                pred_site_id = name.strip().split(f'_{performer_id}_')[0]
                pred_site_id = fix_site_id(pred_site_id)
                pred_site_ids.append(pred_site_id)
        has_positive_match = len(pred_site_ids)
        true_cfsn = heuristics.iarpa_assign_truth_confusion(truth_status, has_positive_match)
        true_confusion_rows.append({
            'true_site_id': true_site_id,
            'pred_site_ids': pred_site_ids,
            'type': true_cfsn,
        })

    for row in assign2.to_dict('records'):
        pred_site_id = row['site model'].split(f'_{performer_id}_')[0]
        pred_site_id = fix_site_id(pred_site_id)
        true_site_ids = []
        truth_match_statuses = []
        if isinstance(row['matched truth sites'], str):
            for name in row['matched truth sites'].split(','):
                true_site_id = name.strip().split('_te_')[0]
                true_site_id = fix_site_id(true_site_id)
                truth_match_statuses.append(site_to_status[true_site_id])
                true_site_ids.append(true_site_id)
        pred_cfsn = heuristics.iarpa_assign_pred_confusion(truth_match_statuses)
        pred_confusion_rows.append({
            'pred_site_id': pred_site_id,
            'true_site_ids': true_site_ids,
            'type': pred_cfsn,
        })

    for row in true_confusion_rows + pred_confusion_rows:
        row['color'] = heuristics.IARPA_CONFUSION_COLORS.get(row['type'])

    """
    True Confusion Spec
    -------------------

    "misc_info":  {
        "confusion": {
            "true_site_id": str,          # redundant site id information,
            "pred_site_ids": List[str],   # the matching predicted site ids,
            "type": str,                  # the type of true confusion assigned by T&E
            "color": str,                 # a named color coercable via kwimage.Color.coerce
        }
    }

    Predicted Confusion Spec
    -------------------

    "misc_info":  {
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

    # Add the confusion info as misc data in new site files and reproject them
    # onto the truth for visualization.
    # pred_sites_fpath = poly_pred_dpath / 'sites_manifest.json'
    # assert pred_sites_fpath.exists()
    # pred_site_fpaths = list(util_gis.coerce_geojson_paths(pred_sites_fpath))

    rm_files = list(true_region_dpath.glob(region_id + '*.geojson'))

    gt_files = list(true_site_dpath.glob(region_id + '*.geojson'))
    sm_files = pred_site_fpaths
    true_site_infos = list(util_gis.coerce_geojson_datas(gt_files, format='json'))
    pred_site_infos = list(util_gis.coerce_geojson_datas(sm_files, format='json'))
    orig_region_infos = list(util_gis.coerce_geojson_datas(rm_files, format='json'))
    assert len(orig_region_infos) == 1

    # Ensure all site data has misc-info
    # Ensure all data cast to site models
    from watch.geoannots.geomodels import SiteModel, SiteModelCollection, RegionModel
    import json

    true_region_model = RegionModel(orig_region_infos[0]['data'])

    for info in it.chain(pred_site_infos, true_site_infos):
        info['data'] = SiteModel(**info['data'])
        info['data'].header['properties'].setdefault('misc_info', {})

    id_to_true_data = {ub.Path(d['fpath']).stem: d for d in true_site_infos}
    id_to_pred_data = {ub.Path(d['fpath']).stem: d for d in pred_site_infos}

    # Add confusion metadata to predicted and truth models
    # https://gis.stackexchange.com/questions/346518/opening-geojson-style-properties-in-qgis
    for row in true_confusion_rows:
        info = id_to_true_data[row['true_site_id']]
        info['data'].header['properties']['misc_info']['confusion'] = row
    for row in pred_confusion_rows:
        info = id_to_pred_data[row['pred_site_id']]
        info['data'].header['properties']['misc_info']['confusion'] = row

    pred_sites = SiteModelCollection([d['data'] for d in pred_site_infos])
    true_sites = SiteModelCollection([d['data'] for d in true_site_infos])

    VALIDATE = 1
    if VALIDATE:
        all_models = SiteModelCollection(pred_sites + true_sites)
        all_models.fixup()
        all_models.validate(workers=8)

    pred_region_model = pred_sites.as_region_model(region=true_region_model.header)
    pred_df = pred_region_model.pandas_summaries()
    true_df = true_region_model.pandas_summaries()
    idx1_to_idx2 = util_gis.geopandas_pairwise_overlaps(pred_df, true_df)

    # Find predicted annotations that have no truth overlap
    non_intersecting_idxs = [idx1 for idx1, idx2s in idx1_to_idx2.items() if not len(idx2s)]
    cand_df = pred_df.iloc[non_intersecting_idxs]

    hard_positive_site_ids = []
    for true_site in true_sites:
        misc = true_site.header['properties']['misc_info']
        if misc['confusion']['type'] in 'gt_false_neg':
            hard_positive_site_ids.append(true_site.header['properties']['site_id'])

    hard_negative_sites = []
    for site_id in cand_df['site_id']:
        pred_site = id_to_pred_data[site_id]['data']
        misc = pred_site.header['properties']['misc_info']
        if misc['confusion']['type'] in 'sm_completely_wrong':
            hard_negative_sites.append(pred_site.deepcopy())

    orig_site_num = int(true_df['site_id'].max().split('_')[-1])
    base_site_num = max(900, orig_site_num)
    for num, hard_neg in enumerate(hard_negative_sites, start=base_site_num):
        header_prop = hard_neg.header['properties']
        header_prop['site_id'] = region_id + f'_{num:04d}'
        header_prop['status'] = 'negative'
        header_prop['model_content'] = 'annotation'
        header_prop['misc_info'].pop('confusion', None)
        header_prop['misc_info']['kwcoco'] = {'weight': 1.5}
        header_prop['comments'] = 'hard_negative'
        for obs in hard_neg.observations():
            props = obs['properties']
            props['current_phase'] = None
            props['is_site_boundary'] = None
            props.pop('predicted_phase_transition', None)
            props.pop('predicted_phase_transition_date', None)

    hard_neg_summaries = [s.as_summary() for s in hard_negative_sites]
    new_region = true_region_model.deepcopy()
    new_region['features'] += hard_neg_summaries
    new_true_sites = [s.deepcopy() for s in true_sites]

    # Upweight hard positive true sites
    print('hard_positive_site_ids = {}'.format(ub.urepr(hard_positive_site_ids, nl=1)))
    new_true_props = [n.header['properties'] for n in new_true_sites] + [s['properties'] for s in new_region.site_summaries()]
    for prop in new_true_props:
        if 'misc_info' not in prop:
            prop['misc_info'] = {}
        prop['misc_info'].pop('confusion', None)
        if prop['site_id'] in hard_positive_site_ids:
            prop['misc_info']['kwcoco'] = {'weight': 2.0}
            if 'comments' not in prop or not prop['comments']:
                prop['comments'] = 'hard_positive'
            else:
                prop['comments'] += ';hard_positive'

    # Add in hard negatives
    new_sites = new_true_sites + hard_negative_sites

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
    cfsn_dpath = config.out_dpath / 'confusion_sites'
    true_cfsn_dpath = (cfsn_dpath / 'true').ensuredir()
    pred_cfsn_dpath = (cfsn_dpath / 'pred').ensuredir()
    for pred_site_id, pred_site in id_to_pred_data.items():
        fpath = pred_cfsn_dpath / (pred_site_id + '.geojson')
        text = json.dumps(pred_site['data'], indent='    ')
        fpath.write_text(text)
    for true_site_id, true_site in id_to_true_data.items():
        fpath = true_cfsn_dpath / (true_site_id + '.geojson')
        text = json.dumps(true_site['data'], indent='    ')
        fpath.write_text(text)

    # Need to build site summaries from site models.

    # Write a new "enriched truth" file that reweights false negatives add
    # false positive as hard negatives.

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

        if 0:
            # Project confusion site models onto kwcoco for visualization
            import kwcoco
            from watch.cli import reproject_annotations
            src_dset = kwcoco.CocoDataset(config.src_kwcoco)
            dst_dset = src_dset.copy()
            dst_dset.fpath = config.dst_kwcoco
            dst_dset.clear_annotations()
            true_site_infos2 = list(util_gis.coerce_geojson_datas(
                id_to_true_data.values(), format='dataframe', allow_raw=True))
            pred_site_infos2 = list(util_gis.coerce_geojson_datas(
                id_to_pred_data.values(), format='dataframe', allow_raw=True))

            for info in pred_site_infos2:
                site_df = info['data']

            for info in true_site_infos2:
                site_df = info['data']
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
            kwargs = true_kwargs
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

            set(dst_dset.annots().lookup('role', None))
            set([x['role'] for x in dst_dset.annots().lookup('misc_info', None)])
            # dst_dset.annots().take([0, 1, 2])
            viz_dpath = cfsn_dpath
            summary_visualization(dst_dset, viz_dpath)
            rich.print(f'Viz Dpath: [link={viz_dpath}]{viz_dpath}[/link]')


def summary_visualization(dst_dset, viz_dpath):
    import kwplot
    import numpy as np

    resolution = '5GSD'

    from watch.utils import util_progress
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

                    from watch.utils import util_time
                    time = util_time.coerce_datetime(coco_img['timestamp'])
                    job.time = time
                    job.coco_img = coco_img

                oldest_time = None
                oldest_img = None
                newest_time = None
                newest_img = None

                job_loader = pman(jobs.as_completed(), total=len(jobs), desc='averaging heatmaps')
                for job in job_loader:
                    im = job.result()

                    if oldest_time is None or oldest_time > job.time:
                        if job.coco_img['sensor_coarse'] == 'S2':
                            oldest_time = job.time
                            oldest_img = im

                    if newest_time is None or newest_time < job.time:
                        if job.coco_img['sensor_coarse'] == 'S2':
                            newest_time = job.time
                            newest_img = im

                    im = kwimage.atleast_3channels(im)
                    running.update(im)
                    # track_ids = dst_dset.annots(dets.data['aids']).lookup('track_id')

            all_dets = kwimage.Detections.concatenate(det_accum)
            all_annots = dst_dset.annots(all_dets.data['aids'])
            all_dets.data['frame_index'] = np.array(all_annots.images.lookup('frame_index'))
            all_dets.data['track_id'] = np.array(all_annots.lookup('track_id'))
            all_dets.data['role'] = np.array(all_annots.lookup('role'))
            all_dets.data['misc_info'] = np.array(all_annots.lookup('misc_info'))

            groupers = list(zip(all_dets.data['role'], all_dets.data['track_id']))
            unique_tids, groupxs = kwarray.group_indices(groupers)

            track_summaries = []
            from shapely.ops import unary_union
            for (role, tid), groupx in zip(unique_tids, groupxs):
                track_dets = all_dets.take(groupx)
                misc_info = track_dets.data['misc_info'][0]
                row = misc_info.copy()
                # row['role'] = role
                # assert row['role'] == role
                sh_poly = unary_union([p.to_shapely() for p in track_dets.data['segmentations']])
                kw_poly = kwimage.MultiPolygon.from_shapely(sh_poly)
                row['poly'] = kw_poly
                track_summaries.append(row)

            # canvas = kwplot.make_heatmask(util_kwimage.exactly_1channel(mean_heatmap), cmap='magma')[:, :, 0:3]

            old_canvas = kwimage.normalize_intensity(oldest_img, axis=2)
            new_canvas = kwimage.normalize_intensity(newest_img, axis=2)

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
                'new': {
                    'true': new_canvas.copy(),
                    'pred': new_canvas.copy(),
                    'cfsn': new_canvas.copy(),
                },
                'old': {
                    'true': old_canvas.copy(),
                    'pred': old_canvas.copy(),
                    'cfsn': old_canvas.copy(),
                },
                'avg': {
                    'true': canvas.copy(),
                    'pred': canvas.copy(),
                    'cfsn': canvas.copy(),
                },
            }

            role_to_summary = ub.udict(ub.group_items(track_summaries, key=lambda x: x['role']))
            print(ub.udict(role_to_summary).map_values(len))

            alpha = 0.6

            for row in ub.ProgIter(role_to_summary['true_confusion']):
                for k1, v1 in canvases.items():
                    v1['true'] = row['poly'].draw_on(v1['true'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                    v1['cfsn'] = row['poly'].draw_on(v1['cfsn'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                # row['poly'].draw_on(canvas_true, fill=False, edgecolor=row['confusion_color'])
                # row['poly'].draw_on(canvas_cfsn, fill=False, edgecolor=row['confusion_color'])

            for row in ub.ProgIter(role_to_summary['pred_confusion']):
                for k1, v1 in canvases.items():
                    v1['pred'] = row['poly'].draw_on(v1['pred'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                    v1['cfsn'] = row['poly'].draw_on(v1['cfsn'], fill=False, edgecolor=row['confusion_color'], alpha=alpha)
                # row['poly'].draw_on(canvas_pred, fill=False, edgecolor=row['confusion_color'])
                # row['poly'].draw_on(canvas_cfsn, fill=False, edgecolor=row['confusion_color'])

            for k1, v1 in canvases.items():
                v1['true'] = kwimage.draw_header_text(v1['true'], 'true confusion')
                v1['pred'] = kwimage.draw_header_text(v1['pred'], 'pred confusion')
                v1['cfsn'] = kwimage.draw_header_text(v1['cfsn'], 'both')

            row1 = kwimage.stack_images(list(canvases['avg'].values()), axis=1, pad=10)
            row2 = kwimage.stack_images(list(canvases['old'].values()), axis=1, pad=10)
            row3 = kwimage.stack_images(list(canvases['new'].values()), axis=1, pad=10)
            row1 = kwimage.ensure_uint255(row1)
            row2 = kwimage.ensure_uint255(row2)
            row3 = kwimage.ensure_uint255(row3)

            final_canvas = kwimage.stack_images([row1, row2, row3], axis=0, pad=5)

            fpath = viz_dpath / f'confusion_{video["name"]}.jpg'
            kwimage.imwrite(fpath, final_canvas)

    from watch import heuristics
    legend_img = kwplot.make_legend_img(heuristics.IARPA_CONFUSION_COLORS)
    kwimage.imwrite(viz_dpath / 'confusion_legend.png', legend_img)


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/watch/mlops/confusor_analysis.py
        python -m watch.mlops.confusor_analysis
    """
    main()
