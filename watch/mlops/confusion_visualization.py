import pandas as pd
import json  # NOQA
from watch.mlops import smart_result_parser
import ubelt as ub


def bas_poly_eval_confusion_analysis(eval_fpath):
    """
    Given an MLops polygon evaluation, make the confusion visualizations.

    Ignore:
        ls /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/aggregate/inspect_simplify_2023-01-18T175853-5
        eval_fpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_bcabbc12/poly_eval.json')

        eval_fpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/pred/flat/sc_poly_viz/sc_poly_viz_id_735ce10d/.pred/sc_poly/sc_poly_id_ff4b875f/.succ/sc_poly_eval/sc_poly_eval_id_1ce902c2/')

        eval_fpath = ub.Path('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/sc_poly_eval/sc_poly_eval_id_efbb2d98/poly_eval.fpath')
    """
    out_dpath = eval_fpath.parent
    eval_fpath.parent / '.pred'
    poly_pred_dpaths = list((eval_fpath.parent / '.pred').glob('*_poly/*'))
    assert len(poly_pred_dpaths) == 1
    poly_pred_dpath = ub.Path(poly_pred_dpaths[0])

    info = smart_result_parser.load_eval_trk_poly(eval_fpath)
    bas_row = info['json_info']['best_bas_rows']['data'][0]
    region_id = bas_row['region_id']
    rho = bas_row['rho']
    tau = bas_row['tau']
    dpath = (eval_fpath.parent / region_id / 'overall/bas')
    assign_fpaths1 = list(dpath.glob(f'detections_tau={tau}_rho={rho}_min_area*.csv'))
    assign_fpaths2 = list(dpath.glob(f'proposals_tau={tau}_rho={rho}_min_area*.csv'))
    assert len(assign_fpaths1) == 1
    assert len(assign_fpaths2) == 1
    assign_fpath1 = assign_fpaths1[0]
    assign_fpath2 = assign_fpaths2[0]

    import watch
    dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    true_site_dpath = dvc_dpath / 'annotations/site_models'
    # true_region_dpath = dvc_dpath / 'annotations/region_models'

    from watch.utils import util_gis

    performer_id = 'kit'
    assign1 = pd.read_csv(assign_fpath1)
    assign2 = pd.read_csv(assign_fpath2)
    # hack: if there are "seq" in the site names, we need to fix those old
    # files by reinvoking.
    if any('_seq_' in m or m.startswith('seq_') for m in assign2['site model'] if m):
        invoke_fpath = eval_fpath.parent / 'invoke.sh'
        import platform
        if 'toothbrush' in invoke_fpath.read_text() and platform.node() != 'toothbrush':
            # Resolve to a local invoke
            text = invoke_fpath.read_text()
            text = text.replace('/home/joncrall/remote/toothbrush', str(ub.Path.home()))
            # Hack out paths
            newline = ''
            for line in text.split('\n'):
                line = line.strip()
                if not line.startswith('#'):
                    if line.endswith('\\'):
                        newline += line.rstrip('\\')
                    else:
                        newline += line + '\n'
            import shlex
            parts = shlex.split(newline)
            ub.Path(ub.argval('--true_site_dpath', argv=parts)).exists()
            ub.Path(ub.argval('--true_region_dpath', argv=parts)).exists()
            ub.Path(ub.argval('--pred_sites', argv=parts)).exists()
            ub.Path(ub.argval('--tmp_dir', argv=parts)).exists()
            ub.Path(ub.argval('--out_dir', argv=parts)).exists()
            ub.Path(ub.argval('--merge_fpath', argv=parts)).exists()
            print(newline)

            pred_fpath = ub.Path(ub.argval('--pred_sites', argv=parts))
            data = json.loads(pred_fpath.read_text())
            old_root = '/home/joncrall/remote/toothbrush'
            new_root = str(ub.Path.home())
            data['files'] = [f.replace(old_root, new_root) for f in data['files']]

            new_pred_fpath = pred_fpath.augment(stemsuffix='2')
            new_pred_fpath.write_text(json.dumps(data))
            fixed_invoke = newline.replace(str(pred_fpath), str(new_pred_fpath))
            print(fixed_invoke)
            _ = ub.cmd(fixed_invoke, verbose=3)

            r"""

            python -m watch.cli.run_metrics_framework --merge=True --name "todo-bas_poly_algo_id_6ad71c72-bas_pxl_algo_id_79393b54-todo" \
                    --true_site_dpath "/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc-ssd/annotations/site_models" --true_region_dpath "/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc-ssd/annotations/region_models" \
                    --pred_sites "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/pred/flat/bas_poly/bas_poly_id_2fc4e8d6/sites_manifest.json" \
                    --tmp_dir "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_9f01d221/tmp" --out_dir "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_9f01d221" --merge_fpath "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_9f01d221/poly_eval.json"


            python -m watch.cli.run_metrics_framework \
                --merge=True \
                --name "todo-bas_poly_algo_id_6ad71c72-bas_pxl_algo_id_a91d4564-todo" \
                --true_site_dpath "/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc-ssd/annotations/site_models" \
                --true_region_dpath "/home/local/KHQ/jon.crall/data/dvc-repos/smart_data_dvc-ssd/annotations/region_models" \
                --pred_sites "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/pred/flat/bas_poly/bas_poly_id_331d9ad9/sites" \
                --tmp_dir "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_1983aca0/tmp" \
                --out_dir "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_1983aca0" \
                --merge_fpath "/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/_testpipe/eval/flat/bas_poly_eval/bas_poly_eval_id_1983aca0/poly_eval.json"

            """
        else:
            info = ub.cmd(f'bash {invoke_fpath}', verbose=3)
        assign1 = pd.read_csv(assign_fpath1)
        assign2 = pd.read_csv(assign_fpath2)
        if any('_seq_' in m or m.startswith('seq_') for m in assign2['site model'] if m):
            raise AssertionError

    ### Assign a confusion label to each truth and predicted annotation
    true_confusion_rows = []
    pred_confusion_rows = []
    site_to_status = {}
    from watch import heuristics
    for row in assign1.to_dict('records'):
        true_site_id = row['truth site'].split('_te_')[0]
        pred_site_ids = []
        truth_status = row['site type']
        site_to_status[true_site_id] = truth_status
        if isinstance(row['matched site models'], str):
            for name in row['matched site models'].split(','):
                pred_site_id = name.strip().split(f'_{performer_id}_')[0]
                pred_site_ids.append(pred_site_id)
        has_positive_match = len(pred_site_ids)
        true_cfsn = heuristics.iarpa_assign_truth_confusion(truth_status, has_positive_match)
        true_confusion_rows.append({
            'true_site_id': true_site_id,
            'pred_site_ids': pred_site_ids,
            'true_confusion': true_cfsn,
            'role': 'true_confusion',
        })

    for row in assign2.to_dict('records'):
        pred_site_id = row['site model'].split(f'_{performer_id}_')[0]
        true_site_ids = []
        truth_match_statuses = []
        if isinstance(row['matched truth sites'], str):
            for name in row['matched truth sites'].split(','):
                true_site_id = name.strip().split('_te_')[0]
                truth_match_statuses.append(site_to_status[true_site_id])
                true_site_ids.append(true_site_id)
        pred_cfsn = heuristics.iarpa_assign_pred_confusion(truth_match_statuses)
        pred_confusion_rows.append({
            'pred_site_id': pred_site_id,
            'true_site_ids': true_site_ids,
            'pred_confusion': pred_cfsn,
            'role': 'pred_confusion',
        })

    for true_row in true_confusion_rows:
        true_row['confusion_color'] = heuristics.IARPA_CONFUSION_COLORS.get(true_row['true_confusion'])
        true_row['role'] = 'true_confusion'

    for pred_row in pred_confusion_rows:
        pred_row['confusion_color'] = heuristics.IARPA_CONFUSION_COLORS.get(pred_row['pred_confusion'])
        pred_row['role'] = 'pred_confusion'

    """
    True Confusion Spec
    -------------------

    "misc_info":  {
        "true_site_id": str,          # redundant site id information,
        "pred_site_ids": List[str],   # the matching predicted site ids,
        "true_confusion": str,        # the type of true confusion assigned by T&E
        "confusion_color": str,       # a named color coercable via kwimage.Color.coerce
        "role": "true_confusion",     # constant
    }

    Predicted Confusion Spec
    -------------------

    "misc_info":  {
        "pred_site_id": str,          # redundant site id information,
        "true_site_ids": List[str],   # the matching predicted site ids,
        "pred_confusion": str,        # the type of predicted confusion assigned by T&E
        "confusion_color": str,       # a named color coercable via kwimage.Color.coerce
        "role": "pred_confusion",     # constant
    }

    # The possible confusion codes and the corresponding confusion_color they
    # will be assigned is.
    IARPA_CONFUSION_COLORS = {}
    IARPA_CONFUSION_COLORS['gt_true_neg'] = 'darkgreen'  # no IARPA color for this, make one up.
    IARPA_CONFUSION_COLORS['gt_true_pos'] = 'lime'
    IARPA_CONFUSION_COLORS['gt_false_pos'] = 'red'
    IARPA_CONFUSION_COLORS['gt_false_neg'] = 'black'
    IARPA_CONFUSION_COLORS['gt_positive_unbounded'] = "darkviolet"
    IARPA_CONFUSION_COLORS['gt_ignore'] = "lightsalmon"
    IARPA_CONFUSION_COLORS['gt_seen'] = "gray"
    IARPA_CONFUSION_COLORS['sm_pos_match'] = "orange"
    IARPA_CONFUSION_COLORS['sm_partially_wrong'] = "aquamarine"
    IARPA_CONFUSION_COLORS['sm_completely_wrong'] = "magenta"
    """

    # confusion vectors -- unused
    if 0:
        pred_to_row = {r['pred_site_id']: r for r in pred_confusion_rows}
        confusion_vectors = []
        for true_row in true_confusion_rows:
            if len(true_row['pred_site_ids']):
                for pred_site_id in true_row['pred_site_ids']:
                    pred_row = pred_to_row[pred_site_id]
                    confusion_vectors.append({
                        'true_site_id': true_row['true_site_id'],
                        'true_confusion': true_row['true_confusion'],
                        'pred_site_id': pred_site_id,
                        'pred_confusion': pred_row['pred_confusion'],
                        'num_other_true': len(true_row['pred_site_ids']) - 1,
                        'num_other_pred': len(pred_row['true_site_ids']) - 1,
                    })
            else:
                confusion_vectors.append({
                    'true_site_id': true_row['true_site_id'],
                    'true_confusion': true_row['true_confusion'],
                    'pred_site_id': None,
                    'pred_confusion': None,
                    'num_other_true': 0,
                    'num_other_pred': 0,
                })

        for pred_row in pred_confusion_rows:
            if not pred_row['true_site_ids']:
                confusion_vectors.append({
                    'pred_site_id': pred_row['pred_site_id'],
                    'pred_confusion': pred_row['pred_confusion'],
                    'true_site_id': None,
                    'true_confusion': None,
                    'num_other_true': 0,
                    'num_other_pred': 0,
                })
    # /confusion vectors -- unused

    # Add the confusion info as misc data in new site files and reproject them
    # onto the truth for visualization.
    # pred_sites_fpath = poly_pred_dpath / 'sites_manifest.json'
    # assert pred_sites_fpath.exists()
    # pred_site_fpaths = list(util_gis.coerce_geojson_paths(pred_sites_fpath))
    pred_site_fpaths = list(util_gis.coerce_geojson_paths(poly_pred_dpath / 'sites'))
    # rm_files = list(true_region_dpath.glob(region_id + '*.geojson'))
    gt_files = list(true_site_dpath.glob(region_id + '*.geojson'))
    sm_files = pred_site_fpaths
    true_site_infos = list(util_gis.coerce_geojson_datas(gt_files, format='json'))
    pred_site_infos = list(util_gis.coerce_geojson_datas(sm_files, format='json'))

    id_to_true_data = {ub.Path(d['fpath']).stem: d for d in true_site_infos}
    id_to_pred_data = {ub.Path(d['fpath']).stem: d for d in pred_site_infos}

    for true_row in true_confusion_rows:
        info = id_to_true_data[true_row['true_site_id']]
        for feat in info['data']['features']:
            if 'misc_info' in feat['properties']:
                feat['properties']['misc_info'].update(true_row)
            else:
                feat['properties']['misc_info'] = true_row.copy()

    for pred_row in pred_confusion_rows:
        info = id_to_pred_data[pred_row['pred_site_id']]
        for feat in info['data']['features']:
            if 'misc_info' in feat['properties']:
                feat['properties']['misc_info'].update(pred_row)
            else:
                feat['properties']['misc_info'] = pred_row.copy()

    # Check misc info is populated correctly and add role to site model
    for pred_site_id, pred_site in id_to_pred_data.items():
        for feat in pred_site['data']['features']:
            props = feat['properties']

            import kwimage
            geom = kwimage.MultiPolygon.coerce(feat['geometry']).to_shapely()
            simple_geom = geom.simplify(0.0002)  # Hack, should do this properly in the tracker
            new_geom = kwimage.MultiPolygon.coerce(simple_geom).to_geojson()
            feat['geometry'] = new_geom
            # misc_info = props['misc_info']
            # print('misc_info = {}'.format(ub.urepr(misc_info, nl=1)))

    for true_site_id, true_site in id_to_true_data.items():
        for feat in true_site['data']['features']:
            props = feat['properties']
            assert 'misc_info' in props
            # misc_info = props['misc_info']
            # print('misc_info = {}'.format(ub.urepr(misc_info, nl=1)))

    cfsn_dpath = out_dpath / 'confusion_sites'
    true_cfsn_dpath = (cfsn_dpath / 'true').ensuredir()
    pred_cfsn_dpath = (cfsn_dpath / 'pred').ensuredir()

    # Dump confusion site models to disk
    for pred_site_id, pred_site in id_to_pred_data.items():
        fpath = pred_cfsn_dpath / (pred_site_id + '.geojson')
        text = json.dumps(pred_site['data'], indent='    ')
        fpath.write_text(text)

    for true_site_id, true_site in id_to_true_data.items():
        fpath = true_cfsn_dpath / (true_site_id + '.geojson')
        text = json.dumps(true_site['data'], indent='    ')
        fpath.write_text(text)

    # Project confusion site models onto kwcoco for visualization
    from watch.cli import project_annotations
    import kwcoco
    src_fpath = poly_pred_dpath / 'poly.kwcoco.json'
    if not src_fpath.exists():
        if ub.Path(src_fpath + '.zip').exists():
            src_fpath = ub.Path(src_fpath + '.zip')
    dst_fpath = out_dpath / 'poly_toviz.kwcoco.json'
    src_dset = kwcoco.CocoDataset(src_fpath)

    if True:
        old_root = '/home/joncrall/remote/toothbrush'
        new_root = str(ub.Path.home())
        for img in src_dset.images().coco_images:
            for obj in img.iter_asset_objs():
                obj['file_name'] = obj['file_name'].replace(old_root, new_root)
        ...
    dst_dset = src_dset.copy()
    dst_dset.fpath = dst_fpath
    cmdline = 0

    true_site_infos2 = list(util_gis.coerce_geojson_datas(
        id_to_true_data.values(), format='dataframe', allow_raw=True))
    pred_site_infos2 = list(util_gis.coerce_geojson_datas(
        id_to_pred_data.values(), format='dataframe', allow_raw=True))

    for info in pred_site_infos2:
        site_df = info['data']

    for info in true_site_infos2:
        site_df = info['data']
        project_annotations.validate_site_dataframe(site_df)

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
    dst_dset = project_annotations.main(cmdline=cmdline, **true_kwargs)
    repr2 = str(dst_dset.annots())
    print(f'repr1={repr1}')
    print(f'repr2={repr2}')
    pred_kwargs['src'] = dst_dset
    dst_dset = project_annotations.main(cmdline=cmdline, **pred_kwargs)
    repr3 = str(dst_dset.annots())
    print(f'repr1={repr1}')
    print(f'repr2={repr2}')
    print(f'repr3={repr3}')

    set(dst_dset.annots().lookup('role', None))
    set([x['role'] for x in dst_dset.annots().lookup('misc_info', None)])
    # dst_dset.annots().take([0, 1, 2])

    summary_visualization(dst_dset, out_dpath)

    if 1:
        # FIXME
        from watch.cli import coco_visualize_videos
        kwargs = dict(
            src=dst_dset,
            smart=True,
            role_order=['true_confusion', 'pred_confusion'],
            resolution='10 GSD',
            # workers=0,
            workers='avail',
            draw_labels=False,
            animate={'frames_per_second': 10},
            draw_imgs=False,
        )
        coco_visualize_videos.main(cmdline=cmdline, **kwargs)

    eval_dpath = ub.Path(dst_dset.fpath).parent
    return eval_dpath

    # TODO:
    # Run coco_align on the different sites or groups of sites to
    # split them by category and inspect them individually.


def summary_visualization(dst_dset, out_dpath):
    import kwplot
    import numpy as np

    resolution = '10GSD'
    viz_dpath = (out_dpath / 'bas_summary_viz').ensuredir()

    from watch.utils import util_progress
    from watch.utils import util_kwimage
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
            executor = ub.Executor('process', max_workers=2)
            # with executor:
            if 1:
                jobs = []
                for coco_img in pman(images.coco_images, desc='submit delay jobs'):
                    delayed = coco_img.delay('salient', resolution=resolution, nodata_method='float')
                    job = executor.submit(delayed.finalize)
                    jobs.append(job)
                from concurrent.futures import as_completed

                jobs = set(jobs)
                job_loader = pman(as_completed(jobs), total=len(jobs), desc='averaging heatmaps')
                for job in job_loader:
                    jobs.remove(job)
                    im = job.result()
                    running.update(im)
                    # track_ids = dst_dset.annots(dets.data['aids']).lookup('track_id')

            all_dets = kwimage.Detections.concatenate(det_accum)
            all_annots = dst_dset.annots(all_dets.data['aids'])
            all_dets.data['frame_index'] = np.array(all_annots.images.lookup('frame_index'))
            all_dets.data['track_id'] = np.array(all_annots.lookup('track_id'))
            all_dets.data['role'] = np.array(all_annots.lookup('role'))
            all_dets.data['misc_info'] = np.array(all_annots.lookup('misc_info'))

            for a, b in zip(all_dets.data['role'], all_dets.data['misc_info']):
                ...

            for ann in dst_dset.anns.values():
                ann['role']
            {ann['misc_info']['role'] for ann in dst_dset.anns.values()}
            {ann['role'] for ann in dst_dset.anns.values()}
            # ...

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
            current = running.current()
            mean_heatmap = current['mean']
            # min_heatmap = current['min']
            # max_heatmap = current['max']
            canvas = kwplot.make_heatmask(util_kwimage.exactly_1channel(mean_heatmap), cmap='viridis')[:, :, 0:3]
            # canvas_raw = canvas.copy()
            canvas_true = canvas.copy()
            canvas_pred = canvas.copy()
            canvas_cfsn = canvas.copy()

            role_to_summary = ub.udict(ub.group_items(track_summaries, key=lambda x: x['role']))
            print(ub.udict(role_to_summary).map_values(len))

            for row in ub.ProgIter(role_to_summary['true_confusion']):
                row['poly'].draw_on(canvas_true, fill=False, edgecolor=row['confusion_color'])
                row['poly'].draw_on(canvas_cfsn, fill=False, edgecolor=row['confusion_color'])

            for row in ub.ProgIter(role_to_summary['pred_confusion']):
                row['poly'].draw_on(canvas_pred, fill=False, edgecolor=row['confusion_color'])
                row['poly'].draw_on(canvas_cfsn, fill=False, edgecolor=row['confusion_color'])

            canvas_true = kwimage.draw_header_text(canvas_true, 'true confusion')
            canvas_pred = kwimage.draw_header_text(canvas_pred, 'pred confusion')
            canvas_cfsn = kwimage.draw_header_text(canvas_cfsn, 'both')

            final_canvas = kwimage.stack_images([canvas_true, canvas_pred, canvas_cfsn], axis=1, pad=10)
            final_canvas = kwimage.ensure_uint255(final_canvas)
            fpath = viz_dpath / f'confusion_{video["name"]}.jpg'
            kwimage.imwrite(fpath, final_canvas)
