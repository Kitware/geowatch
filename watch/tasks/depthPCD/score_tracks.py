#!/usr/bin/env python3
"""
PCD = parallel change detection
"""
import geojson
import json
import os
import ubelt as ub
import scriptconfig as scfg


class ScoreTracksConfig(scfg.DataConfig):
    """
    Filter tracks based on the depth detector.
    """
    input_kwcoco = scfg.Value(None, required=True,
                              help='kwcoco file with the images to use',
                              position=1, alias=['in_file'])

    poly_kwcoco = scfg.Value(None, required=True, help=ub.paragraph(
        '''
            optional: kwcoco file with polygons (alternative to input region /
            sites)
            '''), group='track scoring')

    threshold = scfg.Value(0.4, help=ub.paragraph(
        '''
            threshold to filter polygons, very sensitive
            '''), group='track scoring')

    model_fpath = scfg.Value(None, help='Path to the depthPCD site validation model')

    region_id = scfg.Value(None, help=ub.paragraph(
        '''
            ID for region that sites belong to. If None, try to infer
            from kwcoco file.
            '''), group='convenience')

    input_region = scfg.Value(None, help='The coercable input region model')
    input_sites = scfg.Value(None, help='The coercable input site models')

    out_kwcoco = scfg.Value(None, help=ub.paragraph(
        '''
            The file path to write the "tracked" kwcoco file to.
            '''))

    out_sites_dir = scfg.Value(None, help=ub.paragraph(
        '''
        The directory where site model geojson files will be written.
        '''))

    out_site_summaries_dir = scfg.Value(None, help=ub.paragraph(
        '''
        The directory path where site summary geojson files will be written.
        '''))

    out_sites_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The file path where a manifest of all site models will be written.
        '''))

    out_site_summaries_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        The file path where a manifest of all site summary geojson files will
        be written.
        '''))

    append_mode = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        Append sites to existing region GeoJSON.
        '''), group='behavior')


def score_tracks(poly_coco_dset, img_coco_dset, thresh, model_fpath):
    from watch.tasks.depthPCD.model import getModel, normalize, TPL_DPATH

    import numpy as np
    import cv2
    import kwimage
    import ndsampler
    import pandas as pd
    from tqdm import tqdm

    print('loading site validation model')
    proto_fpath = TPL_DPATH / 'deeplab2/max_deeplab_s_backbone_os16.textproto'
    model = getModel(proto=proto_fpath)

    model.load_weights(model_fpath, by_name=True, skip_mismatch=True)
    # model.load_weights('/media/hdd2/antonio/models/urbanTCDs-use.h5', by_name=True, skip_mismatch=True)

    to_keep = []
    for coco_img in img_coco_dset.images().coco_images:
        if coco_img['sensor_coarse'] == 'WV' and 'red' in coco_img.channels:
            to_keep.append(coco_img['id'])

    dset = img_coco_dset.subset(to_keep)

    sampler = ndsampler.CocoSampler(dset)

    # FIXME: pandas is slow, we likely want to use an alternative impl
    imgs = pd.DataFrame(poly_coco_dset.dataset["images"])
    if "timestamp" not in imgs.columns:
        imgs["timestamp"] = imgs["id"]

    annots = pd.DataFrame(poly_coco_dset.dataset["annotations"])

    if annots.shape[0] == 0:
        print("Nothing to filter")
        return poly_coco_dset

    annots = annots[[
        "id", "image_id", "track_id", "score"
    ]].join(
        imgs[["timestamp"]],
        on="image_id",
    )

    track_ids_to_drop = []
    ann_ids_to_drop = []

    regions = []
    tannots = annots.groupby('track_id', axis=0)
    tq = tqdm(total=len(tannots))
    for track_id, track_group in tannots:

        first_annot_id = track_group["id"].tolist()[0]
        first_image_id = track_group["image_id"].tolist()[0]
        first_coco_img = poly_coco_dset.coco_image(first_image_id)
        first_annot = poly_coco_dset.anns[first_annot_id]
        imgspace_annot_box = kwimage.Box.coerce(first_annot['bbox'], format='xywh')
        vidspace_annot_box = imgspace_annot_box.warp(first_coco_img.warp_vid_from_img)
        ref_coco_img = first_coco_img  # dset.coco_image(dset.dataset['images'][0]['id'])

        if 1:
            # Because we want a higher resolution, we need to scale the requested
            # videospace region down. Looks like quantization errors may happen
            # here not sure how I deal with in the dataloader, it probably needs to
            # be fixed there too.

            res = '2GSD'
            scale_res_from_vidspace = ref_coco_img._scalefactor_for_resolution(space='video', resolution=res)

            # cxy = vidspace_annot_box.to_cxywh().data[0:2]
            # warp_res_from_vidspace = kwimage.Affine.coerce(scale=scale_res_from_vidspace, about=cxy)
            warp_res_from_vidspace = kwimage.Affine.scale(scale_res_from_vidspace)

            # Convert the video space annotation into requested resolution "window space"
            winspace_annot_box = vidspace_annot_box.warp(warp_res_from_vidspace)

            # Convert to center xy/with/height format
            winspace_annot_box = winspace_annot_box.toformat('cxywh')

            # Force the box to be a specific size at our window resolution
            # THIS IS THE BUG
            # winspace_target_box = winspace_annot_box.resize(*force_dsize)

            # Workaround
            winspace_target_box = winspace_annot_box.copy()
            #            size = max(winspace_target_box.data[2], winspace_target_box.data[3])
            winspace_target_box.data[2:4] = (224, 224)

            # Convert the box back to videospace
            vidspace_target_box = winspace_target_box.warp(warp_res_from_vidspace.inv())

            # Get the slice for video space
            vidspace_slice = vidspace_target_box.quantize().to_slice()
        else:
            vidspace_slice = vidspace_annot_box.to_slice()

        # find in second video
        videoName = first_coco_img.video['name']
        vidid = -1
        for v in sampler.dset.index.videos.values():
            if v['name'] == videoName:
                vidid = v['id']
        if vidid == -1:
            tq.update(1)
            continue
        target = {
            'vidid': vidid,
            'channels': 'blue|green|red',
            'allow_augment': False,
            'space_slice': vidspace_slice,
            'use_native_scale': True,
        }
        data = sampler.load_sample(target, with_annots=False, visible_thres=1.0)

        ims = data['im']

        good_ims = []
        for i in ims:
            im = np.stack([ii[..., 0] for ii in i], axis=-1) / (4096 * 2)
            im[im < 0] = 0
            im[im > 1] = 1
            im = (im * 255).astype(np.uint8)
            if im.shape[-1] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if np.mean(im == 0) > .2:
                continue
            good_ims.append(normalize(im))

        # a little average at start vs end
        nAvg = 2
        if len(good_ims) < nAvg + 1:
            tq.update(1)
            continue
        ims = []
        for i in range(nAvg):
            for j in range(-1, -nAvg - 1, -1):
                first = good_ims[i]
                last = good_ims[j]
                ims.append(np.stack([first, 0.5 * first + 0.5 * last, last], axis=-1).astype(np.float32))

        score = np.mean(model.predict(np.array(ims), batch_size=1, verbose=False)[8])
        tq.set_description('%s-%d score %3.2f' % (videoName, track_id, score))
        tq.update(1)
        if 0:

            cv2.namedWindow('main', cv2.WINDOW_GUI_NORMAL)
            i = 1
            first = good_ims[0]
            last = good_ims[-1]
            while 1:
                if i == 1:
                    i = 0
                    im = first
                else:
                    i = 1
                    im = last
                cv2.imshow('main', im / 255)
                q = cv2.waitKey(0)
                if q == 113:
                    break

        #        ks = list(coco_dset.index.videos.keys())

        if score < thresh:  # or coco_dset.index.videos[ks[0]]['name'] == 'AE_R001':
            track_ids_to_drop.append(track_id)
            ann_ids_to_drop.extend(track_group["id"].tolist())
            regions.append(videoName)

    print(f"Dropping {len(ann_ids_to_drop)} annotations from {len(track_ids_to_drop)} tracks.")
    from collections import Counter
    print(Counter(regions))
    if len(ann_ids_to_drop) > 0:
        poly_coco_dset.remove_annotations(ann_ids_to_drop)

    return poly_coco_dset, track_ids_to_drop


def main(**kwargs):
    args = ScoreTracksConfig.cli(cmdline=True, data=kwargs, strict=True)
    import rich
    rich.print('args = {}'.format(ub.urepr(args, nl=1)))

    # Import this first
    print('Importing tensorflow stuff (can take a sec)')
    from watch.tasks.depthPCD.model import getModel, normalize, TPL_DPATH  # NOQA

    from watch.cli.kwcoco_to_geojson import convert_kwcoco_to_iarpa, create_region_header
    from watch.utils import process_context

    import kwcoco
    from kwcoco.util import util_json

    if args.model_fpath is None:
        print('warning: the path to the model was not explicitly specified, '
              'attempting to automatically infer it')
        cand_model_path = ub.Path(os.environ.get('DVC_EXPT_DPATH', '')) / 'models/depthPCD/basicModel2.h5'
        if cand_model_path.exists():
            args.model_fpath = cand_model_path
        else:
            raise IOError(
                f'Attempted to infer model path {cand_model_path}, '
                'but it does not exist. Please specify it explicitly')

    model_fpath = ub.Path(args.model_fpath)
    if not model_fpath.exists():
        raise IOError(f'Specified {model_fpath=} does not exist')

    tracking_output = {
        'type': 'tracking_result',
        'info': [],
        'files': [],
    }
    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_config = util_json.ensure_json_serializable(args.asdict())
    walker = ub.IndexableWalker(jsonified_config)
    for problem in util_json.find_json_unserializable(jsonified_config):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    info = tracking_output['info']

    from watch.geoannots import geomodels
    from watch.utils import util_gis
    region_model = geomodels.RegionModel.coerce(args.input_region)
    input_site_fpaths = util_gis.coerce_geojson_paths(args.input_sites)
    # output_region_fpath = ub.Path(args.output_region_fpath)

    proc_context = process_context.ProcessContext(
        name='watch.tasks.depthPCD.score_tracks', type='process',
        config=jsonified_config,
        track_emissions=False,
    )
    proc_context.start()
    info.append(proc_context.obj)

    img_coco_dset = kwcoco.CocoDataset.coerce(args.input_kwcoco)

    if args.poly_kwcoco is not None:
        poly_coco_dset = kwcoco.CocoDataset.coerce(args.poly_kwcoco)
    else:
        from watch.cli import reproject_annotations
        img_coco_dset = reproject_annotations.main(
            cmdline=0, src=img_coco_dset,
            dst='return',
            region_models=args.input_region,
            status_to_catname={'system_confirmed': 'positive'},
            role='pred_poly',
            validate_checks=False,
            clear_existing=False,
        )
        poly_coco_dset = poly_coco_dset

    coco_dset, track_ids_to_drop = score_tracks(poly_coco_dset, img_coco_dset, args.threshold, model_fpath)

    # TODO:
    # just return the list of tracks that failed the filter. Remove those sites
    # and then just pass the rest through.

    proc_context.stop()
    out_kwcoco = args.out_kwcoco
    if out_kwcoco is not None:
        coco_dset = coco_dset.reroot(absolute=True, check=False)
        # Add tracking audit data to the kwcoco file
        coco_info = coco_dset.dataset.get('info', [])
        coco_info.append(proc_context.obj)
        coco_dset.fpath = out_kwcoco
        ub.Path(out_kwcoco).parent.ensuredir()
        print(f'write to coco_dset.fpath={coco_dset.fpath}')
        coco_dset.dump(out_kwcoco, indent=2)

        # Convert kwcoco to sites
    import safer
    verbose = 1

    if args.out_sites_dir is not None:

        sites_dir = ub.Path(args.out_sites_dir).ensuredir()
        # Also do this in BAS mode
        sites = convert_kwcoco_to_iarpa(coco_dset,
                                        default_region_id=args.region_id,
                                        as_summary=False)
        print(f'{len(sites)=}')
        # write sites to disk
        site_fpaths = []
        for site in ub.ProgIter(sites, desc='writing sites', verbose=verbose):
            site_props = site['features'][0]['properties']
            assert site_props['type'] == 'site'
            site_fpath = sites_dir / (site_props['site_id'] + '.geojson')
            site_fpaths.append(os.fspath(site_fpath))

            with safer.open(site_fpath, 'w', temp_file=not ub.WIN32) as f:
                geojson.dump(site, f, indent=2)

    if args.out_sites_fpath is not None:
        site_tracking_output = tracking_output.copy()
        site_tracking_output['files'] = site_fpaths
        out_sites_fpath = ub.Path(args.out_sites_fpath)
        print(f'Write tracked site result to {out_sites_fpath}')
        with safer.open(out_sites_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(site_tracking_output, file, indent='    ')

    # Convert kwcoco to sites summaries
    if args.out_site_summaries_dir is not None:
        sites = convert_kwcoco_to_iarpa(coco_dset,
                                        default_region_id=args.region_id,
                                        as_summary=True)
        print(f'{len(sites)=}')
        site_summary_dir = ub.Path(args.out_site_summaries_dir).ensuredir()
        # write sites to region models on disk
        groups = ub.group_items(sites, lambda site: site['properties'].pop('region_id'))

        site_summary_fpaths = []
        for region_id, site_summaries in groups.items():

            region_fpath = site_summary_dir / (region_id + '.geojson')
            if args.append_mode and region_fpath.is_file():
                with open(region_fpath, 'r') as f:
                    region = geojson.load(f)
                if verbose:
                    print(f'writing to existing region {region_fpath}')
            else:
                region = geojson.FeatureCollection(
                    [create_region_header(region_id, site_summaries)])
                if verbose:
                    print(f'writing to new region {region_fpath}')
            for site_summary in site_summaries:
                assert site_summary['properties']['type'] == 'site_summary'
                region['features'].append(site_summary)

            site_summary_fpaths.append(os.fspath(region_fpath))
            with safer.open(region_fpath, 'w', temp_file=not ub.WIN32) as f:
                geojson.dump(region, f, indent=2)

    if args.out_site_summaries_fpath is not None:
        site_summary_tracking_output = tracking_output.copy()
        site_summary_tracking_output['files'] = site_summary_fpaths
        out_site_summaries_fpath = ub.Path(args.out_site_summaries_fpath)
        out_site_summaries_fpath.parent.ensuredir()
        print(f'Write tracked site summary result to {out_site_summaries_fpath}')
        with safer.open(out_site_summaries_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(site_summary_tracking_output, file, indent='    ')


r'''
Ignore:

    python -m watch.tasks.depthPCD.score_tracks /media/barcelona/Drop6/bas_baseline/polyb.kwcoco.zip
            --images /media/barcelona/Drop6/valT.kwcoco.zip
            --out_site_summaries_fpath "/media/barcelona/Drop6/tronexperiments/debug/site_summaries_manifest.json"
            --out_site_summaries_dir "/media/barcelona/Drop6/tronexperiments/debug/site_summaries"
            --out_sites_fpath "/media/barcelona/Drop6/tronexperiments/debug/sites_manifest.json"
            --out_sites_dir "/media/barcelona/Drop6/tronexperiments/debug/sites"
            --out_kwcoco "some file with filtered poly if you need"
            --threshold 0.3 (default)


Example:

    ### Run BAS and then run SV on top of it.

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt

    # Predict BAS Heatmaps
    python -m watch.tasks.fusion.predict \
        --package_fpath="$BAS_MODEL_FPATH" \
        --test_dataset=$DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2L.kwcoco.zip \
        --pred_dataset=$DVC_EXPT_DPATH/_test_dzyne_sv/pred_heatmaps.kwcoco.zip \
        --chip_overlap="0.3" \
        --chip_dims="196,196" \
        --time_span="auto" \
        --fixed_resolution="10GSD" \
        --time_sampling="soft4" \
        --drop_unused_frames="True"  \
        --num_workers="4" \
        --devices="0," \
        --batch_size="1" \
        --with_saliency="True" \
        --with_class="False" \
        --with_change="False"

    # Convert Heatmaps to Polygons
    python -m watch.cli.run_tracker \
        --in_file "$DVC_EXPT_DPATH/_test_dzyne_sv/pred_heatmaps.kwcoco.zip" \
        --default_track_fn saliency_heatmaps \
        --track_kwargs '{
            \"agg_fn\": \"probs\",
            \"thresh\": 0.4,
            \"time_thresh\": 0.8,
            \"inner_window_size\": \"1y\",
            \"inner_agg_fn\": \"mean\",
            \"norm_ord\": \"inf\",
            \"resolution\": \"10GSD\",
            \"moving_window_size\": null,
            \"poly_merge_method\": \"v2\",
            \"polygon_simplify_tolerance\": 1,
            \"min_area_square_meters\": 7200,
            \"max_area_square_meters\": 8000000
        }' \
        --clear_annots=True \
        --site_summary 'None' \
        --boundary_region $DVC_DATA_DPATH/annotations/drop6/region_models \
        --out_site_summaries_fpath "$DVC_EXPT_DPATH/_test_dzyne_sv/site_summaries_manifest.json" \
        --out_site_summaries_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/site_summaries" \
        --out_sites_fpath "$DVC_EXPT_DPATH/_test_dzyne_sv/sites_manifest.json" \
        --out_sites_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/sites" \
        --out_kwcoco "$DVC_EXPT_DPATH/_test_dzyne_sv/poly.kwcoco.zip"

    # Score the Initial Predictions
    python -m watch.cli.run_metrics_framework \
        --merge=True \
        --name "todo" \
        --true_site_dpath "$DVC_DATA_DPATH/annotations/drop6/site_models" \
        --true_region_dpath "$DVC_DATA_DPATH/annotations/drop6/region_models" \
        --pred_sites "$DVC_EXPT_DPATH/_test_dzyne_sv/sites_manifest.json" \
        --tmp_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_before/tmp" \
        --out_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_before" \
        --merge_fpath "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_before/poly_eval_before.json"

    # Run the Site Validation Filter
    python -m watch.tasks.depthPCD.score_tracks \
        --input_kwcoco $DVC_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.json \
        --poly_kwcoco $DVC_EXPT_DPATH/_test_dzyne_sv/poly.kwcoco.zip \
        --input_region "$DVC_EXPT_DPATH/_test_dzyne_sv/site_summaries_manifest.json" \
        --input_sites "$DVC_EXPT_DPATH/_test_dzyne_sv/sites_manifest.json" \
        --model_fpath $DVC_EXPT_DPATH/models/depthPCD/basicModel2.h5 \
        --out_site_summaries_fpath "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_site_summaries_manifest.json" \
        --out_site_summaries_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_site_summaries" \
        --out_sites_fpath  "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_sites_manifest.json" \
        --out_sites_dir  "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_sites" \
        --out_kwcoco "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_poly.kwcoco.zip" \
        --threshold 0.4

    # Score the Filtered Predictions
    python -m watch.cli.run_metrics_framework \
        --merge=True \
        --name "todo" \
        --true_site_dpath "$DVC_DATA_DPATH/annotations/drop6/site_models" \
        --true_region_dpath "$DVC_DATA_DPATH/annotations/drop6/region_models" \
        --pred_sites "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_sites_manifest.json" \
        --tmp_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_after/tmp" \
        --out_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_after" \
        --merge_fpath "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_after/poly_eval_after.json"

    python -c "if 1:
        import pandas as pd
        import rich
        import json
        import ubelt as ub
        text1 = ub.Path('$DVC_EXPT_DPATH/_test_dzyne_sv/eval_before/poly_eval_before.json').read_text()
        text2 = ub.Path('$DVC_EXPT_DPATH/_test_dzyne_sv/eval_after/poly_eval_after.json').read_text()
        data1 = json.loads(text1)
        data2 = json.loads(text2)
        df1 = pd.read_json(json.dumps(data1['best_bas_rows']), orient='table')
        df2 = pd.read_json(json.dumps(data2['best_bas_rows']), orient='table')
        print('BEFORE:')
        rich.print(df1)
        print('After:')
        rich.print(df2)
    "
'''
if __name__ == '__main__':
    main()
