#!/usr/bin/env python3
"""
PCD = parallel change detection
"""

import os
import ubelt as ub
import scriptconfig as scfg


class ScoreTracksConfig(scfg.DataConfig):
    """
    Filter tracks based on the depth detector.
    """
    input_kwcoco = scfg.Value(None, required=True, help=ub.paragraph(
        '''
        The input kwcoco file with the high-resolution images to use for the
        depth filter. This does not to cover all sites, any site this does not
        cover will be automatically accepted.
        '''), position=1, alias=['in_file'], group='inputs')

    model_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        Path to the depth_pcd site validation model
        '''), group='inputs')

    input_region = scfg.Value(None, help=ub.paragraph(
        '''
        The coercable input region model with site summaries
        '''), group='inputs')

    out_kwcoco = scfg.Value(None, help=ub.paragraph(
        '''
            The file path to write the "tracked" kwcoco file to.
            '''), group='outputs')


def score_tracks(img_coco_dset, model_fpath):
    from geowatch.tasks.depth_pcd.model import getModel, normalize, TPL_DPATH

    import numpy as np
    import cv2
    import kwimage
    import ndsampler
    import pandas as pd
    from tqdm import tqdm
    import warnings

    print('loading site validation model')
    proto_fpath = TPL_DPATH / 'deeplab2/max_deeplab_s_backbone_os16.textproto'
    use_ln = False
    res = '2GSD'
    if str(model_fpath).find('model4') != -1:
        use_ln = True
        res = '1GSD'
    model = getModel(proto=proto_fpath, use_ln=use_ln)

    model.load_weights(model_fpath, by_name=True, skip_mismatch=True)
    # model.load_weights('/media/hdd2/antonio/models/urbanTCDs-use.h5', by_name=True, skip_mismatch=True)

    to_keep = []
    for coco_img in img_coco_dset.images().coco_images:
        if coco_img['sensor_coarse'] == 'WV' and 'red' in coco_img.channels:
            to_keep.append(coco_img['id'])

    dset = img_coco_dset.subset(to_keep)

    sampler = ndsampler.CocoSampler(dset)

    all_videos = dset.videos()
    all_annots = dset.annots()

    if len(all_annots) == 0:
        print("Nothing to filter")
        return img_coco_dset

    vidid_to_name = all_videos.lookup('name', keepid=True)
    # vidname_to_id = ub.udict(vidid_to_name).invert()
    annot_video_ids = all_annots.images.lookup('video_id')
    annot_timestamp = all_annots.images.lookup('timestamp')
    annot_video_names = list(ub.take(vidid_to_name, annot_video_ids))
    annot_image_ids = all_annots.lookup('image_id')
    annot_track_ids = all_annots.lookup('track_id')

    annot_df = pd.DataFrame({
        'video_id': np.array(annot_video_ids),
        'timestamp': np.array(annot_timestamp),
        'video_name': np.array(annot_video_names),
        'image_id': np.array(annot_image_ids),
        'track_id': np.array(annot_track_ids),
        'id': np.array(all_annots.ids),
    })
    # Group by track and video name.
    trackid_to_group = dict(list(annot_df.groupby('track_id')))

    # TODO: use new kwcoco track mechanisms
    # Ideally the dataset will be passed to us with tracks.
    # This will happen once reproject annotations handles it.
    if 'tracks' not in img_coco_dset.dataset.keys():
        img_coco_dset.dataset['tracks'] = []
    tracks = img_coco_dset.dataset['tracks']
    tq = tqdm(total=len(trackid_to_group))

    for track_id, orig_track_group in trackid_to_group.items():
        # TODO: use new kwcoco track mechanisms
        track_obj = {
            'id': track_id,
            'name': track_id,  # add a name for "future-proofing"
            'score': 1.0,
            'src': 'sv_depth_pcd'
        }
        tracks.append(track_obj)
        # Does the track appear in more than one video?
        video_names = orig_track_group['video_name'].unique()
        if len(video_names) > 1:
            if track_id not in video_names:
                msg = ub.paragraph(
                    f'''
                    track-id {track_id} expected to correspond with video names
                    'in site-cropped datasets
                    ''')
                warnings.warn(msg)
                continue
            # take the "main" video for this track
            track_group = orig_track_group[orig_track_group['video_name'] == track_id]
        else:
            track_group = orig_track_group

        track_group = track_group.sort_values('timestamp')

        video_id = track_group["video_id"].iloc[0]
        video_name = track_group["video_name"].iloc[0]
        image_ids = track_group["image_id"].tolist()
        first_annot_id = track_group["id"].iloc[0]
        first_image_id = image_ids[0]
        first_coco_img = img_coco_dset.coco_image(first_image_id)
        first_annot = img_coco_dset.anns[first_annot_id]

        # Read the location of the first annotation in "image space"
        imgspace_annot_box = kwimage.Box.coerce(first_annot['bbox'], format='xywh')

        # Convert the location of the annotation to "video space"
        vidspace_annot_box = imgspace_annot_box.warp(first_coco_img.warp_vid_from_img)

        # Because we want a higher resolution, we need to scale the requested
        # videospace region down. Looks like quantization errors may happen
        # here not sure how I deal with in the dataloader, it probably needs to
        # be fixed there too.
        scale_res_from_vidspace = first_coco_img._scalefactor_for_resolution(space='video', resolution=res)

        # This is the transform from video space (i.e. the space we use to talk
        # to ndsampler) to the final resolution we want to sample.
        warp_res_from_vidspace = kwimage.Affine.scale(scale_res_from_vidspace)

        # Convert the video space annotation into requested resolution "window space"
        winspace_annot_box = vidspace_annot_box.warp(warp_res_from_vidspace)

        # Convert to center xy/with/height format
        winspace_annot_box = winspace_annot_box.toformat('cxywh')

        # Force the box to be a specific size at our window resolution
        force_dsize = (224, 224)
        winspace_target_box = winspace_annot_box.resize(*force_dsize)

        # Convert the box back to videospace
        vidspace_target_box = winspace_target_box.warp(warp_res_from_vidspace.inv())

        # Get the slice for video space
        # vidspace_slice = vidspace_target_box.quantize().to_slice()
        # vidspace_slice = vidspace_target_box.quantize().to_ltrb().quantize().to_slice()
        vidspace_slice = vidspace_target_box.to_ltrb().quantize().to_slice()

        # The space slice is specified in video space, so to recover the
        # requested resolution, we pass the videospace -> samplespace
        # scalefactor. We could assert that decompose opertion should have
        # zeros everywhere but scale, but we aren't.
        scale = warp_res_from_vidspace.decompose()['scale']

        target = {
            'vidid': video_id,
            'gids': image_ids,
            'channels': 'blue|green|red',
            'allow_augment': False,
            'space_slice': vidspace_slice,

            'use_native_scale': False,
            'scale': scale,
        }
        try:
            data = sampler.load_sample(target, with_annots=False)
        except ValueError:
            print('warning: failed to sample with value error')
            tq.update(1)
            continue
        except Exception:
            # Sample again with more info or debugging and then
            target['verbose_ndsample'] = True
            sampler.load_sample(target, with_annots=False)
            print('warning: failed to sample with unknown exception')
            tq.update(1)
            continue
            # raise

        ims = data['im']

        good_ims = []
        for i in ims:
            im = np.stack([ii[..., 0] for ii in i], axis=-1) / (4096 * 2)
            im = im.clip(0, 1)
            im = (im * 255).astype(np.uint8)
            if im.shape[-1] == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            if np.mean(im == 0) > .2:
                continue
            norm_im = normalize(im)
            good_ims.append(norm_im)

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
        track_obj['score'] = float(score)
        tq.set_description(f'{video_name} score {score:3.2f}')
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

    return img_coco_dset


def main(cmdline=True, **kwargs):
    args = ScoreTracksConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('args = {}'.format(ub.urepr(args, nl=1)))

    # Import this first
    print('Importing tensorflow stuff (can take a sec)')
    from geowatch.tasks.depth_pcd.model import getModel, normalize, TPL_DPATH  # NOQA

    import kwcoco
    from kwcoco.util import util_json
    from geowatch.utils import process_context

    if args.model_fpath is None:
        print('warning: the path to the model was not explicitly specified, '
              'attempting to automatically infer it')
        cand_model_path = ub.Path(os.environ.get('DVC_EXPT_DPATH', '')) / 'models/depth_pcd/basicModel2.h5'
        if cand_model_path.exists():
            args.model_fpath = cand_model_path
        else:
            raise IOError(
                f'Attempted to infer model path {cand_model_path}, '
                'but it does not exist. Please specify it explicitly')

    model_fpath = ub.Path(args.model_fpath)
    if not model_fpath.exists():
        raise IOError(f'Specified {model_fpath=} does not exist')

    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_config = util_json.ensure_json_serializable(args.asdict())
    walker = ub.IndexableWalker(jsonified_config)
    for problem in util_json.find_json_unserializable(jsonified_config):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    proc_context = process_context.ProcessContext(
        name='geowatch.tasks.depth_pcd.score_tracks', type='process',
        config=jsonified_config,
        track_emissions=False,
    )
    proc_context.start()

    img_coco_dset = kwcoco.CocoDataset.coerce(args.input_kwcoco)

    # Project the site polygons onto the kwcoco dataset.
    from geowatch.cli import reproject_annotations
    img_coco_dset = reproject_annotations.main(
        cmdline=0, src=img_coco_dset,
        dst='return',
        region_models=args.input_region,
        status_to_catname={'system_confirmed': 'positive'},
        role='pred_poly',
        validate_checks=False,
        clear_existing=False,
    )

    coco_dset = score_tracks(img_coco_dset, model_fpath)

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


r'''
Example:

    ### Run BAS and then run SV on top of it.

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt

    # Predict BAS Heatmaps
    python -m geowatch.tasks.fusion.predict \
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
    python -m geowatch.cli.run_tracker \
        --in_file "$DVC_EXPT_DPATH/_test_dzyne_sv/pred_heatmaps.kwcoco.zip" \
        --default_track_fn saliency_heatmaps \
        --track_kwargs '{
            "agg_fn": "probs",
            "thresh": 0.4,
            "time_thresh": 0.8,
            "inner_window_size": "1y",
            "inner_agg_fn": "mean",
            "norm_ord": "inf",
            "resolution": "10GSD",
            "moving_window_size": null,
            "poly_merge_method": "v2",
            "polygon_simplify_tolerance": 1,
            "min_area_square_meters": 7200,
            "max_area_square_meters": 8000000
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
    python -m geowatch.cli.run_metrics_framework \
        --merge=True \
        --name "todo" \
        --true_site_dpath "$DVC_DATA_DPATH/annotations/drop6/site_models" \
        --true_region_dpath "$DVC_DATA_DPATH/annotations/drop6/region_models" \
        --pred_sites "$DVC_EXPT_DPATH/_test_dzyne_sv/sites_manifest.json" \
        --tmp_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_before/tmp" \
        --out_dir "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_before" \
        --merge_fpath "$DVC_EXPT_DPATH/_test_dzyne_sv/eval_before/poly_eval_before.json"

    # Run the Site Validation Filter
    python -m geowatch.tasks.depth_pcd.score_tracks \
        --input_kwcoco $DVC_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.json \
        --input_region "$DVC_EXPT_DPATH/_test_dzyne_sv/site_summaries_manifest.json" \
        --input_sites "$DVC_EXPT_DPATH/_test_dzyne_sv/sites_manifest.json" \
        --model_fpath $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5 \
        --out_kwcoco "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_poly.kwcoco.zip" \


        --output_sites_dpath  "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_sites" \
        --output_region_fpath "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_site_summaries.json" \
        --output_site_manifest_fpath  "$DVC_EXPT_DPATH/_test_dzyne_sv/filtered_sites_manifest.json" \
        --threshold 0.4

    # Score the Filtered Predictions
    python -m geowatch.cli.run_metrics_framework \
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


Example in MLOPs:

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    geowatch schedule --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R001_I2LS.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2LS.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-BR_R002_I2LS.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2LS.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2LS.kwcoco.zip
                # - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-AE_R001_I2LS.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims: auto
            bas_pxl.time_span: auto
            bas_pxl.time_sampling: soft4
            bas_poly.thresh:
                # - 0.42
                - 0.425
            bas_poly.inner_window_size: 1y
            bas_poly.inner_agg_fn: mean
            bas_poly.norm_ord: inf
            bas_poly.polygon_simplify_tolerance: 1
            bas_poly.agg_fn: probs
            bas_poly.time_thresh:
                # - 0.65
                - 0.8
            bas_poly.resolution: 10GSD
            bas_poly.moving_window_size: null
            bas_poly.poly_merge_method: 'v2'
            bas_poly.min_area_square_meters: 7200
            bas_poly.max_area_square_meters: 8000000
            bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
            bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
            bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
            bas_pxl.enabled: 1
            bas_pxl_eval.enabled: 0
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_poly_viz.enabled: 0
            sv_crop.enabled: 1
            sv_crop.minimum_size: "256x256@2GSD"
            sv_crop.num_start_frames: 10
            sv_crop.num_end_frames: 10
            sv_crop.context_factor: 1.5

            sv_dino_boxes.enabled: 1
            sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
            sv_dino_boxes.window_dims: 256
            sv_dino_boxes.window_overlap: 0.5
            sv_dino_boxes.fixed_resolution: 3GSD

            sv_dino_filter.enabled: 1
            sv_dino_filter.end_min_score: 0.15
            sv_dino_filter.start_max_score: 1.0
            sv_dino_filter.box_score_threshold: 0.01
            sv_dino_filter.box_isect_threshold: 0.1

            sv_depth_score.enabled: 1
            sv_depth_score.model_fpath: $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
            sv_depth_filter.threshold:
                - 0.20
            #     - 0.4
        submatrices:
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R001_I2LS.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2LS.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-BR_R002_I2LS.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-BR_R002.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2LS.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-CH_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2LS.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-NZ_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-AE_R001_I2LS.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-AE_R001.kwcoco.json
        " \
        --root_dpath="$DVC_EXPT_DPATH/_mlops_test_depth_pcd2" \
        --devices="0," --tmux_workers=2 \
        --backend=serial --queue_name "_mlops_test_depth_pcd2" \
        --pipeline=bas_building_and_depth_vali \
        --skip_existing=1 \
        --run=0

        --pipeline=bas_depth_vali \


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas_building_vali \
    --target \
        "$DVC_EXPT_DPATH/_mlops_test_depth_pcd" \
    --stdout_report="
        top_k: 100
        per_group: 2
        macro_analysis: 0
        analyze: 0
        reference_region: final
        print_models: True
    " \
    --resource_report=0 \
    --query='
        `params.bas_poly.thresh` == 0.425 and
        `params.bas_pxl.package_fpath`.str.contains("V47_epoch47_")
    ' \
    --plot_params="
        enabled: False
        compare_sv_hack: True
        stats_ranking: 0
        min_variations: 1
    " \
    --export_tables=0 \
    --io_workers=10 \
    --output_dpath="$DVC_EXPT_DPATH/_mlops_test_depth_pcd/aggregate" \
    --rois=KR_R002,CH_R001,NZ_R001

    # --rois=KR_R002,

    CH_R001,NZ_R001,KR_R001,BR_R002
'''
if __name__ == '__main__':
    main()
