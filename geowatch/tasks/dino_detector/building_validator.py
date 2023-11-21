r"""
Uses the building detector to validate a construction event.

Given a site model:

* search for the K highest quality images at the start of the sequences.

* search for the K highest quality images at the end of the sequence.

* run the building detector on all of the chosen images.

* test if the site boundary intersects detections in the start images.

* test if the site boundary intersects detections in the end images.


.. code::

    StartTest | EndTest | Result
    ----------+---------+-------
        T     |    T    | Reject
    ----------+---------+-------
        T     |    F    | Reject
    ----------+---------+-------
        F     |    T    | Accept
    ----------+---------+-------
        F     |    F    | Reject
    ----------+---------+-------
        ?     |    T    | Accept
    ----------+---------+-------
        T     |    ?    | Reject
    ----------+---------+-------
        ?     |    F    | Reject
    ----------+---------+-------
        F     |    ?    | Accept
    ----------+---------+-------


Dataflow:

    * BAS outputs a region model with candidate site summaries

    * We should be given a kwcoco path that indexes all of the data we could
      look at. In MLOPs this will be a region cropped kwcoco path that indexes
      existant images on disk. In smartflow this will be a virtual kwcoco file
      that requires network access.
"""
#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class BuildingValidatorConfig(scfg.DataConfig):
    input_kwcoco = scfg.Value(None, help='input')

    input_region = scfg.Value(None)
    input_sites = scfg.Value(None)

    output_region_fpath = scfg.Value(None)
    output_sites_dpath = scfg.Value(None)
    output_site_manifest_fpath = scfg.Value(None)

    box_isect_threshold = scfg.Value(0.1, help='This fraction of the a detected building box must intersect the proposed polygon')
    box_score_threshold = scfg.Value(0.01, help='The detected building boxes must have a score higher than this')
    start_max_score = scfg.Value(1.0, help='The max building score needed in the start sequence')
    end_min_score = scfg.Value(0.1, help='The min building score needed in the end sequence')


IGNORE_CLASS_NAMES = ub.codeblock(
    '''
    Fixed-wing Aircraft
    Small Aircraft
    Cargo Plane
    Helicopter
    Passenger Vehicle
    Small Car
    Bus
    Pickup Truck
    Utility Truck
    Truck
    Cargo Truck
    Truck w/Box
    Truck Tractor
    Trailer
    Truck w/Flatbed
    Truck w/Liquid
    Crane Truck
    Railway Vehicle
    Passenger Car
    Cargo Car
    Flat Car
    Tank car
    Locomotive
    Maritime Vessel
    Motorboat
    Sailboat
    Tugboat
    Barge
    Fishing Vessel
    Ferry
    Yacht
    Container Ship
    Oil Tanker
    Engineering Vehicle
    Tower crane
    Container Crane
    Reach Stacker
    Straddle Carrier
    Mobile Crane
    Dump Truck
    Haul Truck
    Scraper/Tractor
    Front loader/Bulldozer
    Excavator
    Cement Mixer
    Ground Grader
    ''').split('\n')

DONT_IGNORE_CLASSNAMES = ub.codeblock(
    '''
    Aircraft Hangar
    Helipad
    Storage Tank
    Hut/Tent
    Shipping container lot
    Shipping Container
    Pylon
    Tower
    Shed
    Building
    Damaged Building
    Facility
    Construction Site
    Vehicle Lot
    ''').split('\n')


def main(cmdline=1, **kwargs):
    """
    Ignore:
        from geowatch.tasks.dino_detector.building_validator import *  # NOQA
        NODE_DPATH = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_fd298dba'
        from geowatch.utils.partial_format import fsubtemplate
        coco_fpath = fsubtemplate('$NODE_DPATH/pred_boxes.kwcoco.zip')
        from kwutil.util_path import coerce_patterned_paths
        site_summary = ub.Path(fsubtemplate('$NODE_DPATH/.pred/sv_crop/*/.pred/bas_poly/*/site_summaries_manifest.json'))
        site_summary = ub.Path(fsubtemplate('$NODE_DPATH/.pred/valicrop/*/.pred/bas_poly/*/site_summaries_manifest.json'))
        site_summary = coerce_patterned_paths(site_summary)[0]

        output_region_fpath = ub.Path(fsubtemplate('$NODE_DPATH/filtered_site_summaries.geojson'))

        kwargs = {
            'input_kwcoco': coco_fpath,
            'input_region': site_summary,
            'input_sites': (site_summary.parent / 'sites'),
            'output_region_fpath': output_region_fpath,
        }
        cmdline = 0

    Ignore:
        >>> import geowatch
        >>> dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> coco_fpath = dvc_data_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json'
        >>> region_fpath = dvc_data_dpath / 'annotations/drop6_hard_v1/region_models/KR_R001.geojson'

    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict()
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = BuildingValidatorConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = ' + ub.urepr(config, nl=1))

    from geowatch.geoannots import geomodels
    from geowatch.utils import util_gis
    import kwcoco
    from geowatch.cli import reproject_annotations
    from kwutil import util_time
    from kwcoco.util import util_json
    from geowatch.utils import process_context
    import os
    import safer
    import json

    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_config = util_json.ensure_json_serializable(config.asdict())
    walker = ub.IndexableWalker(jsonified_config)
    for problem in util_json.find_json_unserializable(jsonified_config):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)
    filter_output = {
        'type': 'tracking_result',
        'info': [],
        'files': [],
    }
    # Track process info
    proc_context = process_context.ProcessContext(
        name='geowatch.tasks.dino_detector.building_validator', type='process',
        config=jsonified_config,
        track_emissions=False,
    )
    proc_context.start()
    filter_output['info'].append(proc_context.obj)

    input_coco = kwcoco.CocoDataset(config.input_kwcoco)
    region_model = geomodels.RegionModel.coerce(config.input_region)
    output_region_fpath = ub.Path(config.output_region_fpath)

    input_site_fpaths = util_gis.coerce_geojson_paths(config.input_sites)
    # set(input_coco.annots().lookup('track_id', None))

    site_to_site_fpath = ub.udict({
        p.stem: p for p in input_site_fpaths
    })
    input_sites = list(geomodels.SiteModel.coerce_multiple(site_to_site_fpath.values()))

    site_id_to_summary = ub.udict()
    for summary in region_model.site_summaries():
        assert summary.site_id not in site_id_to_summary
        site_id_to_summary[summary.site_id] = summary
    ##

    # Ensure caches
    for summary in site_id_to_summary.values():
        summary['properties'].setdefault('cache', {})
    for site in input_sites:
        site.header['properties'].setdefault('cache', {})

    output_kwcoco = reproject_annotations.main(
        cmdline=0, src=input_coco.copy(),
        dst='return',
        region_models=config.input_region,
        status_to_catname={'system_confirmed': 'positive'},
        role='pred_poly',
        validate_checks=False,
        clear_existing=False,
    )

    # Enrich all sites with features (evidence) and decisions.
    site_to_decisions = {
        s: {
            'type': 'dino_decision',
            'accept': True,
            'why': None,
            'features': None,
        }
        for s in site_id_to_summary.keys()
    }
    for video_id in ub.ProgIter(output_kwcoco.videos(), desc='validate sites'):
        video = output_kwcoco.index.videos[video_id]
        video_name = video['name']  # the vide name should be the site id
        summary = site_id_to_summary[video_name]
        site_id = summary.site_id
        video_images = output_kwcoco.images(video_id=video_id)

        start_date = summary.start_date
        end_date = summary.end_date

        # Get the starting and ending observations
        start_images = []
        end_images = []
        for coco_img in video_images.coco_images:
            img_time = util_time.coerce_datetime(coco_img['date_captured'])
            dist_start = abs(img_time - start_date)
            dist_end = abs(img_time - end_date)
            if dist_end < dist_start:
                end_images.append(coco_img)
            else:
                start_images.append(coco_img)

        start_features = []
        end_features = []
        try:
            for coco_img in start_images:
                feat = building_in_image_features(coco_img, site_id, config)
                feat['datetime'] = coco_img['date_captured']
                feat['type'] = 'start_feature'
                start_features.append(feat)

            for coco_img in end_images:
                feat = building_in_image_features(coco_img, site_id, config)
                feat['datetime'] = coco_img['date_captured']
                feat['type'] = 'end_feature'
                end_features.append(feat)

            if len(start_features) and len(end_features):
                max_start_score = max(f['max_score'] for f in start_features)
                max_end_score = max(f['max_score'] for f in end_features)
                accept = bool(
                    max_start_score <= config.start_max_score and
                    max_end_score >= config.end_min_score
                )
                why = 'threshold'
            else:
                # Unobservable case, automatically accept
                accept = True
                why = 'unobservable'
        except CouldNotValidate:
            accept = True
            why = 'CouldNotValidate'
        decision = site_to_decisions[site_id]
        decision['accept'] = accept
        decision['why'] = why
        decision['features'] = start_features + end_features
        decision = util_json.ensure_json_serializable(decision)
        site_to_decisions[site_id] = decision

    num_accept = sum(d['accept'] for s, d in site_to_decisions.items())
    print(f'Filter to {num_accept} / {len(site_id_to_summary)} sites')

    # Enrich each site summary with the decision reason and update status
    for site_id, decision in site_to_decisions.items():
        sitesum = site_id_to_summary[site_id]

        # Change the status of sites to "system_rejected" instead of droping them
        if not decision['accept']:
            sitesum['properties']['status'] = 'system_rejected'

        if 'cache' not in sitesum['properties']:
            sitesum['properties']['cache'] = {}

        sitesum['properties']['cache']['dino_decision'] = decision

    # Copy the site models and update their header with new summary
    # information.
    output_sites_dpath = ub.Path(config.output_sites_dpath)
    output_sites_dpath.ensuredir()
    out_site_fpaths = []

    for old_site in input_sites:
        old_fpath = site_to_site_fpath[old_site.site_id]
        new_fpath = output_sites_dpath / old_fpath.name
        new_summary = site_id_to_summary[site_id]
        old_site.header['properties']['status'] = new_summary['properties']['status']
        if 'cache' not in old_site.header['properties']:
            old_site.header['properties']['cache'] = {}
        old_site.header['properties']['cache'].update(new_summary['properties']['cache'])
        new_fpath.write_text(old_site.dumps())
        out_site_fpaths.append(new_fpath)

    # Write the updated site summaries in a new region model
    new_summaries = list(site_id_to_summary.values())
    new_region_model = geomodels.RegionModel.from_features(
        [region_model.header] + list(new_summaries))
    output_region_fpath.parent.ensuredir()
    print(f'Write filtered region model to: {output_region_fpath}')
    with safer.open(output_region_fpath, 'w', temp_file=not ub.WIN32) as file:
        json.dump(new_region_model, file, indent=4)

    # from kwutil.util_json import debug_json_unserializable
    # debug_json_unserializable(new_region_model)

    proc_context.stop()

    if config.output_site_manifest_fpath is not None:
        filter_output['files'] = [os.fspath(p) for p in out_site_fpaths]
        print(f'Write filtered site result to {config.output_site_manifest_fpath}')
        with safer.open(config.output_site_manifest_fpath, 'w', temp_file=not ub.WIN32) as file:
            json.dump(filter_output, file, indent=4)


def building_in_image_features(coco_img, site_id, config):
    import numpy as np
    import kwimage
    import geopandas as gpd
    import warnings
    annots = coco_img.annots()
    flags = np.array([r == 'pred_poly' for r in annots.lookup('role', None)], dtype=bool)
    box_annots = annots.compress(~flags)
    poly_annots = annots.compress(flags)
    is_main_poly = [t == site_id for t in poly_annots.lookup('track_id')]
    main_poly_annots = poly_annots.compress(is_main_poly)
    if len(main_poly_annots) > 1:
        warnings.warn('FIXME: Len of "main-poly-annots" is not 1.')
    elif len(main_poly_annots) == 0:
        raise CouldNotValidate('We dont expect to be here')

    flags = [cname not in IGNORE_CLASS_NAMES for cname in box_annots.category_names]
    box_annots = box_annots.compress(flags)

    iooa_thresh = config.box_isect_threshold
    score_thresh = config.box_score_threshold

    box_gdf = gpd.GeoDataFrame({
        'geometry': box_annots.boxes.to_shapely(),
        'class': box_annots.category_names,
        'score': box_annots.lookup('score'),
    })
    box_gdf = box_gdf[box_gdf.score > score_thresh]

    proposal_gdf = gpd.GeoDataFrame({
        'geometry': [
            kwimage.MultiPolygon.coerce(p).to_shapely()
            for p in main_poly_annots.lookup('segmentation')
        ],
    })
    proposal_geom = proposal_gdf.iloc[0]['geometry']

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'invalid value encountered in intersection')
        intersections = box_gdf.intersection(proposal_geom)
        bad_flags = (intersections.isnull() | intersections.is_empty)
        isect_flags = (~bad_flags) & (intersections.is_valid)
        feat = {
            'max_iou': 0,
            'max_iooa': 0,
            'max_score': 0,
            'num_cands': 0,
        }
        if isect_flags.any():
            cand_boxes = box_gdf[isect_flags]
            cand_isects = intersections[isect_flags]
            cand_iooa1 = cand_isects.area / cand_boxes.area
            ioaa_flags = cand_iooa1 > iooa_thresh
            if ioaa_flags.any():
                cand_boxes = cand_boxes[ioaa_flags]

                cand_unions = cand_boxes.union(proposal_geom)
                cand_ious = cand_isects.area / cand_unions.area

                feat['max_iou'] = cand_ious.max()
                feat['max_iooa'] = cand_iooa1.max()
                feat['max_score'] = cand_boxes['score'].max()
                feat['num_cands'] = len(cand_boxes)
        return feat


class CouldNotValidate(Exception):
    ...


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/geowatch/tasks/dino_detector/building_validator.py
        python -m geowatch.tasks.dino_detector.building_validator
    """
    main()

"""

Ignore:
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    geowatch schedule --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R002.kwcoco.zip
                - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-BR_R002.kwcoco.zip
                - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-CH_R001.kwcoco.zip
                - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-NZ_R001.kwcoco.zip
                - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001.kwcoco.zip
                - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-AE_R001.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims:
                - auto
            bas_pxl.time_span:
                - auto
            bas_pxl.time_sampling:
                # - auto
                # - soft5
                - soft4
            bas_poly.thresh:
                - 0.33
                #- 0.38
                #- 0.4
            bas_poly.inner_window_size:
                - 1y
                #- null
            bas_poly.inner_agg_fn:
                - mean
            bas_poly.norm_ord:
                #- 1
                - inf
            bas_poly.polygon_simplify_tolerance:
                - 1
            bas_poly.agg_fn:
                - probs
            bas_poly.resolution:
                - 10GSD
            bas_poly.moving_window_size:
                - null
                #- 1
            bas_poly.poly_merge_method:
                - 'v2'
                #- 'v1'
            bas_poly.min_area_square_meters:
                - 7200
            bas_poly.max_area_square_meters:
                - 8000000
            bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
            bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
            bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
            bas_pxl.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_poly_viz.enabled: 0
            sv_crop.enabled: 1
            sv_crop.minimum_size: "256x256@2GSD"
            sv_crop.num_start_frames: 3
            sv_crop.num_end_frames: 3
            sv_crop.context_factor: 1.5
            sv_dino_boxes.enabled: 1
            sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
            sv_dino_boxes.window_dims:
                - 256
                - 512
                - 768
                - 1024
                # - 1536
            sv_dino_boxes.window_overlap:
                - "0.5"
            sv_dino_boxes.fixed_resolution:
                - "1GSD"
                - "2GSD"
                - "2.5GSD"
                - "3GSD"
                - "3.3GSD"
            sv_dino_filter.box_isect_threshold:
                - 0.1
            sv_dino_filter.box_score_threshold:
                - 0.01
            sv_dino_filter.start_max_score:
                - 1.0
                - 0.8
                # - 0.5
            sv_dino_filter.end_min_score:
                - 0.0
                # - 0.05
                - 0.1
                - 0.15
                - 0.2
                - 0.25
                - 0.3
                # - 0.4
                - 0.5
        submatrices:
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R002.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-BR_R002.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-BR_R002.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-CH_R001.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-CH_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-NZ_R001.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-NZ_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-AE_R001.kwcoco.zip
              sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-AE_R001.kwcoco.json
        " \
        --root_dpath="$DVC_EXPT_DPATH/_mlops_eval10_baseline" \
        --devices="0," --tmux_workers=8 \
        --backend=tmux --queue_name "_mlops_eval10_baseline" \
        --pipeline=bas_building_vali --skip_existing=1 \
        --run=1

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
gwmlops aggregate \
    --pipeline=bas_building_vali \
    --target \
        "$DVC_EXPT_DPATH/_mlops_eval10_baseline" \
    --resource_report=0 \
    --rois='[KR_R002,KR_R001,BR_R002,CH_R001,NZ_R001,AE_R001]' \
    --stdout_report="
        top_k: 5
        per_group: 1
        macro_analysis: 0
        analyze: 0
        reference_region: final
        shorten: 1
    " --eval_nodes="[sv_poly_eval]" \
    --plot_params="
        enabled: True
    "

    --rois='[NZ_R001,KR_R001]' \



        print_models: True

geowatch align \
    --src "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/imgonly-KR_R001.kwcoco.json" \
    --dst "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/sv_crop/valicrop_id_2e8c8dc3/sv_crop.kwcoco.zip" \
    --regions="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/bas_poly/bas_poly_id_dc32b2a6/site_summaries_manifest.json" \
    --site_summary=True \
    --verbose="1" \
    --workers="32" \
    --aux_workers="4" \
    --debug_valid_regions="False" \
    --visualize="False" \
    --keep="img" \
    --geo_preprop="auto"  \
    --minimum_size="256x256@2GSD" \
    --num_start_frames="3" \
    --num_end_frames="3" \
    --context_factor="1.5" \
    --include_sensors="WV" \
    --force_nodata="-9999" \
    --rpc_align_method="orthorectify" \
    --target_gsd="2" \
    --force_min_gsd="2" \
    --convexify_regions="True"

geowatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/sv_crop/valicrop_id_2e8c8dc3/sv_crop.kwcoco.zip --smart

python ~/code/watch/dev/wip/grid_sitevali_crops.py /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/sv_crop/valicrop_id_2e8c8dc3/_viz_*


python -m geowatch.tasks.dino_detector.predict \
    --package_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/kitware/xview_dino.pt" \
    --coco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/sv_crop/valicrop_id_2e8c8dc3/sv_crop.kwcoco.zip" \
    --out_coco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_61b8c2c7/pred_boxes.kwcoco.zip" \
    --device="0" \
    --data_workers="2" \
    --fixed_resolution="1.0GSD" \
    --window_dims="2048" \
    --batch_size="1"

geowatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_61b8c2c7/pred_and_truth.kwcoco.zip \
    --resolution=2GSD \
    --smart \
    --ann_score_thresh=0.3 \
    --viz_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_61b8c2c7/_vizme

python ~/code/watch/dev/wip/grid_sitevali_crops.py --sub=_anns \
    /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_61b8c2c7/_vizme


NODE_DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_fd298dba
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)

geowatch reproject \
        --src "$NODE_DPATH/pred_boxes.kwcoco.zip" \
        --dst "$NODE_DPATH/pred_boxes_with_polys.kwcoco.zip" \
        --region_models "$NODE_DPATH/.pred/sv_crop/*/.pred/bas_poly/*/site_summaries_manifest.json" \
        --status_to_catname="{system_confirmed: positive}" \
        --role=pred_poly \
        --validate_checks=False \
        --clear_existing=False

geowatch reproject \
        --src "$NODE_DPATH/pred_boxes_with_polys.kwcoco.zip" \
        --dst "$NODE_DPATH/pred_and_truth.kwcoco.zip" \
        --region_models="$DVC_DATA_DPATH/annotations/drop6/region_models/*.geojson" \
        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson" \
        --status_to_catname="{system_confirmed: positive}" \
        --role=truth \
        --clear_existing=False

gw visualize --smart 1 \
    --ann_score_thresh 0.5 \
    --draw_labels False \
    --alpha 0.5 \
    --src $NODE_DPATH/pred_and_truth.kwcoco.zip \
    --viz_dpath $NODE_DPATH/_vizme \

python ~/code/watch/dev/wip/grid_sitevali_crops.py \
    --sub=_anns \
    $NODE_DPATH/_vizme

python -m geowatch.tasks.dino_detector.building_validator \
    --input_kwcoco "$NODE_DPATH/pred_boxes.kwcoco.zip" \
    --input_region $NODE_DPATH/.pred/sv_crop/*/.pred/bas_poly/*/site_summaries_manifest.json \
    --output_region_fpath "$NODE_DPATH/filtered_summaries.json" \
    --box_isect_threshold 0.1 \
    --box_score_threshold 0.1 \
    --start_max_score 0.0 \
    --end_min_score 0.3

python -m geowatch.tasks.dino_detector.building_validator \
    --input_kwcoco="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_663bd461/pred_boxes.kwcoco.zip" \
    --input_region="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/bas_poly/bas_poly_id_f8061df6/site_summaries_manifest.json" \
    --input_sites="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/bas_poly/bas_poly_id_f8061df6/sites_manifest.json" \
    --output_region_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/building_validate/building_validate_id_60a56eed/out_region.geojson" \
    --output_sites_dpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/building_validate/building_validate_id_60a56eed/out_sites" \
    --output_site_manifest_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/building_validate/building_validate_id_60a56eed/out_site_manifest.json" \
    --box_isect_threshold="0.1" \
    --box_score_threshold="0.1" \
    --start_max_score="0.1" \
    --end_min_score="0.3"


"""
