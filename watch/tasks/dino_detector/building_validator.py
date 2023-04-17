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
    input_kwcoco_fpath = scfg.Value(None, help='input')


def main(cmdline=1, **kwargs):
    """
    Ignore:
        >>> import watch
        >>> dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
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

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/watch/tasks/dino_detector/building_validator.py
        python -m watch.tasks.dino_detector.building_validator
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
                - auto
                - soft5
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
            sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
            sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
            bas_pxl.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_poly_viz.enabled: 0
            valicrop.enabled: 1
            valicrop.minimum_size: "256x256@2GSD"
            valicrop.num_start_frames: 3
            valicrop.num_end_frames: 3
            valicrop.context_factor: 1.5
            buildings.enabled: 1
            buildings.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
            buildings.window_dims: 1024
            buildings.window_overlap: "0.5"
            buildings.fixed_resolution: "1GSD"
        submatrices:
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001.kwcoco.zip
              valicrop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R002.kwcoco.zip
              valicrop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-BR_R002.kwcoco.zip
              valicrop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-BR_R002.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-CH_R001.kwcoco.zip
              valicrop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-CH_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-NZ_R001.kwcoco.zip
              valicrop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-NZ_R001.kwcoco.json
            - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-AE_R001.kwcoco.zip
              valicrop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-AE_R001.kwcoco.json
        " \
        --root_dpath="$DVC_EXPT_DPATH/_mlops_eval10_baseline" \
        --devices="0," --tmux_workers=2 \
        --backend=tmux --queue_name "_mlops_eval10_baseline" \
        --pipeline=bas_building_vali --skip_existing=1 \
        --run=1

geowatch align \
    --src "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop6/imgonly-KR_R001.kwcoco.json" \
    --dst "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/valicrop/valicrop_id_2e8c8dc3/valicrop.kwcoco.zip" \
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

geowatch visualize /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/valicrop/valicrop_id_2e8c8dc3/valicrop.kwcoco.zip --smart

python ~/code/watch/dev/wip/grid_sitevali_crops.py /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/valicrop/valicrop_id_2e8c8dc3/_viz_*


python -m watch.tasks.dino_detector.predict \
    --package_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/kitware/xview_dino.pt" \
    --coco_fpath="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/valicrop/valicrop_id_2e8c8dc3/valicrop.kwcoco.zip" \
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


NODE_DPATH=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_mlops_eval10_baseline/pred/flat/buildings/buildings_id_fd298dba/

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
geowatch reproject \
        --src "$NODE_DPATH/pred_boxes.kwcoco.zip" \
        --dst "$NODE_DPATH/pred_boxes_with_polys.kwcoco.zip" \
        --region_models "$NODE_DPATH/.pred/valicrop/*/.pred/bas_poly/*/site_summaries_manifest.json" \
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


"""
