#!/bin/bash
__doc__="
This is a simplified version of tutorial 6 that does not require COLD features.
"

### DEFINE VARIABLES

# TODO: add instructions for how to set these if they are unset.
#
# If you can access our DVC repo:
#DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_DATA_DPATH=$HOME/data/dvc-repos/tutorial7-data

# The "name" of the demo dataset we will create
DATASET_SUFFIX=KHQ_Tutorial7_Data

# Set this to where you want to build the dataset
DEMO_DPATH=$DVC_DATA_DPATH/$DATASET_SUFFIX

REGION_ID=KHQ_R001
RAW_DSET_DPATH=$DEMO_DPATH/Aligned-$DATASET_SUFFIX
TIMECOMBO_DSET_DPATH=$DEMO_DPATH/TimeCombine-$DATASET_SUFFIX

# ==================================

# Create a demo region file, and create vairables that point at relevant
# paths, which are by default written in your ~/.cache folder
python -m geowatch.demo.demo_region

REGION_FPATH="$HOME/.cache/geowatch/demo/annotations/${REGION_ID}.geojson"
SITE_GLOBSTR="$HOME/.cache/geowatch/demo/annotations/${REGION_ID}_sites/*.geojson"

mkdir -p "$DEMO_DPATH"

# This is a string code indicating what STAC endpoint we will pull from
SENSORS="sentinel-2-l2a,landsat-c2l2-sr,landsat-c2l2-bt"

# Depending on the STAC endpoint, some parameters may need to change:
# collated - True for IARPA endpoints, Usually False for public data
# requester_pays - True for public landsat
# api_key - A secret for non public data
export REQUESTER_PAYS=True
export SMART_STAC_API_KEY=""
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Construct the TA2-ready dataset.
# This is a cmdqueue pipeline of simpler commands
python -m geowatch.cli.queue_cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --cloud_cover=20 \
    --stac_query_mode=auto \
    --sensors "$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated False \
    --requester_pays=$REQUESTER_PAYS \
    --dvc_dpath="$DEMO_DPATH" \
    --aws_profile=None \
    --region_globstr="$REGION_FPATH" \
    --site_globstr="$SITE_GLOBSTR" \
    --fields_workers=8 \
    --convert_workers=0 \
    --align_workers=26 \
    --cache=1 \
    --skip_existing=0 \
    --ignore_duplicates=1 \
    --target_gsd="10GSD" \
    --visualize=False \
    --max_products_per_region=10 \
    --backend=serial \
    --run=1

# Create a low temporal resolution time-combined dataset
# (We will use this for BAS)
python -m geowatch.cli.coco_time_combine \
    --kwcoco_fpath="$RAW_DSET_DPATH/${REGION_ID}/imgonly-${REGION_ID}-rawbands.kwcoco.zip" \
    --output_kwcoco_fpath="$TIMECOMBO_DSET_DPATH/${REGION_ID}/imgonly-${REGION_ID}-rawbands.kwcoco.zip" \
    --channels="red|green|blue|nir|swir16|swir22|pan|coastal|cirrus|B05|B06|B07|B8A|B09" \
    --resolution="10GSD" \
    --time_window=1y \
    --remove_seasons=winter \
    --merge_method=median \
    --spatial_tile_size=1024 \
    --mask_low_quality=True \
    --start_time=2010-03-01 \
    --assets_dname="raw_bands" \
    --workers=0

geowatch visualize "$RAW_DSET_DPATH/${REGION_ID}/imgonly-${REGION_ID}-rawbands.kwcoco.zip" --smart=1 \
    --channels="(L8,S2):(red|green|blue)"


inspect_stuff(){
    IMGONLY_COCO_FPATH="$RAW_DSET_DPATH/${REGION_ID}/imgonly-${REGION_ID}-rawbands.kwcoco.zip"
    geowatch stats "$IMGONLY_COCO_FPATH"
    geowatch visualize "$IMGONLY_COCO_FPATH" --smart

    geowatch model_stats "$BAS_MODEL_FPATH"
}


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DATASET_SUFFIX=KHQ_Tutorial6_Data
DEMO_DPATH=$DVC_DATA_DPATH/$DATASET_SUFFIX
IMGONLY_COCO_FPATH="$DEMO_DPATH/Aligned-$DATASET_SUFFIX/KHQ_R001/imgonly-KHQ_R001-rawbands.kwcoco.zip"
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
ACSC_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
PIPELINE_OUTPUT_DPATH="$DEMO_DPATH/outputs"
echo "BAS_MODEL_FPATH = $BAS_MODEL_FPATH"
echo "ACSC_MODEL_FPATH = $ACSC_MODEL_FPATH"
test -f "$BAS_MODEL_FPATH" || echo "missing BAS model"
test -f "$ACSC_MODEL_FPATH" || echo "missing ACSC model"


python -m geowatch.tasks.fusion.predict \
    --package_fpath="$BAS_MODEL_FPATH" \
    --test_dataset="$TIMECOMBO_DSET_DPATH/${REGION_ID}/imgonly-${REGION_ID}-rawbands.kwcoco.zip" \
    --pred_dataset="$PIPELINE_OUTPUT_DPATH"/bas_pxl/pred.kwcoco.zip \
    --chip_overlap=0.3 \
    --chip_dims "196,196" \
    --time_span=auto \
    --fixed_resolution=10GSD \
    --time_sampling=soft4 \
    --drop_unused_frames=True \
    --with_saliency=True \
    --with_class=False \
    --with_change=False  \
    --num_workers=2 \
    --devices=0, \
    --batch_size=1


python -m geowatch.cli.run_tracker \
    --input_kwcoco "$PIPELINE_OUTPUT_DPATH"/bas_pxl/pred.kwcoco.zip \
    --default_track_fn saliency_heatmaps \
    --track_kwargs '{
        "agg_fn": "probs",
        "thresh": 0.37,
        "time_thresh": 0.8,
        "inner_window_size": "1y",
        "inner_agg_fn": "max",
        "norm_ord": "inf",
        "moving_window_size": null,
        "poly_merge_method": "v2",
        "polygon_simplify_tolerance": 1,
        "min_area_square_meters": 7200,
        "max_area_square_meters": 8000000
    }' \
    --clear_annots=True \
    --out_site_summaries_fpath "$PIPELINE_OUTPUT_DPATH/bas_poly/site_summaries_manifest.json" \
    --out_site_summaries_dir "$PIPELINE_OUTPUT_DPATH/bas_poly/site_summaries" \
    --out_sites_fpath "$PIPELINE_OUTPUT_DPATH/bas_poly/sites_manifest.json" \
    --out_sites_dir "$PIPELINE_OUTPUT_DPATH/bas_poly/sites" \
    --out_kwcoco "$PIPELINE_OUTPUT_DPATH/bas_poly/poly.kwcoco.zip" \
    --viz_out_dir "$PIPELINE_OUTPUT_DPATH/bas_poly/_viz_tracker" \
    --site_summary=None


python -m geowatch.cli.cluster_sites \
    --context_factor=1.0 \
    --minimum_size=128x128@2GSD \
    --maximum_size=256x256@2GSD \
    --src="$PIPELINE_OUTPUT_DPATH/bas_poly/site_summaries_manifest.json" \
    --dst_dpath="$PIPELINE_OUTPUT_DPATH/ac_clusters" \
    --dst_region_fpath="$PIPELINE_OUTPUT_DPATH/ac_clusters/clustered.geojson" \
    --io_workers=4 \
    --draw_clusters=True \
    --crop_time=True


python -m geowatch.cli.coco_align \
    --src "$RAW_DSET_DPATH/${REGION_ID}/imgonly-${REGION_ID}-rawbands.kwcoco.zip" \
    --dst "$PIPELINE_OUTPUT_DPATH/ac_crops/sitecrop.kwcoco.zip" \
    --regions="$PIPELINE_OUTPUT_DPATH/ac_clusters/clustered.geojson" \
    --site_summary=True \
    --img_workers=16 \
    --aux_workers=2 \
    --verbose=1 \
    --debug_valid_regions=False \
    --visualize=False \
    --keep=img \
    --geo_preprop=auto  \
    --force_nodata=-9999 \
    --include_channels='red|green|blue|nir|quality' \
    --exclude_sensors=L8 \
    --minimum_size=128x128@2GSD \
    --convexify_regions=True \
    --target_gsd=2 \
    --context_factor=1.5 \
    --force_min_gsd=8 \
    --rpc_align_method=orthorectify


python -m geowatch.tasks.fusion.predict \
    --package_fpath="$ACSC_MODEL_FPATH" \
    --test_dataset="$PIPELINE_OUTPUT_DPATH/ac_crops/sitecrop.kwcoco.zip"  \
    --pred_dataset="$PIPELINE_OUTPUT_DPATH/ac_pxl/ac_heatmaps.kwcoco.zip"  \
    --tta_fliprot=0.0 \
    --tta_time=0.0 \
    --chip_overlap=0.3 \
    --fixed_resolution=8GSD \
    --output_space_scale=8GSD \
    --time_span=6m \
    --time_sampling=auto \
    --time_steps=12 \
    --chip_dims=auto \
    --set_cover_algo=None \
    --resample_invalid_frames=3 \
    --observable_threshold=0.0 \
    --mask_low_quality=True \
    --drop_unused_frames=True \
    --write_workers=0 \
    --with_saliency=True \
    --with_class=True \
    --with_change=False \
    --saliency_chan_code=ac_salient  \
    --num_workers=12 \
    --batch_size=1 \
    --devices=0,


python -m geowatch.cli.run_tracker \
    --input_kwcoco "$PIPELINE_OUTPUT_DPATH/ac_pxl/ac_heatmaps.kwcoco.zip" \
    --default_track_fn class_heatmaps \
    --track_kwargs '{"boundaries_as": "polys", "thresh": 0.07, "resolution": "8GSD", "min_area_square_meters": 7200}' \
    --clear_annots=True \
    --out_site_summaries_fpath "$PIPELINE_OUTPUT_DPATH/ac_poly/site_summaries_manifest.json" \
    --out_site_summaries_dir "$PIPELINE_OUTPUT_DPATH/ac_poly/site_summaries" \
    --out_sites_fpath "$PIPELINE_OUTPUT_DPATH/ac_poly/sites_manifest.json" \
    --out_sites_dir "$PIPELINE_OUTPUT_DPATH/ac_poly/sites" \
    --out_kwcoco "$PIPELINE_OUTPUT_DPATH/ac_poly/poly.kwcoco.zip" \
    --site_summary "$PIPELINE_OUTPUT_DPATH/bas_poly/site_summaries_manifest.json" \
    --boundary_region=None


# A basic pipeline can be run as a schedule evaluation pipeline.
python -m geowatch.mlops.schedule_evaluation --params="
    pipeline: full
    matrix:
        ######################
        ## BAS PIXEL PARAMS ##
        ######################

        bas_pxl.package_fpath:
            - $BAS_MODEL_FPATH
        bas_pxl.test_dataset:
            - $IMGONLY_COCO_FPATH
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - [196,196]
        bas_pxl.time_span: auto
        bas_pxl.fixed_resolution: 10GSD
        bas_pxl.time_sampling: soft4

        ######################
        ## BAS POLY PARAMS  ##
        ######################

        bas_poly.thresh: 0.425
        bas_poly.time_thresh: 0.8
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: max
        bas_poly.norm_ord: inf
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000

        ###########################
        ## BAS POLY EVAL PARAMS  ##
        ###########################

        bas_poly_eval.true_site_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/region_models

        ### SV
        sv_crop.enabled: 1
        sv_crop.minimum_size: '256x256@2GSD'
        sv_crop.num_start_frames: 3
        sv_crop.num_end_frames: 3
        sv_crop.context_factor: 1.6

        sv_dino_boxes.enabled: 1
        sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
        sv_dino_boxes.window_dims: 256
        sv_dino_boxes.window_overlap: 0.5
        sv_dino_boxes.fixed_resolution: 3GSD

        sv_dino_filter.enabled: 1
        sv_dino_filter.end_min_score:
            - 0.15
        sv_dino_filter.start_max_score: 1.0
        sv_dino_filter.box_score_threshold: 0.01
        sv_dino_filter.box_isect_threshold: 0.1

        sv_depth_score.enabled: 1
        sv_depth_score.model_fpath:
            - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
        sv_depth_filter.threshold:
            - 0.10

        ##########################
        ## Cluster Sites Params ##
        ##########################
        cluster_sites.context_factor: 1.5
        cluster_sites.minimum_size: '128x128@8GSD'
        cluster_sites.maximum_size: '1024x1024@8GSD'


        ########################
        ## SC CROPPING PARAMS ##
        ########################

        sc_crop.force_nodata: -9999
        sc_crop.include_channels: 'red|green|blue|nir|quality'
        sc_crop.exclude_sensors: 'L8'
        sc_crop.minimum_size: '128x128@8GSD'
        sc_crop.convexify_regions: True
        sc_crop.target_gsd: 2
        sc_crop.context_factor: 1.5
        sc_crop.force_min_gsd: 8
        sc_crop.img_workers: 16
        sc_crop.aux_workers: 2

        #####################
        ## SC PIXEL PARAMS ##
        #####################

        sc_pxl.package_fpath:
            - $ACSC_MODEL_FPATH
        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3
        sc_pxl.fixed_resolution: 8GSD
        sc_pxl.output_space_scale: 8GSD
        sc_pxl.time_span: 6m
        sc_pxl.time_sampling: auto
        sc_pxl.time_steps: 12
        sc_pxl.chip_dims: auto
        sc_pxl.set_cover_algo: null
        sc_pxl.resample_invalid_frames: 3
        sc_pxl.observable_threshold: 0.0
        sc_pxl.mask_low_quality: true
        sc_pxl.drop_unused_frames: true
        sc_pxl.num_workers: 12
        sc_pxl.batch_size: 1
        sc_pxl.write_workers: 0

        #####################
        ## SC POLY PARAMS  ##
        #####################

        sc_poly.thresh: 0.07
        sc_poly.boundaries_as: polys
        sc_poly.resolution: 8GSD
        sc_poly.min_area_square_meters: 7200

        ##########################
        ## SC POLY EVAL PARAMS  ##
        ##########################

        sc_poly_eval.true_site_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/site_models
        sc_poly_eval.true_region_dpath: $LORES_DVC_DATA_DPATH/annotations/drop6/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        bas_pxl.enabled: 1
        bas_poly.enabled: 1
        sc_crop.enabled: 1
        sc_pxl.enabled: 1
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 0
        bas_pxl_eval.enabled: 0
        bas_poly_eval.enabled: 0
        sc_pxl_eval.enabled: 0
        bas_poly_viz.enabled: 0
        sc_poly_viz.enabled: 0

    submatrices1:
        - bas_pxl.test_dataset: $IMGONLY_COCO_FPATH
          sc_crop.crop_src_fpath: $IMGONLY_COCO_FPATH
    " \
    --root_dpath="$DVC_EXPT_DPATH/_demo_khq" \
    --queue_name "_demo_khq" \
    --devices="0,1" \
    --backend=tmux --tmux_workers=6 \
    --cache=1 --skip_existing=0 --run=0

