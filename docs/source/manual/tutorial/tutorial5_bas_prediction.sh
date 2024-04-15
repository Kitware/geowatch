#!/bin/bash
__doc__="
BAS Prediciton
==============

This tutorial outlines how to run BAS prediction on an arbitrary region.
"

# If you can access our DVC repo:
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

REGION_FPATH=$DVC_DATA_DPATH/annotations/drop6/region_models/KR_R001.geojson
SITE_GLOBSTR=None

BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt

BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt


# If you don't have access to our DVC repo (or you want a quicker test), you
# can generate a demo region by uncommenting the following code.
#xdoctest -m watch.demo.demo_region demo_khq_region_fpath
#REGION_FPATH="$HOME/.cache/watch/demo/annotations/KHQ_R001.geojson"
#SITE_GLOBSTR="$HOME/.cache/watch/demo/annotations/KHQ_R001_sites/*.geojson"

# The "name" of the new dataset
DATASET_SUFFIX=Tutorial5-Demo

# Set this to where you want to build the dataset
DEMO_DPATH=$DVC_DATA_DPATH/$DATASET_SUFFIX
mkdir -p "$DEMO_DPATH"

# This is a string code indicating what STAC endpoint we will pull from
# Some options depend on permissions or have other dependencies.
# For options, see:

# ~/code/watch/watch/stac/stac_search_builder.py
#SENSORS="sentinel-2-l2a"
SENSORS="sentinel-2-l1c"

echo "
DVC_DATA_DPATH=$DVC_DATA_DPATH
DVC_EXPT_DPATH=$DVC_EXPT_DPATH
REGION_FPATH=$REGION_FPATH
DEMO_DPATH=$DEMO_DPATH
"

# Requires:
# pip install planetary_computer
#SENSORS="planetarycomputer_s2"

# Depending on the STAC endpoint, some parameters may need to change:
# collated - True for IARPA endpoints, Usually False for public data
# requester_pays - True for public landsat
# api_key - A secret for non public data
export REQUESTER_PAYS=True
export SMART_STAC_API_KEY=""
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

#if [ -e "$HOME/code/watch/secrets" ]; then
#    source ~/code/watch/secrets/secrets
#fi

# Construct the TA2-ready dataset
# This is a cmd_queue CLI.  This will generate a DAG of bash commands and
# optionally execute them. Set --run=0 to do a dry run to see what will try to
# do before it does.
python -m geowatch.cli.queue_cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --cloud_cover=5 \
    --stac_query_mode=auto \
    --sensors "$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated False \
    --include_channels="red|green|blue|nir" \
    --requester_pays=$REQUESTER_PAYS \
    --dvc_dpath="$DEMO_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_FPATH" \
    --site_globstr="$SITE_GLOBSTR" \
    --fields_workers=8 \
    --convert_workers=8 \
    --align_workers=26 \
    --cache=0 \
    --force_nodata=-9999 \
    --ignore_duplicates=1 \
    --target_gsd=10 \
    --visualize=False \
    --max_products_per_region=10 \
    --backend=serial --run=1

# Note: Depending on the size of the region / number of available images /
# number of workers, this command can take a LOT of memory. This example is
# fairly tame, but if your system has low memory you may need to dial down some
# parameters.

BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt

PREDICTION_DPATH=$DEMO_DPATH/Aligned-$DATASET_SUFFIX

inspect_stuff(){
    geowatch stats "$DEMO_DPATH/Aligned-$DATASET_SUFFIX/data.kwcoco.json"

    geowatch visualize "$DEMO_DPATH/Aligned-$DATASET_SUFFIX/data.kwcoco.json" --smart

    #BAS_MODEL_FPATH=$DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt
    #cd "$DVC_EXPT_DPATH"
    #dvc pull -r aws models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt.dvc
    geowatch model_stats "$BAS_MODEL_FPATH"
}


python -m geowatch.tasks.fusion.predict \
    --test_dataset="$DEMO_DPATH/Aligned-$DATASET_SUFFIX/data.kwcoco.json" \
    --package_fpath="$BAS_MODEL_FPATH" \
    --pred_dataset="$PREDICTION_DPATH/data_heatmaps.kwcoco.json" \
    --quality_threshold=0 \
    --use_cloudmask=False \
    --fixed_resolution=10GSD \
    --devices="0,"

#    #--input_resolution=10GSD \
#    #--window_resolution=10GSD \
#    #--output_resolution=10GSD \
#    #--quality_threshold=0 \
#    #--use_cloudmask=False \
#    #--force_bad_frames=True

geowatch visualize "$PREDICTION_DPATH/data_heatmaps.kwcoco.json" \
    --channels="red|green|blue,salient" --smart


# Convert Heatmaps to Polygons
python -m geowatch.cli.run_tracker \
    --in_file "$PREDICTION_DPATH/data_heatmaps.kwcoco.json" \
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
    --boundary_region "$DVC_DATA_DPATH"/annotations/drop6/region_models \
    --out_site_summaries_fpath "$PREDICTION_DPATH/site_summaries_manifest.json" \
    --out_site_summaries_dir "$PREDICTION_DPATH/site_summaries" \
    --out_sites_fpath "$PREDICTION_DPATH/sites_manifest.json" \
    --out_sites_dir "$PREDICTION_DPATH/sites" \
    --out_kwcoco "$PREDICTION_DPATH/poly.kwcoco.zip"

geowatch visualize "$PREDICTION_DPATH/poly.kwcoco.zip" \
    --viz_dpath="$PREDICTION_DPATH"/_viz_polys \
    --draw_labels=False \
    --channels="red|green|blue,salient" --smart

#### Crop big images to the geojson regions
##AWS_DEFAULT_PROFILE=iarpa AWS_REQUEST_PAYER='requester' python -m geowatch.cli.coco_align \
##    --src "$HOME/remote/Ooo/data/dvc-repos/smart_data_dvc-ssd/Tutorial5-Demo/Uncropped-Tutorial5-Demo/data_KR_R001_fielded.kwcoco.json" \
##    --dst "$HOME/remote/Ooo/data/dvc-repos/smart_data_dvc-ssd/Tutorial5-Demo/Aligned-Tutorial5-Demo/imgonly-KR_R001.kwcoco.json" \
##    --regions "$HOME/remote/Ooo/data/dvc-repos/smart_data_dvc-ssd/annotations/drop6/region_models/KR_R001.geojson" \
##    --context_factor=1 \
##    --geo_preprop=auto \
##    --keep=img \
##    --force_nodata=None \
##    --include_channels="red|green|blue|nir" \
##    --exclude_channels="None" \
##    --visualize=False \
##    --debug_valid_regions=False \
##    --rpc_align_method orthorectify \
##    --verbose=0 \
##    --aux_workers=0 \
##    --target_gsd=10 \
##    --force_min_gsd=None \
##    --workers=26 \
##    --hack_lazy="dry"

###gdalinfo /vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-2-l2a/23/K/PQ/2019/6/S2B_23KPQ_20190623_0_L2A/B09.tif
###gdalinfo /vsicurl/https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-2-l2a/52/S/DG/2021/7/S2A_52SDG_20210731_0_L2A/B08.tif


###gdalwarp -overwrite -multi --debug off -t_srs epsg:32652 -of COG \
###    -te 128.649453 37.64368 128.734073 37.683356 \
###    -te_srs epsg:4326 -wm 1500 \
###    -co OVERVIEWS=AUTO -co BLOCKSIZE=256 -co COMPRESS=DEFLATE -co NUM_THREADS=2 \
###    --config GDAL_CACHEMAX 1500 \
###    /vsicurl/https:/sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-2-l2a/52/S/DG/2021/7/S2A_52SDG_20210731_0_L2A/B08.tif \
###    /home/joncrall/remote/Ooo/data/dvc-repos/smart_data_dvc-ssd/Tutorial5-Demo/Aligned-Tutorial5-Demo/KR_R001/S2/affine_warp/crop_20210731T020000Z_N37.643680E128.649453_N37.683356E128.734073_S2_0/.tmpwarp.crop_20210731T020000Z_N37.643680E128.649453_N37.683356E128.734073_S2_0_nir.tif



__notes__="

Downloading The Model (and Prebuilt Demo Dataset)

The following is instructions for downloading the model and optionally the
pre-constructed kwcoco dataset (which is an alternative from constructing it
from STAC yourself via prepare_ta2_dataset).

We host it on two locations: data.kitware.com and via IPFS.


Download from data.kitware.com (via Girder)
-------------------------------------------

Both data and the model are available at this URL:
https://data.kitware.com/#user/598a19658d777f7d33e9c18b/folder/645a7cde58fefcae81a9f72b


Programatically:


.. code:: bash

    pip install girder-client
    girder-client --api-url https://data.kitware.com/api/v1 download --parent-type folder 645a7cde58fefcae81a9f72b tutorial5-data
    unzip tutorial5-data/Aligned-Tutorial5-Demo.zip -d tutorial5-data

    # The demo dataset
    ls tutorial5-data/Aligned-Tutorial5-Demo

    # The model
    ls tutorial5-data/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt


Download from IPFS
------------------

# Both data and model are available at this URL
https://ipfs.io/ipfs/bafybeiflbygurj5efyf5j7yevcw6cw3ebvgrc3hiau7e7rncwb4u562vca

Relevant IPFS CIDS:

* bafybeiflbygurj5efyf5j7yevcw6cw3ebvgrc3hiau7e7rncwb4u562vca - tutorial-5-data
* bafybeicsgpwom66kyp7nhuqy7zv4aaxo3x3ilfptuckh3ahi7xr6nal7jy - tutorial-5-data/Aligned-Tutorial5-Demo/
* bafybeiegbappwipoelo532fh3wtrircuhmk4poyuqjzudulme742xnc5hi - Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
* bafybeiamg3pqo2q3xefxlvstqeixxvf4wiazpjqleznwhgmwhhdfthebxq - Aligned-Tutorial5-Demo.zip  # The data but as an zipped archive


.. code:: bash

    # The model only:
    curl -LJO https://ipfs.io/ipfs/bafybeicydgguvhts4ejcnyausvw6sff453htpxp5tktcigdyy6hdkuovgy/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt

    # The pre-built kwcoco dataset (downloading a directory seems to require wget)
    wget -r https://ipfs.io/ipfs/bafybeiflbygurj5efyf5j7yevcw6cw3ebvgrc3hiau7e7rncwb4u562vca/Aligned-Tutorial5-Demo


.. code:: bash

    # OR if you have ipfs / kubo installed
    ipfs get --progress bafybeiflbygurj5efyf5j7yevcw6cw3ebvgrc3hiau7e7rncwb4u562vca

    # OR use the standalone ipget binary
    # https://github.com/ipfs/ipget
    # https://dist.ipfs.tech/#ipget
    ipget --progress bafybeiflbygurj5efyf5j7yevcw6cw3ebvgrc3hiau7e7rncwb4u562vca

"
