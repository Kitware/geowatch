#!/bin/bash
__doc__="
Notes:
    ~/code/watch/dev/poc/remove_empty_kwcoco_files.py
"
source "$HOME"/code/watch/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=ta1-10m-tsmoothed-acc-3
DATASET_SUFFIX=Drop7-Smooth-V2
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Construct the TA2-ready dataset
python -m geowatch.cli.queue_cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=20 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DVC_DATA_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --requester_pays=False \
    --fields_workers=8 \
    --convert_workers=0 \
    --align_workers=4 \
    --align_aux_workers=0 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --visualize=0 \
    --target_gsd=10 \
    --cache=0 \
    --verbose=100 \
    --skip_existing=0 \
    --force_min_gsd=2.0 \
    --force_nodata=-9999 \
    --hack_lazy=False \
    --backend=tmux \
    --tmux_workers=8 \
    --run=1


# COLD FEATURES on Raw Data
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Aligned-Drop7-Smooth-V2
python -m geowatch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/*/imgann-*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_invariants2=0 \
    --with_sam=0 \
    --with_materials=0 \
    --with_depth=0 \
    --with_cold=1 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, \
    --cold_workermode=process \
    --cold_workers=8 \
    --tmux_workers=16 \
    --backend=tmux --run=0
    #--base_fpath "$BUNDLE_DPATH"/*/imganns-*[0-9].kwcoco.zip \


##########
# 3-month
##########

#python ~/code/watch-smartflow-dags/reproduce_mlops.py imgonly-US_R006.kwcoco.zip
# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/watch/cli/queue_cli/prepare_time_combined_dataset.py \
    --regions=all \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7-Smooth-V2 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-SmoothMedianNoWinter3Month10GSD \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/region_models \
    --spatial_tile_size=1024 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=4 \
    --time_window=3m \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --run=1

# Compute all non-COLD features
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-SmoothMedianNoWinter3Month10GSD
python -m geowatch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/imganns-*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=1 \
    --with_invariants2=1 \
    --with_sam=1 \
    --with_materials=1 \
    --with_mae=1 \
    --with_depth=0 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, \
    --tmux_workers=1 --backend=tmux \
    --run=1


##########
# 1-month
##########

#python ~/code/watch-smartflow-dags/reproduce_mlops.py imgonly-US_R006.kwcoco.zip
# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/watch/cli/queue_cli/prepare_time_combined_dataset.py \
    --regions=all \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7-Smooth-V2 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-SmoothMedianNoWinterMonthly10GSD \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/region_models \
    --spatial_tile_size=1024 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=4 \
    --time_window=1m \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --run=1


# Compute all non-COLD features
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-SmoothMedianNoWinterMonthly10GSD
python -m geowatch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/imganns-*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=1 \
    --with_invariants2=1 \
    --with_sam=1 \
    --with_materials=1 \
    --with_mae=1 \
    --with_depth=0 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, \
    --tmux_workers=1 --backend=tmux \
    --run=0
