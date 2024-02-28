#!/bin/bash

source "$HOME"/code/watch-smartflow-dags/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
#SENSORS=TA1-S2-L8-WV-PD-ACC-3

#'ta1-datacube-ara-4',
#'ta1-dsm-ara-4',

#'ta1-ls-ara-4',
#'ta1-pd-ara-4',
#'ta1-s2-ara-4',
#'ta1-wv-ara-4',

#'ta1-wv-ara-4t1',
#'ta1-wv-ara-4t2',
#'ta1-dsm-ara-4t1',
#'ta1-dsm-ara-4t2',


#SENSORS="ta1-ls-ara-4,ta1-pd-ara-4,ta1-s2-ara-4,ta1-wv-ara-4"
#SENSORS="sentinel-2-l2a,landsat-c2l2-sr,planet-dove,worldview-nitf"
#SENSORS="sentinel-2-l2a,landsat-c2l2-sr,planet-dove"
SENSORS="sentinel-2-l2a,landsat-c2l2-sr,planet-dove,worldview-nitf"

DATASET_SUFFIX=Drop8-L2

# Just KR1
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models/KR_R001.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models/KR_R001*.geojson"

# All Regions
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*_*0*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*_*0*_*.geojson"

# T&E Regions Only
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*_R0*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*_R0*_*.geojson"

#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*_T*.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*_T*_*.geojson"


export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export REQUESTER_PAYS=True
#export SMART_STAC_API_KEY=""

# Construct the TA2-ready dataset
python -m geowatch.cli.prepare_ta2_dataset \
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
    --requester_pays=$REQUESTER_PAYS \
    --fields_workers=8 \
    --convert_workers=0 \
    --align_workers=4 \
    --align_aux_workers=0 \
    --ignore_duplicates=1 \
    --visualize=0 \
    --target_gsd="10GSD" \
    --cache=0 \
    --verbose=100 \
    --skip_existing=0 \
    --force_min_gsd=2.0 \
    --force_nodata=-9999 \
    --align_tries=1 \
    --asset_timeout="10 minutes" \
    --image_timeout="30 minutes" \
    --hack_lazy=False \
    --backend=tmux \
    --tmux_workers=4 \
    --sensor_to_time_window='
        S2: 2 weeks
        L8: 2 weeks
        PD: 2 weeks
    ' \
    --run=1


#export AWS_REQUEST_PAYER=requester
#python -m geowatch.cli.coco_add_watch_fields \
#    --src=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Uncropped-Drop8-L2/data_KR_R001.kwcoco.zip \
#    --dst=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Uncropped-Drop8-L2/data_KR_R001_fielded.kwcoco.zip \
#    --overwrite=False \
#    --workers=8 \
#    --enable_video_stats=False \
#    --target_gsd=10GSD \
#    --remove_broken=True \
#    --skip_populate_errors=False

#AWS_DEFAULT_PROFILE=iarpa AWS_REQUEST_PAYER='requester' python -m geowatch.cli.coco_align \
#   --regions "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/region_models/KR_R001.geojson" \
#   --context_factor=1 \
#   --geo_preprop=auto \
#   --keep=img \
#   --force_nodata=-9999 \
#   --include_channels="None" \
#   --exclude_channels="None" \
#   --visualize=False \
#   --debug_valid_regions=False \
#   --rpc_align_method orthorectify \
#   --sensor_to_time_window  "
#        S2: 2 weeks
#        L8: 2 weeks
#        PD: 2 weeks " \
#   --verbose=100 \
#   --aux_workers=0 \
#   --target_gsd=10GSD \
#   --force_min_gsd=2.0 \
#   --workers=4 \
#   --tries=2 \
#   --hack_lazy=False \
#   --src=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Uncropped-Drop8-L2/data_KR_R001_fielded.kwcoco.zip \
#   --dst=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop8-L2/KR_R001/imgonly-KR_R001-rawbands.kwcoco.zip \
#   --dst_bundle_dpath=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Aligned-Drop8-L2


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
cd "$DVC_DATA_DPATH/Aligned-Drop8-L2"

git pull


# Add regions where kwcoco files exist
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
cd "$DVC_DATA_DPATH/Aligned-Drop8-L2"
python -c "
import ubelt as ub
root = ub.Path('.').absolute()
regions_dpaths_with_kwcoco = sorted({p.parent for p in root.glob('*/*.kwcoco.zip')})
to_add = []
for dpath in regions_dpaths_with_kwcoco:
    to_add.extend(list(dpath.glob('*.kwcoco.zip')))
    to_add.extend(list(dpath.glob('S2')))
    to_add.extend(list(dpath.glob('WV')))
    to_add.extend(list(dpath.glob('WV1')))
    to_add.extend(list(dpath.glob('PD')))
    to_add.extend(list(dpath.glob('L8')))
print(len(to_add))

import simple_dvc as sdvc
dvc_repo = sdvc.SimpleDVC.coerce(root)
dvc_repo.add(to_add, verbose=3)
"

git pull
git add -- */.gitignore
git commit -am "Add L2 version of Drop8"
git push

# Push kwcoco files first
# Push L8 next
# Push S2 next, then PD and WV
dvc push -r aws -- */*.kwcoco.zip.dvc && \
dvc push -r aws -- */L8.dvc && \
dvc push -r aws -- */S2.dvc && \
dvc push -r aws -- */PD.dvc && \
dvc push -r aws -- */WV.dvc



##########################
# Build Median BAS Dataset
##########################

# shellcheck disable=SC2155
export SRC_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=hdd)
# shellcheck disable=SC2155
export DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)

export SRC_BUNDLE_DPATH=$SRC_DVC_DATA_DPATH/Aligned-Drop8-L2
export DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop8-L2-Median10GSD-V1

export TRUTH_DPATH=$SRC_DVC_DATA_DPATH/annotations/drop8
export TRUTH_REGION_DPATH="$SRC_DVC_DATA_DPATH/annotations/drop8/region_models"

echo "
SRC_DVC_DATA_DPATH=$SRC_DVC_DATA_DPATH
DST_DVC_DATA_DPATH=$DST_DVC_DATA_DPATH

SRC_BUNDLE_DPATH=$SRC_BUNDLE_DPATH
DST_BUNDLE_DPATH=$DST_BUNDLE_DPATH

TRUTH_REGION_DPATH=$TRUTH_REGION_DPATH
"

# shellcheck disable=SC2155
export REGION_IDS_STR=$(python -c "if 1:
    import pathlib
    import os
    TRUTH_REGION_DPATH = os.environ.get('TRUTH_REGION_DPATH')
    SRC_BUNDLE_DPATH = os.environ.get('SRC_BUNDLE_DPATH')
    region_dpath = pathlib.Path(TRUTH_REGION_DPATH)
    src_bundle = pathlib.Path(SRC_BUNDLE_DPATH)
    region_fpaths = list(region_dpath.glob('*_[RC]*.geojson'))
    region_names = [p.stem for p in region_fpaths]
    final_names = []
    for region_name in region_names:
        coco_fpath = src_bundle / region_name / f'imgonly-{region_name}-rawbands.kwcoco.zip'
        if coco_fpath.exists():
            final_names.append(region_name)
    print(' '.join(sorted(final_names)))
    ")
#export REGION_IDS_STR="CN_C000 KW_C001 CO_C001"

# Dump regions to a file
# FIXME: tmp_region_names.yaml is not a robust interchange.
python -c "if 1:
    from kwutil.util_yaml import Yaml
    import os
    import ubelt as ub
    dpath = ub.Path(os.environ['DST_BUNDLE_DPATH']).ensuredir()
    tmp_fpath = dpath / 'tmp_region_names.yaml'
    REGION_IDS_STR = os.environ.get('REGION_IDS_STR')
    final_names = [p.strip() for p in REGION_IDS_STR.split(' ') if p.strip()]
    text = Yaml.dumps(final_names)
    print(text)
    tmp_fpath.write_text(text)
"
#REGION_IDS_STR="CN_C000 KW_C001 CO_C001 SA_C001 VN_C002"

echo "REGION_IDS_STR = $REGION_IDS_STR"
# shellcheck disable=SC2206
REGION_IDS_ARR=($REGION_IDS_STR)
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
done

# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
python -m geowatch.cli.queue_cli.prepare_time_combined_dataset \
    --regions="$DST_BUNDLE_DPATH/tmp_region_names.yaml" \
    --reproject=False \
    --input_bundle_dpath="$SRC_BUNDLE_DPATH" \
    --output_bundle_dpath="$DST_BUNDLE_DPATH" \
    --spatial_tile_size=1024 \
    --merge_method=median \
    --mask_low_quality=True \
    --tmux_workers=1 \
    --time_window=6months \
    --combine_workers=1 \
    --resolution=10GSD \
    --backend=tmux \
    --skip_existing=1 \
    --cache=1 \
    --run=1 --print-commands


python -m cmd_queue new "reproject_for_bas"
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    if test -f "$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip"; then
        python -m cmd_queue submit --jobname="reproject-$REGION_ID" -- reproject_for_bas \
            geowatch reproject_annotations \
                --src "$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip" \
                --dst "$DST_BUNDLE_DPATH/$REGION_ID/imganns-$REGION_ID-rawbands.kwcoco.zip" \
                --io_workers="avail/2" \
                --region_models="$TRUTH_DPATH/region_models/${REGION_ID}.geojson" \
                --site_models="$TRUTH_DPATH/site_models/${REGION_ID}_*.geojson"
    else
        echo "Missing imgonly kwcoco for $REGION_ID"
    fi
done
python -m cmd_queue show "reproject_for_bas"
python -m cmd_queue run --workers=8 "reproject_for_bas"


python -m geowatch.cli.prepare_splits \
    --src_kwcocos "$DST_BUNDLE_DPATH"/*/imganns*-rawbands.kwcoco.zip \
    --dst_dpath "$DST_BUNDLE_DPATH" \
    --suffix=rawbands \
    --backend=tmux --tmux_workers=6 \
    --splits split6 \
    --run=1


dvc add -v -- \
    */raw_bands \
    */imgonly-*-rawbands.kwcoco.zip \
    */imganns-*-rawbands.kwcoco.zip \
    data_train_rawbands_split6_*.kwcoco.zip \
    data_vali_rawbands_split6_*.kwcoco.zip

git commit -m "Update Drop8-L2-Median10GSD-V1 10mGSD BAS" && \
git push && \
dvc push -r aws -R . -vvv
