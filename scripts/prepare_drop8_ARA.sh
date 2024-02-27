#!/bin/bash
__doc__="
This process was done on horologic

Setting up the AWS bucket and DVC repo
"

source "$HOME"/code/watch-smartflow-dags/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
#SENSORS="ta1-ls-ara-4,ta1-pd-ara-4,ta1-s2-ara-4,ta1-wv-ara-4"
SENSORS="ta1-ls-ara-4,ta1-pd-ara-4,ta1-s2-ara-4"

DATASET_SUFFIX=Drop8-ARA

# Just KR1
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models/KR_R001.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models/KR_R001*.geojson"


# NOTE: Ensure the annotations/drop8.dvc data is pulled, otherwise there is an error.

# All Regions
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*.geojson"

#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/CO_C009.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/CO_C009_*.geojson"

REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/KW_C001.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/KW_C001_*.geojson"


# iMerit Regions Only
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*_C*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*_C*_*.geojson"

#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/KR_T001.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/KR_T001_*.geojson"
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*_T0*.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*_T0*_*.geojson"

# T&E Regions Only
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*_R*.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*_R*_*.geojson"


export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export REQUESTER_PAYS=False
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
        #WV: 2 weeks
    ' \
    --run=1


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
cd "$DVC_DATA_DPATH/Aligned-Drop8-ARA"

git pull

# Add a few files from KR_R001 to start with so people have data
dvc add -vvv -- \
    KR_R001/L8 \
    KR_R001/S2 \
    KR_R001/WV \
    KR_R001/imganns-*-rawbands.kwcoco.zip \
    KR_R001/imgonly-*-rawbands.kwcoco.zip

git commit -am "Add KR_R001"
git push
dvc push -r aws -R KR_R001 -vvv


# Add more select regions
dvc add -vvv -- \
    BR_R002/L8 \
    BR_R002/S2 \
    BR_R002/WV \
    BR_R002/PD \
    BR_R002/*.kwcoco.zip \
    HK_T003/L8 \
    HK_T003/S2 \
    HK_T003/WV \
    HK_T003/PD \
    HK_T003/*.kwcoco.zip

git commit -am "Add BR_R002 and HK_T003"
git push
dvc push -r aws -R . -vvv

# Add regions where kwcoco files exist
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"
# shellcheck disable=SC2164
cd "$DVC_DATA_DPATH/Aligned-Drop8-ARA"
python -c "
import ubelt as ub
root = ub.Path('.').absolute()
regions_dpaths_with_kwcoco = sorted({p.parent for p in root.glob('*/*.kwcoco.zip')})
to_add = []
for dpath in regions_dpaths_with_kwcoco:
    to_add.extend(list(dpath.glob('*.kwcoco.zip')))
    # to_add.extend(list(dpath.glob('S2')))
    # to_add.extend(list(dpath.glob('WV')))
    # to_add.extend(list(dpath.glob('PD')))
    # to_add.extend(list(dpath.glob('L8')))

import simple_dvc as sdvc
dvc_repo = sdvc.SimpleDVC.coerce(root)
dvc_repo.add(to_add, verbose=3)
"


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
# shellcheck disable=SC2164
cd "$DVC_DATA_DPATH/Aligned-Drop8-ARA"
git pull
git add -- */.gitignore
git commit -am "Add rest of drop8 regions"
git push


# Push kwcoco files first
dvc push -vvv -r aws -- */*.kwcoco.zip.dvc

# Push sensor data in a given order
dvc push -vvv -r aws -- */PD.dvc && \
dvc push -vvv -r aws -- */L8.dvc && \
dvc push -vvv -r aws -- */S2.dvc && \
dvc push -vvv -r aws -- */WV.dvc

#
# -- to pull
#

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
# shellcheck disable=SC2164
cd "$DVC_DATA_DPATH/Aligned-Drop8-ARA"
dvc pull -vvv -r aws -- */*.kwcoco.zip.dvc

# Pull data from specific sensors
dvc pull -vvv -r aws -- */PD.dvc
dvc pull -vvv -r aws -- */L8.dvc
dvc pull -vvv -r aws -- */S2.dvc
dvc pull -vvv -r aws -- */WV.dvc



##########################
# Build Cropped SC Dataset
##########################

# shellcheck disable=SC2155
export SRC_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=hdd)
# shellcheck disable=SC2155
export DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)

export SRC_BUNDLE_DPATH=$SRC_DVC_DATA_DPATH/Aligned-Drop8-ARA
export DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop8-Cropped2GSD-V1

export TRUTH_DPATH=$SRC_DVC_DATA_DPATH/annotations/drop8
export TRUTH_REGION_DPATH="$SRC_DVC_DATA_DPATH/annotations/drop8/region_models"

echo "
SRC_DVC_DATA_DPATH=$SRC_DVC_DATA_DPATH
DST_DVC_DATA_DPATH=$DST_DVC_DATA_DPATH

SRC_BUNDLE_DPATH=$SRC_BUNDLE_DPATH
DST_BUNDLE_DPATH=$DST_BUNDLE_DPATH

TRUTH_REGION_DPATH=$TRUTH_REGION_DPATH
"

REGION_IDS_STR=$(python -c "if 1:
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
#REGION_IDS_STR="CN_C000 KW_C001 SA_C001 CO_C001 VN_C002"

echo "REGION_IDS_STR = $REGION_IDS_STR"
# shellcheck disable=SC2206
REGION_IDS_ARR=($REGION_IDS_STR)
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
done


### Cluster and Crop Jobs
python -m cmd_queue new "crop_for_sc_queue"
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    REGION_GEOJSON_FPATH=$TRUTH_REGION_DPATH/$REGION_ID.geojson
    REGION_CLUSTER_DPATH=$DST_BUNDLE_DPATH/$REGION_ID/clusters
    SRC_KWCOCO_FPATH=$SRC_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip

    CRP_KWCOCO_FPATH=$DST_BUNDLE_DPATH/$REGION_ID/_cropped_imgonly-$REGION_ID-rawbands.kwcoco.zip
    DST_KWCOCO_FPATH=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
    if ! test -f "$DST_KWCOCO_FPATH"; then

        if ! test -d "$REGION_CLUSTER_DPATH/_viz_clusters/"; then
            # TODO: need a ".done" file instead of using _viz_clusters as the check
            CLUSTER_JOBNAME="cluster-$REGION_ID"
            cmd_queue submit --jobname="cluster-$REGION_ID" --depends="None" -- crop_for_sc_queue \
                python -m geowatch.cli.cluster_sites \
                    --src "$REGION_GEOJSON_FPATH" \
                    --minimum_size "256x256@2GSD" \
                    --dst_dpath "$REGION_CLUSTER_DPATH" \
                    --draw_clusters True
        else
            CLUSTER_JOBNAME="None"
        fi

        if ! test -f "$CRP_KWCOCO_FPATH"; then
            # TODO: should coco-align should have the option to remove nan images?
            CROP_JOBNAME="crop-$REGION_ID"
            python -m cmd_queue submit --jobname="crop-$REGION_ID" --depends="$CLUSTER_JOBNAME" -- crop_for_sc_queue \
                python -m geowatch.cli.coco_align \
                    --src "$SRC_KWCOCO_FPATH" \
                    --dst "$CRP_KWCOCO_FPATH" \
                    --regions "$REGION_CLUSTER_DPATH/*.geojson" \
                    --rpc_align_method orthorectify \
                    --workers=10 \
                    --aux_workers=2 \
                    --force_nodata=-9999 \
                    --context_factor=1.0 \
                    --minimum_size="256x256@2GSD" \
                    --force_min_gsd=2.0 \
                    --convexify_regions=True \
                    --target_gsd=2.0 \
                    --geo_preprop=False \
                    --sensor_to_time_window "
                        S2: 1month
                        PD: 1month
                        L8: 1month
                    " \
                    --keep img
                    #--exclude_sensors=L8 \
        else
            CROP_JOBNAME="None"
        fi

        if ! test -f "$DST_KWCOCO_FPATH"; then
            # Cleanup the data, remove bad images that are nearly all nan.
            python -m cmd_queue submit --jobname="removebad-$REGION_ID" --depends="$CROP_JOBNAME" -- crop_for_sc_queue \
                geowatch remove_bad_images \
                    --src "$CRP_KWCOCO_FPATH" \
                    --dst "$DST_KWCOCO_FPATH" \
                    --delete_assets False \
                    --interactive False \
                    --channels "red|green|blue|pan" \
                    --workers "0" \
                    --overview 0
                    #--workers "avail/2" \
        fi
    fi
done
python -m cmd_queue show "crop_for_sc_queue"
python -m cmd_queue run --workers=8 "crop_for_sc_queue"

python ~/code/watch/dev/poc/find_and_remove_unregistered_images.py


## Hack fixup
#python -m cmd_queue new "crop_for_sc_queue"
## sdvc unprotect -- */*.kwcoco*.zip
#for REGION_ID in "${REGION_IDS_ARR[@]}"; do
#    REGION_GEOJSON_FPATH=$TRUTH_REGION_DPATH/$REGION_ID.geojson
#    REGION_CLUSTER_DPATH=$DST_BUNDLE_DPATH/$REGION_ID/clusters
#    SRC_KWCOCO_FPATH=$SRC_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip

#    CRP_KWCOCO_FPATH=$DST_BUNDLE_DPATH/$REGION_ID/_cropped_imgonly-$REGION_ID-rawbands.kwcoco.zip
#    DST_KWCOCO_FPATH=$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip
#    if test -f "$DST_KWCOCO_FPATH"; then
#        mv "$DST_KWCOCO_FPATH" "$CRP_KWCOCO_FPATH"
#        #echo "DST_KWCOCO_FPATH = $DST_KWCOCO_FPATH"
#    fi
#done


### Reproject Annotation Jobs
python -m cmd_queue new "reproject_for_sc"
# shellcheck disable=SC3054
for REGION_ID in "${REGION_IDS_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
    if ! test -f "$DST_BUNDLE_DPATH/$REGION_ID/imganns-$REGION_ID-rawbands.kwcoco.zip"; then
        if test -f "$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip"; then
            python -m cmd_queue submit --jobname="reproject-$REGION_ID" -- reproject_for_sc \
                geowatch reproject_annotations \
                    --src "$DST_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip" \
                    --dst "$DST_BUNDLE_DPATH/$REGION_ID/imganns-$REGION_ID-rawbands.kwcoco.zip" \
                    --io_workers="avail/2" \
                    --region_models="$TRUTH_DPATH/region_models/${REGION_ID}.geojson" \
                    --site_models="$TRUTH_DPATH/site_models/${REGION_ID}_*.geojson"
        fi
    fi
done
python -m cmd_queue show "reproject_for_sc"
python -m cmd_queue run --workers=2 "reproject_for_sc"


python -m geowatch.cli.prepare_splits \
    --src_kwcocos "$DST_BUNDLE_DPATH"/*/imganns*-rawbands.kwcoco.zip \
    --dst_dpath "$DST_BUNDLE_DPATH" \
    --suffix=rawbands \
    --backend=tmux --tmux_workers=2 \
    --splits split6 \
    --run=1





#dvc add -vvv -- */clusters
#dvc add -vvv -- \
#    */*/L8 \
#    */*/S2 \
#    */*/WV \
#    */*/PD
#dvc add -vvv -- \
#    */imgonly-*-rawbands.kwcoco.zip \
#    */imganns-*-rawbands.kwcoco.zip

dvc add -vvv -- \
    *_rawbands_*.kwcoco.zip \
    */imgonly-*-rawbands.kwcoco.zip \
    */imganns-*-rawbands.kwcoco.zip \
    */*/L8 \
    */*/S2 \
    */*/WV \
    */*/PD && \
git commit -m "Update Drop8 Crop SC" && \
git push && \
dvc push -r aws -R . -vvv


##########################
# Build Median BAS Dataset
##########################

# shellcheck disable=SC2155
export SRC_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=hdd)
# shellcheck disable=SC2155
export DST_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)

export SRC_BUNDLE_DPATH=$SRC_DVC_DATA_DPATH/Aligned-Drop8-ARA
export DST_BUNDLE_DPATH=$DST_DVC_DATA_DPATH/Drop8-Median10GSD-V1

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
    --combine_workers=4 \
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

git commit -m "Update Drop8 Median 10mGSD BAS" && \
git push && \
dvc push -r aws -R . -vvv
