#!/bin/bash
__doc__="
This process was done on horologic

Setting up the AWS bucket and DVC repo
"

source "$HOME"/code/watch-smartflow-dags/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
SENSORS="ta1-ls-ara-4,ta1-pd-ara-4,ta1-s2-ara-4,ta1-wv-ara-4"

DATASET_SUFFIX=Drop8-ARA

# Just KR1
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models/KR_R001.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models/KR_R001*.geojson"


# NOTE: Ensure the annotations/drop8.dvc data is pulled, otherwise there is an error.

# All Regions
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/region_models/*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop8/site_models/*.geojson"

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
    --align_workers=16 \
    --align_aux_workers=0 \
    --ignore_duplicates=1 \
    --visualize=0 \
    --target_gsd="10GSD" \
    --cache=0 \
    --verbose=100 \
    --skip_existing=0 \
    --force_min_gsd=2.0 \
    --force_nodata=-9999 \
    --align_tries=0 \
    --asset_timeout="10 minutes" \
    --image_timeout="30 minutes" \
    --hack_lazy=False \
    --backend=tmux \
    --tmux_workers=16
    #--sensor_to_time_window='
    #    S2: 2 weeks
    #    L8: 2 weeks
    #    PD: 2 weeks
    #' \
    #--run=1


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
cd "$DVC_DATA_DPATH/Aligned-Drop8-ARA"
python -c "
import ubelt as ub
root = ub.Path('.').absolute()
regions_dpaths_with_kwcoco = sorted({p.parent for p in root.glob('*/*.kwcoco.zip')})
to_add = []
for dpath in regions_dpaths_with_kwcoco:
    to_add.extend(list(dpath.glob('*.kwcoco.zip')))
    to_add.extend(list(dpath.glob('S2')))
    to_add.extend(list(dpath.glob('WV')))
    to_add.extend(list(dpath.glob('PD')))
    to_add.extend(list(dpath.glob('L8')))

import simple_dvc as sdvc
dvc_repo = sdvc.SimpleDVC.coerce(root)
dvc_repo.add(to_add, verbose=3)
"

git pull
git add -- */.gitignore
git commit -am "Add rest of drop8 regions"
git push

# Push kwcoco files first
dvc push -r aws -- */*.kwcoco.zip.dvc

# Push L8 next
dvc push -r aws -- */L8.dvc

# Push S2 next, then PD and WV
dvc push -r aws -- */S2.dvc && \
dvc push -r aws -- */PD.dvc && \
dvc push -r aws -- */WV.dvc