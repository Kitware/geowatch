source "$HOME"/code/watch/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-3
DATASET_SUFFIX=Drop7
test -e "$DVC_DATA_DPATH/annotations/drop7/region_models"
echo $?
#REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop7/region_models/*_C*.geojson"
#SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop7/site_models/*_C*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

simple_dvc request "$DVC_DATA_DPATH/annotations/drop7" --verbose


# Train Regions
# US_C001, NG_C001, QA_C001, RU_C001, CN_C001, PE_C004, IN_C000, IN_C000,
# SN_C000, CO_C009, US_C016

# Valiation Regions
# CN_C000, KW_C001, SA_C001, CO_C001, VN_C002
#
# # WOrldview-only
# MY_C000, BR_C010, BO_C001, PH_C001



# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=20 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DVC_DATA_DPATH" \
    --aws_profile=iarpa \
    --regions="
        - $DVC_DATA_DPATH/annotations/drop7/region_models/CN_C000.geojson
        - $DVC_DATA_DPATH/annotations/drop7/region_models/KW_C001.geojson
        - $DVC_DATA_DPATH/annotations/drop7/region_models/SA_C001.geojson
        - $DVC_DATA_DPATH/annotations/drop7/region_models/CO_C001.geojson
        - $DVC_DATA_DPATH/annotations/drop7/region_models/VN_C002.geojson
    " \
    --sites="
        - $DVC_DATA_DPATH/annotations/drop7/site_models/CN_C000_*.geojson
        - $DVC_DATA_DPATH/annotations/drop7/site_models/KW_C001_*.geojson
        - $DVC_DATA_DPATH/annotations/drop7/site_models/SA_C001_*.geojson
        - $DVC_DATA_DPATH/annotations/drop7/site_models/CO_C001_*.geojson
        - $DVC_DATA_DPATH/annotations/drop7/site_models/VN_C002_*.geojson
    " \
    --requester_pays=False \
    --fields_workers=8 \
    --convert_workers=0 \
    --align_workers=8 \
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


python -c "if 1:

    import ubelt as ub
    root = ub.Path('.')
    region_dpaths = [p for p in root.ls() if p.is_dir()]
    region_dpaths = [p for p in region_dpaths if '_C' in p.name]

    rois = 'CN_C000,KW_C001,SA_C001,CO_C001,VN_C002'.split(',')
    region_dpaths = [root / n for n in rois]

    for dpath in region_dpaths:
        import xdev
        xdev.tree_repr(dpath, max_depth=1)
        print(dpath.ls())

"
