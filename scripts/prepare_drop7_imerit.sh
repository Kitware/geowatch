source "$HOME"/code/watch/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-3
DATASET_SUFFIX=Drop7
test -e "$DVC_DATA_DPATH/annotations/drop7/region_models"
echo $?
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop7/region_models/*_C*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop7/site_models/*_C*.geojson"

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


simple_dvc validate_sidecar "$DVC_DATA_DPATH/annotations/drop7"


    #--regions="
    #    - $DVC_DATA_DPATH/annotations/drop7/region_models/CN_C000.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/region_models/KW_C001.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/region_models/SA_C001.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/region_models/CO_C001.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/region_models/VN_C002.geojson
    #" \
    #--sites="
    #    - $DVC_DATA_DPATH/annotations/drop7/site_models/CN_C000_*.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/site_models/KW_C001_*.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/site_models/SA_C001_*.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/site_models/CO_C001_*.geojson
    #    - $DVC_DATA_DPATH/annotations/drop7/site_models/VN_C002_*.geojson
    #" \

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=20 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DVC_DATA_DPATH" \
    --regions="$REGION_GLOBSTR" \
    --sites="$SITE_GLOBSTR" \
    --aws_profile=iarpa \
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
    --cache=1 \
    --verbose=100 \
    --skip_existing=1 \
    --force_min_gsd=2.0 \
    --force_nodata=-9999 \
    --hack_lazy=False \
    --backend=tmux \
    --tmux_workers=8 \
    --run=1

dvc add CN_C000/L8 KW_C001/L8 SA_C001/L8 CO_C001/L8 VN_C002/L8 CN_C000/S2 KW_C001/S2 SA_C001/S2 CO_C001/S2 VN_C002/S2 KW_C001/WV

dvc push -vv -r aws CN_C000/*.dvc KW_C001/*.dvc SA_C001/*.dvc CO_C001/*.dvc VN_C002/*.dvc
dvc pull -vv -r namek_hdd CN_C000/*.dvc KW_C001/*.dvc SA_C001/*.dvc CO_C001/*.dvc VN_C002/*.dvc


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
python -c "if 1:

import ubelt as ub
bundle_dpath = ub.Path('.').absolute()
all_paths = list(bundle_dpath.glob('*_[C]*'))
region_dpaths = []
for p in all_paths:
    if p.is_dir() and str(p.name)[2] == '_':
        region_dpaths.append(p)

import cmd_queue
queue = cmd_queue.Queue.create('tmux', size=16)

for dpath in region_dpaths:
    print(ub.urepr(dpath.ls()))

for dpath in region_dpaths:
    region_id = dpath.name
    fnames = [
        f'imgonly-{region_id}.kwcoco.zip',
        f'imganns-{region_id}.kwcoco.zip',
    ]
    for fname in fnames:
        old_fpath = bundle_dpath / fname
        new_fpath = dpath / fname
        if old_fpath.exists() and not new_fpath.exists():
            queue.submit(f'kwcoco move {old_fpath} {new_fpath}')

queue.run()
"

dvc add CN_C000/*.kwcoco.zip KW_C001/*.kwcoco.zip SA_C001/*.kwcoco.zip  CO_C001/*.kwcoco.zip  VN_C002/*.kwcoco.zip
dvc push -r aws CN_C000/*.kwcoco.zip KW_C001/*.kwcoco.zip SA_C001/*.kwcoco.zip  CO_C001/*.kwcoco.zip  VN_C002/*.kwcoco.zip
dvc pull -r namek_hdd -- */*.kwcoco.zip.dvc

dvc pull -R namek_hdd -- CN_C000 KW_C001 SA_C001 CO_C001 VN_C002


#python ~/code/watch-smartflow-dags/reproduce_mlops.py imgonly-US_R006.kwcoco.zip
# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/watch/cli/queue_cli/prepare_time_combined_dataset.py \
    --regions="
      - CN_C000
      #- KW_C001
      #- SA_C001
      #- CO_C001
      #- VN_C002
    " \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/region_models \
    --spatial_tile_size=1024 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=4 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --skip_existing=0 \
    --cache=1 \
    --run=0



###---
#

geowatch site_stats --regions /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/region_models/CN_C000.geojson

python -m watch reproject \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-iMERIT/CN_C000/imgonly-CN_C000-fielded.kwcoco.zip \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-iMERIT/CN_C000/imganns-CN_C000.kwcoco.zip \
    --status_to_catname="positive_excluded: positive" \
    --regions="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/region_models/CN_C000.geojson" \
    --sites="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/site_models/CN_C000_*.geojson"
