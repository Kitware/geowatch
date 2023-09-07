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


#simple_dvc validate_sidecar "$DVC_DATA_DPATH/annotations/drop7"


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
    #
    #
rm -rf 	Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/ Drop6-MeanYear10GSD/ Drop6_Mean3Month10GSD/ Drop6_MeanYear/ Drop7-Cropped2GSD-V2/
    #

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


# Reorganize kwcoco files to be inside of their region folder
python -c "if 1:

import os
os.chdir('/home/joncrall/remote/namek/data/dvc-repos/smart_data_dvc/Aligned-Drop7')

import ubelt as ub
bundle_dpath = ub.Path('.').absolute()
all_paths = list(bundle_dpath.glob('*_[C]*'))
region_dpaths = []
for p in all_paths:
    if p.is_dir() and str(p.name)[2] == '_':
        region_dpaths.append(p)

region_dpaths

import cmd_queue
queue = cmd_queue.Queue.create('tmux', size=16)

for dpath in region_dpaths:
    print(ub.urepr(dpath.ls()))

existing = []
for dpath in region_dpaths:
    region_id = dpath.name
    fnames = [
        f'imgonly-{region_id}.kwcoco.zip',
        f'imganns-{region_id}.kwcoco.zip',
    ]
    for fname in fnames:
        old_fpath = bundle_dpath / fname
        new_fpath = dpath / fname
        if old_fpath.exists():
            if new_fpath.exists():
                existing.append(new_fpath)
            if not new_fpath.exists():
                queue.submit(f'kwcoco move {old_fpath} {new_fpath}')

queue.write_network_text()
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
    --regions="['US_C010', 'NG_C000', 'KW_C001', 'SA_C005', 'US_C001', 'CO_C009',
        'US_C014', 'CN_C000', 'PE_C004', 'IN_C000', 'PE_C001', 'SA_C001', 'US_C016',
        'AE_C001', 'US_C011', 'AE_C002', 'PE_C003', 'RU_C000', 'CO_C001', 'US_C000',
        'US_C012', 'AE_C003', 'CN_C001', 'QA_C001', 'SN_C000']
        # 'VN_C002',
    " \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop7/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop7/region_models \
    --spatial_tile_size=512 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=2 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --skip_existing=0 \
    --cache=1 \
    --run=1


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m watch.cli.prepare_splits \
    --base_fpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/*/imganns*-*_[RC]*.kwcoco.zip \
    --suffix=rawbands \
    --backend=tmux --tmux_workers=6 \
    --run=0


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
kwcoco move \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/US_C010/data_train_rawbands_split6.kwcoco.zip \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/data_train_rawbands_split6.kwcoco.zip

kwcoco move \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/US_C010/data_vali_rawbands_split6.kwcoco.zip \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/data_vali_rawbands_split6.kwcoco.zip


python -m watch reproject \
    --src "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/.kwcoco.zip \
    --dst "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/.kwcoco.zip \
    --status_to_catname="positive_excluded: positive" \
    --regions="$DVC_DATA_DPATH"/drop7/region_models/*.geojson \
    --sites="$DVC_DATA_DPATH"/annotations/drop7/site_models/*.geojson


dvc add -vv -- */raw_bands
dvc add -vv -- */imganns-*.kwcoco.zip


###---
#

geowatch site_stats --regions /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/region_models/CN_C000.geojson

python -m watch reproject \
    --src /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-iMERIT/CN_C000/imgonly-CN_C000-fielded.kwcoco.zip \
    --dst /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-iMERIT/CN_C000/imganns-CN_C000.kwcoco.zip \
    --status_to_catname="positive_excluded: positive" \
    --regions="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/region_models/CN_C000.geojson" \
    --sites="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/annotations/drop6_hard_v1/site_models/CN_C000_*.geojson"


# Run BAS on iMERIT


# PREEVAL 14 BAS+SV Grid
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.utils.simple_dvc request \
    "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt \
    "$DVC_EXPT_DPATH"/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt \
    "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt \
    "$DVC_EXPT_DPATH"/models/fusion/uconn/D7-MNW10_coldL8S2-cv-a0-a1-b1-c1-rmse-split6_eval11_Norm_lr1e4_bs48_focal/epoch=16-step=374.pt \
    "$DVC_EXPT_DPATH"/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
TEST_DPATH=$DVC_EXPT_DPATH/_test/_imeritbas
geowatch schedule --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-iMERIT/CN_C000/imganns-CN_C000.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.2
            - 0.25
            - 0.3
            - 0.325
            - 0.35
            - 0.375
            - 0.39
            - 0.4
            - 0.4125
            - 0.425
            - 0.435
            - 0.45
            - 0.5
            - 0.6
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.95
            - 0.925
            - 0.9
            - 0.85
            - 0.825
            - 0.8
            - 0.775
            - 0.75
            - 0.7
            - 0.6
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$TEST_DPATH" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux \
    --pipeline=bas \
    --skip_existing=1 \
    --run=1 --help

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
TEST_DPATH=$DVC_EXPT_DPATH/_test/_imeritbas
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $TEST_DPATH
    " \
    --output_dpath="$TEST_DPATH/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        - bas_pxl_eval
    " \
    --plot_params="
        enabled: 1
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - resolved_params.bas_poly.time_thresh
            - resolved_params.bas_poly.thresh
            - resolved_params.bas_pxl.package_fpath
    " \
    --stdout_report="
        top_k: 13
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 1
        show_csv: 0
    "


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python -m watch.mlops.confusor_analysis \
    --metrics_node_dpath /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_test/_imeritbas/eval/flat/bas_poly_eval/bas_poly_eval_id_fd88699a/ \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop7/region_models \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop7/site_models \
    --viz_site_case=True



# Compute all feature except COLD
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="ssd")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-iMERIT
python -m watch.cli.prepare_teamfeats \
    --src_kwcocos "$BUNDLE_DPATH"/*/imganns-*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=1 \
    --with_invariants2=1 \
    --with_sam=0 \
    --with_materials=0 \
    --with_mae=1 \
    --with_depth=0 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, --tmux_workers=8 --backend=tmux --run=1

dvc add -- */teamfeats
