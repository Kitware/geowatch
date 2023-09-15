__doc__="
SeeAlso:
    ~/code/watch/dev/poc/prepare_time_combined_dataset.py
"


# Demo with slurm
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R002.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 0
        bas_poly_eval.enabled: 0
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/slurm_demo" \
    --backend=slurm --queue_name "_slurm_demo" \
    --pipeline=bas --skip_existing=1 \
    --devices="0,1" \
    --slurm_options '
    account: public-default
    partition: general-gpu
    ntasks: 1
    cpus_per_task: 4
    gres: "gpu:1"
    ' \
    --print-commands \
    --run=0


#### Eval9 Models (Namek)

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python -m watch.cli.split_videos \
    --src "$DVC_DATA_DPATH/Drop4-BAS/data_train.kwcoco.json" \
          "$DVC_DATA_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
    --io_workers=4 \
    --dst "$DVC_DATA_DPATH/Drop4-BAS/{src_name}_{video_name}.kwcoco.zip"


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.cli.prepare_teamfeats \
    --base_fpath \
       "$DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001.kwcoco.zip" \
       "$DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002.kwcoco.zip" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=0 \
    --with_invariants2=1 \
    --invariant_resolution=30GSD \
    --kwcoco_ext=".kwcoco.zip" \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=0 \
    --gres='0,' --workers=2 --backend=tmux --run=1


# /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V12/lightning_logs/version_4/packages/package_epoch160_step163840.pt



DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
echo "
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step108.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch102_step52736.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch104_step53760.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch138_step71168.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch140_step72192.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch143_step73728.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch149_step76800.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch159_step81920.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch78_step40448.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch98_step50688.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step4305.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step1024.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step8247.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch2_step1536.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch3_step2048.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch4_step2560.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch5_step3072.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch6_step3584.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch7_step3908.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step302.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step38851.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch1_step77702.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch2_step98789.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch3_step155404.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch4_step194255.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch5_step233106.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch6_step252174.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v1_epoch2_step116553.pt
" > "$DVC_EXPT_DPATH"/namek_model_candidates_2023-02.yml



DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step108.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch102_step52736.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch104_step53760.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch138_step71168.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch140_step72192.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch143_step73728.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch149_step76800.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch159_step81920.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch78_step40448.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch98_step50688.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step4305.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step1024.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step8247.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch2_step1536.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch3_step2048.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch5_step3072.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch6_step3584.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch7_step3908.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step302.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step38851.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch1_step77702.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch2_step98789.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch3_step155404.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch4_step194255.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch5_step233106.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch6_step252174.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v1_epoch2_step116553.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.zip
            - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002_uky_invariants.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001_uky_invariants.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007_uky_invariants.kwcoco.zip

            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
            #- 256,256
        bas_pxl.time_span:
            - auto
            #- hardish3+pairwise+distribute
        bas_pxl.time_sampling:
            - auto
            #- 1m
            #- 3m
        bas_poly.thresh:
            #- 0.07
            #- 0.08
            #- 0.09
            - 0.1
            #- 0.11
            #- 0.12
            #- 0.13
            #- 0.14
            #- 0.15
            - 0.16
            - 0.17
            - 0.18
            #- 0.19
            - 0.2
            #- 0.21
            #- 0.22
            #- 0.23
            #- 0.24
            - 0.25
            #- 0.3
            #- 0.35
            - 0.4
            #- 0.45
            - 0.5
            #- 0.55
            - 0.6
            #- 0.65
            - 0.7
            #- 0.75
            #- 0.8
            #- 0.85
        bas_poly.polygon_simplify_tolerance:
            #- 0.5
            - 1
            #- 3
        bas_poly.agg_fn:
            - probs
            #- binary
            - rescaled_probs
            #- rescaled_binary
            #- mean_normalized
            #- frequency_weighted_mean
        bas_poly.resolution:
            - 10GSD
            #- 15GSD
            - 30GSD
        bas_poly.moving_window_size:
            - null
            - 100
            #- 200
            #- 300
            #- 400
        bas_poly.min_area_sqkm:
            - 0.0072
            - 0.072
        bas_poly.max_area_sqkm:
            #- null
            #- 1.00
            #- 2.00
            #- 2.25
            #- 3.25
            - 8
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/region_models
        bas_pxl.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval" \
    --devices="0,1" --tmux_workers=8 --print_commands=0 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=1






# NAMEK Pixel Eval
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step108.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step0.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch0_step0.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch27_step229376.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch29_step245760.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch0_step0.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch44_step46014.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_V4_v0_epoch0_step307.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch7_step3908.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v1_epoch0_step512.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step302.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch6_step252174.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGR_V4/Drop4_BAS_2022_12_15GSD_BGR_V4_v0_epoch0_step1354.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6_v0_epoch0_step0.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6_v0_epoch18_step55860.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch0_step172.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch27_step14078.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch0_step105.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch6_step3584.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v1_epoch5_step3072.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_BGRNSH_V1/Drop4_TuneV323_BAS_BGRNSH_V1_v0_epoch0_step86.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch0_step1172.pt'
            - '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch21_step60082.pt'
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch102_step52736.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch104_step53760.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch138_step71168.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch140_step72192.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch143_step73728.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch149_step76800.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch159_step81920.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch78_step40448.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch98_step50688.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step4305.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step1024.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step8247.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch2_step1536.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch3_step2048.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch5_step3072.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch6_step3584.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch7_step3908.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step302.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step38851.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch1_step77702.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch2_step98789.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch3_step155404.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch4_step194255.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch5_step233106.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch6_step252174.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v1_epoch2_step116553.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.zip
            - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.zip
            - $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002_uky_invariants.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001_uky_invariants.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007_uky_invariants.kwcoco.zip

            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/region_models
        bas_poly.thresh:
            #- 0.07
            #- 0.08
            #- 0.09
            - 0.1
            #- 0.11
            #- 0.12
            #- 0.13
            #- 0.14
            #- 0.15
            - 0.16
            - 0.17
            - 0.18
            #- 0.19
            - 0.2
            #- 0.21
            #- 0.22
            #- 0.23
            #- 0.24
            - 0.25
            - 0.3
            #- 0.35
            - 0.4
            #- 0.45
            - 0.5
            #- 0.55
            - 0.6
            #- 0.65
            - 0.7
            #- 0.75
            - 0.8
            #- 0.85
        bas_poly.polygon_simplify_tolerance:
            #- 0.5
            - 1
            #- 3
        bas_poly.agg_fn:
            - probs
            - rescaled_probs
            #- binary
            #- rescaled_binary
            #- mean_normalized
            #- frequency_weighted_mean
        bas_poly.resolution:
            - 10GSD
            #- 15GSD
            - 30GSD
        bas_poly.moving_window_size:
            - null
            - 100
            #- 200
            #- 300
            #- 400
        bas_poly.min_area_sqkm:
            - 0.0072
            - 0.072
        bas_poly.max_area_sqkm:
            #- null
            #- 1.00
            #- 2.00
            #- 2.25
            #- 3.25
            - 8
        bas_pxl.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval" \
    --devices="0,1" --tmux_workers=2 --print_commands=0 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=1



DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step1024.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.zip
                - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002_uky_invariants.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001_uky_invariants.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007_uky_invariants.kwcoco.zip

                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims:
                - auto
                #- 256,256
            bas_pxl.time_span:
                - auto
                #- hardish3+pairwise+distribute
            bas_pxl.time_sampling:
                - auto
                #- 1m
                #- 3m
            bas_poly.thresh:
                #- 0.07
                #- 0.08
                #- 0.09
                - 0.1
                #- 0.11
                #- 0.12
                #- 0.13
                #- 0.14
                #- 0.15
                - 0.16
                - 0.17
                - 0.18
                #- 0.19
                - 0.2
                #- 0.21
                #- 0.22
                #- 0.23
                #- 0.24
                - 0.25
                #- 0.3
                #- 0.35
                #- 0.4
                #- 0.45
                - 0.5
                #- 0.55
                #- 0.6
                #- 0.65
                - 0.7
                #- 0.75
                #- 0.8
                #- 0.85
            bas_poly.polygon_simplify_tolerance:
                #- 0.5
                - 1
                #- 3
            bas_poly.agg_fn:
                - probs
                - rescaled_probs
                #- binary
                #- rescaled_binary
                #- mean_normalized
                #- frequency_weighted_mean
            bas_poly.resolution:
                - 10GSD
                #- 15GSD
                - 30GSD
            bas_poly.moving_window_size:
                - null
                - 100
                #- 200
                #- 300
                #- 400
            bas_poly.min_area_sqkm:
                - 0.0072
                - 0.072
            bas_poly.max_area_sqkm:
                #- null
                #- 1.00
                #- 2.00
                #- 2.25
                #- 3.25
                - 8
            bas_pxl.enabled: 0
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=1



DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch102_step52736.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims:
                - auto
            bas_pxl.time_span:
                - auto
            bas_pxl.time_sampling:
                - auto
            bas_poly.thresh:
                - 0.1
            bas_poly.polygon_simplify_tolerance:
                - 1
            bas_poly.agg_fn:
                - probs
            bas_poly.resolution:
                - 10GSD
            bas_poly.moving_window_size:
                - null
            bas_poly.max_area_sqkm:
                - 8
            bas_pxl.enabled: 0
            bas_poly.enabled: 0
            bas_poly_eval.enabled: 0
            bas_pxl_eval.enabled: 0
            bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=0



#### QUICK Temporal Sampling Checks


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                #- $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch159_step163840.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step8247.pt

            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.zip
                - $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002_uky_invariants.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001_uky_invariants.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007_uky_invariants.kwcoco.zip

                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R001.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_US_R007.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_BR_R002.kwcoco.zip
                #- $DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001.kwcoco.zip
            bas_pxl.chip_overlap: 0.3
            bas_pxl.chip_dims:
                - auto
                #- 256,256
                #- 196,196
                #- 320,320
            bas_pxl.time_span:
                - null
                - auto
                #- 1m
                #- 3m
            bas_poly.thresh:
                #- 0.1
                - 0.16
                - 0.17
                - 0.18
                #- 0.2
                - 0.25
                #- 0.4
                - 0.5
                #- 0.55
                - 0.7
            bas_poly.polygon_simplify_tolerance:
                - 1
            bas_poly.agg_fn:
                - probs
                #- binary
                #- rescaled_probs
                #- rescaled_binary
                #- mean_normalized
                #- frequency_weighted_mean
            bas_poly.resolution:
                - 10GSD
                #- 15GSD
                - 30GSD
            bas_poly.moving_window_size:
                - null
                - 100
            bas_poly.min_area_square_meters:
                - 7200
                - null
                - 72000
            bas_poly.max_area_square_meters:
                - 8000000
            bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/site_models
            bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/region_models
            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 0
            bas_pxl.batch_size: 1
            bas_pxl.num_workers: 2

        submatrices:
            - bas_pxl.time_span: null
              bax_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch159_step163840.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step8247.pt
              bas_pxl.time_kernel:
                  - '-1y,-3m,-1w,0,1w,3m,1y'
                  - '-6m,-3m,-1w,0,1w,3m,6m'
              bas_pxl.time_sampling:
                  - auto
                  - soft4
                  - soft5
              bas_pxl.window_space_scale: 10GSD
            - bas_pxl.time_span: null
              bax_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
              bas_pxl.time_kernel:
                  - '-1y,-3m,0,3m,1y'
              bas_pxl.time_sampling:
                  - auto
                  - soft4
                  - soft5
              bas_pxl.window_space_scale: 10GSD

        include:
            - bas_pxl.window_space_scale: 10GSD
              bas_pxl.input_space_scale: 10GSD
              bas_pxl.output_space_scale: 10GSD

    " \
    --root_dpath="$DVC_EXPT_DPATH/_timekernel_test_drop4" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_timekernel_test_drop4" \
    --pipeline=bas --skip_existing=1 \
    --print_varied=0  \
    --run=1


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
echo "DVC_DATA_DPATH = $DVC_DATA_DPATH"

#"$DVC_DATA_DPATH/Drop6/imganns-KR*.kwcoco.zip" \
python -m watch.cli.prepare_teamfeats \
    --base_fpath \
       "$DVC_DATA_DPATH/Drop6/imganns-AE_R001.kwcoco.zip" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=0 \
    --with_invariants2=1 \
    --invariant_resolution=10GSD \
    --kwcoco_ext=".kwcoco.zip" \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=0 \
    --gres='0,' --workers=1 --backend=tmux --run=1

#rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/./Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6 .



#### QUICK BAS CHECKS

python -m watch.mlops.manager "list" --dataset_codes Drop6 Drop4-BAS

####
# SPLIT 1 SMALL TEST
####

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.cluster_sites \
        --src "$DVC_DATA_DPATH/annotations/drop6/region_models/KR_R002.geojson" \
        --dst_dpath "$DVC_DATA_DPATH"/ValiRegionSmall/geojson \
        --draw_clusters True

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.coco_align \
    --src "$DVC_DATA_DPATH"/Drop6/combo_imganns-KR_R002_L.kwcoco.json \
    --dst "$DVC_DATA_DPATH"/ValiRegionSmall/small_KR_R002_odarcigm.kwcoco.zip \
    --regions "$DVC_DATA_DPATH"/ValiRegionSmall/geojson/SUB_KR_R002_n007_odarcigm.geojson \
    --minimum_size="128x128@10GSD" \
    --context_factor=1 \
    --geo_preprop=auto \
    --force_nodata=-9999 \
    --site_summary=False \
    --target_gsd=5 \
    --aux_workers=8 \
    --workers=8


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.cli.prepare_teamfeats \
    --base_fpath \
       "$DVC_DATA_DPATH/ValiRegionSmall/small_KR_R002_odarcigm.kwcoco.zip" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=0 \
    --with_invariants2=1 \
    --with_cold=0 \
    --invariant_resolution=30GSD \
    --kwcoco_ext=".kwcoco.zip" \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=0 \
    --gres='0,' --workers=2 --backend=tmux --run=1

python -m watch.mlops.manager "list" --dataset_codes Drop6  | grep -v split2

# SPLIT 1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $HOME/code/watch/dev/reports/split1_all_models.yaml
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/ValiRegionSmall/combo_small_KR_R002_odarcigm_I2.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 0
        bas_poly_eval.enabled: 0
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_split1_eval_small" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_namek_split1_eval_small" \
    --pipeline=bas --skip_existing=1 \
    --run=1




# SPLIT 1 - filter1 analysis
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $HOME/code/watch/dev/reports/split1_models_filter1.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R002.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-NZ_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-CH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R002.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly.thresh:
            #- 0.1
            - 0.17
            - 0.2
            - 0.4
            - 0.5
            - 0.8
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.resolution:
            - 10GSD
        bas_poly.moving_window_size:
            - null
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_split1_eval_filter1_MeanYear10GSD" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_namek_split1_eval_filter1_MeanYear10GSD" \
    --pipeline=bas --skip_existing=1 \
    --run=1


# ###################
# SPLIT 2 - SMALL TEST
# ###################

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.cluster_sites \
        --src "$DVC_DATA_DPATH/annotations/drop6/region_models/NZ_R001.geojson" \
        --dst_dpath "$DVC_DATA_DPATH"/ValiRegionSmall/geojson/NZ_R001 \
        --draw_clusters True

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.coco_align \
    --src "$DVC_DATA_DPATH"/Drop6/combo_imganns-NZ_R001_L.kwcoco.json \
    --dst "$DVC_DATA_DPATH"/ValiRegionSmall/small_NZ_R001_swnykmah.kwcoco.zip \
    --regions "$DVC_DATA_DPATH"/ValiRegionSmall/geojson/NZ_R001/SUB_NZ_R001_n031_swnykmah.geojson \
    --minimum_size="128x128@10GSD" \
    --context_factor=1 \
    --geo_preprop=auto \
    --force_nodata=-9999 \
    --site_summary=False \
    --target_gsd=5 \
    --aux_workers=8 \
    --workers=8


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.cli.prepare_teamfeats \
    --base_fpath \
       "$DVC_DATA_DPATH/ValiRegionSmall/small_NZ_R001_swnykmah.kwcoco.zip" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=0 \
    --with_invariants2=1 \
    --with_cold=0 \
    --invariant_resolution=30GSD \
    --kwcoco_ext=".kwcoco.zip" \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=0 \
    --gres='0,' --workers=2 --backend=tmux --run=1


python -m watch.mlops.manager "list" --dataset_codes Drop6  | grep split2

/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch18_step15200.pt

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $HOME/code/watch/dev/reports/split2_all_models.yaml
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/ValiRegionSmall/combo_small_NZ_R001_swnykmah_I2.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 0
        bas_poly_eval.enabled: 0
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_split2_eval_small" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_namek_split2_eval_small" \
    --pipeline=bas --skip_existing=1 \
    --run=0



#fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V24/Drop6_BAS_scratch_landcover_10GSD_split2_V24_epoch126_step25400.pt.dvc
#fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V24/Drop6_BAS_scratch_landcover_10GSD_split2_V24_epoch16_step3400.pt.dvc
#fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V24/Drop6_BAS_scratch_landcover_10GSD_split2_V24_epoch183_step36800.pt.dvc
#fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V24/Drop6_BAS_scratch_landcover_10GSD_split2_V24_epoch18_step3800.pt.dvc
#fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V24/Drop6_BAS_scratch_landcover_10GSD_split2_V24_epoch211_step42400.pt.dvc
#fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V24/Drop6_BAS_scratch_landcover_10GSD_split2_V24_epoch218_step43800.pt.dvc
#fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch1133_step71442.pt.dvc
#fusion/Drop6/packages/Drop6_BAS_scratch_validation_10GSD_split2_V34/.gitignore
#fusion/Drop6/packages/Drop6_BAS_scratch_validation_10GSD_split2_V34/Drop6_BAS_scratch_validation_10GSD_split2_V34_epoch36_step2331.pt.dvc




#################################################################
# SPLIT 6 - Time Model Checks
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2_epoch31_step8192.pt
        bas_pxl.test_dataset:
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-NZ_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-CH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BH_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R002.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-AE_R002.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly.thresh:
            #- 0.1
            #- 0.17
            - 0.2
            - 0.25
            - 0.3
            #- 0.4
            #- 0.5
            #- 0.8
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.resolution:
            - 10GSD
        bas_poly.moving_window_size:
            - null
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_toothbrush_eval_split6_MeanYear10GSD" \
    --devices="0," --tmux_workers=1 \
    --backend=serial --queue_name "_toothbrush_eval_split6_MeanYear10GSD" \
    --pipeline=bas --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas \
    --stdout_report=True \
    --target \
        "$DVC_EXPT_DPATH/_toothbrush_eval_split6_MeanYear10GSD"

##########
#

# New Stuff - 03-25
#
# SeeAlso:
# ~/code/watch/dev/poc/prepare_time_combined_dataset.py

python -m watch.mlops.manager "list" --dataset_codes Drop6-MeanYear10GSD

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42/Drop6_TCombo1Year_BAS_10GSD_split6_V42_epoch27_step7168.pt
            - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
            #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V41_cont2_epoch41_step10532.pt
        bas_pxl.test_dataset:
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R002.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-CH_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BH_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-NZ_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-AE_R001.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly.thresh:
            #- 0.1
            #- 0.17
            #- 0.2
            #- 0.25
            #- 0.27
            #- 0.27
            #- 0.3
            - 0.33
            - 0.35
            - 0.38
            - 0.4
            - 0.42
            #- 0.5
            #- 0.7
        bas_poly.inner_window_size:
            - 1y
            - null
        bas_poly.inner_agg_fn:
            - mean
        bas_poly.norm_ord:
            - 1
            #- 2
            - inf
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.resolution:
            - 10GSD
        bas_poly.moving_window_size:
            - null
            - 1
            #- 2
            #- 3
            #- 4
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_split6_toothbrush_meanyear" \
    --devices="0,1" --tmux_workers=8 \
    --backend=tmux --queue_name "_split6_toothbrush_meanyear" \
    --pipeline=bas --skip_existing=1 \
    --run=1

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_split6_toothbrush_meanyear
    " \
    --resource_report=True \
    --rois=KR_R001,KR_R002,CH_R001,NZ_R001,AE_R001,BH_R001,BR_R002,BR_R001 \
    --stdout_report="
        top_k: 3
        per_group: 1
        macro_analysis: 1
        analyze: 0
        reference_region: final
    "
    #--plot_params=True \
    #--output_dpath="$DVC_EXPT_DPATH"/_split6_toothbrush_meanyear/_aggregate


# Eval10 baseline
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
            #- soft5
            #- soft4
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
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_mlops_eval10_baseline" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_mlops_eval10_baseline" \
    --pipeline=bas --skip_existing=1 \
    --run=1

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_mlops_eval10_baseline
    " \
    --output_dpath="$DVC_EXPT_DPATH/_mlops_eval10_baseline/aggregate" \
    --resource_report=0 \
    --rois="[KR_R002,BR_R002,CH_R001,NZ_R001,KR_R001,AE_R001]" \
    --stdout_report="
        top_k: 1
        per_group: 1
        macro_analysis: 1
        analyze: 0
        reference_region: final
    "


#Prep models
python -c "if 1:
from kwutil.util_yaml import Yaml
from watch.utils import simple_dvc
import watch
import platform
host = platform.node()
expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
dvc = simple_dvc.SimpleDVC(expt_dvc_dpath)
cand_list_fpath = expt_dvc_dpath / 'model_candidates/split1_shortlist_v3.yaml'
suffixes = Yaml.coerce(cand_list_fpath)
resolved_fpaths = [os.fspath(expt_dvc_dpath / s) for s in suffixes]
new_cand_fpath = cand_list_fpath.augment(prefix=host + '_')
new_cand_fpath.write_text(Yaml.dumps(resolved_fpaths))
print(new_cand_fpath)

dvc.pull(resolved_fpaths)
"

geowatch manager "pull packages" --dataset_codes Drop6-MeanYear10GSD-V2 --yes
geowatch manager "list packages" --dataset_codes Drop6-MeanYear10GSD-V2 --yes

# SITE VISIT 2022-04 SPLIT 1 Analysis
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            #- $HOME/code/watch/dev/reports/split1_all_models.yaml
            #- $HOME/code/watch/dev/reports/split1_shortlist_v2.yaml
            - $DVC_EXPT_DPATH/model_candidates/namek_split1_shortlist_v4.yaml
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-BR_R002_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-AE_R001_I2L.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.input_space_scale:
            - 10GSD
        bas_pxl.time_sampling:
            - soft5
            - soft4
        bas_poly.thresh:
            - 0.25
            - 0.275
            - 0.3
            - 0.325
            - 0.35
            - 0.375
            - 0.4
            - 0.425
            - 0.45
        bas_poly.time_thresh:
            - 1.0
            - 0.8
        bas_poly.inner_window_size:
            - 1y
            #- null
        bas_poly.inner_agg_fn:
            - mean
        bas_poly.norm_ord:
            #- 1
            #- 2
            - inf
        bas_poly.resolution:
            - 10GSD
        bas_poly.moving_window_size:
            - null
            #- 1
            #- 2
            #- 3
            #- 4
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0

    submatrices:
        - bas_pxl.input_space_scale: 10GSD
          bas_pxl.window_space_scale: 10GSD
          bas_pxl.output_space_scale: 10GSD
          bas_poly.resolution:
              - 10GSD
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_split1_eval_filter1_MeanYear10GSD-V2" \
    --devices="0,1" --tmux_workers=6 \
    --backend=tmux --queue_name "_namek_split1_eval_filter1_MeanYear10GSD-V2" \
    --pipeline=bas --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas \
    --target \
        "$DVC_EXPT_DPATH/_namek_split1_eval_filter1_MeanYear10GSD-V2" \
    --rois=KR_R001,KR_R002,CH_R001,NZ_R001,BR_R002 \
    --resource_report=0 \
    --stdout_report="
        top_k: 20
        per_group: 2
        macro_analysis: 0
        analyze: 0
        reference_region: final
        print_models: 1
    "
    #--rois=KR_R002 \

python -c "if 1:
from watch.mlops.aggregate import *  # NOQA
import watch
data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
cmdline = 0
kwargs = {
    'target': [expt_dvc_dpath / '_namek_split1_eval_filter1_MeanYear10GSD'],
    'pipeline': 'bas',
    'io_workers': 10,
}
config = AggregateEvluationConfig.cli(cmdline=cmdline, data=kwargs)
eval_type_to_aggregator = coerce_aggregators(config)

reference_region = 'KR_R002'

agg.build_macro_tables(['KR_R002'])
_ = agg.report_best()

agg = eval_type_to_aggregator['bas_pxl_eval']
agg = eval_type_to_aggregator['bas_poly_eval']
"



# SITE VISIT 2022-04 SPLIT 2 Analysis
# OOO Variant Analysis
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/model_candidates/split2_mixed_ooo.yaml
            #- /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V45/Drop6_TCombo1Year_BAS_10GSD_split6_V45_epoch73_step18944.pt
            #- /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch6_step22939.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-BR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-CH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-NZ_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-AE_R001.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.input_space_scale:
            - 10GSD
        bas_pxl.time_sampling:
            - soft5
        bas_poly.thresh:
            - 0.3
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.moving_window_size:
            - null
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0

    submatrices:
        - bas_pxl.input_space_scale: 10GSD
          bas_pxl.window_space_scale: 10GSD
          bas_pxl.output_space_scale: 10GSD
          bas_poly.resolution:
              - 10GSD
    " \
    --root_dpath="$DVC_EXPT_DPATH/_ooo_split2_eval_filter1_MeanYear10GSD-V2" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --queue_name "_ooo_split2_eval_filter1_MeanYear10GSD-V2" \
    --pipeline=bas --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas \
    --target \
        "$DVC_EXPT_DPATH/_ooo_split2_eval_filter1_MeanYear10GSD-V2" \
    --resource_report=True \
    --stdout_report="
        top_k: 30
        per_group: 2
        macro_analysis: 0
        analyze: 0
        # reference_region: KR_R002
        # print_models: True
    "
    #--rois=KR_R002 \




### Build namek aggregate
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_namek_split1_eval_filter1_MeanYear10GSD-V2
        - $DVC_EXPT_DPATH/_namek_split1_eval_filter1_MeanYear10GSD
        - $DVC_EXPT_DPATH/_namek_split1_eval_filter1
        - $DVC_EXPT_DPATH/_namek_split1_eval_small
        - $DVC_EXPT_DPATH/_namek_split2_eval_small
        - $DVC_EXPT_DPATH/_quick_split1_checks
        - $DVC_EXPT_DPATH/_timekernel_test_drop4
        - $DVC_EXPT_DPATH/_mlops_eval10_baseline
    " \
    --export_tables=True \
    --output_dpath="$DVC_EXPT_DPATH/aggregate_results/namek"


### Build toothbrush aggregate
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas_building_vali \
    --target "
        - $DVC_EXPT_DPATH/_mlops_eval10_baseline
        - $DVC_EXPT_DPATH/_toothbrush_eval_split6_MeanYear10GSD
        - $DVC_EXPT_DPATH/_split6_toothbrush_meanyear
    " \
    --export_tables=True \
    --output_dpath="$DVC_EXPT_DPATH/aggregate_results/toothbrush"


### Build toothbrush aggregate
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas_building_vali \
    --target "
        - $DVC_EXPT_DPATH/_ooo_split2_eval_filter1_MeanYear10GSD-V2
    " \
    --export_tables=True \
    --output_dpath="$DVC_EXPT_DPATH/aggregate_results/ooo"


# New  VISIT 2022-04 SPLIT 2 Analysis
# OOO Variant Analysis
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/model_candidates/split2_mixed_ooo.yaml
            #- /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V45/Drop6_TCombo1Year_BAS_10GSD_split6_V45_epoch73_step18944.pt
            #- /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch6_step22939.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-BR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-CH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-NZ_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-KR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/imganns-AE_R001.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.input_space_scale:
            - 10GSD
        bas_pxl.time_sampling:
            - soft5
        bas_poly.thresh:
            - 0.3
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.moving_window_size:
            - null
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0

    submatrices:
        - bas_pxl.input_space_scale: 10GSD
          bas_pxl.window_space_scale: 10GSD
          bas_pxl.output_space_scale: 10GSD
          bas_poly.resolution:
              - 10GSD
    " \
    --root_dpath="$DVC_EXPT_DPATH/_ooo_split2_eval_filter1_MeanYear10GSD-V2" \
    --devices="0,1" --tmux_workers=2 \
    --backend=tmux --queue_name "_ooo_split2_eval_filter1_MeanYear10GSD-V2" \
    --pipeline=bas --skip_existing=1 \
    --run=1



### Evaluate promissing landcover models on namek
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V45/Drop6_TCombo1Year_BAS_10GSD_split6_V45_epoch73_step18944.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V48/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V48_epoch106_step6848.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-BR_R002_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R001_I2L.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-AE_R001_I2L.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - '196,196'
        bas_pxl.time_span:
            - auto
        bas_pxl.fixed_resolution:
            - 10GSD
        bas_pxl.time_sampling:
            - auto
            - soft5
            - soft4
        bas_poly.thresh:
            - 0.25
            - 0.3
            - 0.35
            - 0.4
            - 0.45
            - 0.5
        bas_poly.time_thresh:
            - 1.0
            - 0.9
            - 0.8
        bas_poly.inner_window_size:
            - 1y
        bas_poly.inner_agg_fn:
            - mean
        bas_poly.norm_ord:
            - inf
        bas_poly.resolution:
            - 10GSD
        bas_poly.moving_window_size:
            - null
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
        sv_crop.enabled: 1
        sv_crop.minimum_size: 256x256@3GSD
        sv_crop.num_start_frames: 3
        sv_crop.num_end_frames: 3
        sv_crop.context_factor: 1.6
        sv_dino_boxes.enabled: 1
        sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
        sv_dino_boxes.window_dims:
            #- 256
            - 512
            #- 768
            #- 1024
            #- 1536
        sv_dino_boxes.window_overlap:
            - 0.5
        sv_dino_boxes.fixed_resolution:
            #- 1GSD
            #- 2GSD
            #- 2.5GSD
            - 3GSD
            #- 3.3GSD
        sv_dino_filter.box_isect_threshold:
            - 0.1
        sv_dino_filter.box_score_threshold:
            - 0.01
        sv_dino_filter.start_max_score:
            - 1.0
            #- 0.9
            # - 0.8
            # - 0.5
        sv_dino_filter.end_min_score:
            - 0.0
            #- 0.05
            - 0.1
            #- 0.15
            - 0.2
            #- 0.25
            #- 0.3
            # - 0.4
            #- 0.5
    submatrices:
        - bas_pxl.fixed_resolution: 10GSD
          bas_poly.resolution:
              - 10GSD
        - bas_pxl.fixed_resolution: 8GSD
          bas_poly.resolution:
              - 8GSD
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Drop6/imgonly-KR_R001.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-BR_R002_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Drop6/imgonly-BR_R002.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Drop6/imgonly-CH_R001.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Drop6/imgonly-NZ_R001.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-AE_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Drop6/imgonly-AE_R001.kwcoco.json
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_split6_landcover_MeanYear10GSD-V2" \
    --devices="0,1" --tmux_workers=8 \
    --backend=tmux --queue_name "_namek_split6_landcover_MeanYear10GSD-V2" \
    --pipeline=bas_building_vali --skip_existing=1 \
    --run=1



### Evaluate promissing landcover models on toothbrush
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V45/Drop6_TCombo1Year_BAS_10GSD_split6_V45_epoch73_step18944.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V48/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V48_epoch106_step6848.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-BR_R002_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R001_I2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-AE_R001_I2L.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            #- auto
            - '196,196'
            #- '256,256'
            #- '320,320'
        bas_pxl.time_span:
            - auto
        bas_pxl.fixed_resolution:
            - 10GSD
        bas_pxl.time_sampling:
            #- auto
            #- soft5
            - soft4
        bas_poly.thresh:
            - 0.25
            - 0.275
            - 0.3
            - 0.325
            - 0.35
            - 0.375
            - 0.4
            - 0.425
            - 0.45
        bas_poly.time_thresh:
            #- 1.0
            #- 0.95
            #- 0.9
            #- 0.85
            - 0.8
            - 0.75
            - 0.70
        bas_poly.inner_window_size:
            - 1y
        bas_poly.inner_agg_fn:
            - mean
        bas_poly.norm_ord:
            #- 1
            #- 2
            - inf
        bas_poly.resolution:
            - 10GSD
        bas_poly.moving_window_size:
            - null
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
        sv_crop.enabled: 1
        sv_crop.minimum_size: 256x256@3GSD
        sv_crop.num_start_frames: 3
        sv_crop.num_end_frames: 3
        sv_crop.context_factor: 1.6
        sv_dino_boxes.enabled: 1
        sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
        sv_dino_boxes.window_dims:
            #- 256
            - 320
            #- 512
            #- 768
            #- 1024
            #- 1536
        sv_dino_boxes.window_overlap:
            - 0.5
        sv_dino_boxes.fixed_resolution:
            #- 1GSD
            #- 2GSD
            #- 2.5GSD
            - 3GSD
            #- 3.3GSD
            #- 4.0GSD
        sv_dino_filter.box_isect_threshold:
            - 0.1
        sv_dino_filter.box_score_threshold:
            - 0.01
        sv_dino_filter.start_max_score:
            - 1.0
            #- 0.9
            # - 0.8
            # - 0.5
        sv_dino_filter.end_min_score:
            #- 0.0
            #- 0.05
            - 0.1
            - 0.15
            - 0.2
            - 0.25
            - 0.3
            # - 0.4
            #- 0.5
    submatrices1:
        - bas_pxl.fixed_resolution: 10GSD
          bas_poly.resolution:
              - 10GSD
        - bas_pxl.fixed_resolution: 8GSD
          bas_poly.resolution:
              - 8GSD
    submatrices2:
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R001.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-KR_R002.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-BR_R002_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-BR_R002.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-CH_R001.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-NZ_R001.kwcoco.json
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-AE_R001_I2L.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_DATA_DPATH/Drop6/imgonly-AE_R001.kwcoco.json
    " \
    --root_dpath="$DVC_EXPT_DPATH/_toothbrush_split6_landcover_MeanYear10GSD-V2" \
    --devices="0,1" --tmux_workers=8 \
    --backend=tmux --queue_name "_toothbrush_split6_landcover_MeanYear10GSD-V2" \
    --pipeline=bas_building_vali --skip_existing=1 \
    --run=1



DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas_building_vali \
    --target \
        "$DVC_EXPT_DPATH/_toothbrush_split6_landcover_MeanYear10GSD-V2" \
    --stdout_report="
        top_k: 10
        per_group: 2
        macro_analysis: 0
        analyze: 0
        reference_region: final
        # print_models: True
    " \
    --resource_report=0 \
    --plot_params=0 \
    --export_tables=0 \
    --output_dpath="$DVC_EXPT_DPATH/_toothbrush_split6_landcover_MeanYear10GSD-V2/aggregate" \
    --rois=KR_R001,KR_R002,CH_R001,NZ_R001,BR_R002



#DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
#geowatch aggregate \
#    --pipeline=bas \
#    --target "
#        - $DVC_EXPT_DPATH/aggregate_results/toothbrush/*.csv.zip
#        - $DVC_EXPT_DPATH/aggregate_results/ooo/*.csv.zip
#        - $DVC_EXPT_DPATH/aggregate_results/namek/*.csv.zip
#    " \
#    --rois=KR_R001,KR_R002,CH_R001,NZ_R001,BR_R002 \
#    --stdout_report="
#        top_k: 3
#        per_group: 2
#        macro_analysis: 0
#        analyze: 0
#        reference_region: final
#        print_models: False
#    " \
#    --resource_report=False \
#    --plot_params=False \
#    --export_tables=False \




#Prep models
python -c "if 1:
    from kwutil.util_yaml import Yaml
    from watch.utils import simple_dvc
    import watch
    import platform
    import os
    host = platform.node()
    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    dvc = simple_dvc.SimpleDVC(expt_dvc_dpath)
    cand_list_fpath = expt_dvc_dpath / 'model_candidates/split1_shortlist_v4_top.yaml'
    suffixes = Yaml.coerce(cand_list_fpath)
    resolved_fpaths = [os.fspath(expt_dvc_dpath / s) for s in suffixes]
    new_cand_fpath = cand_list_fpath.augment(prefix=host + '_')
    new_cand_fpath.write_text(Yaml.dumps(resolved_fpaths))
    print(new_cand_fpath)

    path = resolved_fpaths
    dvc.pull(resolved_fpaths, remote='aws')
"




# SITE VISIT 2022-04 SPLIT 1 Analysis
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            #- $DVC_EXPT_DPATH/model_candidates/namek_split1_shortlist_v4_top.yaml
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_MultiModal_Resume/Drop6_MultiModal_Resume_epoch2_step96.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_MultiModal_Resume/Drop6_MultiModal_Resume_epoch3_step128.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-NoWinterMedian10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56_epoch268_step7801.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-NoWinterMedian10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56_epoch359_step10440.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch78_step5056.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V53/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V53_epoch0_step0.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch46_step3008.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch30_step1984.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch10_step704.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58_epoch0_step0.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58_epoch218_step11607.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58_epoch10_step440.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59_epoch146_step7791.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58_epoch0_step10.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59_epoch0_step0.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch85_step5504.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58_epoch11_step457.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V58_epoch278_step14787.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59/Drop6_TCombo1Year_BAS_10GSD_V2_invariants_landcover_split6_V59_epoch57_step16124.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-KR_R002_I2LS.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-CH_R001_I2LS.kwcoco.zip
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD-V2/combo_imganns-NZ_R001_I2LS.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
            - [196,196]
        bas_pxl.time_span:
            - auto
        bas_pxl.input_space_scale:
            - 10GSD
            - 5GSD
        bas_pxl.time_sampling:
            - soft4
        bas_poly.thresh:
            #- 0.1
            #- 0.2
            - 0.275
            - 0.3
            - 0.325
            - 0.35
            - 0.375
            - 0.39
            - 0.4
            - 0.41
            - 0.42
            - 0.425
            - 0.430
            - 0.45
            - 0.5
            - 0.6
        bas_poly.time_thresh:
            - 0.8
            - 0.7
            - 0.6
            - 0.5
            #- 0.4
        bas_poly.inner_window_size:
            - 1y
            #- null
        bas_poly.inner_agg_fn:
            - mean
            - max
        bas_poly.norm_ord:
            - inf
        bas_poly.moving_window_size:
            - null
            #- 1
            #- 2
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0

    submatrices:
        - bas_pxl.input_space_scale: 10GSD
          bas_pxl.window_space_scale: 10GSD
          bas_pxl.output_space_scale: 10GSD
          bas_poly.resolution: 10GSD
        - bas_pxl.input_space_scale: 5GSD
          bas_pxl.window_space_scale: 5GSD
          bas_pxl.output_space_scale: 5GSD
          bas_poly.resolution: 5GSD
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_preeval12" \
    --devices="0,1" --tmux_workers=6 \
    --backend=tmux --queue_name "_namek_preeval12" \
    --pipeline=bas --skip_existing=1 \
    --run=1



DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch aggregate \
    --pipeline=bas \
    --target \
        "$DVC_EXPT_DPATH/_namek_preeval12" \
    --stdout_report="
        top_k: 10
        per_group: 2
        macro_analysis: 0
        analyze: 0
        reference_region: final
        print_models: True
    " \
    --resource_report=0 \
    --plot_params=0 \
    --export_tables=0 \
    --io_workers=10 \
    --output_dpath="$DVC_EXPT_DPATH/_namek_preeval12/aggregate" \
    --rois=KR_R002,NZ_R001
    #--rois=KR_R002,CH_R001,NZ_R001
    #--inspect=fcfdpnldzxzv \
    #--rois="KR_R002,"
    #
    #--inspect=kdvkheujolhb \
    #--pipeline=bas_building_vali \


# Initial eval over Drop7
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/model_candidates/toothbrush_split1_shortlist_v4_top.yaml
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_MultiModal_Resume/Drop6_MultiModal_Resume_epoch2_step96.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_MultiModal_Resume/Drop6_MultiModal_Resume_epoch3_step128.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-NoWinterMedian10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56_epoch268_step7801.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-NoWinterMedian10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_invar_split6_V56_epoch359_step10440.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch78_step5056.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V53/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V53_epoch0_step0.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch46_step3008.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch30_step1984.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57/Drop6_TCombo1Year_BAS_10GSD_V2_sam_landcover_split6_V57_epoch10_step704.pt

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_I2LS.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_I2LS.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_I2LS.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            #- auto
            - [196,196]
        bas_pxl.time_span:
            - auto
        bas_pxl.input_space_scale:
            - 10GSD
        bas_pxl.time_sampling:
            - soft4
        bas_poly.thresh:
            - 0.4
            - 0.425
            - 0.45
            - 0.5
            - 0.6
        bas_poly.time_thresh:
            - 0.8
            - 0.6
            - 0.5
        bas_poly.inner_window_size:
            - 1y
        bas_poly.inner_agg_fn:
            - max
        bas_poly.norm_ord:
            - inf
        bas_poly.moving_window_size:
            - null
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0

    submatrices:
        - bas_pxl.input_space_scale: 10GSD
          bas_pxl.window_space_scale: 10GSD
          bas_pxl.output_space_scale: 10GSD
          bas_poly.resolution: 10GSD
    " \
    --root_dpath="$DVC_EXPT_DPATH/_toothbrush_drop7_nowinter" \
    --devices="0,1" --tmux_workers=6 \
    --backend=tmux --queue_name "_toothbrush_drop7_nowinter" \
    --pipeline=bas --skip_existing=1 \
    --run=1





- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V61/Drop7-MedianNoWinter10GSD_bgr_split6_V61_epoch359_step15480.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V61/Drop7-MedianNoWinter10GSD_bgr_split6_V61_epoch359_step15480.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V63/Drop7-MedianNoWinter10GSD_bgr_split6_V63_epoch359_step15480.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V63/Drop7-MedianNoWinter10GSD_bgr_split6_V63_epoch359_step15480.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch75_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch76_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch92_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch76_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch95_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64_epoch120_step10406.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64_epoch39_step3440.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch148_step6407.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch301_step12986.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60_epoch12_step559.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60_epoch163_step7052.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60_epoch163_step7052.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65_epoch115_stepNone.pt
- /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65_epoch182_stepNone.pt




python -m watch.mlops.manager "push packages" --dataset_codes Drop7-MedianNoWinter10GSD --yes
python -m watch.mlops.manager "pull packages" --dataset_codes Drop7-MedianNoWinter10GSD --yes

# New model eval over Drop7
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V61/Drop7-MedianNoWinter10GSD_bgr_split6_V61_epoch359_step15480.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch301_step12986.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch148_step6407.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V63/Drop7-MedianNoWinter10GSD_bgr_split6_V63_epoch359_step15480.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60_epoch12_step559.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_split6_V60_epoch163_step7052.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt
            #####
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch11_step1001.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64_epoch120_step10406.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64_epoch39_step3440.pt
            ####
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch75_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch92_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch76_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch95_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_mat_mae_scratch_split6_V64_epoch266_step22962.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65_epoch115_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65/Drop7-MedianNoWinter10GSD_landcover_invar_scratch_split6_V65_epoch182_stepNone.pt

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
            - [196,196]
        bas_pxl.time_span:
            - auto
        bas_pxl.input_space_scale:
            - 10GSD
        bas_pxl.time_sampling:
            - soft4
        bas_poly.thresh:
            - 0.4
            - 0.425
            - 0.45
            - 0.5
            - 0.6
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
            - 0.5
        bas_poly.inner_window_size:
            - 1y
        bas_poly.inner_agg_fn:
            - max
        bas_poly.norm_ord:
            - inf
        bas_poly.moving_window_size:
            - null
        bas_poly.poly_merge_method:
            - 'v2'
        bas_poly.polygon_simplify_tolerance:
            - 1
        bas_poly.agg_fn:
            - probs
        bas_poly.min_area_square_meters:
            - 7200
        bas_poly.max_area_square_meters:
            - 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0

    submatrices:
        - bas_pxl.input_space_scale: 10GSD
          bas_pxl.window_space_scale: 10GSD
          bas_pxl.output_space_scale: 10GSD
          bas_poly.resolution: 10GSD
    " \
    --root_dpath="$DVC_EXPT_DPATH/_toothbrush_drop7_nowinter" \
    --devices="0,1" --tmux_workers=6 \
    --backend=tmux --queue_name "_toothbrush_drop7_nowinter" \
    --pipeline=bas --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_toothbrush_drop7_nowinter
    " \
    --output_dpath="$DVC_EXPT_DPATH/_toothbrush_drop7_nowinter/aggregate" \
    --resource_report=0 \
    --plot_params="
        enabled: 1
        compare_sv_hack: True
        stats_ranking: 0
        min_variations: 1
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: KR_R002
    " \
    --rois="KR_R002,"


# Pull out baseline tables
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_toothbrush_drop7_nowinter
    " \
    --output_dpath="$DVC_EXPT_DPATH/_toothbrush_drop7_nowinter/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        - bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
    " \
    --stdout_report="
        top_k: 100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
    " \
    --rois="KR_R002"

    #--query='
    #    (df["params.bas_poly.thresh"] == 0.425) &
    #    (df["params.bas_poly.time_thresh"] == 0.8) &
    #    (df["params.bas_pxl.chip_dims"].apply(str).str.contains("196")) &
    #    (df["params.bas_pxl.package_fpath"].str.contains("Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026"))
    #'

#`params.bas_poly.thresh` == 0.425 and
#`params.bas_poly.time_thresh` == 0.8 and
#`params.bas_pxl.chip_dims`.apply(`str`).str.contains("196") and
#`params.bas_pxl.package_fpath`.str.contains("Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026")




#### HOROLOGIC SV PARAM SWEEP ####
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.utils.simple_dvc request \
    "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt \
    "$DVC_EXPT_DPATH"/models/depth_pcd/basicModel2.h5 \
    "$DVC_EXPT_DPATH"/models/depth_pcd/model3.h5

geowatch schedule --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.3
            - 0.325
            - 0.35
            - 0.375
            - 0.4
            - 0.425
            - 0.45
            - 0.5
            - 0.6
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
        sv_crop.enabled: 1
        sv_crop.minimum_size: '256x256@2GSD'
        sv_crop.num_start_frames: 3
        sv_crop.num_end_frames: 3
        sv_crop.context_factor: 1.5

        sv_dino_boxes.enabled: 1
        sv_dino_boxes.package_fpath: $DVC_EXPT_DPATH/models/kitware/xview_dino.pt
        sv_dino_boxes.window_dims: 256
        sv_dino_boxes.window_overlap: 0.5
        sv_dino_boxes.fixed_resolution: 3GSD

        sv_dino_filter.enabled: 1
        sv_dino_filter.end_min_score: 0.15
        sv_dino_filter.start_max_score: 1.0
        sv_dino_filter.box_score_threshold: 0.01
        sv_dino_filter.box_isect_threshold: 0.1

        sv_depth_score.enabled: 1
        sv_depth_score.model_fpath:
            - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
            - $DVC_EXPT_DPATH/models/depth_pcd/model3.h5
        sv_depth_filter.threshold:
            - 0.5
            - 0.10
            - 0.15
            - 0.20
            - 0.25
            - 0.30
            - 0.35
            - 0.40
            - 0.45
    submatrices:
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/KR_R001/imgonly-KR_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/AE_R001/imgonly-AE_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/BR_R002/imgonly-BR_R002.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/CH_R001/imgonly-CH_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/NZ_R001/imgonly-NZ_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/PE_R001/imgonly-PE_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/BR_R004/imgonly-BR_R004.kwcoco.zip
    " \
    --root_dpath="$DVC_EXPT_DPATH/_horologic_sv_sweep" \
    --devices="0,1,2,3" --tmux_workers=8 \
    --backend=tmux --queue_name "_horologic_sv_sweep" \
    --pipeline=bas_building_and_depth_vali \
    --skip_existing=1 \
    --run=0


# Pull out baseline tables
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_horologic_sv_sweep
    " \
    --output_dpath="$DVC_EXPT_DPATH/_horologic_sv_sweep/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        #- sv_poly_eval
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 1
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
    " \
    --rois="KR_R002,PE_R001,NZ_R001,CH_R001,KR_R001,AE_R001,BR_R002,BR_R004"
    #--rois="PE_R001"

    #--rois="KR_R002"

### Helper to build SV crop dataset submatrix
python -c "if 1:
    import ubelt as ub
    regions = ['KR_R001', 'KR_R002', 'AE_R001', 'BR_R002', 'CH_R001', 'NZ_R001', 'PE_R001', 'BR_R004']
    feature_code = 'EI2LMSC'
    dollar = chr(36)
    dvc_var1 = dollar + 'DVC_DATA_DPATH'
    dvc_var2 = dollar + 'DVC_HDD_DATA_DPATH'
    for region_id in regions:
        print(ub.indent(ub.codeblock(
            f'''
            - bas_pxl.test_dataset: {dvc_var1}/Drop7-MedianNoWinter10GSD/combo_imganns-{region_id}_{feature_code}.kwcoco.zip
              sv_crop.crop_src_fpath: {dvc_var2}/Aligned-Drop7/{region_id}/imgonly-{region_id}.kwcoco.zip
          '''), ' ' * 8))
"



HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation --params="
    matrix:
        #####################
        ## SC PIXEL PARAMS ##
        #####################

        sc_pxl.test_dataset:
          - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip
          #- $HIRES_DVC_DATA_DPATH/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip
        sc_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3
        sc_pxl.input_space_scale: 2GSD
        sc_pxl.window_space_scale: 2GSD
        sc_pxl.output_space_scale: 2GSD
        #sc_pxl.time_span: 6m
        #sc_pxl.time_sampling: auto
        #sc_pxl.time_steps: 12
        #sc_pxl.chip_dims: auto
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
        sc_poly.resolution: 2GSD
        sc_poly.min_area_square_meters: 7200

        ##########################
        ## SC POLY EVAL PARAMS  ##
        ##########################

        sc_poly_eval.true_site_dpath: $HIRES_DVC_DATA_DPATH/annotations/drop6/site_models
        sc_poly_eval.true_region_dpath: $HIRES_DVC_DATA_DPATH/annotations/drop6/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 1
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0
    " \
    --pipeline=sc \
    --root_dpath="$DVC_EXPT_DPATH/_drop7_baseline_sc_truth" \
    --queue_name "_drop7_baseline_sc_truth" \
    --devices=",1" \
    --backend=tmux --tmux_workers=6 \
    --cache=1 --skip_existing=1 --run=1


# Dzyne agg
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/aggregate_results/dzyne/bas_poly_eval_2023-07-10T131639-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/dzyne/bas_poly_eval_2023-07-10T164254-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/dzyne/sv_poly_eval_2023-07-10T164254-5.csv.zip
    " \
    --output_dpath="$DVC_EXPT_DPATH/_tmp/_dzyne_csvs/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        #- sv_poly_eval
        - bas_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - resolved_params.sv_depth_filter.threshold
            - resolved_params.sv_depth_score.model_fpath
            - resolved_params.bas_poly.thresh
            - resolved_params.bas_pxl.channels
    " \
    --stdout_report="
        top_k: 1
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
    " \
    --rois="KR_R002,PE_R001,NZ_R001,CH_R001,KR_R001,AE_R001,BR_R002,BR_R004" --embed



# Change the "pipeline" to the one you ran MLOPs with.
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/YOUR_MLOPS_OUTPUT_DIR
    " \
    --export_tables=True \
    --output_dpath="$DVC_EXPT_DPATH/YOUR_MLOPS_OUTPUT_DIR/aggregate"



### REPRODUCE DZYNE


# Pre Agg

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_mlops_eval10_baseline
        - $DVC_EXPT_DPATH/_drop7_nowinter_baseline_joint_bas_sc
        - $DVC_EXPT_DPATH/_drop7_nowinter_baseline
        - $DVC_EXPT_DPATH/_split6_toothbrush_meanyear
        - $DVC_EXPT_DPATH/_timekernel_test_drop4
        - $DVC_EXPT_DPATH/_toothbrush_eval_split6_MeanYear10GSD
        - $DVC_EXPT_DPATH/_toothbrush_split6_landcover_MeanYear10GSD-V2
    " \
    --output_dpath="$DVC_EXPT_DPATH/_preagg_toothbrush/aggregate" \
    --export_tables=True

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_namek_sv_sweep
        - $DVC_EXPT_DPATH/_timekernel_test_drop4
        - $DVC_EXPT_DPATH/_toothbrush_drop7_nowinter
        - $DVC_EXPT_DPATH/_namek_split6_landcover_MeanYear10GSD-V2
        - $DVC_EXPT_DPATH/_namek_split1_eval_filter1_MeanYear10GSD-V2
        - $DVC_EXPT_DPATH/_namek_split1_eval_filter1_MeanYear10GSD
        - $DVC_EXPT_DPATH/_namek_split1_eval_filter1
        - $DVC_EXPT_DPATH/_namek_preeval12
        - $DVC_EXPT_DPATH/_namek_eval
        - $DVC_EXPT_DPATH/_mlops_eval10_baseline
    " \
    --output_dpath="$DVC_EXPT_DPATH/_preagg_namek/aggregate" \
    --export_tables=True

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_horologic_sv_sweep
    " \
    --output_dpath="$DVC_EXPT_DPATH/_horologic_sv_sweep/aggregate" \
    --export_tables=True


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_ooo_split2_eval_filter1_MeanYear10GSD-V2
    " \
    --output_dpath="$DVC_EXPT_DPATH/_ooo_split2_eval_filter1_MeanYear10GSD-V2/aggregate" \
    --export_tables=True


# Big aggregation
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/aggregate_results/dzyne/bas_poly_eval_2023-07-10T131639-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/dzyne/bas_poly_eval_2023-07-10T164254-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/dzyne/sv_poly_eval_2023-07-10T164254-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/horologic/bas_poly_eval_2023-07-10T155903-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/horologic/sv_poly_eval_2023-07-10T155903-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/namek/bas_poly_eval_2023-04-19T113433-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/namek/bas_poly_eval_2023-07-10T161857-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/namek/bas_pxl_eval_2023-04-19T113433-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/namek/bas_pxl_eval_2023-07-10T161857-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/namek/sv_poly_eval_2023-07-10T161857-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/toothbrush/bas_poly_eval_2023-04-19T105718-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/toothbrush/bas_poly_eval_2023-07-10T150132-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/toothbrush/bas_pxl_eval_2023-04-19T105718-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/toothbrush/bas_pxl_eval_2023-07-10T150132-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/toothbrush/sv_poly_eval_2023-04-19T105718-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/toothbrush/sv_poly_eval_2023-07-10T150132-5.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/uconn/COLD_candidates_0705.zip
        #- $DVC_EXPT_DPATH/aggregate_results/uconn/bas_poly_eval_2023-06-20T140324-5_COLD.csv
        - $DVC_EXPT_DPATH/aggregate_results/wu/bas_pxl_eval_2023-07-11T180910+0.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/wu/bas_poly_eval_2023-07-11T180910+0.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/wu/bas_pxl_eval_2023-07-11T181515+0.csv.zip
        - $DVC_EXPT_DPATH/aggregate_results/wu/bas_poly_eval_2023-07-11T181515+0.csv.zip
    " \
    --output_dpath="$DVC_EXPT_DPATH/_bigagg/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sv_poly_eval
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - resolved_params.sv_depth_filter.threshold
            - resolved_params.sv_depth_score.model_fpath
            - resolved_params.bas_poly.thresh
            - resolved_params.bas_pxl.channels
    " \
    --stdout_report="
        top_k: 1
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
    " \
    --rois="KR_R002,PE_R001,NZ_R001,CH_R001,KR_R001,AE_R001,BR_R002,BR_R004"
    #--rois="KR_R002,KR_R001"
    #--rois="KR_R002"
    #--rois="PE_R001"



# Big aggregation
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        /home/joncrall/quicklinks/toothbrush_smart_expt_dvc/aggregate_results/wu/
/home/joncrall/quicklinks/toothbrush_smart_expt_dvc/aggregate_results/wu/
    " \
    --output_dpath="$DVC_EXPT_DPATH/_bigagg/aggregate" \
    --resource_report=0 \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - resolved_params.sv_depth_filter.threshold
            - resolved_params.sv_depth_score.model_fpath
            - resolved_params.bas_poly.thresh
            - resolved_params.bas_pxl.channels
    " \
    --stdout_report="
        top_k: 1
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
    " \
    --rois="KR_R002,PE_R001,NZ_R001,CH_R001,KR_R001,AE_R001,BR_R002,BR_R004"
    #--rois="KR_R002,KR_R001"
    #--rois="KR_R002"
    #--rois="PE_R001"

    #--eval_nodes="
    #    - sv_poly_eval
    #    - bas_poly_eval
    #    #- bas_pxl_eval
    #" \

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
# Change the "pipeline" to the one you ran MLOPs with.
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/YOUR_MLOPS_OUTPUT_DIR
    " \
    --export_tables=True \
    --output_dpath="$DVC_EXPT_DPATH/YOUR_MLOPS_OUTPUT_DIR/aggregate"



# PREEVAL 14 BAS+SV Grid
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.utils.simple_dvc request \
    "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt \
    "$DVC_EXPT_DPATH"/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt \
    "$DVC_EXPT_DPATH"/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt \
    "$DVC_EXPT_DPATH"/models/fusion/uconn/D7-MNW10_coldL8S2-cv-a0-a1-b1-c1-rmse-split6_eval11_Norm_lr1e4_bs48_focal/epoch=16-step=374.pt \
    "$DVC_EXPT_DPATH"/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt

geowatch schedule --params="
    matrix:
        bas_pxl.package_fpath:
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            - $DVC_EXPT_DPATH/models/fusion/uconn/D7-MNW10_coldL8S2-cv-a0-a1-b1-c1-rmse-split6_eval11_Norm_lr1e4_bs48_focal/epoch=16-step=374.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.35
            - 0.3625
            - 0.375
            - 0.375
            - 0.38125
            - 0.3875
            - 0.39375
            - 0.4
            - 0.425
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop6/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop6/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
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
            - 0.05
            - 0.10
            - 0.13
            - 0.15
        sv_dino_filter.start_max_score: 1.0
        sv_dino_filter.box_score_threshold: 0.01
        sv_dino_filter.box_isect_threshold: 0.1

        sv_depth_score.enabled: 1
        sv_depth_score.model_fpath:
            - $DVC_EXPT_DPATH/models/depth_pcd/basicModel2.h5
        sv_depth_filter.threshold:
            - 0.10
            - 0.15
            - 0.25
    submatrices:
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/KR_R001/imgonly-KR_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-AE_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/AE_R001/imgonly-AE_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R002_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/BR_R002/imgonly-BR_R002.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-CH_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/CH_R001/imgonly-CH_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-NZ_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/NZ_R001/imgonly-NZ_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-PE_R001_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/PE_R001/imgonly-PE_R001.kwcoco.zip
        - bas_pxl.test_dataset: $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-BR_R004_EI2LMSC.kwcoco.zip
          sv_crop.crop_src_fpath: $DVC_HDD_DATA_DPATH/Aligned-Drop7/BR_R004/imgonly-BR_R004.kwcoco.zip
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval13_sweep" \
    --devices="0,1" --tmux_workers=8 \
    --backend=tmux --queue_name "_namek_eval13_sweep" \
    --pipeline=bas_building_and_depth_vali \
    --skip_existing=1 \
    --run=1

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_namek_eval13_sweep
    " \
    --output_dpath="$DVC_EXPT_DPATH/_namek_eval13_sweep/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sv_poly_eval
        #- bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 13
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --rois="KR_R002,NZ_R001,CH_R001,KR_R001"

    #--query='
    #    (df["param_hashid"] == "kaniujrrtybn")
    #' \
    #--rois="KR_R002,PE_R001,NZ_R001,CH_R001,KR_R001,AE_R001,BR_R002,BR_R004"
    #--rois="PE_R001"

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_namek_eval13_sweep_old_metrics
    " \
    --output_dpath="$DVC_EXPT_DPATH/_namek_eval13_sweep_old_metrics/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sv_poly_eval
        #- bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 13
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --rois="KR_R002,NZ_R001,CH_R001,KR_R001"



# PREEVAL 14 AC Grid

python -m watch.mlops.manager "list" --dataset_codes Drop7-Cropped2GSD

HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
TRUTH_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation --params="
    matrix:
        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.test_dataset:
          - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip
          - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip
          - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/CH_R001/CH_R001.kwcoco.zip
          - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/NZ_R001/NZ_R001.kwcoco.zip
        sc_pxl.package_fpath:
            # Shortlist
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch74_step6450.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch20_step903.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch217_step9374.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch326_step28122.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch486_step20921.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            # Tested in KR1
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch1_step172.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch2_step258.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch74_step6450.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch0_step86.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch27_step700.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch19_step500.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch25_step650.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch0_step0.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch28_step710.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch22_step575.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch10_step275.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch398_step17157.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch468_step20167.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch486_step20921.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch268_step11567.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch389_step16770.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch299_step12900.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch76_step3311.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch545_step23478.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch1_step86.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch4_step215.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch2_step129.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch538_step23177.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch0_step43.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch5_step249.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch448_step19307.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch3_step172.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch20_step903.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch77_step6708.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch75_step6536.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch78_step6794.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch48_step4214.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch60_step5246.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch336_step28982.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch326_step28122.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch302_step26058.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch202_step17458.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch216_step18662.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch374_step16125.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch272_step11739.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch217_step9374.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch129_step5590.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt

        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3
        #sc_pxl.input_space_scale: 2GSD
        #sc_pxl.window_space_scale: 2GSD
        #sc_pxl.output_space_scale: 2GSD
        #sc_pxl.time_span: 6m
        #sc_pxl.time_sampling: auto
        #sc_pxl.time_steps: 12
        #sc_pxl.chip_dims: auto
        sc_pxl.set_cover_algo: null
        sc_pxl.resample_invalid_frames: 3
        sc_pxl.observable_threshold: 0.0
        sc_pxl.mask_low_quality: true
        sc_pxl.drop_unused_frames: true
        sc_pxl.num_workers: 12
        sc_pxl.batch_size: 1
        sc_pxl.write_workers: 0

        ########################
        ## AC/SC POLY PARAMS  ##
        ########################

        sc_poly.thresh: 0.07
        sc_poly.boundaries_as: polys
        #sc_poly.resolution: 2GSD
        sc_poly.min_area_square_meters: 7200

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $TRUTH_DVC_DATA_DPATH/annotations/drop6/site_models
        sc_poly_eval.true_region_dpath: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 1
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0

    submatrices:
        - sc_pxl.test_dataset: $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip
          sc_poly.site_summary: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models/KR_R001.geojson
        - sc_pxl.test_dataset: $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip
          sc_poly.site_summary: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models/KR_R002.geojson
        - sc_pxl.test_dataset: $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/CH_R001/CH_R001.kwcoco.zip
          sc_poly.site_summary: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models/CH_R001.geojson
        - sc_pxl.test_dataset: $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/NZ_R001/NZ_R001.kwcoco.zip
          sc_poly.site_summary: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models/NZ_R001.geojson
    " \
    --pipeline=sc \
    --root_dpath="$DVC_EXPT_DPATH/_toothbrush_preeval14_ac_eval_v2" \
    --queue_name "_toothbrush_preeval14_ac_eval_v2" \
    --devices="0,1" \
    --backend=tmux --tmux_workers=6 \
    --cache=1 --skip_existing=1 --run=1

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_toothbrush_preeval14_ac_eval_v2
    " \
    --output_dpath="$DVC_EXPT_DPATH/_toothbrush_preeval14_ac_eval_v2/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
        #- sv_poly_eval
        #- bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
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
    " \
    --rois="KR_R002,NZ_R001,CH_R001,KR_R001"


# EVAL 15 BAS GRID
#
python -m watch.mlops.manager "pull packages" --dataset_codes Drop7-MedianNoWinter10GSD-NoMask --yes
python -m watch.mlops.manager "list packages" --dataset_codes Drop7-MedianNoWinter10GSD-NoMask --yes
python -m watch.mlops.manager "list packages" --dataset_codes Drop7-MedianNoWinter10GSD Drop7-MedianNoWinter10GSD-NoMask --yes

geowatch manager "pull packages" --dataset_codes Drop6-MeanYear10GSD-V2 --yes
geowatch manager "list packages" --dataset_codes Drop6-MeanYear10GSD-V2 --yes

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
sdvc request "
    - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
    - $DVC_EXPT_DPATH/models/fusion/uconn/D7-MNW10_coldL8S2-cv-a0-a1-b1-c1-rmse-split6_eval11_Norm_lr1e4_bs48_focal/epoch=16-step=374.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch12_step2223.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch9_step1710.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch11_step2052.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch13_step2394.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch0_step131.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch7_step126.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch3_step64.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch1_step32.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch0_step16.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch2_step48.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch5_step96.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch42_step3698.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch33_step2924.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch28_step2494.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch25_step2236.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V61/Drop7-MedianNoWinter10GSD_bgr_split6_V61_epoch359_step15480.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch94_step4069.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch301_step12986.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch195_step8406.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch194_step8385.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60/Drop7-MedianNoWinter10GSD_landcover_invar_cold_sam_scratch_split6_V60_epoch148_step6407.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch28_step2494.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch25_step2236.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch42_step3698.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch7_step688.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch49_step4300.pt

    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch28_step2494.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch49_step4300.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch25_step2236.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch7_step688.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch42_step3698.pt

"


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)
geowatch schedule --params="
    pipeline: joint_bas_sv_sc

    # Convinience argument
    smart_highres_bundle: $DVC_HDD_DATA_DPATH/Aligned-Drop7

    matrix:
        bas_pxl.package_fpath:
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch10_step176.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch7_step126.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch28_step2494.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R002/imganns-KR_R002.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CH_R001/imganns-CH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/NZ_R001/imganns-NZ_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R001/imganns-KR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CN_C000/imganns-CN_C000.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.4
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
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

        ########################
        ## AC/SC CROP PARAMS  ##
        ########################
        sc_crop.target_gsd: 8GSD
        sc_crop.minimum_size: '128x128@8GSD'
        sc_crop.force_min_gsd: 8GSD
        sc_crop.context_factor: 1.5
        sc_crop.rpc_align_method: affine_warp
        sc_crop.sensor_to_time_window:
            - 'S2: 1month'

        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3
        sc_pxl.input_space_scale: 8GSD
        sc_pxl.window_space_scale: 8GSD
        sc_pxl.output_space_scale: 8GSD
        #sc_pxl.time_span: 6m
        #sc_pxl.time_sampling: auto
        #sc_pxl.time_steps: 12
        #sc_pxl.chip_dims: auto
        sc_pxl.set_cover_algo: null
        sc_pxl.resample_invalid_frames: 3
        sc_pxl.observable_threshold: 0.0
        sc_pxl.mask_low_quality: false
        sc_pxl.drop_unused_frames: true
        #sc_pxl.num_workers: 12
        #sc_pxl.batch_size: 1
        sc_pxl.write_workers: 0

        ########################
        ## AC/SC POLY PARAMS  ##
        ########################

        sc_poly.thresh: 0.07
        sc_poly.boundaries_as: polys
        sc_poly.resolution: 8GSD
        sc_poly.min_area_square_meters: 7200

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 1
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval15_bas_nomask" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_namek_eval15_bas_nomask" \
    --skip_existing=1 \
    --run=1

##### MASKED VARIANT
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)
geowatch schedule --params="
    pipeline: joint_bas_sv_sc

    # Convinience argument
    smart_highres_bundle: $DVC_HDD_DATA_DPATH/Aligned-Drop7

    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch10_step176.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch7_step126.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V70/Drop7-MedianNoWinter10GSD_bgrn_split6_V70_epoch28_step2494.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/imganns-KR_R002.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/imganns-CH_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/imganns-NZ_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/imganns-KR_R001.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-iMERIT/CN_C000/imganns-CN_C000.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.4
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        sv_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
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
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval15_bas_yesmask" \
    --devices="0,1" --tmux_workers=4 \
    --backend=tmux --queue_name "_namek_eval15_bas_yesmask" \
    --skip_existing=1 \
    --run=1
    #--pipeline=bas_building_and_depth_vali \


# NO MASK
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_namek_eval15_bas_nomask
    " \
    --output_dpath="$DVC_EXPT_DPATH/_namek_eval15_bas_nomask/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sv_poly_eval
        #- bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 13
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --rois="KR_R002,CN_C000"

    #--rois="KR_R002,NZ_R001,CH_R001,KR_R001,CN_C000"

# VS MASKED
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_namek_eval15_bas_yesmask
    " \
    --output_dpath="$DVC_EXPT_DPATH/_namek_eval15_bas_yesmask/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sv_poly_eval
        #- bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 13
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 0
        show_csv: 0
    " \
    --rois="KR_R002,CN_C000"

    #--rois="KR_R002,NZ_R001,CH_R001,KR_R001,CN_C000"

# Both
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas_building_and_depth_vali \
    --target "
        - $DVC_EXPT_DPATH/_namek_eval15_bas_nomask
        - $DVC_EXPT_DPATH/_namek_eval15_bas_yesmask
    " \
    --output_dpath="$DVC_EXPT_DPATH/_namek_eval15_bas_nomask/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sv_poly_eval
        #- bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 13
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002"

    #,NZ_R001,CH_R001,KR_R001,CN_C000"



simple_dvc request "
    - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
"

# Toothbrush nomask
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=hdd)
geowatch schedule --params="
    pipeline: full

    # Convinience argument
    smart_highres_bundle: $DVC_HDD_DATA_DPATH/Aligned-Drop7

    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R002/imganns-KR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CN_C000/imganns-CN_C000.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/BR_R002/imganns-BR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/CH_R001/imganns-CH_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/NZ_R001/imganns-NZ_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-NoMask/KR_R001/imganns-KR_R001.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.37
            #- 0.3875
            #- 0.4
            #- 0.41
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        bas_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
        bas_poly_viz.enabled: 0
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
        ## AC/SC CROP PARAMS  ##
        ########################
        sc_crop.target_gsd: 2GSD
        sc_crop.minimum_size: '128x128@2GSD'
        sc_crop.force_min_gsd: 2GSD
        sc_crop.context_factor: 1.0
        sc_crop.rpc_align_method: affine_warp
        sc_crop.sensor_to_time_window:
            - 'S2: 1month'
            #- 'S2: 3month'
            #- 'S2: 0.5month'

        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.package_fpath:
            #- $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            - $HOME/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop7-Cropped2GSD/runs/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/lightning_logs/version_3/checkpoints/epoch=0-step=342-val_loss=4.648.ckpt.pt
        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3
        sc_pxl.input_space_scale: 2GSD
        sc_pxl.window_space_scale: 2GSD
        sc_pxl.output_space_scale: 2GSD
        sc_pxl.chip_dims: '128,128'
        #sc_pxl.time_span: 6m
        #sc_pxl.time_sampling: auto
        #sc_pxl.time_steps: 12
        #sc_pxl.chip_dims: auto
        sc_pxl.set_cover_algo: null
        sc_pxl.resample_invalid_frames: 3
        sc_pxl.observable_threshold: 0.0
        sc_pxl.mask_low_quality: false
        sc_pxl.drop_unused_frames: true
        #sc_pxl.num_workers: 12
        #sc_pxl.batch_size: 1
        sc_pxl.write_workers: 0

        ########################
        ## AC/SC POLY PARAMS  ##
        ########################

        sc_poly.thresh:
            #- 0.00001
            #- 0.01
            #- 0.05
            #- 0.08
            - 0.07
            #- 0.09
            - 0.10
            - 0.15
            - 0.20
            #- 0.25
            - 0.30
            #- 0.325
            #- 0.35
            #- 0.375
            #- 0.40
        sc_poly.boundaries_as: polys
        sc_poly.resolution: 8GSD
        sc_poly.min_area_square_meters: 7200

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/drop7/site_models
        sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/drop7/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_crop.enabled: 1
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 1
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_toothbrush_eval15_bas_nomask" \
    --devices="0," --tmux_workers=1 \
    --backend=tmux --queue_name "_toothbrush_eval15_bas_nomask" \
    --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=full \
    --target "
        - $DVC_EXPT_DPATH/_toothbrush_eval15_bas_nomask
    " \
    --output_dpath="$DVC_EXPT_DPATH/_namek_eval15_bas_nomask/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
        #- sv_poly_eval
        #- bas_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sv_depth_filter.threshold
            - params.sv_depth_score.model_fpath
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 40
        per_group: .inf
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002"


# TEST SRI's MODELS ON YARDRAT
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
# (cd "$DVC_EXPT_DPATH"; git pull)
sdvc request --paths="
    #- $DVC_EXPT_DPATH/models/fusion/uconn/D7-MNW10_coldL8S2-cv-a0-a1-b1-c1-rmse-split6_eval11_Norm_lr1e4_bs48_focal/epoch=16-step=374.pt
    # - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62/Drop7-MedianNoWinter10GSD_bgr_cold_split6_V62_epoch359_step15480.pt

    #- $DVC_EXPT_DPATH/models/wu/bas/wu_bas_mae_10GSD_epoch=31-step_checkpint=352.pt
    #- $DVC_EXPT_DPATH/models/wu/bas/wu_bas_mae_10GSD_epoch=49-step_scratch=550.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
    #- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
    - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
    - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt
    "

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=ssd)
DVC_HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
geowatch schedule --params="
    pipeline: bas

    # Convinience argument
    # smart_highres_bundle: $DVC_HDD_DATA_DPATH/Aligned-Drop7

    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/wu/bas/wu_bas_mae_10GSD_epoch=31-step_checkpint=352.pt
            - $DVC_EXPT_DPATH/models/wu/bas/wu_bas_mae_10GSD_epoch=49-step_scratch=550.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-iMERIT/CN_C000/combo_imganns-CN_C000_EI2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-iMERIT/KW_C001/combo_imganns-KW_C001_EI2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-iMERIT/CO_C001/combo_imganns-CO_C001_EI2L.kwcoco.zip
            - $DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            - 0.35
            - 0.375
            - 0.4
            - 0.425
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
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
    --root_dpath="$DVC_EXPT_DPATH/_yardrat_bas_test" \
    --devices="0," --tmux_workers=4 \
    --backend=tmux --queue_name "_yardrat_bas_test" \
    --skip_existing=1 \
    --run=1

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_yardrat_bas_test
    " \
    --output_dpath="$DVC_EXPT_DPATH/_yardrat_bas_test/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.bas_poly.thresh
    " \
    --stdout_report="
        top_k: 11
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002,CN_C000,KW_C001,CO_C001"
