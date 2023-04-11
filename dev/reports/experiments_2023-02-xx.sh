__doc__="
SeeAlso:
    ~/code/watch/dev/poc/prepare_time_combined_dataset.py


"

#### Eval9 Models (Namek)

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=hdd)
python -m watch.cli.split_videos \
    --src "$DVC_DATA_DPATH/Drop4-BAS/data_train.kwcoco.json" \
          "$DVC_DATA_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
    --io_workers=4 \
    --dst "$DVC_DATA_DPATH/Drop4-BAS/{src_name}_{video_name}.kwcoco.zip"


DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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



DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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



DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

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
    --devices="0,1" --queue_size=8 --print_commands=0 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=1






# NAMEK Pixel Eval
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)

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
    --devices="0,1" --queue_size=2 --print_commands=0 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=1



DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0,1" --queue_size=2 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=1



DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0,1" --queue_size=2 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=0



#### QUICK Temporal Sampling Checks


DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0,1" --queue_size=4 \
    --backend=tmux --queue_name "_timekernel_test_drop4" \
    --pipeline=bas --skip_existing=1 \
    --print_varied=0  \
    --run=1


DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.cluster_sites \
        --src "$DVC_DATA_DPATH/annotations/drop6/region_models/KR_R002.geojson" \
        --dst_dpath "$DVC_DATA_DPATH"/ValiRegionSmall/geojson \
        --draw_clusters True

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
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


DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=hdd)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0,1" --queue_size=4 \
    --backend=tmux --queue_name "_namek_split1_eval_small" \
    --pipeline=bas --skip_existing=1 \
    --run=1


# SPLIT 1 - filter1 analysis
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0,1" --queue_size=4 \
    --backend=tmux --queue_name "_namek_split1_eval_filter1_MeanYear10GSD" \
    --pipeline=bas --skip_existing=1 \
    --run=1


# ###################
# SPLIT 2 - SMALL TEST
# ###################

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.cluster_sites \
        --src "$DVC_DATA_DPATH/annotations/drop6/region_models/NZ_R001.geojson" \
        --dst_dpath "$DVC_DATA_DPATH"/ValiRegionSmall/geojson/NZ_R001 \
        --draw_clusters True

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
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


DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0,1" --queue_size=4 \
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
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0," --queue_size=1 \
    --backend=serial --queue_name "_toothbrush_eval_split6_MeanYear10GSD" \
    --pipeline=bas --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
    --devices="0,1" --queue_size=8 \
    --backend=tmux --queue_name "_split6_toothbrush_meanyear" \
    --pipeline=bas --skip_existing=1 \
    --run=1

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
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
        macro_analysis: 0
        analyze: 0
        reference_region: final
    "
    #--plot_params=True \
    #--output_dpath="$DVC_EXPT_DPATH"/_split6_toothbrush_meanyear/_aggregate


DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2/Drop6_TCombo1Year_BAS_10GSD_split6_V42_cont2_epoch3_step941.pt
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
        bas_poly.thresh:
            - 0.35
            - 0.38
            - 0.4
        bas_poly.inner_window_size:
            - 1y
            - null
        bas_poly.inner_agg_fn:
            - mean
        bas_poly.norm_ord:
            - 1
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
    --root_dpath="$DVC_EXPT_DPATH/_mlops_output" \
    --devices="0,1" --queue_size=8 \
    --backend=slurm --queue_name "_mlops_output" \
    --pipeline=bas --skip_existing=1 \
    --print_commands=1
    --run=0



# SITE VISIT 2022-04 SPLIT 1 Analysis
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation --params="
    matrix:
        bas_pxl.package_fpath:
            - $HOME/code/watch/dev/reports/split1_models_filter1.pt
        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R002.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-KR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-NZ_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-CH_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BH_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R001.kwcoco.zip
            #- $DVC_DATA_DPATH/Drop6-MeanYear10GSD/imganns-BR_R002.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims:
            - auto
        bas_pxl.time_span:
            - auto
        bas_pxl.time_sampling:
            - auto
        bas_poly.input_resolution:
            - 10GSD
        bas_poly.output_resolution:
            - 10GSD
        bas_poly.window_resolution:
            - 10GSD
        bas_poly.thresh:
            - 0.4
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
    --devices="0,1" --queue_size=4 \
    --backend=tmux --queue_name "_namek_split1_eval_filter1_MeanYear10GSD" \
    --pipeline=bas --skip_existing=1 \
    --run=1
