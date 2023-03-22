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
        --dst_dpath $DVC_DATA_DPATH/ValiRegionSmall/geojson \
        --draw_clusters True

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.coco_align \
    --src $DVC_DATA_DPATH/Drop6/combo_imganns-KR_R002_L.kwcoco.json \
    --dst $DVC_DATA_DPATH/ValiRegionSmall/small_KR_R002_odarcigm.kwcoco.zip \
    --regions $DVC_DATA_DPATH/ValiRegionSmall/geojson/SUB_KR_R002_n007_odarcigm.geojson \
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
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch27_step28672.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch28_step29696.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch21_step22528.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch26_step27648.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch32_step33792.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch34_step35840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch29_step30720.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch24_step25600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch37_step38225.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch31_step32768.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch33_step34816.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_v0_epoch0_step277.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_epoch186_step382976.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_epoch0_step232.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_epoch0_step277.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_v0_epoch0_step232.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2_epoch311_step637457.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch80_step33210.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch21_step9020.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch86_step35670.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch82_step34030.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch40_step16810.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch80_step33210.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch21_step9020.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch71_step29520.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch0_step410.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch23_step9840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch3_step13108.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch0_step25.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch90_step37310.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch23_step9840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch40_step16810.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v1_epoch1_step6554.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch90_step37310.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch86_step35670.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch84_step34850.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch0_step3277.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch0_step410.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch71_step29520.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch2_step9831.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch3_step13108.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch89_step36900.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch84_step34850.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch89_step36900.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch82_step34030.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch32_step13530.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch2_step9831.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_epoch1_step6554.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch0_step25.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v0_epoch32_step13530.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5_v1_epoch0_step3277.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V4/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V4_epoch25_step53248.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch8_step29493.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch2_step9831.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch1_step6554.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch6_step22939.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch11_step36914.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v1_epoch0_step3277.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch9_step32770.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch3_step13108.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch9_step32770.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch7_step26216.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch1_step6554.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch4_step16385.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch7_step26216.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch3_step13108.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch8_step29493.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch6_step22939.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch0_step3277.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_v0_epoch11_step36914.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch4_step16385.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4_epoch2_step9831.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch16_step17408.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch29_step30029.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch7_step8192.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch4_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch5_step6144.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch19_step20480.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch11_step12288.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch4_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch28_step29696.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch19_step20480.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch29_step30029.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch2_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch11_step12288.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch3_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch16_step17408.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch28_step29696.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v1_epoch7_step8192.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch5_step6144.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_v0_epoch3_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_tune_3GSD_allheads/Drop6_BAS_tune_3GSD_allheads_epoch2_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v1_epoch1_step5462.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch1_step5462.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch0_step2731.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch2_step8193.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch0_step2731.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch12_step35503.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v1_epoch9_step27310.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch14_step40965.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch11_step32772.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch3_step10924.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch5_step16386.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch10_step30041.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch3_step10924.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch12_step35503.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch5_step16386.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch10_step30041.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_epoch9_step27310.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch14_step40965.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch11_step32772.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2_v0_epoch2_step8193.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch98_step101376.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch22_step23552.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch0_step58.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch122_step125952.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch0_step677.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch30_step31744.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch72_step74752.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch30_step31744.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch43_step45056.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch72_step74752.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch53_step55296.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch0_step677.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch159_step163840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch22_step23552.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch53_step55296.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch98_step101376.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch85_step88064.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch85_step88064.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch0_step58.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch0_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch116_step119808.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch43_step45056.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch122_step125952.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_epoch0_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch159_step163840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch116_step119808.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch93_step110074.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch159_step187360.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch70_step83141.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch53_step63234.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch136_step160427.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch107_step126468.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch57_step67918.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch91_step107732.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch56_step66747.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch29_step30720.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch63_step65536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch7_step8192.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v1_epoch0_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v1_epoch18_step19456.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch35_step36864.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch3_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch26_step27648.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch66_step68608.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch73_step74777.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch14_step15360.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch4_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v1_epoch6_step7168.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch57_step59392.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch4_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch57_step59392.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch8_step9216.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch69_step71680.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch7_step8192.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch69_step71680.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch8_step9216.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch26_step27648.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch18_step19456.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v1_epoch14_step15360.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch35_step36864.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch3_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch39_step40960.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch29_step30720.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch63_step65536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch6_step7168.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch5_step6144.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v1_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch39_step40960.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch5_step6144.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch66_step68608.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_epoch0_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_v0_epoch73_step74777.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch20_step57351.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch119_step327720.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch36_step101047.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch21_step60082.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch39_step54640.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v1_epoch3_step10924.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch0_step259.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch43_step59666.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch30_step42346.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch25_step35516.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch17_step24588.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch32_step45078.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch120_step328801.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch116_step319527.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch9_step13660.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch39_step54640.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch36_step101047.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch43_step59666.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch119_step327720.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch0_step255.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch24_step34150.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch0_step255.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch0_step1619.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch25_step35516.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch116_step319527.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch37_step51908.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch33_step46444.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch0_step2731.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch17_step24588.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch39_step109240.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch32_step45078.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch0_step711.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch39_step109240.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch120_step328801.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch3_step10924.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v1_epoch0_step2731.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch33_step46444.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch7_step21848.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch30_step42346.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch4_step13655.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch4_step13655.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch9_step13660.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch0_step711.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch24_step34150.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch0_step259.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch0_step1619.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch21_step60082.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch7_step21848.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch20_step57351.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1_epoch37_step51908.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V3_singlehead/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V3_singlehead_epoch31_step65536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V3_singlehead/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V3_singlehead_epoch50_step103455.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V3_singlehead/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V3_singlehead_v0_epoch50_step103455.pt

            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step23012.pt.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step24.pt.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step7501.pt.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch149_step76800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch1_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch98_step50688.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch82_step42496.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch79_step40960.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch143_step73728.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch159_step81920.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch2_step1536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step108.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch104_step53760.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch102_step52736.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch140_step72192.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch85_step44032.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch3_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch81_step41984.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch4_step2560.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch78_step40448.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v1_epoch0_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch138_step71168.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch86_step44544.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch83_step43008.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch76_step39424.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGR_V4/Drop4_BAS_2022_12_15GSD_BGR_V4_v0_epoch0_step1354.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch7_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch0_step7.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch56_step29184.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch1_step1969.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch13_step7168.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch49_step25600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch1_step706.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch0_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch5_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch31_step16384.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch0_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch29_step15360.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch3_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch47_step24576.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch39_step20480.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch11_step5936.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch28_step14848.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch46_step24064.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch0_step10.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch48_step25088.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch10_step5632.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch4_step2560.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v0_epoch9_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v1_epoch2_step1536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V7/Drop4_BAS_10GSD_BGRNSH_invar_V7_v1_epoch1_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6_v0_epoch18_step55860.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6_v0_epoch0_step5578.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch21_step11264.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch4_step2560.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch29_step15360.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch13_step7168.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch16_step8704.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch10_step5632.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch146_step75264.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v1_epoch2_step1536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch6_step3584.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch120_step61952.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch117_step60416.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch0_step500.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch131_step67584.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch123_step63488.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v1_epoch0_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch90_step46592.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch95_step49152.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch33_step17408.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch30_step15872.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch7_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch14_step7680.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch23_step12288.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch103_step53248.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch9_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch8_step4608.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch159_step81920.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v1_epoch3_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v0_epoch124_step64000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_v1_epoch1_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch28_step29696.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch25_step26624.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch29_step30720.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch0_step152.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch4_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch24_step25600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch12_step13312.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch19_step20480.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch23_step24576.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch26_step27648.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V7/Drop4_BAS_15GSD_BGRNSH_invar_V7_v0_epoch30_step31003.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step302.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch6_step252174.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch5_step233106.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch1_step77702.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch4_step194255.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch3_step155404.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch2_step98789.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v1_epoch2_step116553.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step38851.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch18_step51889.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch21_step60082.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch19_step54620.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch11_step32772.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch0_step1172.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch9_step27310.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch20_step57351.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch14_step40965.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v1_epoch15_step43696.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3_v0_epoch17_step49158.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_BGRNSH_V1/Drop4_TuneV323_BAS_BGRNSH_V1_v0_epoch0_step12.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_BGRNSH_V1/Drop4_TuneV323_BAS_BGRNSH_V1_v0_epoch0_step86.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch71_step73728.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch13_step14336.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch14_step15360.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch33_step34816.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch60_step62464.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch59_step61440.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch28_step29696.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch2_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch67_step69632.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch11_step12288.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch74_step76224.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch49_step51200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch39_step40960.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch37_step38912.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch57_step59392.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch66_step68608.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch9_step10240.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch38_step39936.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch34_step35840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch27_step28672.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch15_step16384.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch12_step13312.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch69_step71680.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch10_step11264.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch16_step17408.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch70_step72704.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12/Drop4_BAS_10GSD_BGRNSH_invar_V12_v0_epoch26_step27648.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch15_step131072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch24_step204800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch29_step245760.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch16_step139264.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch19_step163840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch9_step81920.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch20_step172032.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch27_step229376.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch22_step188416.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1/Drop4_BAS_2022_12_10GSD_BGRN_V11_CONT1_v0_epoch4_step40960.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_V4_v0_epoch0_step307.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch0_step172.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch0_step2.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch4_step2560.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch22_step11776.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch21_step11264.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch19_step10240.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch25_step13312.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch15_step8192.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v1_epoch14_step7680.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch0_step80.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch0_step185.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch27_step14078.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch20_step10752.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch0_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch10_step5632.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch3_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch21_step10752.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch1_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch2_step1536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch26_step13824.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch1_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch18_step9728.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch16_step8704.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch13_step7168.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V10/Drop4_BAS_BGRNSH_invar_V10_v0_epoch17_step9216.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch0_step105.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch7_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch8_step4608.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch0_step155.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch4_step2560.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch0_step162.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v1_epoch0_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch3_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch6_step3584.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v0_epoch0_step213.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v1_epoch5_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v1_epoch1_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_BGRNSH_invar_V7_alt/Drop4_BAS_BGRNSH_invar_V7_alt_v1_epoch2_step1536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch7_step3908.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch3_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch4_step2560.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v1_epoch0_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch2_step1536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step8247.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch5_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch6_step3584.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step4305.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v1_epoch0_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch15_step15697.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch8_step9216.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v1_epoch39_step40960.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch1_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch18_step19456.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch3_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch10_step11264.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch9_step10240.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch5_step6144.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch11_step12288.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch19_step20240.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch7_step8192.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch41_step43008.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch37_step38912.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch0_step20591.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch30_step31744.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch10_step10734.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch40_step41984.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch44_step46014.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v1_epoch0_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch42_step44032.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch0_step39.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch16_step17408.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch4_step2122.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch0_step301.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch4_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch0_step81.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch2_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch28_step29696.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch0_step128.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch35_step36864.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch3_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch25_step26624.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch6_step7168.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch33_step34816.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch12_step13312.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4_v0_epoch2_step1536.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont_v0_epoch4_step5120.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont_v0_epoch2_step3072.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont_v0_epoch1_step2048.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont_v0_epoch3_step4096.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop4-BAS/packages/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont/Drop4_BAS_10GSD_BGRNSH_invar_V12_cont_v1_epoch0_step1024.pt
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


# ###################
# SPLIT 2 - SMALL TEST
# ###################

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.cluster_sites \
        --src "$DVC_DATA_DPATH/annotations/drop6/region_models/NZ_R001.geojson" \
        --dst_dpath $DVC_DATA_DPATH/ValiRegionSmall/geojson/NZ_R001 \
        --draw_clusters True

DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
python -m watch.cli.coco_align \
    --src $DVC_DATA_DPATH/Drop6/combo_imganns-NZ_R001_L.kwcoco.json \
    --dst $DVC_DATA_DPATH/ValiRegionSmall/small_NZ_R001_swnykmah.kwcoco.zip \
    --regions $DVC_DATA_DPATH/ValiRegionSmall/geojson/NZ_R001/SUB_NZ_R001_n031_swnykmah.geojson \
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
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V6/Drop6_BAS_scratch_landcover_10GSD_split2_V6_epoch4_step16385.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V6/Drop6_BAS_scratch_landcover_10GSD_split2_V6_epoch2_step9831.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V6/Drop6_BAS_scratch_landcover_10GSD_split2_V6_epoch3_step13108.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V6/Drop6_BAS_scratch_landcover_10GSD_split2_V6_epoch0_step3277.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V6/Drop6_BAS_scratch_landcover_10GSD_split2_V6_epoch1_step6554.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V6/Drop6_BAS_scratch_landcover_10GSD_split2_V6_epoch14_step49155.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V6/Drop6_BAS_scratch_landcover_10GSD_split2_V6_epoch23_step76331.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch9_step10670.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch48_step52283.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch2_step4800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch8_step9603.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch49_step52283.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch61_step66108.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch4_step7408.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch12_step13871.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch2_step3201.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch0_step1600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch1_step3200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch0_step1067.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch3_step6400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch0_step861.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10/Drop6_BAS_scratch_big_landcover_10GSD_split2_V10_epoch0_step147.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch25_step10400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch104_step42000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch34_step14000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch0_step1.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch38_step15600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch21_step8800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont/Drop6_BAS_scratch_landcover_10GSD_split2_V4_cont_epoch46_step18800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13_epoch3_step6400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13_epoch2_step4800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13_epoch10_step16258.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13_epoch1_step3200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13_epoch0_step1600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13/Drop6_BAS_scratch_big_landcover_10GSD_split2_V13_epoch4_step8000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont_epoch3_step6158.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont_epoch2_step4800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont_epoch0_step1600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont_epoch1_step3200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch184_step4625.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch466_step11675.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch1_step38.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch212_step5325.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch0_step25.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch622_step15575.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch431_step10800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch0_step11.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V23/Drop6_BAS_sits_raw_eGSD_split2_V23_epoch257_step6450.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch74_step1875.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch0_step9.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch64_step1625.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch194_step4875.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch286_step7175.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch71_step1800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch57_step1450.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch0_step65.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_10GSD_split2_V19/Drop6_BAS_sits_raw_10GSD_split2_V19_epoch62_step1575.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V24/Drop6_BAS_sits_raw_eGSD_split2_V24_epoch208_step5225.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V24/Drop6_BAS_sits_raw_eGSD_split2_V24_epoch154_step3875.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V24/Drop6_BAS_sits_raw_eGSD_split2_V24_epoch219_step5500.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V24/Drop6_BAS_sits_raw_eGSD_split2_V24_epoch101_step2550.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V24/Drop6_BAS_sits_raw_eGSD_split2_V24_epoch131_step3300.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V24/Drop6_BAS_sits_raw_eGSD_split2_V24_epoch271_step6776.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V24/Drop6_BAS_sits_raw_eGSD_split2_V24_epoch235_step5900.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_V2/Drop6_BAS_scratch_raw_10GSD_split2_V2_epoch2_step2400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_V2/Drop6_BAS_scratch_raw_10GSD_split2_V2_epoch62_step50000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_V2/Drop6_BAS_scratch_raw_10GSD_split2_V2_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_V2/Drop6_BAS_scratch_raw_10GSD_split2_V2_epoch16_step13600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_V2/Drop6_BAS_scratch_raw_10GSD_split2_V2_epoch45_step36800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_V2/Drop6_BAS_scratch_raw_10GSD_split2_V2_epoch11_step9600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_V2/Drop6_BAS_scratch_raw_10GSD_split2_V2_epoch0_step800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2/Drop6_BAS_scratch_raw_10GSD_split2_epoch20_step16800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2/Drop6_BAS_scratch_raw_10GSD_split2_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2/Drop6_BAS_scratch_raw_10GSD_split2_epoch1_step1600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2/Drop6_BAS_scratch_raw_10GSD_split2_epoch4_step4000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2/Drop6_BAS_scratch_raw_10GSD_split2_epoch16_step13600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2/Drop6_BAS_scratch_raw_10GSD_split2_epoch25_step20800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2/Drop6_BAS_scratch_raw_10GSD_split2_epoch11_step9600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch4_step4000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch6_step5600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch0_step800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch3_step3200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch19_step16000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch53_step43200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V3/Drop6_BAS_scratch_landcover_10GSD_split2_V3_epoch7_step6400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V30/Drop6_BAS_scratch_landcover_10GSD_split2_V30_epoch1_step116.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V30/Drop6_BAS_scratch_landcover_10GSD_split2_V30_epoch0_step64.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch16_step6800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch5_step2400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch3_step1600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch0_step1.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch0_step400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch0_step534.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch4_step2000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch26_step10496.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch9_step4000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V4/Drop6_BAS_scratch_landcover_10GSD_split2_V4_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch14_step9600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch9_step6400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch3_step2560.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch10_step7040.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch13_step8960.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch0_step640.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch2_step1920.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch5_step3840.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch1_step1280.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch220_step141440.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V8/Drop6_BAS_scratch_landcover_10GSD_split2_V8_epoch4_step3200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch8_step7200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch28_step23200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch2_step2400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch14_step12000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch1_step1600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch9_step8000.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch51_step41116.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2/Drop6_BAS_scratch_landcover_10GSD_split2_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch79_step7900.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch7_step800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch0_step33.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch0_step38.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch78_step7900.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch11_step1200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch14_step1500.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch8_step900.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont_epoch6_step700.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31_epoch1_step126.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31_epoch0_step256.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31_epoch1_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31_epoch2_step177.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31_epoch2_step522.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31_epoch0_step15.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch31_step3200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch36_step3700.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch32_step3300.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch21_step2200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch473_step47400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch16_step1700.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch0_step134.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch1_step139.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2_epoch0_step77.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_VITB16_1k_landcover_10GSD_split2_V8/Drop6_BAS_VITB16_1k_landcover_10GSD_split2_V8_epoch0_step63.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_VITB16_1k_landcover_10GSD_split2_V8/Drop6_BAS_VITB16_1k_landcover_10GSD_split2_V8_epoch0_step43.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_VITB16_1k_landcover_10GSD_split2_V8/Drop6_BAS_VITB16_1k_landcover_10GSD_split2_V8_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V17/Drop6_BAS_scratch_big_landcover_10GSD_split2_V17_epoch0_step2.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12_epoch61_step66154.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12_epoch0_step1067.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12_epoch1_step2134.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12_epoch2_step3201.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12_epoch3_step4268.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12_epoch20_step22407.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12_epoch4_step5335.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V20/Drop6_BAS_sits_raw_eGSD_split2_V20_epoch54_step1375.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V20/Drop6_BAS_sits_raw_eGSD_split2_V20_epoch0_step25.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V20/Drop6_BAS_sits_raw_eGSD_split2_V20_epoch2_step75.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V20/Drop6_BAS_sits_raw_eGSD_split2_V20_epoch3_step100.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V20/Drop6_BAS_sits_raw_eGSD_split2_V20_epoch1_step50.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V20/Drop6_BAS_sits_raw_eGSD_split2_V20_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_sits_raw_eGSD_split2_V20/Drop6_BAS_sits_raw_eGSD_split2_V20_epoch4_step125.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V32/Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V32_epoch64_step4095.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch569_step35910.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch615_step38808.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch67_step4284.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch604_step38115.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch608_step38367.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch9_step630.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V33/Drop6_BAS_scratch_landcover_10GSD_split2_V33_epoch74_step4725.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch4_step640.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch1_step256.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch2_step384.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch0_step128.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch4_step1280.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch5_step1386.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch3_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch60_step7689.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch3_step1024.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch2_step768.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch42_step5504.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch1_step512.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V18/Drop6_BAS_scratch_landcover_10GSD_split2_V18_epoch0_step256.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8/Drop6_BAS_scratch_raw_10GSD_split2_smt8_epoch15_step12800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8/Drop6_BAS_scratch_raw_10GSD_split2_smt8_epoch23_step19200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8/Drop6_BAS_scratch_raw_10GSD_split2_smt8_epoch21_step17600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8/Drop6_BAS_scratch_raw_10GSD_split2_smt8_epoch32_step26400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8/Drop6_BAS_scratch_raw_10GSD_split2_smt8_epoch25_step20800.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_raw_10GSD_split2_smt8/Drop6_BAS_scratch_raw_10GSD_split2_smt8_epoch17_step14400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V9/Drop6_BAS_scratch_big_landcover_10GSD_split2_V9_epoch0_step378.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch44_step48015.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch3_step4268.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch45_step48015.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch1_step2134.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch0_step1067.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch34_step37345.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch77_step82159.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_epoch4_step5335.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch0_step0.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch13_step11200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch2_step2400.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch18_step15200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch8_step7200.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch1_step1600.pt
            - /home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_scratch_landcover_10GSD_split2_V5/Drop6_BAS_scratch_landcover_10GSD_split2_V5_epoch0_step800.pt

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
    --run=1


