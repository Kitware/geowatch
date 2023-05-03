# Note: Optimize Macro BAS F1 average across regions



####
# Need to precompute invariants

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
python -m watch.cli.split_videos \
    --src "$DVC_DATA_DPATH/Drop4-BAS/data_train.kwcoco.json" \
          "$DVC_DATA_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
    --io_workers=2 \
    --dst "$DVC_DATA_DPATH/Drop4-BAS/{src_name}_{video_name}.kwcoco.zip"


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
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
    --gres='0,1' --workers=2 --backend=tmux --run=1


####

# Real inputs, this actually will run something given the DVC repos
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

SC_MODEL=$DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
BAS_MODEL=$DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt

python -m watch.mlops.schedule_evaluation \
    --pipeline=joint_bas_sc_nocrop \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $BAS_MODEL
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
            bas_pxl.time_sampling: auto
            bas_pxl.window_space_scale: 15GSD
            bas_pxl.input_space_scale: window
            bas_poly.moving_window_size: null
            bas_poly.thresh: 0.1
            sc_pxl.window_space_scale: 8GSD
            sc_pxl.input_space_scale: window
            sc_poly.thresh: 0.1
            sc_poly.use_viterbi: 0
            sc_pxl.package_fpath: $SC_MODEL
            sc_poly_viz.enabled: 0
            sc_pxl_eval.enabled: 0
    " \
    --root_dpath=./my_dag_runs \
    --devices="0,1" --queue_size=2 --backend=serial \
    --cache=1 --skip_existing=0 --run=1


#####################
## BAS-Only Evaluation 
## ------------------
## Assumes the ground truth is the BAS input
#####################

python -m watch.mlops.repackager \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_2/checkpoints/epoch=1-step=77702.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_3/checkpoints/epoch=1-step=77702-v1.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_3/checkpoints/epoch=5-step=233106.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_0/checkpoints/epoch=0-step=4305.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_3/checkpoints/epoch=0-step=512-v1.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_3/checkpoints/epoch=4-step=2560.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_3/checkpoints/epoch=6-step=3584.ckpt \
    "$HOME/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_10GSD_BGRN_V11/lightning_logs/version_1/checkpoints/epoch=4-step=2560.ckpt" \
    "$HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V12/lightning_logs/version_1/checkpoints/epoch=16-step=17408.ckpt" \
    "$HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V12/lightning_logs/version_1/checkpoints/epoch=71-step=73728.ckpt" \
    "$HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_0/checkpoints/epoch=16-step=8704.ckpt" \
    "$HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_1/checkpoints/epoch=90-step=46592.ckpt" \
    "$HOME/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_1/checkpoints/epoch=159-step=81920.ckpt" 



#### Eval8 Models

echo "
#- $DVC_EXPT_DPATH/bas_native_epoch44.pt
- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
- $DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_0/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305.pt
- $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_0/checkpoints/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
#- $DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_10GSD_BGRN_V11/lightning_logs/version_1/checkpoints/Drop4_BAS_2022_12_10GSD_BGRN_V11_epoch=86-step=44544.pt
#- $DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_3/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V5_epoch=1-step=77702-v1.pt
#- $DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_3/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=512-v1.pt
#- $DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_3/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=4-step=2560.pt
#- $DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_3/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=6-step=3584.pt
#- $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V12/lightning_logs/version_1/checkpoints/Drop4_BAS_10GSD_BGRNSH_invar_V12_epoch=16-step=17408.pt
#- $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V12/lightning_logs/version_1/checkpoints/Drop4_BAS_10GSD_BGRNSH_invar_V12_epoch=71-step=73728.pt
#- $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_1/checkpoints/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=90-step=46592.pt
#- $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_1/checkpoints/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=159-step=81920.pt
" > "$DVC_EXPT_DPATH"/toothbrush_model_candidates_2023-01.yml 


#### Eval9 Models (Namek)

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
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_15GSD_BGRNSH_invar_V8/Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch0_step108.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch100_step51712.pt
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch102_step52736.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch104_step53760.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch138_step71168.pt   
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch140_step72192.pt   
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch143_step73728.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch149_step76800.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch159_step81920.pt   
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch78_step40448.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_10GSD_BGRN_V11/Drop4_BAS_2022_12_10GSD_BGRN_V11_v0_epoch98_step50688.pt    
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step0.pt         
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch0_step4305.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step1024.pt      
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch1_step8247.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch2_step1536.pt      
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch3_step2048.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch5_step3072.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch6_step3584.pt      
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_v0_epoch7_step3908.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V5/Drop4_BAS_2022_12_15GSD_BGRN_V5_v0_epoch0_step302.pt
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
                #- $DVC_DATA_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.zip
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
                - 0.45
                - 0.5
                - 0.55
                - 0.6
                - 0.65
                - 0.7
                - 0.75
                - 0.8
                - 0.85
            bas_poly.polygon_simplify_tolerance:
                #- 0.5
                - 1
                #- 3
            bas_poly.agg_fn:
                - probs
                #- binary
                #- rescaled_probs
                #- rescaled_binary
                #- mean_normalized
                #- frequency_weighted_mean
            bas_poly.resolution:
                - 10GSD
                - 15GSD
                - 30GSD
            bas_poly.moving_window_size: 
                - null
                #- 100
                #- 200
                #- 300
                #- 400
            #bas_poly.min_area_sqkm:
            #    - 0.072
            #    #- 0.031
            #    - 0.001
            bas_poly.max_area_sqkm:
                #- null
                #- 1.00
                #- 2.00
                #- 2.25
                #- 3.25
                - 8
            bas_pxl.enabled: 0
            bas_poly.enabled: 0
            bas_poly_eval.enabled: 0
            bas_pxl_eval.enabled: 0
            bas_poly_viz.enabled: 0
            include:
                - bas_pxl.chip_dims: 256,256
                  bas_pxl.window_space_scale: 10GSD
                  bas_pxl.input_space_scale: 10GSD
                  bas_pxl.output_space_scale: 10GSD
                - bas_pxl.chip_dims: 256,256
                  bas_pxl.window_space_scale: 15GSD
                  bas_pxl.input_space_scale: 15GSD
                  bas_pxl.output_space_scale: 15GSD
                - bas_pxl.chip_dims: 256,256
                  bas_pxl.window_space_scale: 30GSD
                  bas_pxl.input_space_scale: 30GSD
                  bas_pxl.output_space_scale: 30GSD
            ##    - bas_pxl.chip_dims: auto
            ##      bas_pxl.window_space_scale: auto
            ##      bas_pxl.input_space_scale: auto
            ##      bas_pxl.output_space_scale: auto
    " \
    --root_dpath="$DVC_EXPT_DPATH/_namek_eval" \
    --devices="0,1" --queue_size=2 \
    --backend=tmux --queue_name "bas-namek-evaluation-grid" \
    --pipeline=bas --skip_existing=1 \
    --run=0


#######


# Real data
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_native_epoch44.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/BR_R002.kwcoco.json

            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt

            bas_pxl.window_space_scale:
                #- auto
                - 15GSD
                - 30GSD
            bas_pxl.input_space_scale: window
            bas_pxl.output_space_scale: window
            bas_pxl.chip_dims:
                - auto
                - 256,256
            bas_pxl.time_sampling:
                - auto
            bas_poly.moving_window_size:
                - null
                - 100
                - 200
                - 300
            bas_poly.thresh:
                - 0.1
                #- 0.13
                #- 0.2
            sc_pxl.chip_dims:
                - auto
                - 256,256
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: window
            sc_pxl.output_space_scale: window
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0
            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 1
            sc_pxl.enabled: 0
            sc_poly.enabled: 0
            sc_poly_eval.enabled: 0
            sc_pxl_eval.enabled: 0
            sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testpipe2" \
    --devices="0,1" --queue_size=2 \
    --backend=tmux \
    --pipeline=joint_bas_sc_nocrop \
    --cache=1 --skip_existing=0 \
    --rprint=1 --run=0

#--max_configs=1 \

# Real data
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_native_epoch44.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-BAS/BR_R002.kwcoco.json

            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt

            bas_pxl.window_space_scale:
                - auto
                - 15GSD
                - 30GSD
            bas_pxl.chip_dims:
                - auto
                - 256,256
            bas_pxl.time_sampling:
                - auto
            bas_pxl.input_space_scale:
                - window
            bas_poly.moving_window_size:
                - null
                - 100
                - 200
                - 300
            bas_poly.thresh:
                - 0.1
                - 0.13
                - 0.2
            sc_pxl.chip_dims:
                - auto
                - 256,256
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: window
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0

            bas_pxl.enabled: 0
            bas_poly.enabled: 1
            sc_pxl.enabled: 0
            sc_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 0
            sc_poly_eval.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testpipe" \
    --devices="0,1" --queue_size=3 \
    --backend=tmux \
    --pipeline=joint_bas_sc_nocrop \
    --cache=1 \
    --queue_name="eval-existing" \
    --run=1 --rprint=1


# Real data
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                #- $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
                - $DVC_EXPT_DPATH/connor-models/2022-12-19/bas_native_rgb_2022-12-19.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_upsampled_epoch28.pt
                #- /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/bas_native_epoch44.pt
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop4-BAS/KR_R001.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/KR_R002.kwcoco.json
                #- $DVC_DATA_DPATH/Drop4-BAS/BR_R002.kwcoco.json
            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            bas_pxl.window_space_scale:
                - auto
                #- 15GSD
                #- 30GSD
            bas_pxl.chip_dims:
                - auto
                #- 256,256
            bas_pxl.time_sampling:
                - auto
            bas_pxl.input_space_scale:
                - auto
            bas_poly.moving_window_size:
                - null
                #- 100
                #- 200
                #- 300
            bas_poly.thresh:
                - 0.1
                #- 0.13
                #- 0.2
            sc_pxl.chip_dims:
                - auto
                #- 256,256
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: window
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0

            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            sc_pxl.enabled: 0
            sc_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 1
            sc_poly_eval.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_evaluations" \
    --devices="0," --queue_size=1 \
    --backend=tmux \
    --pipeline=bas --rprint=1 \
    --cache=1 \
    --queue_name="eval-models" \
    --run=0


#####################
## SC-Only Evaluation 
## ------------------
## Assumes the ground truth is the BAS input
#####################
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

# Dataset munging
# ---------------
if [ ! -f "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R001_sites.kwcoco.json ]; then
    # Split into a kwcoco file per site
    python -m watch.cli.split_videos "$DVC_DATA_DPATH"/Drop4-SC/data_vali.kwcoco.json
    python -m watch.cli.split_videos "$DVC_DATA_DPATH"/Drop4-SC/data_train.kwcoco.json
    # Combine sites per region
    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R001_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R001_sites.kwcoco.json
    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R002_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_vali_KR_R002_sites.kwcoco.json
    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_vali_US_R007_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_vali_US_R007_sites.kwcoco.json

    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_train_AE_R001_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_train_AE_R001_sites.kwcoco.json
    kwcoco union "$DVC_DATA_DPATH"/Drop4-SC/data_train_BR_R002_*_box.kwcoco.json \
        --dst "$DVC_DATA_DPATH"/Drop4-SC/data_train_BR_R002_sites.kwcoco.json
    # Remove the temporary split sites
    rm "$DVC_DATA_DPATH"/Drop4-SC/data_vali_*_box.kwcoco.json
fi



python -m watch.mlops.repackager --force=True \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=35-step=486072.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=12-step=175526-v1.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=21-step=297044-v2.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=32-step=445566.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=36-step=499574.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=37-step=513076.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=37-step=513076.ckpt \
    "$HOME"/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/epoch=89-step=1215180.ckpt



DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            sc_pxl.package_fpath:
                # - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
                # - $DVC_EXPT_DPATH/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=35-step=486072.pt
                - '/home/joncrall/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=12-step=175526-v1.pt'
                - '/home/joncrall/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=21-step=297044-v2.pt'
                - '/home/joncrall/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=32-step=445566.pt'
                - '/home/joncrall/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=36-step=499574.pt'
                - '/home/joncrall/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=37-step=513076.pt'
                - '/home/joncrall/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-SC/runs/Drop4_tune_V30_V1/lightning_logs/version_6/checkpoints/Drop4_tune_V30_V1_epoch=89-step=1215180.pt'
                
            sc_pxl.test_dataset:
                #- $DVC_DATA_DPATH/Drop4-SC/data_vali.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-SC/data_vali_KR_R001_sites.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-SC/data_vali_KR_R002_sites.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-SC/data_vali_US_R007_sites.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-SC/data_train_BR_R002_sites.kwcoco.json
                - $DVC_DATA_DPATH/Drop4-SC/data_train_AE_R001_sites.kwcoco.json
            sc_poly.site_summary:
                - $DVC_DATA_DPATH/annotations/region_models/*.geojson
            sc_poly_eval.true_region_dpath: $DVC_DATA_DPATH/annotations/region_models
            sc_poly_eval.true_site_dpath: $DVC_DATA_DPATH/annotations/site_models
            sc_pxl.chip_dims:
                - auto
                - 256,256
            sc_poly.thresh:
                - 0.07
                - 0.1
                - 0.13
            sc_poly.use_viterbi:
                - 0
            sc_pxl.enabled: 1
            sc_poly.enabled: 1
            sc_poly_eval.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly_viz.enabled: 0
            include:
                - sc_pxl.chip_dims: 256,256
                  sc_pxl.window_space_scale: 8GSD
                  sc_pxl.input_space_scale: 8GSD
                  sc_pxl.output_space_scale: 8GSD
                - sc_pxl.chip_dims: 256,256
                  sc_pxl.window_space_scale: 4GSD
                  sc_pxl.input_space_scale: 4GSD
                  sc_pxl.output_space_scale: 4GSD
                - sc_pxl.chip_dims: auto
                  sc_pxl.window_space_scale: auto
                  sc_pxl.input_space_scale: auto
                  sc_pxl.output_space_scale: auto
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testsc" \
    --devices="0," --queue_size=1 \
    --backend=tmux \
    --pipeline=sc \
    --cache=1 --skip_existing=1 \
    --rprint=0 --run=1
    

# Real data
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            bas_pxl.window_space_scale:
                - auto
            bas_pxl.test_dataset:
                - $DVC_DATA_DPATH/Drop6/KR_R001_BAS.kwcoco.json
            sitecrop.crop_src_fpath:
                - $DVC_DATA_DPATH/Drop6/KR_R001.kwcoco.json
            bas_pxl.input_space_scale: window
            bas_pxl.output_space_scale: window
            bas_pxl.chip_dims:
                - auto
            bas_pxl.time_sampling:
                - auto
            bas_poly.moving_window_size:
                - null
            bas_poly.thresh:
                - 0.15
            sc_pxl.chip_dims:
                - auto
            sc_pxl.window_space_scale:
                - auto
            sc_pxl.input_space_scale: window
            sc_pxl.output_space_scale: window
            sc_poly.thresh:
                - 0.1
            sc_poly.use_viterbi:
                - 0
            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 1
            sc_pxl.enabled: 1
            sc_poly.enabled: 1
            sc_poly_eval.enabled: 1
            sc_pxl_eval.enabled: 0
            sc_poly_viz.enabled: 0
    " \
    --root_dpath="$DVC_EXPT_DPATH/_testpipe2" \
    --devices="0,1" --queue_size=1 \
    --backend=tmux \
    --pipeline=joint_bas_sc_nocrop \
    --cache=1 --skip_existing=0 \
    --rprint=1 --run=0
