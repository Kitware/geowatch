#### Small Extended Baseline Evaluation


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m watch.mlops.manager "list packages" --dataset_codes Drop7-MedianNoWinter10GSD-NoMask --yes
python -m watch.mlops.manager "list packages" --dataset_codes Drop7-Cropped2GSD

python -m watch.mlops.schedule_evaluation --params="
    pipeline: sc

    matrix:
        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.test_dataset:
            - $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands-small.kwcoco.zip
            #- $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip
            #- $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands-small.kwcoco.zip
            #- $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands-small.kwcoco.zip

        sc_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch=189-step=12160-val_loss=2.881.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch1_step172.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch2_step258.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch74_step6450.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch0_step86.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch398_step17157.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch468_step20167.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch486_step20921.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch268_step11567.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch389_step16770.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V10/Drop7-Cropped2GSD_SC_bgrn_split6_V10_epoch299_step12900.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch76_step3311.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch545_step23478.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch1_step86.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch4_step215.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch2_step129.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch538_step23177.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch0_step43.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch5_step249.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch448_step19307.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch3_step172.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V09/Drop7-Cropped2GSD_SC_bgrn_split6_V09_epoch20_step903.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch165_step14276.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch173_step14964.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch149_step12900.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch186_step16082.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V81_epoch88_step7654.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch103_step8944.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch54_step4730.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch95_step8256.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch104_step9030.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V17/Drop7-Cropped2GSD_SC_bgrn_split6_V17_epoch84_step7310.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch236_step20382.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch250_step21586.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch311_step26832.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch230_step19866.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch0_step86.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch240_step20726.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V14_epoch1_step172.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch39_step1280.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch36_step1184.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch37_step1216.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch35_step1152.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_gnt_sgd_split6_V86_epoch38_step1248.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch41_step3612.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch91_step31464.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch24_step8550.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch36_step12654.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch13_step4788.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch89_step30780.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch14_step5130.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch6_step2394.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch11_step4104.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch22_step1978.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch0_step342.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch37_step3268.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch57_step19836.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch55_step19152.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch138_step11954.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch185_step15996.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch148_step12814.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch165_step14276.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80/Drop7-Cropped2GSD_SC_bgrn_snp_split6_V80_epoch173_step14964.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch70_step6106.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch54_step4730.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch45_step3956.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch42_step3698.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13/Drop7-Cropped2GSD_SC_bgrn_depth_split6_V13_epoch51_step4472.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch17_step2304.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch147_step9472.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch57_step3712.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch5_step384.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch11_step1536.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch189_step12160.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch6_step448.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch4_step640.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch7_step1024.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch51_step3328.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch0_step1.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch8_step1152.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch18_step2316.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch8_step576.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch166_step10688.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch9_step640.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch193_step12416.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch0_step64.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch187_step12032.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch55_step3584.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch56_step3648.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch7_step512.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch49_step3200.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch1_step81.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch0_step128.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86/Drop7-Cropped2GSD_SC_bgrn_snp_sgd_split6_V86_epoch54_step3520.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85_epoch127_step4096.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85_epoch120_step3872.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85_epoch103_step3328.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85_epoch115_step3712.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85/Drop7-Cropped2GSD_SC_bgrn_gnt_snp_split6_V85_epoch23_step768.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch77_step6708.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch75_step6536.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch78_step6794.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch48_step4214.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V05/Drop7-Cropped2GSD_SC_bgrn_split6_V05_epoch60_step5246.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch336_step28982.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch326_step28122.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch302_step26058.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch202_step17458.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch216_step18662.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch374_step16125.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch272_step11739.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch518_step22317.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch217_step9374.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch129_step5590.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch521_step22446.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04_epoch8_step774.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04_epoch9_step860.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04_epoch103_step8944.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04_epoch13_step1204.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04_epoch16_step1462.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04/Drop7-Cropped2GSD_SC_bgrn_peonly_small_V04_epoch11_step1032.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch27_step700.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch19_step500.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch25_step650.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch0_step0.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch28_step710.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch22_step575.pt
            #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_petest_V06/Drop7-Cropped2GSD_SC_bgrn_petest_V06_epoch10_step275.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch1_step16514.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v1_epoch0_step5778.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch2_step17334.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch0_step171.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch11_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch86_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch57_step67918.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch0_step0.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch73_step999148.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch32_step445566.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch3_step64.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch13_step2394.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch10_step176.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch6_step83790.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v1_epoch12_step175526.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch0_step11970.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch96_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch76_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch93_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch76_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V63/Drop7-MedianNoWinter10GSD_bgr_split6_V63_epoch359_step15480.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch78_step1066658.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step6587.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch12_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step1661.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch34_step35840.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch136_step160427.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch97_step1313549.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v1_epoch69_step945140.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch53_step63234.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch31_step32768.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch28_step391558.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch71_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch72_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch71_step972144.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch7_step126.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch28_step29696.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_2GSD_V3/package_epoch0_step57.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch4_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch1_step11556.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v1_epoch60_step823622.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch3_step64.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v1_epoch3_step47880.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_2GSD_V3/Drop4_tune_V30_2GSD_V3_v0_epoch0_step57.pt
            - $DVC_EXPT_DPATH/models/fusion/eval-2022-11-sc/packages/Drop4_SC_ManuallyCobbled/Drop4_tune_V30_V2-SC-epoch0_step11970.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch70_step83141.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch6_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch30_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch75_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/package_epoch3_step22551.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch9_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch37_step513076.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch0_step3819.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch4_step80.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch38_step526578.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch2_step48.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v3_epoch9_step135020.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch7_step95760.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch1_step23940.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch69_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch5_step96.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch7_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch0_step131.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v2_epoch21_step297044.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch11_step2052.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch75_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch5_step71820.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch56_step66747.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch11_step1001.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch7_step95760.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V63/Drop7-MedianNoWinter10GSD_bgr_split6_V63_epoch0_step0.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v1_epoch47_step648096.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step11970.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch0_step16.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V79_epoch8_step144.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch107_step126468.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch21_step22528.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch80_step1093662.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch89_step1215180.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_2GSD_V3/Drop4_tune_V30_2GSD_V3_v0_epoch0_step13119.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_2GSD_V3/Drop4_tune_V30_2GSD_V3_v0_epoch2_step29798.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v1_epoch2_step35910.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v1_epoch24_step337550.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch95_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch1_step32.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch65_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch8_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch12_step2223.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch92_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch0_step0.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch93_step110074.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V66_epoch72_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch16_step2907.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V77_epoch9_step1710.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch0_step0.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch159_step187360.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step171.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/package_epoch0_step0.pt.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch0_step337.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67/Drop7-MedianNoWinter10GSD_bgrn_scratch_split6_V67_epoch88_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split1_5GSD_V13/Drop6_JOINT_Split1_5GSD_V13_epoch91_step107732.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch33_step34816.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch25_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch29_step30720.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch4_step59850.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V68/Drop7-MedianNoWinter10GSD_bgrn_split6_V68_epoch34_stepNone.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch37_step38225.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch0_step1661.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_v0_epoch3_step22551.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-NoMask/packages/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78/Drop7-MedianNoWinter10GSD_bgrn_mixed_split6_V78_epoch2_step48.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch36_step499574.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD/packages/Drop7-MedianNoWinter10GSD_bgr_split6_V61/Drop7-MedianNoWinter10GSD_bgr_split6_V61_epoch359_step15480.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch32_step33792.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch24_step25600.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V2/Drop4_tune_V30_V2_v0_epoch0_step6587.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch0_step13502.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_V1/Drop4_tune_V30_V1_v0_epoch35_step486072.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch27_step28672.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_JOINT_Split2_5GSD_V13/Drop6_JOINT_Split2_5GSD_V13_epoch26_step27648.pt
            - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_2GSD_V3/Drop4_tune_V30_2GSD_V3_v0_epoch1_step26238.pt


        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3

        sc_pxl.fixed_resolution:
            - 8GSD
            #- 4GSD
            #- 2GSD

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

        sc_poly.thresh:
         #- 0.07
         #- 0.10
         - 0.10
         #- 0.30
        sc_poly.boundaries_as: polys
        sc_poly.min_area_square_meters: 7200
        sc_poly.resolution: 8GSD

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $BUNDLE_DPATH/bas_small_truth/site_models
        sc_poly_eval.true_region_dpath: $BUNDLE_DPATH/bas_small_truth/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 0
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0

    submatrices:

        # Point each region to the polygons that AC/SC will score

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/KR_R002.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/KW_C501.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/CO_C501.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/CN_C500.geojson

    " \
    --root_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_v1" \
    --queue_name "_ac_static_small_baseline_v1" \
    --devices="1," \
    --backend=tmux --tmux_workers=4 \
    --cache=1 --skip_existing=1 --run=1


HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
TRUTH_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m watch.mlops.aggregate \
    --pipeline=sc \
    --target "
        - $DVC_EXPT_DPATH/_ac_static_small_baseline_v1
    " \
    --output_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_v1/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - sc_poly_eval
    " \
    --plot_params="
        enabled: 0
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.sc_poly.thresh
    " \
    --stdout_report="
        top_k: 100
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002"
    #--rois="KR_R002,KW_C501,CO_C501,CN_C500"


#####
## Inspecting


# Drop7 Model
cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_a62f6099/


# Drop4 Model
cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_47bde2af/


# Drop 7 CN_5000
cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_6582a840/

eog /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_a5bc5e5b/confusion_analysis/site_viz/_flat/KR_R002_0017-vs-KR_R002_0031.jpg


# SINGLE MODEL DIVE
HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
BUNDLE_DPATH=$HIRES_DVC_DATA_DPATH/Drop7-StaticACTestSet-2GSD

python -m watch.mlops.schedule_evaluation --params="
    pipeline: sc

    matrix:
        ########################
        ## AC/SC PIXEL PARAMS ##
        ########################

        sc_pxl.test_dataset:
            - $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands-small.kwcoco.zip
            #- $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip
            #- $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands-small.kwcoco.zip
            #- $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands-small.kwcoco.zip

        sc_pxl.package_fpath:
            - $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt

        sc_pxl.tta_fliprot: 0.0
        sc_pxl.tta_time: 0.0
        sc_pxl.chip_overlap: 0.3

        sc_pxl.fixed_resolution:
            #- 8GSD
            #- 4GSD
            - 2GSD

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

        sc_poly.thresh:
         #- 0.07
         #- 0.10
         - 0.20
         #- 0.30
        sc_poly.boundaries_as: polys
        sc_poly.min_area_square_meters: 7200
        sc_poly.resolution: 8GSD
        sc_poly.site_score_thresh:
            - 0.45
        sc_poly.smoothing:
            - 0.66

        #############################
        ## AC/SC POLY EVAL PARAMS  ##
        #############################

        sc_poly_eval.true_site_dpath: $BUNDLE_DPATH/bas_small_truth/site_models
        sc_poly_eval.true_region_dpath: $BUNDLE_DPATH/bas_small_truth/region_models

        ##################################
        ## HIGH LEVEL PIPELINE CONTROLS ##
        ##################################
        sc_pxl.enabled: 1
        sc_pxl_eval.enabled: 1
        sc_poly.enabled: 1
        sc_poly_eval.enabled: 1
        sc_poly_viz.enabled: 0

    submatrices:

        # Point each region to the polygons that AC/SC will score

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KR_R002/imgonly-KR_R002-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/KR_R002.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/KW_C001/imgonly-KW_C001-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/KW_C501.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CO_C001/imgonly-CO_C001-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/CO_C501.geojson

        - sc_pxl.test_dataset: $BUNDLE_DPATH/CN_C000/imgonly-CN_C000-rawbands-small.kwcoco.zip
          sc_poly.site_summary: $BUNDLE_DPATH/bas_small_output/region_models/CN_C500.geojson

    " \
    --root_dpath="$DVC_EXPT_DPATH/_ac_static_small_baseline_v1" \
    --queue_name "_ac_static_small_baseline_v1" \
    --devices="0," \
    --backend=serial --tmux_workers=1 \
    --cache=1 --skip_existing=0 --run=1



### TEST NEW PARAMS
python -m watch.cli.run_tracker \
    --in_file "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_pxl/sc_pxl_id_fc2b4d4e/pred.kwcoco.zip" \
    --default_track_fn class_heatmaps \
    --track_kwargs '{"boundaries_as": "polys", "thresh": 0.2, "min_area_square_meters": 7200, "resolution": "8GSD"}' \
    --clear_annots=True \
    --out_site_summaries_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_9d81331e/site_summaries_manifest.json" \
    --out_site_summaries_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_9d81331e/site_summaries" \
    --out_sites_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_9d81331e/sites_manifest.json" \
    --out_sites_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_9d81331e/sites" \
    --out_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_9d81331e/poly.kwcoco.zip" \
    --site_score_thresh=0.4 \
    --smoothing=0.66 \
    --site_summary=/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002.geojson \
    --boundary_region=None


python -m watch.cli.run_metrics_framework \
    --merge=True \
    --name "todo-sc_poly_algo_id_7f666f77-sc_pxl_algo_id_5d3e8b55-todo" \
    --true_site_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/site_models" \
    --true_region_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models" \
    --pred_sites "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_9d81331e/sites_manifest.json" \
    --tmp_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_a312914f/tmp" \
    --out_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_a312914f" \
    --merge_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_a312914f/poly_eval.json" \
    --enable_viz=False




##### BEFORE CHANGES BASELINE
#
python -m watch.cli.run_tracker \
    --in_file "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_pxl/sc_pxl_id_6189d572/pred.kwcoco.zip" \
    --default_track_fn class_heatmaps \
    --track_kwargs '{"boundaries_as": "polys", "thresh": 0.2, "min_area_square_meters": 7200, "resolution": "8GSD"}' \
    --clear_annots=True \
    --out_site_summaries_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e280fb40/site_summaries_manifest.json" \
    --out_site_summaries_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e280fb40/site_summaries" \
    --out_sites_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e280fb40/sites_manifest.json" \
    --out_sites_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e280fb40/sites" \
    --out_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e280fb40/poly.kwcoco.zip" \
    --site_summary=/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002.geojson \
    --boundary_region=None


### Command 4 / 4 - sc_poly_eval_id_62fab2ed
python -m watch.cli.run_metrics_framework \
    --merge=True \
    --name "todo-sc_poly_algo_id_900574bb-sc_pxl_algo_id_ac244673-todo" \
    --true_site_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/site_models" \
    --true_region_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models" \
    --pred_sites "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e280fb40/sites_manifest.json" \
    --tmp_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_62fab2ed/tmp" \
    --out_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_62fab2ed" \
    --merge_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_62fab2ed/poly_eval.json" \
    --enable_viz=False


## REAL BASELINE?
python -m watch.cli.run_tracker \
    --in_file "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_pxl/sc_pxl_id_fc2b4d4e/pred.kwcoco.zip" \
    --default_track_fn class_heatmaps \
    --track_kwargs '{"boundaries_as": "polys", "thresh": 0.2, "min_area_square_meters": 7200, "resolution": "8GSD"}' \
    --clear_annots=True \
    --site_summary '/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002.geojson' \
    --boundary_region 'None' \
    --out_site_summaries_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_a0da6abb/site_summaries_manifest.json" \
    --out_site_summaries_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_a0da6abb/site_summaries" \
    --out_sites_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_a0da6abb/sites_manifest.json" \
    --out_sites_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_a0da6abb/sites" \
    --out_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_a0da6abb/poly.kwcoco.zip"


python -m watch.cli.run_metrics_framework \
    --merge=True \
    --name "todo-sc_poly_algo_id_900574bb-sc_pxl_algo_id_5d3e8b55-todo" \
    --true_site_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/site_models" \
    --true_region_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models" \
    --pred_sites "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_a0da6abb/sites_manifest.json" \
    --tmp_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_372c0c95/tmp" \
    --out_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_372c0c95" \
    --merge_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_372c0c95/poly_eval.json" \
    --enable_viz=False \


## SHOULD REPRODUCE BASELINE
python -m watch.cli.run_tracker \
    --in_file "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_pxl/sc_pxl_id_fc2b4d4e/pred.kwcoco.zip" \
    --default_track_fn class_heatmaps \
    --track_kwargs '{"boundaries_as": "polys", "thresh": 0.2, "min_area_square_meters": 7200, "resolution": "8GSD"}' \
    --clear_annots=True \
    --out_site_summaries_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e14677a2/site_summaries_manifest.json" \
    --out_site_summaries_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e14677a2/site_summaries" \
    --out_sites_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e14677a2/sites_manifest.json" \
    --out_sites_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e14677a2/sites" \
    --out_kwcoco "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e14677a2/poly.kwcoco.zip" \
    --site_score_thresh=None \
    --smoothing=None \
    --site_summary=/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_output/region_models/KR_R002.geojson \
    --boundary_region=None

python -m watch.cli.run_metrics_framework \
    --merge=True \
    --name "todo-sc_poly_algo_id_e5608e61-sc_pxl_algo_id_5d3e8b55-todo" \
    --true_site_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/site_models" \
    --true_region_dpath "/media/joncrall/flash1/smart_drop7/Drop7-StaticACTestSet-2GSD/bas_small_truth/region_models" \
    --pred_sites "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/pred/flat/sc_poly/sc_poly_id_e14677a2/sites_manifest.json" \
    --tmp_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_e97651d4/tmp" \
    --out_dir "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_e97651d4" \
    --merge_fpath "/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_e97651d4/poly_eval.json" \
    --enable_viz=False
