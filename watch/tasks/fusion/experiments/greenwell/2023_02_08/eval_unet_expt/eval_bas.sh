# Determine the paths to your SMART data and experiment repositories.
# DATA_DVC_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
# DATA_DVC_DPATH=/home/local/KHQ/connor.greenwell/data/dvc-repos/smart_data_dvc/Drop4-BAS
DATA_DVC_DPATH=/flash/smart_data_dvc/Drop6
EXPT_DVC_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware=auto)/training/horologic/connor.greenwell/sym_lightning_logs


echo "
EXPT_DVC_DPATH=$EXPT_DVC_DPATH
DATA_DVC_DPATH=$DATA_DVC_DPATH
"


            # bas_pxl.package_fpath:
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=378-step=9475.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=382-step=9575.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=390-step=9775.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=428-step=10725.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=379-step=9500.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=441-step=11050.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=479-step=12000.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=440-step=11025.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=459-step=11500.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=406-step=10175.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=442-step=11075.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=421-step=10550.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=366-step=9175.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=431-step=10800.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=450-step=11275.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=427-step=10700.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=472-step=11825.pt
            #     - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=454-step=11375.pt


# python -m watch.mlops.schedule_evaluation \
#     --params="
#         matrix:
#             bas_pxl.package_fpath:
#                 - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=441-step=11050.pt
#                 - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=479-step=12000.pt
#                 - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=472-step=11825.pt

#             bas_pxl.test_dataset:
#                 - $DATA_DVC_DPATH/imganns-KR_R001.kwcoco.zip
#                 - $DATA_DVC_DPATH/imganns-KR_R002.kwcoco.zip
#                 - $DATA_DVC_DPATH/imganns-US_R007.kwcoco.zip
#                 - $DATA_DVC_DPATH/imganns-BR_R001.kwcoco.zip
#                 - $DATA_DVC_DPATH/imganns-BR_R002.kwcoco.zip
#                 - $DATA_DVC_DPATH/imganns-AE_R001.kwcoco.zip

#             bas_pxl.channels: auto
#             bas_pxl.chip_dims: 128, 128
#             bas_pxl.chip_overlap: 0.3
#             bas_pxl.window_space_scale: auto
#             bas_pxl.output_space_scale: auto
#             bas_pxl.input_space_scale: auto
#             bas_pxl.time_span: auto
#             bas_pxl.time_sampling: auto

#             bas_poly.moving_window_size: null
#             bas_poly.min_area_sqkm: null
#             bas_poly.max_area_sqkm: null
#             bas_poly.max_area_behavior: 'ignore'
#             bas_poly.response_thresh: null
#             bas_poly.time_thresh: null
#             bas_poly.morph_kernel: 0
#             bas_poly.thresh:
#                 - 0.05
#                 - 0.1
#                 - 0.15
#                 - 0.2
#                 - 0.25
#                 - 0.3
#                 - 0.35
#                 - 0.4
#                 - 0.45
#                 - 0.5
#                 - 0.55
#                 - 0.6
#                 - 0.65
#                 - 0.7
#                 - 0.75
#                 - 0.8
#                 - 0.85
#                 - 0.9
#                 - 0.95
#             bas_poly.agg_fn: rescaled_probs

#             bas_pxl.enabled: 1
#             bas_poly.enabled: 1
#             bas_poly_eval.enabled: 1
#             bas_pxl_eval.enabled: 1
#             bas_poly_viz.enabled: 0
#     " \
#     --root_dpath="$EXPT_DVC_DPATH/_evaluations/unet_expt" \
#     --devices="0" --queue_size=4 \
#     --backend=tmux --queue_name "baseline-queue" \
#     --virtualenv_cmd="conda activate watch_py3.9" \
#     --pipeline=bas \
#     --run=1

python -m watch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_3/checkpoints/epoch=441-step=11050.pt
                - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_UNet/lightning_logs/version_2/checkpoints/epoch=479-step=12000.pt
                - $EXPT_DVC_DPATH/Drop4_BAS_S2L8_NoDecoderHetModel/lightning_logs/version_2/checkpoints/epoch=472-step=11825.pt

            bas_pxl.test_dataset:
                - $DATA_DVC_DPATH/imganns-KR_R001.kwcoco.zip
                - $DATA_DVC_DPATH/imganns-KR_R002.kwcoco.zip
                - $DATA_DVC_DPATH/imganns-US_R007.kwcoco.zip
                - $DATA_DVC_DPATH/imganns-BR_R001.kwcoco.zip
                - $DATA_DVC_DPATH/imganns-BR_R002.kwcoco.zip
                - $DATA_DVC_DPATH/imganns-AE_R001.kwcoco.zip

            bas_pxl.channels: auto
            bas_pxl.chip_dims: 128, 128
            bas_pxl.chip_overlap: 0.3
            bas_pxl.window_space_scale: auto
            bas_pxl.output_space_scale: auto
            bas_pxl.input_space_scale: auto
            bas_pxl.time_span: auto
            bas_pxl.time_sampling: auto

            bas_poly.moving_window_size: null
            bas_poly.min_area_sqkm: null
            bas_poly.max_area_sqkm: null
            bas_poly.max_area_behavior: 'ignore'
            bas_poly.response_thresh: null
            bas_poly.time_thresh: null
            bas_poly.morph_kernel: 0
            bas_poly.thresh:
                - 0.025
                - 0.05
                - 0.075
                - 0.1
                - 0.125
                - 0.15
                - 0.175
                - 0.2
                - 0.225
                - 0.25
                - 0.275
                - 0.3
                - 0.325
                - 0.35
                - 0.375
                - 0.4
                - 0.425
                - 0.45
                - 0.475
                - 0.5
            bas_poly.agg_fn: probs

            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 0
    " \
    --root_dpath="$EXPT_DVC_DPATH/_evaluations/unet_expt" \
    --devices="0" --queue_size=4 \
    --backend=tmux --queue_name "baseline-queue" \
    --virtualenv_cmd="conda activate watch_py3.9" \
    --pipeline=bas \
    --run=1