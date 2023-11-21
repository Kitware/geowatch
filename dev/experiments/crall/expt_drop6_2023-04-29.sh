#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6-MeanYear10GSD-V2
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_I2L_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_I2L_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50_test3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=2 python -m geowatch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 11
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
    window_resolution     : 10.0GSD
    input_resolution      : 10.0GSD
    output_resolution     : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    absolute_weighting     : True
    modality_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    #use_grid_negatives     : 'cleared'
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights       : '1:1'
        #class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 0.01
        global_saliency_weight : 1.00
        multimodal_reduce      : learned_linear
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 64
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360

torch_globals:
    float32_matmul_precision: auto

initializer:
    #init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD/packages/Drop6_TCombo1Year_BAS_10GSD_split6_V46/Drop6_TCombo1Year_BAS_10GSD_split6_V46_epoch118_step22253.pt
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6-MeanYear10GSD-V2/runs/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V50/lightning_logs/version_0/checkpoints/epoch=77-step=1248.ckpt
"


#!/bin/bash

