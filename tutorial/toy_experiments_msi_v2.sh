#!/bin/bash
__doc__="""
This tutorial expands on ~/code/watch/tutorial/toy_experiments_msi.sh and
trains different models with varied hyperparameters. The comments in this
tutorial will be sparse. Be sure to read the previous tutorial and compare
these fit invocation with the default one.
"""

# Define wherever you want to store results
DVC_DATA_DPATH=$HOME/data/dvc-repos/toy_data_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/toy_expt_dvc

NUM_TOY_TRAIN_VIDS="${NUM_TOY_TRAIN_VIDS:-100}"  # If variable not set or null, use default.
NUM_TOY_VALI_VIDS="${NUM_TOY_VALI_VIDS:-5}"  # If variable not set or null, use default.
NUM_TOY_TEST_VIDS="${NUM_TOY_TEST_VIDS:-2}"  # If variable not set or null, use default.

# Generate toy datasets
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}/data.kwcoco.json
#TEST_FPATH=$DVC_DATA_DPATH/vidshapes_msi_test${NUM_TOY_TEST_VIDS}/data.kwcoco.json 

generate_data(){
    mkdir -p "$DVC_DATA_DPATH"

    kwcoco toydata --key="vidshapes${NUM_TOY_TRAIN_VIDS}-frames5-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}" --verbose=1

    kwcoco toydata --key="vidshapes${NUM_TOY_VALI_VIDS}-frames5-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}"  --verbose=1

    kwcoco toydata --key="vidshapes${NUM_TOY_TEST_VIDS}-frames6-randgsize-speed0.2-msi-multisensor" \
        --bundle_dpath "$DVC_DATA_DPATH/vidshapes_msi_test${NUM_TOY_TEST_VIDS}" --verbose=1
}

if [[ ! -e "$TRAIN_FPATH" ]]; then
    generate_data
fi


# Training with the baseline MultiModalModel
DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_V002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MAX_STEPS=1000
TARGET_LR=3e-4
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"
python -m watch.tasks.fusion fit --config "
    data:
        num_workers          : 4
        train_dataset        : $TRAIN_FPATH
        vali_dataset         : $VALI_FPATH
        channels             : '$CHANNELS'
        time_steps           : 5
        chip_dims            : 128
        batch_size           : 2
    model:
        class_path: MultimodalTransformer
        init_args:
            name        : $EXPERIMENT_NAME
            arch_name   : smt_it_stm_p8 
            window_size : 8
            dropout     : 0.1
            global_saliency_weight: 1.0 
            global_class_weight:    1.0 
            global_change_weight:   0.0 
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.OneCycleLR
      init_args:
        max_lr: $TARGET_LR
        total_steps: $MAX_STEPS
        anneal_strategy: cos
        pct_start: 0.05
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: $TARGET_LR
        weight_decay: 1e-5
        betas:
          - 0.9
          - 0.99
    trainer:
      accumulate_grad_batches: 1
      default_root_dir     : $DEFAULT_ROOT_DIR
      accelerator          : gpu 
      devices              : 0,
      #devices             : 0,1
      #strategy            : ddp 
      check_val_every_n_epoch: 1
      enable_checkpointing: true
      enable_model_summary: true
      log_every_n_steps: 5
      logger: true
      max_steps: $MAX_STEPS
      num_sanity_val_steps: 0
      replace_sampler_ddp: true
      track_grad_norm: 2
    initializer:
        init: noop
"




rm -rf "$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME"

# Training with the HeterogeneousModel using a very small backbone
#
# TODO:
# * Debug why ddp isn't working
# * Debug why checkpoint restart isn't working
DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_V002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MAX_STEPS=10000
TARGET_LR=3e-4
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"
python -m watch.tasks.fusion fit --config "
    seed_everything: 123
    data:
        num_workers          : 4
        train_dataset        : $TRAIN_FPATH
        vali_dataset         : $VALI_FPATH
        channels             : '$CHANNELS'
        time_steps           : 5
        chip_dims            : 128
        batch_size           : 2
        max_epoch_length     : 5
    model:
      class_path: watch.tasks.fusion.methods.HeterogeneousModel
      init_args:
        name        : $EXPERIMENT_NAME
        token_width : 16
        token_dim   : 64
        position_encoder:
          class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
          init_args:
            in_dims   : 3
            max_freq  : 3
            num_freqs : 16
        backbone:
          class_path: watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder
          init_args:
            encoder_depth   : 2
            decoder_depth   : 0
            dim             : 160
            queries_dim     : 96
            logits_dim      : 64
            latent_dim_head : 8
        spatial_scale_base     : 1.0
        temporal_scale_base    : 1.0
        global_change_weight   : 0.0
        global_class_weight    : 0.0
        global_saliency_weight : 1.0
        saliency_loss          : dicefocal
        decoder                : simple_conv
    #lr_scheduler:
    #  class_path: torch.optim.lr_scheduler.OneCycleLR
    #  init_args:
    #    max_lr: $TARGET_LR
    #    total_steps: $MAX_STEPS
    #    anneal_strategy: cos
    #    pct_start: 0.05
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: $TARGET_LR
        weight_decay: 1e-5
        #betas:
        #  - 0.9
        #  - 0.99
    trainer:
      accumulate_grad_batches: 1
      default_root_dir     : $DEFAULT_ROOT_DIR
      accelerator          : gpu 
      devices              : 0,
      #devices             : 0,1
      #strategy            : ddp 
      check_val_every_n_epoch: 1
      enable_checkpointing: true
      enable_model_summary: true
      log_every_n_steps: 5
      logger: true
      max_steps: $MAX_STEPS
      num_sanity_val_steps: 0
      replace_sampler_ddp: true
      track_grad_norm: 2
    initializer:
        init: noop
"


# If you Ctrl+C the previous model after it has saved an end-of-epoch
# checkpoint, you can restart by pointing at that checkpoint.
# NOTE: CURRENTLY BUGGED AND NOT SURE WHY IT IS NOT WORKING
#CKPT_FPATH=$(python -c "import pathlib; print(list(pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs').glob('*/package-interupt/*.ckpt'))[0])")
CKPT_FPATH=$(python -c "import pathlib; print(list(pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs').glob('*/checkpoints/*.ckpt'))[0])")
CONFIG_FPATH=$(python -c "import pathlib; print(sorted(pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs').glob('*/config.yaml'))[-1])")
echo "CONFIG_FPATH = $CONFIG_FPATH"
echo "CKPT_FPATH = $CKPT_FPATH"

python -m watch.tasks.fusion fit --config "$CONFIG_FPATH" \
    --ckpt_path="$CKPT_FPATH"
