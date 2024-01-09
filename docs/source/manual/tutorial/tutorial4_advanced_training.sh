#!/bin/bash
__doc__="
This tutorial expands on ~/code/watch/tutorial/toy_experiments_msi.sh and
trains different models with varied hyperparameters. The comments in this
tutorial will be sparse. Be sure to read the previous tutorial and compare
these fit invocation with the default one.
"

# Define wherever you want to store results
# We will register them with geowatch DVC to abstract away the
# machine specific paths in the rest of the code.
# Running this multiple times is idempotent
geowatch dvc add --path "$HOME/data/dvc-repos/toy_data_dvc" --tag "toy_data_dvc" --name "toy_data_dvc"
geowatch dvc add --path "$HOME/data/dvc-repos/toy_expt_dvc" --tag "toy_expt_dvc" --name "toy_expt_dvc"

# Now we can access the registered paths as such
DVC_DATA_DPATH=$(geowatch dvc --tags "toy_data_dvc")
DVC_EXPT_DPATH=$(geowatch dvc --tags "toy_expt_dvc")

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


__doc__="
###############################################
# DEMO: MultimodalTransformer with LightningCLI
###############################################
"

# Training with the baseline MultiModalModel
DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_LightningCLI
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MAX_STEPS=1000
TARGET_LR=3e-4
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"
python -m geowatch.tasks.fusion fit --config "
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
      #devices              : 0,
      devices             : 0,1
      strategy            : ddp
      check_val_every_n_epoch: 1
      enable_checkpointing: true
      enable_model_summary: true
      log_every_n_steps: 5
      logger: true
      max_steps: $MAX_STEPS
      num_sanity_val_steps: 0
      replace_sampler_ddp: true
      track_grad_norm: 2
      callbacks:
          - class_path: pytorch_lightning.callbacks.ModelCheckpoint
            init_args:
                monitor: val_loss
                mode: min
                save_top_k: 5
                filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
                save_last: true
    initializer:
        init: noop
"


__doc__="
####################################################
# DEMO: MultiGPU Training with MultimodalTransformer
####################################################

The following command trains a MultimodalTransformer model on two GPUs with DDP

It seems to be the case that something in our system can cause DDP to hang with
100% reported GPU utilization (even though it really isn't doing anything).

References:
    https://github.com/Lightning-AI/lightning/issues/11242
    https://github.com/Lightning-AI/lightning/issues/10947
    https://github.com/Lightning-AI/lightning/issues/5319
    https://github.com/Lightning-AI/lightning/discussions/6501#discussioncomment-553152
    https://discuss.pytorch.org/t/ddp-via-lightning-fabric-training-hang-with-100-gpu-utilization/181046
    https://discuss.pytorch.org/t/single-machine-ddp-issue-on-a6000-gpu/134869/5

So far we may be able to avoid this if we do some combination of the following:
    * Disable pl_ext.callbacks.BatchPlotter
        - does cause the issue by itself, but seemingly only if we try to put
          in rank guards.

    * Disable pl.callbacks.LearningRateMonitor
    * Disable pl.callbacks.ModelCheckpoint
"

DVC_DATA_DPATH=$(geowatch dvc --tags "toy_data_dvc")
DVC_EXPT_DPATH=$(geowatch dvc --tags "toy_expt_dvc")
NUM_TOY_TRAIN_VIDS="${NUM_TOY_TRAIN_VIDS:-100}"  # If variable not set or null, use default.
NUM_TOY_VALI_VIDS="${NUM_TOY_VALI_VIDS:-5}"  # If variable not set or null, use default.
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}/data.kwcoco.json

DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_MultiModal_DDP
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MAX_STEPS=10000
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"

export CUDA_VISIBLE_DEVICES=0,1
export DISABLE_TENSORBOARD_PLOTTER=1
export DISABLE_BATCH_PLOTTER=1
export DDP_WORKAROUND=0
python -m geowatch.tasks.fusion fit --config "
seed_everything: 8675309
data:
    num_workers          : 2
    train_dataset        : $TRAIN_FPATH
    vali_dataset         : $VALI_FPATH
    channels             : '$CHANNELS'
    time_steps           : 5
    chip_dims            : 128
    batch_size           : 4
    max_epoch_length     : 1024
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto
        name                   : $EXPERIMENT_NAME
        class_weights          : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p8
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 5
        change_head_hidden     : 5
        global_change_weight   : 0.00
        global_class_weight    : 0.00
        global_saliency_weight : 1.00
        perterb_scale          : 0.001
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr          : $TARGET_LR
    total_steps     : $MAX_STEPS
    anneal_strategy : cos
    pct_start       : 0.05
trainer:
    accumulate_grad_batches: 2
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    #devices              : 0,
    devices              : 0,1
    strategy             : ddp
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto
"


__doc__="

########################################################
# DEMO: MultiGPU Training with Heterogeneous Transformer
########################################################

The following command trains a HeterogeneousModel model on two GPUs with DDP
"

DVC_DATA_DPATH=$(geowatch dvc --tags "toy_data_dvc")
DVC_EXPT_DPATH=$(geowatch dvc --tags "toy_expt_dvc")
NUM_TOY_TRAIN_VIDS="${NUM_TOY_TRAIN_VIDS:-100}"  # If variable not set or null, use default.
NUM_TOY_VALI_VIDS="${NUM_TOY_VALI_VIDS:-5}"  # If variable not set or null, use default.
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_msi_train${NUM_TOY_TRAIN_VIDS}/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_msi_vali${NUM_TOY_VALI_VIDS}/data.kwcoco.json

DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_Heterogeneous_DDP
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MAX_STEPS=10000
TARGET_LR=3e-4
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"
DDP_WORKAROUND=1 python -m geowatch.tasks.fusion fit --config "
    seed_everything: 8675309
    data:
        num_workers          : 2
        train_dataset        : $TRAIN_FPATH
        vali_dataset         : $VALI_FPATH
        channels             : '$CHANNELS'
        time_steps           : 5
        chip_dims            : 128
        batch_size           : 4
        max_epoch_length     : 1024
    model:
      class_path: watch.tasks.fusion.methods.HeterogeneousModel
      init_args:
        name                   : $EXPERIMENT_NAME
        token_width            : 8
        token_dim              : 256
        position_encoder       : auto
        backbone               : small
        global_change_weight   : 0.0
        global_class_weight    : 0.0
        global_saliency_weight : 1.0
        saliency_loss          : focal
        decoder                : simple_conv
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.OneCycleLR
      init_args:
        max_lr          : $TARGET_LR
        total_steps     : $MAX_STEPS
        anneal_strategy : cos
        pct_start       : 0.05
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr           : $TARGET_LR
        weight_decay : 1e-5
    trainer:
      default_root_dir : $DEFAULT_ROOT_DIR
      max_steps        : $MAX_STEPS
      accelerator      : gpu
      devices          : 0,1
      strategy        : ddp
"


__doc__="
###############################################
# DEMO: Restarting from an existing checkpoint
###############################################

The following demo illustrates how to restart from an end-of-epoch checkpoint.
To run this demo you will need to run the training command, wait for it to
complete one epoch (which is only a few seconds), and then kill the job with
ctrl+C. Then it shows how to restart given the checkpoint that was written.
"

# Training with the HeterogeneousModel using a very small backbone
DATASET_CODE=ToyDataMSI
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyDataMSI_Demo_CheckpointRestart
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
# Fresh start
rm -rf "$DEFAULT_ROOT_DIR"
#
# Write a config to disk:
mkdir -p "$DEFAULT_ROOT_DIR"
CONFIG_FPATH="$DEFAULT_ROOT_DIR"/restart_demo_config.yaml
CHANNELS="(*):(disparity|gauss,X.2|Y:2:6,B1|B8a,flowx|flowy|distri)"
MAX_STEPS=10000
TARGET_LR=3e-4
echo "
    seed_everything: 123
    data:
        num_workers          : 4
        train_dataset        : $TRAIN_FPATH
        vali_dataset         : $VALI_FPATH
        channels             : '$CHANNELS'
        time_steps           : 5
        chip_dims            : 128
        batch_size           : 2
        max_epoch_length     : 100
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
" > "$CONFIG_FPATH"

# Train with the above config for at least 1 epoch (should be very short)
# And then Ctrl+C to kill it
python -m geowatch.tasks.fusion fit --config "$CONFIG_FPATH"

# The following command should grab the most recent checkpoint
CKPT_FPATH=$(python -c "import pathlib; print(list(pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs').glob('*/checkpoints/*.ckpt'))[0])")
echo "CKPT_FPATH = $CKPT_FPATH"

# Calling fit again, but passing in the checkpoint should restart training from
# where it left off.
python -m geowatch.tasks.fusion fit --config "$CONFIG_FPATH" --ckpt_path="$CKPT_FPATH"
