DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(smartwatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_wsmall_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_wsmall_split2.kwcoco.zip
# CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_scratch_10GSD_split2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=100000

# --shm-size=30g 
docker run -it \
    --env WATCH_GRID_WORKERS=0 \
    --user $(id -u):$(id -g) \
    --ipc=host \
    --memory=30g --cpus=8 --gpus='"device=1"' \
    --runtime=nvidia \
    --mount type=bind,source="$DVC_DATA_DPATH",target="$DVC_DATA_DPATH" \
    --mount type=bind,source="$DVC_EXPT_DPATH",target="$DVC_EXPT_DPATH" \
    --mount type=bind,source="/data/dvc-caches",target="/data/dvc-caches" \
    --mount type=bind,source="/data/connor.greenwell/cache",target="/.cache" \
    --mount type=bind,source="/data/connor.greenwell/config",target="/.config" \
    "watch/experiment-small_sites_positive:latest" \
    conda run --no-capture-output -n watch \
    python -m watch.tasks.fusion fit --config "
data:
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 5
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs: true
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout: 0.
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives: false
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
  num_workers: 0
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 128
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone:
      class_path: watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder
      init_args:
        encoder_depth: 6
        decoder_depth: 0
        dim: 224
        queries_dim: 96
        logits_dim: 128
        latent_dim_head: 1024
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: dicefocal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: linear
    pct_start: 0.05
optimizer:
  class_path: torch.optim.Adam
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-2
    betas:
      - 0.9
      - 0.98
    eps: 1e-10
trainer:
  accumulate_grad_batches: 1
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        mode: min
        save_top_k: 5
        auto_insert_metric_name: true
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu
  devices              : 0,
  #devices              : 0,1
  #strategy             : ddp
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  #log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"