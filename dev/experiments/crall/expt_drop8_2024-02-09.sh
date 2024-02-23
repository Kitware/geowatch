#!/bin/bash


size_parameter_adjuster(){
    python -c "if 1:
        import sympy
        import ubelt as ub
        limit_train_batches, batch_size, accumulate_grad_batches, max_epochs, MAX_STEPS = sympy.symbols(
            'limit_train_batches, batch_size, accumulate_grad_batches, max_epochs, MAX_STEPS')

        subs = {
            limit_train_batches: $ITEMS_PER_EPOCH,
            batch_size: $BATCH_SIZE,
            accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES,
            max_epochs: $MAX_EPOCHS,
            MAX_STEPS: $MAX_STEPS,
        }

        effective_batch_size = accumulate_grad_batches * batch_size
        #steps_per_epoch = sympy.floor(limit_train_batches / effective_batch_size)
        steps_per_epoch = limit_train_batches / effective_batch_size
        total_steps = max_epochs * steps_per_epoch
        total_steps.subs(subs)

        effective_batch_size_ = effective_batch_size.subs(subs).evalf()

        print(f'{effective_batch_size_=}')

        # The training progress iterator should show this number as the total number
        import math
        train_epoch_prog_iters = math.ceil((limit_train_batches / batch_size).subs(subs).evalf())

        diff = MAX_STEPS - total_steps
        curr_diff = diff.subs(subs)
        print(f'curr_diff={curr_diff.evalf()}')

        if curr_diff > 0:
            print('Not enough total steps to fill MAX_STEPS')
        else:
            print('MAX STEPS will stop training short')

        for k, v in subs.items():
            print('--- Possible Adjustment For ---')
            print(k)
            tmp_subs = (ub.udict(subs) - {k})
            solutions = sympy.solve(diff.subs(tmp_subs), k)
            solutions = [s.evalf() for s in solutions]
            print(solutions)
    "
}


prepare_splits(){
    DVC_DATA_DPATH=$(geowatch_dvc --tags=phase3_data --hardware="hdd")
    python -m geowatch.cli.prepare_splits \
        --src_kwcocos="$DVC_DATA_DPATH"/Aligned-Drop8-ARA/*/imganns-*.kwcoco.zip \
        --dst_dpath "$DVC_DATA_DPATH"/Aligned-Drop8-ARA \
        --suffix=rawbands \
        --backend=tmux --tmux_workers=6 \
        --splits=split6 \
        --run=0
}

# ARA All Sensor Retrain

export CUDA_VISIBLE_DEVICES="2"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='hdd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop8-ARA
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_allsensors_V2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4


PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")

echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
DEVICES        = $DEVICES
ACCELERATOR    = $ACCELERATOR
STRATEGY       = $STRATEGY

DDP_WORKAROUND = $DDP_WORKAROUND

TARGET_LR      = $TARGET_LR
WEIGHT_DECAY   = $WEIGHT_DECAY
PERTERB_SCALE  = $PERTERB_SCALE
"


MAX_STEPS=10000
MAX_EPOCHS=720
ITEMS_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6

DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.6
  neg_to_pos_ratio        : 1.0
  normalize_inputs        : 1024
  normalize_perframe      : false
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : $ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: false
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    input_channels: null
    input_sensorchan: null
    learning_rate: 0.001
    lr_scheduler: CosineAnnealingLR
    modulate_class_weights: ''
    multimodal_reduce: learned_linear
    name: unnamed_model
    negative_change_weight: 0.01
    ohem_ratio: null
    optimizer: RAdam
    perterb_scale          : $PERTERB_SCALE
    positional_dims: 48
    positive_change_weight: 1
    rescale_nans: null
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
    stream_channels: 16
    tokenizer: linconv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay : $WEIGHT_DECAY
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : $ACCELERATOR
  devices              : $DEVICES
  strategy             : $STRATEGY
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: 256
  limit_train_batches: $ITEMS_PER_EPOCH
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
          monitor: val_loss
          mode: min
          save_top_k: 5
          filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
          save_last: true
torch_globals:
    float32_matmul_precision: auto
initializer:
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"


# --------

# ARA All Sensor Retrain

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='hdd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop8-ARA
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_allsensors_V1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4


PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")

echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
DEVICES        = $DEVICES
ACCELERATOR    = $ACCELERATOR
STRATEGY       = $STRATEGY

DDP_WORKAROUND = $DDP_WORKAROUND

TARGET_LR      = $TARGET_LR
WEIGHT_DECAY   = $WEIGHT_DECAY
PERTERB_SCALE  = $PERTERB_SCALE
"


MAX_STEPS=10000
MAX_EPOCHS=720
ITEMS_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6

DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.6
  neg_to_pos_ratio        : 1.0
  normalize_inputs        : 1024
  normalize_perframe      : false
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : $ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: false
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    input_channels: null
    input_sensorchan: null
    learning_rate: 0.001
    lr_scheduler: CosineAnnealingLR
    modulate_class_weights: ''
    multimodal_reduce: learned_linear
    name: unnamed_model
    negative_change_weight: 0.01
    ohem_ratio: null
    optimizer: RAdam
    perterb_scale          : $PERTERB_SCALE
    positional_dims: 48
    positive_change_weight: 1
    rescale_nans: null
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
    stream_channels: 16
    tokenizer: linconv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay : $WEIGHT_DECAY
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : $ACCELERATOR
  devices              : $DEVICES
  strategy             : $STRATEGY
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 1
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: 256
  limit_train_batches: $ITEMS_PER_EPOCH
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
          monitor: val_loss
          mode: min
          save_top_k: 5
          filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
          save_last: true
torch_globals:
    float32_matmul_precision: auto
initializer:
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"

# --------
# ARA All Sensor Retrain

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='hdd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop8-ARA
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_51a7651a.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_da6461b7.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_allsensors_V3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4


PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")

echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
DEVICES        = $DEVICES
ACCELERATOR    = $ACCELERATOR
STRATEGY       = $STRATEGY

DDP_WORKAROUND = $DDP_WORKAROUND

TARGET_LR      = $TARGET_LR
WEIGHT_DECAY   = $WEIGHT_DECAY
PERTERB_SCALE  = $PERTERB_SCALE
"


MAX_STEPS=10000
MAX_EPOCHS=720
ITEMS_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6

DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.6
  neg_to_pos_ratio        : 1.0
  normalize_inputs        : 1024
  normalize_perframe      : false
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : $ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: false
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    input_channels: null
    input_sensorchan: null
    learning_rate: 0.001
    lr_scheduler: CosineAnnealingLR
    modulate_class_weights: ''
    multimodal_reduce: learned_linear
    name: unnamed_model
    negative_change_weight: 0.01
    ohem_ratio: null
    optimizer: RAdam
    perterb_scale          : $PERTERB_SCALE
    positional_dims: 48
    positive_change_weight: 1
    rescale_nans: null
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
    stream_channels: 16
    tokenizer: linconv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.1
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay : $WEIGHT_DECAY
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : $ACCELERATOR
  devices              : $DEVICES
  strategy             : $STRATEGY
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: 256
  limit_train_batches: $ITEMS_PER_EPOCH
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
          monitor: val_loss
          mode: min
          save_top_k: 5
          filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
          save_last: true
torch_globals:
    float32_matmul_precision: auto
initializer:
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"


# --------
# ARA All Sensor Retrain V4, changed schduler peak to 0.3, using higher LR

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='hdd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop8-ARA
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_51a7651a.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_da6461b7.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_allsensors_V4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")

echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
DEVICES        = $DEVICES
ACCELERATOR    = $ACCELERATOR
STRATEGY       = $STRATEGY

DDP_WORKAROUND = $DDP_WORKAROUND

TARGET_LR      = $TARGET_LR
WEIGHT_DECAY   = $WEIGHT_DECAY
PERTERB_SCALE  = $PERTERB_SCALE
"


MAX_STEPS=10000
MAX_EPOCHS=720
ITEMS_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6

DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.6
  neg_to_pos_ratio        : 1.0
  normalize_inputs        : 1024
  normalize_perframe      : false
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : $ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: false
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    input_channels: null
    input_sensorchan: null
    learning_rate: 0.001
    lr_scheduler: CosineAnnealingLR
    modulate_class_weights: ''
    multimodal_reduce: learned_linear
    name: unnamed_model
    negative_change_weight: 0.01
    ohem_ratio: null
    optimizer: RAdam
    perterb_scale          : $PERTERB_SCALE
    positional_dims: 48
    positive_change_weight: 1
    rescale_nans: null
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
    stream_channels: 16
    tokenizer: linconv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.3
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay : $WEIGHT_DECAY
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : $ACCELERATOR
  devices              : $DEVICES
  strategy             : $STRATEGY
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 50
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: 256
  limit_train_batches: $ITEMS_PER_EPOCH
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
          monitor: val_loss
          mode: min
          save_top_k: 5
          filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
          save_last: true
torch_globals:
    float32_matmul_precision: auto
initializer:
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"


# --------
# ARA All Sensor Retrain SCRATCH

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="2"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='hdd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop8-ARA
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_51a7651a.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_da6461b7.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_allsensors_scratch_V5
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.01)")
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
DEVICES=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(','.join(list(map(str, range(n)))) + ',')
")
ACCELERATOR=gpu
STRATEGY=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print('ddp' if n > 1 else 'auto')
")
DDP_WORKAROUND=$(python -c "if 1:
    import os
    n = len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    print(int(n > 1))
")

echo "
CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES
DEVICES        = $DEVICES
ACCELERATOR    = $ACCELERATOR
STRATEGY       = $STRATEGY

DDP_WORKAROUND = $DDP_WORKAROUND

TARGET_LR      = $TARGET_LR
WEIGHT_DECAY   = $WEIGHT_DECAY
PERTERB_SCALE  = $PERTERB_SCALE
"


MAX_STEPS=10000
MAX_EPOCHS=720
ITEMS_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6

DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=4 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_space_scale      : 10.0GSD
  input_space_scale       : 10.0GSD
  output_space_scale      : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.6
  neg_to_pos_ratio        : 1.0
  normalize_inputs        : 1024
  normalize_perframe      : false
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-3y,-2.5y,-2y,-1.5y,-1y,0,1y,1.5y,2y,2.5y,3y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : $ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p16
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: false
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    input_channels: null
    input_sensorchan: null
    learning_rate: 0.001
    lr_scheduler: CosineAnnealingLR
    modulate_class_weights: ''
    multimodal_reduce: learned_linear
    name: unnamed_model
    negative_change_weight: 0.01
    ohem_ratio: null
    optimizer: RAdam
    perterb_scale          : $PERTERB_SCALE
    positional_dims: 48
    positive_change_weight: 1
    rescale_nans: null
    saliency_head_hidden: 6
    saliency_loss: focal
    saliency_weights: auto
    stream_channels: 16
    tokenizer: linconv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.3
optimizer:
  class_path: torch.optim.AdamW
  init_args:gt
    weight_decay : $WEIGHT_DECAY
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : $ACCELERATOR
  devices              : $DEVICES
  strategy             : $STRATEGY
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 1
  logger: true
  max_epochs: $MAX_EPOCHS
  num_sanity_val_steps: 0
  limit_val_batches: $ITEMS_PER_EPOCH
  limit_train_batches: $ITEMS_PER_EPOCH
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
          monitor: val_loss
          mode: min
          save_top_k: 5
          filename: '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
          save_last: true
torch_globals:
    float32_matmul_precision: auto
initializer:
    init: noop
"


# shellcheck disable=SC2155
export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase3_data")
# shellcheck disable=SC2155
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase3_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "status" --dataset_codes "Aligned-Drop8-ARA"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Aligned-Drop8-ARA"
#python -m geowatch.mlops.manager "add packages" --dataset_codes "Aligned-Drop8-ARA"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Aligned-Drop8-ARA"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Aligned-Drop8-ARA"
