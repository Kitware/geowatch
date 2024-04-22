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
    python -m geowatch.cli.queue_cli.prepare_splits \
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
# ARA All Sensor Finetune V4, changed schduler peak to 0.3, using higher LR

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


# --------
# ARA All Sensor Median Scratch V7, changed schduler peak to 0.3, lower perterb (horologic)

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_d2af9008.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_a1a7138e.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_Median10GSD_allsensors_scratch_V7
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")
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
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"


# --------
# ARA All Sensor Median Scratch V8, changed schduler peak to 0.3, lower perterb (horologic), back to lr=3e-3

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="2"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_d2af9008.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_a1a7138e.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_Median10GSD_allsensors_scratch_V8
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=3e-3

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")
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
    init: $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"


# --------
# ARA All Sensor Median Finetune V8, changed schduler peak to 0.3, lower perterb (namek)

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
DVC_EXPT2_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
#ls $KWCOCO_BUNDLE_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_Median10GSD_allsensors_V9
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")
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

#kwcoco validate $TRAIN_FPATH
#kwcoco validate $VALI_FPATH

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
    init: $DVC_EXPT2_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"



# --------
# ARA All Sensor Median Finetune V8, changed schduler peak to 0.3, lower perterb (horologic)

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="3"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
DVC_EXPT2_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
#ls $KWCOCO_BUNDLE_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_Median10GSD_allsensors_V10
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")
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

#kwcoco validate $TRAIN_FPATH
#kwcoco validate $VALI_FPATH

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
    init: $DVC_EXPT2_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
"


### Package up models for evaluation
# shellcheck disable=SC2155
#export DVC_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
# shellcheck disable=SC2155
export DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase3_expt")
cd "$DVC_EXPT_DPATH"
python -m geowatch.mlops.manager "status" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Drop7-MedianNoWinter10GSD-V2"

python -m geowatch.mlops.manager "status"        --dataset_codes "Drop8-Median10GSD-V1"
python -m geowatch.mlops.manager "push packages" --dataset_codes "Drop8-Median10GSD-V1"
python -m geowatch.mlops.manager "list packages" --dataset_codes "Drop8-Median10GSD-V1"


# Broken, why?
#- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch1_step16.pt





__initial_model_list="
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch187_step2632.pt
- $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch0_step1.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch245_step3444.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch261_step3668.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch270_step3794.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch320_step4494.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch328_step4606.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V2/Drop8_allsensors_V2_epoch30_step434.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V2/Drop8_allsensors_V2_epoch58_step826.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V2/Drop8_allsensors_V2_epoch59_step840.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V2/Drop8_allsensors_V2_epoch61_step868.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V2/Drop8_allsensors_V2_epoch64_step910.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch0_step14.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch1_step14.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch390_step5474.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch391_step5488.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch397_step5572.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch441_step6188.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch478_step6706.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch714_step10001.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V4/Drop8_allsensors_V4_epoch33_step476.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V4/Drop8_allsensors_V4_epoch43_step616.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V4/Drop8_allsensors_V4_epoch44_step630.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V4/Drop8_allsensors_V4_epoch51_step728.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V4/Drop8_allsensors_V4_epoch60_step854.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_scratch_V5/Drop8_allsensors_scratch_V5_epoch21_step308.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_scratch_V5/Drop8_allsensors_scratch_V5_epoch24_step350.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_scratch_V5/Drop8_allsensors_scratch_V5_epoch26_step378.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_scratch_V5/Drop8_allsensors_scratch_V5_epoch44_step630.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_scratch_V5/Drop8_allsensors_scratch_V5_epoch51_step728.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V01/Drop7_finetune_COLD_phase3_V01_epoch226_step4994.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch120_step2662.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch214_step4730.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch45_step1012.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch60_step1342.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch63_step1408.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch79_step1760.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch88_step1958.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch208_step4598.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch209_step4620.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch226_step4994.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch15_step1712.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch16_step1819.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch17_step1926.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch18_step2033.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch18_step2052.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch19_step2140.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch20_step2268.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch26_step2916.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch32_step3564.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch35_step3888.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch36_step3996.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch40_step4428.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch41_step4536.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch43_step4752.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch45_step4968.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch35_step3888.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch40_step4428.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch47_step5184.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch5_step648.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch86_step9396.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch167_step2352.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch172_step2422.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch177_step2492.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch180_step2534.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch182_step2562.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch0_step14.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch168_step2366.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch172_step2422.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch179_step2520.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch184_step2590.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch110_step1554.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch133_step1876.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch142_step2002.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch185_step2604.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch97_step1372.pt
"

### Evaluate
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop8
MLOPS_DPATH=$DVC_EXPT_DPATH/_preeval20_bas_grid

MODEL_SHORTLIST="
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch245_step3444.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch261_step3668.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V1/Drop8_allsensors_V1_epoch328_step4606.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V2/Drop8_allsensors_V2_epoch58_step826.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V2/Drop8_allsensors_V2_epoch64_step910.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch390_step5474.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch391_step5488.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V3/Drop8_allsensors_V3_epoch397_step5572.pt
- $DVC_EXPT_DPATH/models/fusion/Aligned-Drop8-ARA/packages/Drop8_allsensors_V4/Drop8_allsensors_V4_epoch51_step728.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch45_step1012.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch79_step1760.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V02/Drop7_finetune_COLD_phase3_V02_epoch88_step1958.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch208_step4598.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch209_step4620.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_finetune_COLD_phase3_V03/Drop7_finetune_COLD_phase3_V03_epoch226_step4994.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch172_step2422.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch180_step2534.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_V10/Drop8_Median10GSD_allsensors_V10_epoch182_step2562.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch172_step2422.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch179_step2520.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch187_step2632.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch110_step1554.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch185_step2604.pt
- $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V8/Drop8_Median10GSD_allsensors_scratch_V8_epoch97_step1372.pt
- $DVC_EXPT_DPATH/models/fusion/uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch35_step3888.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch40_step4428.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch47_step5184.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V05/Drop7_scratch_V05_epoch86_step9396.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch41_step4536.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch43_step4752.pt
- $DVC_EXPT_DPATH/models/fusion/Drop7-MedianNoWinter10GSD-V2/packages/Drop7_scratch_V04/Drop7_scratch_V04_epoch45_step4968.pt
"

mkdir -p "$MLOPS_DPATH"
echo "$MODEL_SHORTLIST" > "$MLOPS_DPATH/shortlist.yaml"

cat "$MLOPS_DPATH/shortlist.yaml"

geowatch schedule --params="
    pipeline: bas

    matrix:
        bas_pxl.package_fpath: $MLOPS_DPATH/shortlist.yaml

        bas_pxl.test_dataset:
            - $DVC_DATA_DPATH/Drop8-Median10GSD-V1/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-Median10GSD-V1/CN_C000/imganns-CN_C000-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-Median10GSD-V1/KW_C001/imganns-KW_C001-rawbands.kwcoco.zip
            - $DVC_DATA_DPATH/Drop8-Median10GSD-V1/CO_C001/imganns-CO_C001-rawbands.kwcoco.zip
        bas_pxl.chip_overlap: 0.3
        bas_pxl.chip_dims: auto
        bas_pxl.time_span: auto
        bas_pxl.time_sampling: soft4
        bas_poly.thresh:
            #- 0.10
            #- 0.20
            - 0.30
            - 0.325
            - 0.35
            - 0.375
            - 0.4
            - 0.425
        bas_poly.inner_window_size: 1y
        bas_poly.inner_agg_fn: mean
        bas_poly.norm_ord: inf
        bas_poly.polygon_simplify_tolerance: 1
        bas_poly.agg_fn: probs
        bas_poly.time_thresh:
            - 0.8
            #- 0.6
        bas_poly.resolution: 10GSD
        bas_poly.moving_window_size: null
        bas_poly.poly_merge_method: 'v2'
        bas_poly.min_area_square_meters: 7200
        bas_poly.max_area_square_meters: 8000000
        bas_poly.boundary_region: $TRUTH_DPATH/region_models
        bas_poly_eval.true_site_dpath: $TRUTH_DPATH/site_models
        bas_poly_eval.true_region_dpath: $TRUTH_DPATH/region_models
        bas_pxl.enabled: 1
        bas_pxl_eval.enabled: 1
        bas_poly_viz.enabled: 0
        bas_poly.enabled: 1
        bas_poly_eval.enabled: 1
    " \
    --root_dpath="$DVC_EXPT_DPATH/_preeval20_bas_grid" \
    --devices="0,1," --tmux_workers=4 \
    --backend=tmux --queue_name "_preeval20_bas_grid" \
    --skip_existing=1 \
    --run=1


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
MLOPS_DPATH=$DVC_EXPT_DPATH/_preeval20_bas_grid
python -m geowatch.cli.experimental.fixup_predict_kwcoco_metadata \
    --coco_fpaths "$MLOPS_DPATH/pred/flat/bas_pxl/*/pred.kwcoco.zip"


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $DVC_EXPT_DPATH/_preeval20_bas_grid
    " \
    --output_dpath="$DVC_EXPT_DPATH/_preeval20_bas_grid/aggregate" \
    --resource_report=0 \
    --eval_nodes="
        - bas_poly_eval
        #- bas_pxl_eval
    " \
    --plot_params="
        enabled: 1
        stats_ranking: 0
        min_variations: 1
        params_of_interest:
            - params.bas_poly.thresh
            - resolved_params.bas_pxl.channels
    " \
    --stdout_report="
        top_k: 10
        per_group: 1
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: final
        concise: 1
        show_csv: 0
    " \
    --rois="KR_R002,CN_C000,KW_C001,CO_C001"
    #--rois="KR_R002"
    #--rois="KR_R002,CN_C000"
    #--rois="CN_C000"


# Restrict
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"

REMOTE_DVC_EXPT_DPATH=$HOME/remote/yardrat/data/dvc-repos/smart_phase3_expt
python -m geowatch.mlops.aggregate \
    --pipeline=bas \
    --target "
        - $REMOTE_DVC_EXPT_DPATH/_preeval20_bas_grid
    " \
    --rois="KR_R002,CN_C000,KW_C001,CO_C001"
    --output_dpath="$DVC_EXPT_DPATH/_preeval20_bas_grid3/aggregate" \
    --snapshot

    #\
    #--resource_report=0 \
    #--eval_nodes="
    #    - bas_poly_eval
    #    #- bas_pxl_eval
    #" \
    #--plot_params="
    #    enabled: 1
    #    stats_ranking: 0
    #    min_variations: 1
    #    params_of_interest:
    #        #- params.bas_poly.thresh
    #        #- resolved_params.bas_pxl.channels
    #        - resolved_params.bas_pxl_fit.initializer.init
    #        #- normalized_params.bas_pxl_fit.initializer.init
    #        #- resolved_bas_pxl_fit.initializer.init
    #" \
    #--stdout_report="
    #    top_k: 10
    #    per_group: 1
    #    macro_analysis: 0
    #    analyze: 0
    #    print_models: True
    #    reference_region: final
    #    concise: 1
    #    show_csv: 0
    #" \

    ##--query "df['resolved_params.bas_pxl_fit.initializer.init'] != 'noop'" \

ipython -i -c "if 1:
    fpath = '/home/joncrall/.cache/xdev/snapshot_states/state_2024-03-11T104255-5.pkl'
    from xdev.embeding import load_snapshot
    load_snapshot(fpath, globals())

    rois = ['KR_R002', 'CN_C000', 'KW_C001', 'CO_C001']
    agg.build_macro_tables(rois)

    label_mappings = {
        'packages/Drop7-MedianNoWinter10GSD_bgrn_split6_V74/Drop7-MedianNoWinter10GSD_bgrn_split6_V74_epoch46_step4042.pt': 'D7-bgrn-V74',
        'uconn/D7-V2-COLD-candidate/epoch=203-step=4488.pt': 'D7-COLD-Eval18',
    }

    plot_config = {
        'min_variations': 1,
        'params_of_interest': ['resolved_params.bas_pxl_fit.initializer.init'],
        'label_mappings': label_mappings,
    }



"


# --------
# ARA All Sensor AC FineTune, no class rebalancing, First Pass

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-ARA-Cropped2GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
#ls $KWCOCO_BUNDLE_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_n041_9010cf6d.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_n006_2e7a3d8f.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8-ARA-Cropped2GSD-V1_allsensors_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5

WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")

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

MAX_STEPS=122880
MAX_EPOCHS=360
TRAIN_BATCHES_PER_EPOCH=2048
ACCUMULATE_GRAD_BATCHES=6
BATCH_SIZE=4
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"

#kwcoco validate $TRAIN_FPATH
#kwcoco validate $VALI_FPATH

DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 5
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_resolution       : 2.0GSD
  input_resolution        : 2.0GSD
  output_resolution       : 2.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.6
  neg_to_pos_ratio        : 1.0
  normalize_inputs        : 1024
  normalize_perframe      : false
  normalize_peritem       : 'blue|green|red|nir|pan'
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : False
  use_grid_negatives      : False
  verbose                 : 1
  max_epoch_length        : $ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
  observable_threshold   : 0.0
  quality_threshold      : 0.0
  weight_dilate          : 10
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p24
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: 0
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 0.05
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
"



# --------
# Point Based BAS V1 on yardrat

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-ARA-Median10GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_pointannv1_split6_n043_486cc4af.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_pointannv1_split6_n005_ba3fd747.kwcoco.zip
CHANNELS="(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8_ARA_Median10GSD_allsensors_pointannsv1_v1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4

PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")
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
MAX_EPOCHS=121
TRAIN_BATCHES_PER_EPOCH=2666
ACCUMULATE_GRAD_BATCHES=32
BATCH_SIZE=6
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")


python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"


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
  dist_weights            : True
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
  max_epoch_length        : $TRAIN_ITEMS_PER_EPOCH
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
  limit_val_batches: $TRAIN_BATCHES_PER_EPOCH
  limit_train_batches: $TRAIN_BATCHES_PER_EPOCH
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop8-Median10GSD-V1/packages/Drop8_Median10GSD_allsensors_scratch_V7/Drop8_Median10GSD_allsensors_scratch_V7_epoch187_step2632.pt
"


# --------
# ARA All Sensor AC FineTune, with Scotts rebalance, First Pass

#export CUDA_VISIBLE_DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware='ssd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware='hdd')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop8-ARA-Cropped2GSD-V1
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
#ls $KWCOCO_BUNDLE_DPATH
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6_n041_9010cf6d.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6_n006_2e7a3d8f.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(L8,S2,PD,WV):(blue|green|red)"
EXPERIMENT_NAME=Drop8-ARA-Cropped2GSD-V1_allsensors_rebalance_V002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5


#export VALI_FPATH
#export TRAIN_FPATH
## Ensure we can do a postgres database
#python -c "if 1:
#    import geowatch
#    import os
#    import ubelt as ub
#    vali_fpath = ub.Path(os.environ.get('VALI_FPATH'))
#    train_fpath = ub.Path(os.environ.get('TRAIN_FPATH'))
#    vali_dset = geowatch.coerce_kwcoco(vali_fpath, sqlview='postgresql')
#    train_dset = geowatch.coerce_kwcoco(train_fpath, sqlview='postgresql')
#    print(vali_dset._cached_hashid())
#    print(train_dset._cached_hashid())
#    print(vali_dset._orig_coco_fpath())
#    print(train_dset._orig_coco_fpath())
#"

WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
PERTERB_SCALE=$(python -c "print($TARGET_LR * 0.003)")

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

MAX_STEPS=122880
MAX_EPOCHS=360
TRAIN_BATCHES_PER_EPOCH=2048
ACCUMULATE_GRAD_BATCHES=6
BATCH_SIZE=4
TRAIN_ITEMS_PER_EPOCH=$(python -c "print($TRAIN_BATCHES_PER_EPOCH * $BATCH_SIZE)")

python -m geowatch.cli.experimental.recommend_size_adjustments \
    --MAX_STEPS=$MAX_STEPS \
    --MAX_EPOCHS=$MAX_EPOCHS \
    --BATCH_SIZE=$BATCH_SIZE \
    --ACCUMULATE_GRAD_BATCHES=$ACCUMULATE_GRAD_BATCHES \
    --TRAIN_BATCHES_PER_EPOCH="$TRAIN_BATCHES_PER_EPOCH" \
    --TRAIN_ITEMS_PER_EPOCH="$TRAIN_ITEMS_PER_EPOCH"

#kwcoco validate $TRAIN_FPATH
#kwcoco validate $VALI_FPATH

DDP_WORKAROUND=$DDP_WORKAROUND WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
data:
  batch_size              : $BATCH_SIZE
  num_workers             : 4
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 9
  chip_dims               : 196,196
  window_resolution       : 2.0GSD
  input_resolution        : 2.0GSD
  output_resolution       : 2.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.6
  normalize_inputs        : 1024
  normalize_perframe      : false
  normalize_peritem       : 'blue|green|red|nir|pan'
  resample_invalid_frames : 3
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1.0y,-0.5y,-0.25y,-0.08y,0.0y,0.08y,0.25y,0.5y,1.0y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : False
  use_grid_negatives      : False
  verbose                 : 1
  max_epoch_length        : $TRAIN_ITEMS_PER_EPOCH
  mask_low_quality        : false
  mask_samecolor_method   : null
  observable_threshold   : 0.0
  quality_threshold      : 0.0
  weight_dilate          : 10
  #sqlview                : postgresql
  neg_to_pos_ratio       : null
  balance_options :
      - attribute: old_has_class_of_interest
        weights:
           True:  0.6
           False: 0.4
      - attribute: contains_phase
        weights:
            False: 0.3
            True: 0.7
      - attribute: phases
        default_weight: 0.01
        weights:
            'No Activity': 0.2
            'Site Preparation': 0.3
            'Active Construction': 0.3
            'Post Construction': 0.2
      - attribute: region
model:
  class_path: watch.tasks.fusion.methods.MultimodalTransformer
  init_args:
    arch_name: smt_it_stm_p24
    attention_impl: exact
    attention_kwargs: null
    backbone_depth: null
    change_head_hidden: 6
    change_loss: cce
    class_head_hidden: 6
    class_loss: dicefocal
    class_weights: auto
    config: null
    continual_learning: 0
    decoder: mlp
    decouple_resolution: false
    dropout: 0.1
    focal_gamma: 2.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 0.5
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
    predictable_classes:
        - background
        - No Activity
        - Site Preparation
        - Active Construction
        - Post Construction
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
  limit_val_batches: $TRAIN_BATCHES_PER_EPOCH
  limit_train_batches: $TRAIN_BATCHES_PER_EPOCH
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
    init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84/Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548.pt
"
