#!/bin/bash
__doc__='

# Sync yardrat
DVC_EXPT_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware=auto)
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6/runs/ "$DVC_EXPT_DPATH"

'

# Register wherever your data lives
geowatch_dvc add --name=smart_data --path="$HOME"/data/dvc-repos/smart_data_dvc --hardware=hdd --priority=100 --tags=phase2_data
geowatch_dvc add --name=smart_expt --path="$HOME"/data/dvc-repos/smart_expt_dvc --hardware=hdd --priority=100 --tags=phase2_expt
geowatch_dvc list

PHASE1_DATA_DPATH=$(geowatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
echo "PHASE1_DATA_DPATH = $PHASE1_DATA_DPATH"
echo "PHASE2_DATA_DPATH = $PHASE2_DATA_DPATH"
echo "PHASE2_EXPT_DPATH = $PHASE2_EXPT_DPATH"
INITIAL_STATE_V323="$PHASE1_DATA_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt


export CUDA_VISIBLE_DEVICES=0
PHASE2_DATA_DPATH=$HOME/data/dvc-repos/smart_data_dvc
PHASE2_EXPT_DPATH=$HOME/data/dvc-repos/smart_expt_dvc
#PHASE2_DATA_DPATH=$(geowatch_dvc)
#PHASE2_DATA_DPATH=$(geowatch_dvc --hardware="hdd")
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
#CHANNELS="blue|green|red|nir|swir16|swir22"
CHANNELS="blue|green|red"
INITIAL_STATE="noop"
EXPERIMENT_NAME=Drop4_BASELINE_Template
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='focal' \
    --class_loss='focal' \
    --num_workers=4 \
    --accelerator="gpu" \
    --devices "0," \
    --batch_size=1 \
    --accumulate_grad_batches=4 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_dims=380 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft2+distribute \
    --time_span=6m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --decoder=mlp \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=160 \
    --patience=160 \
    --max_epoch_length=2048 \
    --draw_interval=5min \
    --num_draw=1 \
    --eval_after_fit=False \
    --amp_backend=apex \
    --dist_weights=0 \
    --use_centered_positives=True \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --space_scale="30GSD" \
    --set_cover_algo=approx \
    --init="$INITIAL_STATE" \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="noop" \
       --package_fpath="$PACKAGE_FPATH" \
        --train_dataset="$TRAIN_FPATH" \
         --vali_dataset="$VALI_FPATH" \
         --test_dataset="$TEST_FPATH" \
         --num_sanity_val_steps=0 \
         --dump "$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml"


#### Retrain on Drop4
export CUDA_VISIBLE_DEVICES=1
PHASE1_DATA_DPATH=$(geowatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
INITIAL_STATE_V323="$PHASE1_DATA_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$INITIAL_STATE_V323
EXPERIMENT_NAME=Drop4_BAS_Retrain_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --accelerator="gpu" \
    --devices "0," \
    --num_workers=4 \
    --dist_weights=1 \
    --chip_dims=380,380 \
    --time_steps=11 \
    --batch_size=1 \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160 


#### Retrain on Drop4
export CUDA_VISIBLE_DEVICES=1
PHASE1_DATA_DPATH=$(geowatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
INITIAL_STATE_V323="$PHASE1_DATA_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$INITIAL_STATE_V323
EXPERIMENT_NAME=Drop4_BAS_Retrain_V002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red|nir|swir16|swir22" \
    --accelerator="gpu" \
    --devices "0," \
    --num_workers=4 \
    --chip_dims=380,380 \
    --time_steps=11 \
    --batch_size=1 \
    --dist_weights=1 \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160 


### ----------------


#### Continue training on 10GSD From V001
export CUDA_VISIBLE_DEVICES=0
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
#ls -altr $PHASE2_EXPT_DPATH/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_Retrain_V001/lightning_logs/version_3/checkpoints/
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$PHASE2_EXPT_DPATH/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_Retrain_V001/lightning_logs/version_3/checkpoints/epoch=129-step=66560.ckpt
EXPERIMENT_NAME=Drop4_BAS_Continue_10GSD_BGR_V003
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --window_space_scale="10GSD" \
    --chip_dims=380,380 \
    --time_steps=11 \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160

#### Continue training on 15GSD From V001 with dicefocal
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$PHASE2_EXPT_DPATH/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_Retrain_V001/lightning_logs/version_3/checkpoints/epoch=129-step=66560.ckpt
EXPERIMENT_NAME=Drop4_BAS_Continue_15GSD_BGR_V004
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --saliency_loss='dicefocal' \
    --window_space_scale="15GSD" \
    --chip_dims=320,320 \
    --max_epoch_length=16384 \
    --time_steps=11 \
    --batch_size=1 \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160


#### Continue training on 10GSD From V002
export CUDA_VISIBLE_DEVICES=2
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$PHASE2_EXPT_DPATH/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_Retrain_V002/lightning_logs/version_0/checkpoints/epoch=73-step=37888.ckpt
EXPERIMENT_NAME=Drop4_BAS_Continue_10GSD_BGRNSH_V005
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red|nir|swir16|swir22" \
    --window_space_scale="10GSD" \
    --max_epoch_length=4096 \
    --chip_dims=380,380 \
    --time_steps=11 \
    --batch_size=1 \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160 

#### Start from scratch at 10GSD
export CUDA_VISIBLE_DEVICES=3
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
#ls -altr $PHASE2_EXPT_DPATH/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_Retrain_V001/lightning_logs/version_3/checkpoints/
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$PHASE2_EXPT_DPATH/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_Retrain_V001/lightning_logs/version_3/checkpoints/epoch=129-step=66560.ckpt
EXPERIMENT_NAME=Drop4_BAS_Scratch_10GSD_BGR_V006
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --window_space_scale="10GSD" \
    --chip_dims=380,380 \
    --time_steps=11 \
    --batch_size=1 \
    --max_epoch_length=4096 \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160


#### On Toothbrush
#### Start from scratch at NATIVE GSD
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_Scratch_20GSD_BGRN_V007
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="blue|green|red|nir" \
    --saliency_loss='dicefocal' \
    --dist_weights=0 \
    --space_scale="20GSD" \
    --window_space_scale="20GSD" \
    --chip_dims=160,160 \
    --time_steps=5 \
    --batch_size=8 \
    --accumulate_grad_batches=1 \
    --max_epochs=160 \
    --patience=160 \
    --decouple_resolution=0


#### On Horologic (Train a joint saliency + change + class network)
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_BGR_15GSD_multihead_perceiver_V008
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=perceiver \
    --channels="blue|green|red" \
    --num_workers=5 \
    --global_change_weight=0.65 \
    --global_class_weight=0.60 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --space_scale="15GSD" \
    --window_space_scale="15GSD" \
    --chip_dims=128,128 \
    --time_steps=8 \
    --batch_size=8 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --stream_channels=32 \
    --saliency_head_hidden=4 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=8048 \
    --max_epochs=240 \
    --patience=240 \
    --backbone_depth=8 \
    --decouple_resolution=0


#### On Ooo
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_BGR_10GSD_multihead_V009
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="blue|green|red" \
    --num_workers=3 \
    --global_change_weight=0.75 \
    --global_class_weight=0.30 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --chip_dims=128,128 \
    --time_steps=8 \
    --batch_size=4 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --stream_channels=32 \
    --saliency_head_hidden=4 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=8048 \
    --max_epochs=240 \
    --patience=240 \
    --decouple_resolution=0 --auto_resume

export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=/home/joncrall/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_BGR_10GSD_multihead_V009/lightning_logs/version_3/checkpoints/epoch=49-step=100600.ckpt
EXPERIMENT_NAME=Drop4_BAS_BGR_10GSD_multihead_V009_cont
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="blue|green|red" \
    --num_workers=3 \
    --global_change_weight=0.001 \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --chip_dims=128,128 \
    --time_steps=8 \
    --batch_size=4 \
    --change_head_hidden=3 \
    --class_head_hidden=3 \
    --stream_channels=32 \
    --saliency_head_hidden=3 \
    --accumulate_grad_batches=4 \
    --max_epoch_length=8048 \
    --max_epochs=240 \
    --patience=240 \
    --optimizer=RAdam \
    --learning_rate=3e-4 \
    --decouple_resolution=0 


export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_BGR_10GSD_V010
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --num_workers=3 \
    --global_change_weight=0.75 \
    --global_class_weight=0.30 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --chip_dims=128,128 \
    --time_steps=8 \
    --batch_size=4 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --stream_channels=32 \
    --learning_rate=3e-4 \
    --accumulate_grad_batches=2 \
    --max_epoch_length=8048 \
    --max_epochs=240 \
    --patience=240 \
    --decouple_resolution=0 --auto_resume


#### On Yardrat

export CUDA_VISIBLE_DEVICES=0
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
#INITIAL_STATE=$PHASE2_EXPT_DPATH/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_Retrain_V001/lightning_logs/version_3/checkpoints/epoch=129-step=66560.ckpt
INITIAL_STATE=noop 
EXPERIMENT_NAME=Drop4_BAS_15GSD_BGR_Scratch_V011
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --saliency_loss='dicefocal' \
    --window_space_scale="15GSD" \
    --chip_dims=320,320 \
    --max_epoch_length=16384 \
    --global_change_weight=0.75 \
    --global_class_weight=0.30 \
    --global_saliency_weight=1.00 \
    --num_workers=8 \
    --time_steps=7 \
    --batch_size=16 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --accumulate_grad_batches=1 \
    --max_epochs=1600 \
    --patience=1600


# On Horologic
#### Retrain on Drop4
export CUDA_VISIBLE_DEVICES=0
PHASE1_DATA_DPATH=$(geowatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
INITIAL_STATE_V323="$PHASE1_DATA_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$INITIAL_STATE_V323
EXPERIMENT_NAME=Drop4_BAS_Retrain_V012
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red|nir|swir16|swir22" \
    --accelerator="gpu" \
    --devices "0," \
    --num_workers=4 \
    --chip_dims=380,380 \
    --class_loss='dicefocal' \
    --global_change_weight=0.75 \
    --global_class_weight=0.30 \
    --global_saliency_weight=1.00 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --time_steps=5 \
    --batch_size=4 \
    --dist_weights=1 \
    --accumulate_grad_batches=4 \
    --max_epoch_length=16384 \
    --max_epochs=1600 \
    --patience=1600 


# On Namek
#### Retrain on Drop4
export CUDA_VISIBLE_DEVICES=1
PHASE1_DATA_DPATH=$(geowatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
INITIAL_STATE_V323="$PHASE1_DATA_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$INITIAL_STATE_V323
EXPERIMENT_NAME=Drop4_BAS_Retrain_V013
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red|nir|swir16|swir22" \
    --accelerator="gpu" \
    --devices "0," \
    --num_workers=4 \
    --chip_dims=256,256 \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.75 \
    --global_class_weight=0.30 \
    --global_saliency_weight=1.00 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --time_steps=7 \
    --batch_size=4 \
    --dist_weights=1 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=16384 \
    --max_epochs=1600 \
    --patience=1600 

## On Toothbrush
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_BGR_10GSD_V014
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --num_workers=8 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --input_space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --chip_dims=128,128 \
    --time_steps=8 \
    --batch_size=64 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --stream_channels=32 \
    --in_features_pos=128 \
    --learning_rate=3e-4 \
    --max_epoch_length=8048 \
    --resample_invalid_frames=0 \
    --use_cloudmask=0 \
    --max_epochs=640 \
    --patience=340 \
    --decouple_resolution=0 --help


export CUDA_VISIBLE_DEVICES=0
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_BGR_10GSD_V015
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --num_workers=8 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --input_space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --chip_dims=128,128 \
    --time_steps=10 \
    --batch_size=32 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --stream_channels=32 \
    --in_features_pos=128 \
    --learning_rate=3e-4 \
    --max_epoch_length=8048 \
    --resample_invalid_frames=0 \
    --use_cloudmask=0 \
    --max_epochs=640 \
    --patience=340 \
    --decouple_resolution=0

export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_BGR_10GSD_V015_continue_v2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --arch_name=smt_it_stm_p8 \
    --channels="blue|green|red" \
    --num_workers=6 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --input_space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --chip_dims=128,128 \
    --time_steps=8 \
    --batch_size=48 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --stream_channels=32 \
    --in_features_pos=128 \
    --learning_rate=3e-4 \
    --max_epoch_length=16384 \
    --resample_invalid_frames=0 \
    --limit_val_batches=0 \
    --use_cloudmask=0 \
    --max_epochs=640 \
    --patience=340 \
    --decouple_resolution=0 \
    --draw_interval=1year \
    --num_draw=0 \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_15/checkpoints/epoch=3-step=4096.ckpt
    
#/home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_BGR_10GSD_V015/lightning_logs/version_0/checkpoints/epoch=43-step=2772.ckpt 



# Fit 
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=combo_train_I.kwcoco.json
VALI_FNAME=combo_vali_I.kwcoco.json
TEST_FNAME=combo_vali_I.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_invariants_30GSD_V016
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion fit \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --data.time_steps=3 \
    --data.chip_size=96 \
    --data.batch_size=2 \
    --data.input_space_scale=30GSD \
    --data.window_space_scale=30GSD \
    --data.output_space_scale=30GSD \
    --model=watch.tasks.fusion.methods.HeterogeneousModel \
    --model.name="$EXPERIMENT_NAME" \
    --optimizer=torch.optim.AdamW \
    --optimizer.lr=1e-3 \
    --trainer.accelerator="gpu" \
    --trainer.devices="0," \
    --data.channels="red|green|blue,invariants:17" 

python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_BAS_baseline_20220812.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --arch_name=smt_it_stm_p8 \
    --channels="L8:(red|green|blue,invariants:17)" \
    --num_workers=6 \
    --global_saliency_weight=1.00 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --change_loss='dicefocal' \
    --dist_weights=0 \
    --input_space_scale="30GSD" \
    --window_space_scale="30GSD" \
    --output_space_scale="30GSD" \
    --chip_dims=192,192 \
    --time_steps=8 \
    --batch_size=8 \
    --change_head_hidden=4 \
    --class_head_hidden=4 \
    --saliency_head_hidden=4 \
    --stream_channels=32 \
    --in_features_pos=128 \
    --learning_rate=3e-4 \
    --max_epoch_length=16384 \
    --resample_invalid_frames=0 \
    --use_cloudmask=0 \
    --max_epochs=640 \
    --patience=340 \
    --decouple_resolution=0 \
    --draw_interval=10minutes \
    --num_draw=2 \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_invariants_30GSD_V016/lightning_logs/version_4/checkpoints/epoch=15-step=8192.ckpt

#--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_invariants_30GSD_V016/lightning_logs/version_2/checkpoints/epoch=1-step=1024-v2.ckpt

#--init=/home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_BGR_10GSD_V015/lightning_logs/version_0/checkpoints/epoch=43-step=2772.ckpt 
#--init=/home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_BGR_10GSD_V015/lightning_logs/version_0/checkpoints/epoch=43-step=2772.ckpt 
#ls /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_invariants_30GSD_V016/lightning_logs/version_2/checkpoints/1024-v1.ckpt



### AAGGG FLASHFS!

PHASE2_DATA_DPATH_HDD=$(geowatch_dvc --tags="phase2_data" --hardware="hdd")
echo "PHASE2_DATA_DPATH_HDD = $PHASE2_DATA_DPATH_HDD"
PHASE2_DATA_DPATH_SSD=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
rsync -avprPR "$PHASE2_DATA_DPATH_SSD"/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/./_assets "$PHASE2_DATA_DPATH_HDD"/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/
rsync -avp "$PHASE2_DATA_DPATH_SSD"/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/*.kwcoco.json "$PHASE2_DATA_DPATH_HDD"/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/


rsync -avprPR toothbrush:data/dvc-repos/smart_data_dvc/.dvc/./cache "$HOME"/data/dvc-repos/smart_data_dvc/.dvc/


rsync -avprPR "$PHASE2_DATA_DPATH_HDD"/Drop4-BAS/./_assets "$PHASE2_DATA_DPATH_SSD"/Drop4-BAS
rsync -avp "$PHASE2_DATA_DPATH_HDD"/Drop4-BAS/*.kwcoco.json "$PHASE2_DATA_DPATH_SSD"/Drop4-BAS


PHASE2_DATA_DPATH_HDD=$(geowatch_dvc --tags="phase2_data" --hardware="hdd")
PHASE2_DATA_DPATH_SSD=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
rsync -avprPR "$PHASE2_DATA_DPATH_HDD"/Drop4-SC/./_assets "$PHASE2_DATA_DPATH_SSD"/Drop4-SC
rsync -avp "$PHASE2_DATA_DPATH_HDD"/Drop4-SC/*.kwcoco.json "$PHASE2_DATA_DPATH_SSD"/Drop4-SC




### Another attempt to resurect v323

### Run on OOO

export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='hdd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop4_TuneV323_BAS_BGRNSH_V1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
INITIAL_STATE_V323="$DVC_EXPT_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-5 \
    --weight_decay=1e-5 \
    --input_space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=True \
    --time_steps=11 \
    --channels="$CHANNELS" \
    --time_sampling=soft2+distribute \
    --time_span=6m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=False \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="$INITIAL_STATE_V323"
    #--config="$WORKDIR/configs/drop3_abalate1.yaml" \



### Run on Namek

export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='hdd')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='hdd')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop4_TuneV323_BAS_30GSD_BGRNSH_V2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
INITIAL_STATE_V323="$DVC_EXPT_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --input_space_scale="15GSD" \
    --window_space_scale="15GSD" \
    --output_space_scale="15GSD" \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=True \
    --time_steps=5 \
    --channels="$CHANNELS" \
    --time_sampling=soft2+distribute \
    --time_span=6m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=False \
    --normalize_inputs=128 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/training/namek/joncrall/Drop4-BAS/runs/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/lightning_logs/version_3/package-interupt/package_epoch0_step41.pt
    #--init=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/training/namek/joncrall/Drop4-BAS/runs/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/lightning_logs/version_2/package-interupt/package_epoch0_step23012.pt 
    #--init=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/training/namek/joncrall/Drop4-BAS/runs/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/lightning_logs/version_1/package-interupt/package_epoch0_step7501.pt
    #--init="$INITIAL_STATE_V323"
    #--config="$WORKDIR/configs/drop3_abalate1.yaml" \


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop4_TuneV323_BAS_10GSD_BGRNSH_V3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
INITIAL_STATE_V323="$DVC_EXPT_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --input_space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=True \
    --time_steps=5 \
    --channels="$CHANNELS" \
    --time_sampling=soft2+distribute \
    --time_span=6m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=False \
    --normalize_inputs=128 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/training/namek/joncrall/Drop4-BAS/runs/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/lightning_logs/version_4/package-interupt/package_epoch0_step23012.pt
    #--init=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/training/namek/joncrall/Drop4-BAS/runs/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/lightning_logs/version_3/package-interupt/package_epoch0_step41.pt
    #--init=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/training/namek/joncrall/Drop4-BAS/runs/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/lightning_logs/version_2/package-interupt/package_epoch0_step23012.pt 
    #--init=/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/training/namek/joncrall/Drop4-BAS/runs/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/lightning_logs/version_1/package-interupt/package_epoch0_step7501.pt
    #--init="$INITIAL_STATE_V323"
    #--config="$WORKDIR/configs/drop3_abalate1.yaml" \




### Toothbrush
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --saliency_weights="1:200" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0.00 \
    --global_class_weight=1e-17 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-3 \
    --chip_dims=224,224 \
    --window_space_scale="10GSD" \
    --input_space_scale="10GSD" \
    --output_space_scale="30GSD" \
    --accumulate_grad_batches=8 \
    --batch_size=2 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --time_steps=7 \
    --channels="$CHANNELS" \
    --neg_to_pos_ratio=0.1 \
    --time_sampling=soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=128 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --resample_invalid_frames=1 \
    --quality_threshold=0.8 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt

    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_15/package-interupt/package_epoch4_step5120.pt

    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_14/package-interupt/package_epoch10_step10734.pt 

    # /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_13
    # /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_13/package-interupt/package_epoch44_step46014.pt

    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_12/package-interupt/package_epoch15_step15697.pt

    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_11/package-interupt/package_epoch4_step2122.pt


    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_4/package-interupt/package_epoch0_step20591.pt
    

    
    #--init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
#/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRNSH_BGR_V4/lightning_logs/version_2/package-interupt/package_epoch0_step301.pt
#"$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt


### Ooo
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir"
EXPERIMENT_NAME=Drop4_BAS_2022_12_15GSD_BGRN_V5
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --weight_decay=1e-7 \
    --input_space_scale="15GSD" \
    --window_space_scale="15GSD" \
    --output_space_scale="15GSD" \
    --chip_dims=224,224 \
    --accumulate_grad_batches=4 \
    --batch_size=1 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --time_steps=11 \
    --channels="$CHANNELS" \
    --time_sampling=soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=False \
    --normalize_inputs=128 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --use_cloudmask=1 \
    --num_sanity_val_steps=0 \
    --init=/home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_2/checkpoints/epoch=1-step=77702.ckpt

    #--init=/home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_0/package-interupt/package_epoch0_step38851.pt

    #--init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
    #--init="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
    #--init
    
smartwatch model_stats /home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_0/package-interupt/package_epoch0_step38851.pt
/home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_2/package-interupt/package_epoch2_step98789.pt

    #--init="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt


### Horologic
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop4_BAS_2022_12_15GSD_BGRNSH_V5
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=3.14e-4 \
    --weight_decay=1e-7 \
    --chip_dims=224,224 \
    --window_space_scale="8GSD" \
    --input_space_scale="8GSD" \
    --output_space_scale="30GSD" \
    --accumulate_grad_batches=4 \
    --batch_size=2 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=1 \
    --dist_weights=False \
    --time_steps=11 \
    --channels="$CHANNELS" \
    --time_sampling=soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=30min \
    --num_draw=1 \
    --use_centered_positives=True \
    --normalize_inputs=256 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --resample_invalid_frames=1 \
    --use_cloudmask=1 \
    --num_sanity_val_steps=0 \
    --init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt


### Yardrat Heterogeneous Test
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="(S2,L8):blue|green|red|nir"
EXPERIMENT_NAME=Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion fit \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --model=HeterogeneousModel \
    --model.name=$EXPERIMENT_NAME \
    --model.init_args.global_change_weight=0.00 \
    --model.init_args.global_class_weight=0.00 \
    --model.init_args.global_saliency_weight=1.00 \
    --model.init_args.class_loss='focal' \
    --model.init_args.saliency_loss='focal' \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --data.test_dataset="$TEST_FPATH" \
    --data.channels="$CHANNELS" \
    --data.chip_dims=128,128 \
    --data.window_space_scale="30GSD" \
    --data.input_space_scale="30GSD" \
    --data.output_space_scale="30GSD" \
    --data.time_steps=3 \
    --data.time_sampling=soft2-contiguous-hardish3\
    --data.time_span=3m-6m-1y \
    --data.temporal_dropout=0.5 \
    --data.dist_weights=False \
    --data.neg_to_pos_ratio=0.2 \
    --data.use_centered_positives=True \
    --data.resample_invalid_frames=1 \
    --data.quality_threshold=0.6 \
    --data.normalize_inputs=128 \
    --data.batch_size=16 \
    --data.num_workers=6 \
    --trainer.accumulate_grad_batches=8 \
    --trainer.max_epochs=160 \
    --trainer.accelerator="gpu" \
    --trainer.devices "0," \
    --trainer.amp_backend=apex \
    --trainer.num_sanity_val_steps=0 \
    --optimizer=torch.optim.AdamW \
    --optimizer.init_args.lr=3e-3 \
    --optimizer.init_args.weight_decay=1e-7 


rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/./Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6 .

## Cant resume easilly
# --ckpt_path=/home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_2022_12_H_15GSD_BGRN_BGR_V6/lightning_logs/version_1/package-interupt/package_epoch0_step5578.pt

    #--init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
    #--trainer.max_epoch_length=16384  \
    #--draw_interval=1min \
    #--num_draw=4 \
    #--decoder=mlp \
    #--arch_name=smt_it_stm_p8 \
    #--trainer.patience=160 \


# Hard coded invariants on yardrat
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

python -m watch.tasks.invariants.predict \
    --input_kwcoco=$DVC_DATA_DPATH/Drop4-BAS/data_train.kwcoco.json \
    --output_kwcoco=$DVC_DATA_DPATH/Drop4-BAS/data_train_invar13_30GSD.kwcoco.json \
    --pretext_package=$DVC_EXPT_DPATH/models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt \
    --input_space_scale=10GSD  \
    --window_space_scale=10GSD \
    --patch_size=256 \
    --do_pca 0 \
    --patch_overlap=0.3 \
    --num_workers="8" \
    --write_workers 1 \
    --tasks before_after pretext

python -m watch.tasks.invariants.predict \
    --input_kwcoco=$DVC_DATA_DPATH/Drop4-BAS/data_vali.kwcoco.json \
    --output_kwcoco=$DVC_DATA_DPATH/Drop4-BAS/data_vali_invar13_30GSD.kwcoco.json \
    --pretext_package=$DVC_EXPT_DPATH/models/uky/uky_invariants_2022_12_17/TA1_pretext_model/pretext_package.pt \
    --input_space_scale=60GSD  \
    --window_space_scale=60GSD \
    --patch_size=256 \
    --do_pca 0 \
    --patch_overlap=0.3 \
    --num_workers="8" \
    --write_workers 1 \
    --tasks before_after pretext


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
python -m watch.cli.split_videos \
    --src "$DVC_DATA_DPATH/Drop4-BAS/data_train.kwcoco.json" \
          "$DVC_DATA_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
    --dst_dpath "$DVC_DATA_DPATH/Drop4-BAS/"


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
#data_train_PE_C001.kwcoco.json

python -m watch.cli.prepare_teamfeats \
    --base_fpath \
        "$DVC_DATA_DPATH/Drop4-BAS/data_train_PE_C001.kwcoco.json" \
        "$DVC_DATA_DPATH/Drop4-BAS/data_train_AE_R001.kwcoco.json" \
        "$DVC_DATA_DPATH/Drop4-BAS/data_train_US_C001.kwcoco.json" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=0 \
    --with_invariants2=1 \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=0 \
    --gres=0, --workers=1 --backend=tmux --run=0


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
python -m watch.cli.prepare_teamfeats \
    --base_fpath \
       "$DVC_DATA_DPATH/Drop4-BAS/data_train_*.kwcoco.json" \
       "$DVC_DATA_DPATH/Drop4-BAS/data_vali_*.kwcoco.json" \
    --expt_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_materials=0 \
    --with_invariants=0 \
    --with_invariants2=1 \
    --with_depth=0 \
    --do_splits=0 \
    --skip_existing=0 \
    --gres=0,1 --workers=2 --backend=tmux --run=0

kwcoco union --src ./*_train_*_uky_invariants*.kwcoco.json --dst combo_train_I2.kwcoco.json
kwcoco union --src ./*_vali_*_uky_invariants*.kwcoco.json --dst combo_vali_I2.kwcoco.json


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
rsync -avpP --include="*invariants*.kwcoco.json" --exclude="*" yardrat:data/dvc-repos/smart_data_dvc/Drop4-BAS/ "$DVC_DATA_DPATH/Drop4-BAS/"
rsync -avprPR yardrat:data/dvc-repos/smart_data_dvc/Drop4-BAS/./_assets/pred_invariants "$DVC_DATA_DPATH/Drop4-BAS/"

rsync -avpP --include="*invariants*.kwcoco.json" --exclude="*" "$DVC_DATA_DPATH/Drop4-BAS/" horologic:data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/ 
rsync -avprPR "$DVC_DATA_DPATH/Drop4-BAS/"./_assets/pred_invariants horologic:data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/ 




DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_0 "$DVC_EXPT_DPATH"



### Toothbrush Invariants
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_I2.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,invariants.0:17"
EXPERIMENT_NAME=Drop4_BAS_BGRNSH_invar_V7_alt
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --saliency_weights="auto" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=1.00 \
    --global_class_weight=0 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-8 \
    --weight_decay=1e-5 \
    --chip_dims=128,128 \
    --window_space_scale="10GSD" \
    --input_space_scale="10GSD" \
    --output_space_scale="30GSD" \
    --accumulate_grad_batches=4 \
    --batch_size=8 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=2 \
    --dist_weights=False \
    --time_steps=11 \
    --channels="$CHANNELS" \
    --neg_to_pos_ratio=0.5 \
    --time_sampling=soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=128 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --resample_invalid_frames=3 \
    --quality_threshold=0.8 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_BGRNSH_invar_V7_alt/lightning_logs/version_3/package-interupt/package_epoch0_step213.pt


    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V7/lightning_logs/version_4/package-interupt/package_epoch56_step29184.pt
    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V7/lightning_logs/version_3/package-interupt/package_epoch1_step706.pt
    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V7/lightning_logs/version_2/package-interupt/package_epoch11_step5936.pt 
#--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V7/lightning_logs/version_1/checkpoints/epoch=29-step=30720.ckpt
#--init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
#/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V7/lightning_logs/version_1/package-interupt/package_epoch30_step31003.pt


### Yardrat Invariants
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_I2.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
CHANNELS="blue|green|red|nir,invariants.0:17"
EXPERIMENT_NAME=Drop4_BAS_15GSD_BGRNSH_invar_V8
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --saliency_weights="1:200" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=5e-5 \
    --weight_decay=1e-3 \
    --chip_dims=196,196 \
    --window_space_scale="10GSD" \
    --input_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --accumulate_grad_batches=8 \
    --batch_size=4 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --time_steps=5 \
    --channels="$CHANNELS" \
    --neg_to_pos_ratio=0.1 \
    --time_sampling=uniform-soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=1024 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --resample_invalid_frames=3 \
    --quality_threshold=0.8 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --init=/home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_0/checkpoints/epoch=30-step=15872.ckpt


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8 "$DVC_EXPT_DPATH"

    #--init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
#save package_fpath = /home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_15GSD_BGRNSH_invar_V8/lightning_logs/version_0/package-interupt/package_epoch33_step17408.pt


### Horologic Invariants
export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_I2.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
CHANNELS="(S2,L8):(blue|green|red|nir,invariants.0:17)"
EXPERIMENT_NAME=Drop4_BAS_15GSD_BGRN_invar_V9
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion fit \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --model=HeterogeneousModel \
    --model.name=$EXPERIMENT_NAME \
    --model.init_args.global_change_weight=0.00 \
    --model.init_args.global_class_weight=0.00 \
    --model.init_args.global_saliency_weight=1.00 \
    --model.init_args.class_loss='focal' \
    --model.init_args.saliency_loss='focal' \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --data.test_dataset="$TEST_FPATH" \
    --data.channels="$CHANNELS" \
    --data.chip_dims=128,128 \
    --data.window_space_scale="10GSD" \
    --data.input_space_scale="10GSD" \
    --data.output_space_scale="30GSD" \
    --data.time_steps=5 \
    --data.time_sampling=uniform-soft2-contiguous-hardish3 \
    --data.time_span=3m-6m-1y \
    --data.temporal_dropout=0.5 \
    --data.dist_weights=False \
    --data.neg_to_pos_ratio=0.2 \
    --data.use_centered_positives=True \
    --data.resample_invalid_frames=3 \
    --data.mask_low_quality=True \
    --data.quality_threshold=0.6 \
    --data.observable_threshold=0.6 \
    --data.normalize_inputs=2048 \
    --data.batch_size=16 \
    --data.num_workers=6 \
    --trainer.accumulate_grad_batches=8 \
    --trainer.max_epochs=160 \
    --trainer.accelerator="gpu" \
    --trainer.devices "0," \
    --trainer.amp_backend=apex \
    --trainer.num_sanity_val_steps=0 \
    --optimizer=torch.optim.AdamW \
    --optimizer.init_args.lr=3e-4 \
    --optimizer.init_args.weight_decay=1e-3

### Toothbrush Invariants - V10
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_I2.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,invariants.0:17"
EXPERIMENT_NAME=Drop4_BAS_BGRNSH_invar_V10
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --saliency_weights="auto" \
    --class_loss='focal' \
    --saliency_loss='focal' \
    --global_change_weight=0 \
    --global_class_weight=0 \
    --global_saliency_weight=1.0 \
    --learning_rate=1e-4 \
    --weight_decay=1e-8 \
    --chip_dims=128,128 \
    --window_space_scale="10GSD" \
    --input_space_scale="10GSD" \
    --output_space_scale="30GSD" \
    --accumulate_grad_batches=4 \
    --batch_size=8 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
    --time_steps=11 \
    --channels="$CHANNELS" \
    --neg_to_pos_ratio=0.5 \
    --time_sampling=soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --resample_invalid_frames=3 \
    --quality_threshold=0.3 \
    --mask_low_quality=True \
    --num_sanity_val_steps=0 \
    --normalize_peritem="blue|green|red|nir|swir16|swir22" \
    --max_epoch_length=16384 \
    --num_workers=4 \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_BGRNSH_invar_V10/lightning_logs/version_8/package-interupt/package_epoch27_step14078.pt

    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_BGRNSH_invar_V10/lightning_logs/version_7/package-interupt/package_epoch0_step512.pt
    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_BGRNSH_invar_V10/lightning_logs/version_1/package-interupt/package_epoch4_step2560.pt
    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_BGRNSH_invar_V10/lightning_logs/version_0/package-interupt/package_epoch21_step10752.pt 
    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop4_BAS_BGRNSH_invar_V10/lightning_logs/version_0/package-interupt/package_epoch21_step10752.pt 


### Ooo run - V 10... again
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir"
EXPERIMENT_NAME=Drop4_BAS_2022_12_10GSD_BGRN_V11
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-5 \
    --weight_decay=1e-8 \
    --input_space_scale="10GSD" \
    --window_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --chip_dims=224,224 \
    --neg_to_pos_ratio=0.4 \
    --accumulate_grad_batches=16 \
    --batch_size=2 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=2 \
    --dist_weights=False \
    --time_steps=7 \
    --channels="$CHANNELS" \
    --normalize_peritem="blue|green|red|nir" \
    --time_sampling=soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=False \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --mask_low_quality=True \
    --use_cloudmask=1 \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --init=/home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_3/package-interupt/package_epoch7_step3908.pt

    #--init=/home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_0/package-interupt/package_epoch1_step8247.pt
    #--init=/home/joncrall/remote/Ooo/data/dvc-repos/smart_expt_dvc/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V5/lightning_logs/version_3/package-interupt/package_epoch6_step252174.pt


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
rsync -avprPR ooo:data/dvc-repos/smart_expt_dvc/./training/Ooo/joncrall/Drop4-BAS/runs "$DVC_EXPT_DPATH"
rsync -avprPR ooo:data/dvc-repos/smart_expt_dvc/./training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10 "$DVC_EXPT_DPATH"
rsync -avprPR ooo:data/dvc-repos/smart_expt_dvc/./training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_10GSD_BGRN_V11 "$DVC_EXPT_DPATH"

Drop4_BAS_2022_12_10GSD_BGRN_V11  Drop4_BAS_2022_12_15GSD_BGRN_V10  Drop4_BAS_2022_12_15GSD_BGRN_V5  Drop4_TuneV323_BAS_BGRNSH_V1


### Yardrat Invariants Scratch
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_I2.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
CHANNELS="blue|green|red|nir,invariants.0:17"
EXPERIMENT_NAME=Drop4_BAS_10GSD_BGRNSH_invar_V12
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --saliency_weights="1:100" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_class_weight=1e-5 \
    --global_change_weight=1e-5 \
    --global_saliency_weight=1.00 \
    --learning_rate=2e-5 \
    --weight_decay=1e-3 \
    --chip_dims=196,196 \
    --window_space_scale="10GSD" \
    --input_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --accumulate_grad_batches=4 \
    --batch_size=4 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --time_steps=5 \
    --channels="$CHANNELS" \
    --normalize_peritem="blue|green|red|nir" \
    --neg_to_pos_ratio=0.3 \
    --time_sampling=uniform-soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --multimodal_reduce=learned \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --resample_invalid_frames=3 \
    --lr_scheduler=CosineAnnealingLR \
    --mask_low_quality=True \
    --use_cloudmask=1 \
    --observable_threshold=0.5 \
    --quality_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --multimodal_reduce=learned_linear \
    --max_epoch_length=16384 

export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_I2.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_I2.kwcoco.json
CHANNELS="blue|green|red|nir,invariants.0:17"
EXPERIMENT_NAME=Drop4_BAS_10GSD_BGRNSH_invar_V12_cont
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --saliency_weights="1:70" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_class_weight=1e-5 \
    --global_change_weight=1e-5 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-2 \
    --chip_dims=196,196 \
    --window_space_scale="10GSD" \
    --input_space_scale="10GSD" \
    --output_space_scale="10GSD" \
    --accumulate_grad_batches=4 \
    --batch_size=4 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --time_steps=6 \
    --channels="$CHANNELS" \
    --normalize_peritem="blue|green|red|nir" \
    --neg_to_pos_ratio=0.5 \
    --time_sampling=uniform-soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --multimodal_reduce=learned \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --accelerator="gpu" \
    --devices "0," \
    --amp_backend=apex \
    --resample_invalid_frames=3 \
    --lr_scheduler=CosineAnnealingLR \
    --mask_low_quality=True \
    --use_cloudmask=1 \
    --observable_threshold=0.5 \
    --quality_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --multimodal_reduce=learned_linear \
    --max_epoch_length=16384  \
    --init=/home/local/KHQ/jon.crall/data/dvc-repos/smart_expt_dvc/training/yardrat/jon.crall/Drop4-BAS/runs/Drop4_BAS_10GSD_BGRNSH_invar_V12/lightning_logs/version_1/package-interupt/package_epoch74_step76224.pt

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop4-BAS/runs "$DVC_EXPT_DPATH"


### Continue of of the BGRN 10 model
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.json
CHANNELS="(L8,S2,PD,WV):(blue|green|red|nir)"
EXPERIMENT_NAME=Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal-gamma4' \
    --saliency_loss='dicefocal-gamma4' \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-2 \
    --input_space_scale="3.3GSD" \
    --window_space_scale="3.3GSD" \
    --output_space_scale="3.3GSD" \
    --chip_dims=320,320 \
    --neg_to_pos_ratio=0.4 \
    --batch_size=6 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --time_steps=5 \
    --time_span="6month" \
    --channels="$CHANNELS" \
    --saliency_weights="1:1" \
    --class_weights="auto:Site Preparation*0+1,Active Construction*0+3,Post Construction*0+1,No Activity*0+1,background*0+1,negative*0+1" \
    --time_sampling=uniform-soft2-contiguous-hardish3\
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=1min \
    --num_draw=3 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=False \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=True \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --balance_areas=True \
    --devices="0," \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/lightning_logs/version_5/package-interupt/package_epoch43_step59666.pt


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.json
CHANNELS="(L8,S2,PD,WV):(blue|green|red|nir)"
EXPERIMENT_NAME=Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal-gamma4' \
    --saliency_loss='dicefocal-gamma4' \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.30 \
    --learning_rate=1e-2 \
    --weight_decay=1e-6 \
    --input_space_scale="3.3GSD" \
    --window_space_scale="3.3GSD" \
    --output_space_scale="3.3GSD" \
    --chip_dims=196,196 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=6 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=3 \
    --time_steps=12 \
    --time_span="1month" \
    --channels="$CHANNELS" \
    --saliency_weights="1:1" \
    --class_weights="auto" \
    --time_sampling=uniform-soft2-contiguous-hardish3+pairwise+distribute \
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=2 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=False \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --balance_areas=False \
    --devices="0," \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/lightning_logs/version_1/package-interupt/package_epoch0_step2731.pt


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="(L8,S2,PD,WV):(blue|green|red|nir)"
EXPERIMENT_NAME=Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal-gamma4' \
    --saliency_loss='dicefocal-gamma4' \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.30 \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --input_space_scale="3.3GSD" \
    --window_space_scale="3.3GSD" \
    --output_space_scale="3.3GSD" \
    --chip_dims=196,196 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=6 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=3 \
    --time_steps=12 \
    --time_span="1month" \
    --channels="$CHANNELS" \
    --saliency_weights="1:1" \
    --class_weights="auto" \
    --time_sampling=uniform-soft2-contiguous-hardish3+pairwise+distribute \
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=2 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=False \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --balance_areas=False \
    --devices="0," \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/lightning_logs/version_1/package-interupt/package_epoch0_step2731.pt


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD,WV,WV1):(blue|green|red|nir)"
EXPERIMENT_NAME=Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal-gamma4' \
    --saliency_loss='dicefocal-gamma4' \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.30 \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --input_space_scale="3.3GSD" \
    --window_space_scale="3.3GSD" \
    --output_space_scale="3.3GSD" \
    --chip_dims=196,196 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=5 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=3 \
    --time_steps=12 \
    --time_span="1month" \
    --channels="$CHANNELS" \
    --saliency_weights="1:1" \
    --class_weights="auto" \
    --time_sampling=uniform-soft3-contiguous-hardish3+pairwise+distribute \
    --time_span=3m-6m-1y \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=True \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --balance_areas=True \
    --devices="0," \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop4-BAS/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT3/lightning_logs/version_2/checkpoints/epoch=21-step=60082.ckpt


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal-gamma4' \
    --saliency_loss='dicefocal-gamma4' \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.30 \
    --learning_rate=3e-4 \
    --weight_decay=1e-2 \
    --input_space_scale="3.3GSD" \
    --window_space_scale="3.3GSD" \
    --output_space_scale="3.3GSD" \
    --chip_dims=196,196 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=5 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=3 \
    --time_steps=14 \
    --channels="$CHANNELS" \
    --saliency_weights="1:1" \
    --class_weights="auto" \
    --time_sampling=uniform-soft3-contiguous-hardish3+pairwise+distribute \
    --time_span=3m-6m-1m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --weight_dilate=5 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=True \
    --normalize_inputs=16384 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=True \
    --mask_samecolor_method=None \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=2048 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --balance_areas=True \
    --devices="0," \
    --init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT5/lightning_logs/version_0/package-interupt/package_epoch3_step13108.pt


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_2022_12_10GSD_BGRN_V12
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --input_space_scale="3.3GSD" \
    --window_space_scale="3.3GSD" \
    --output_space_scale="3.3GSD" \
    --chip_dims=196,196 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=16 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --time_steps=7 \
    --channels="$CHANNELS" \
    --saliency_weights="1:1" \
    --class_weights="auto" \
    --time_sampling=uniform-soft3-contiguous-hardish3+distribute \
    --time_span=3m-6m-1m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --weight_dilate=5 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=True \
    --normalize_inputs=16384 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=True \
    --mask_samecolor_method=None \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --balance_areas=True \
    --devices="0," \
    --init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt  \
    --limit_val_batches=0.25


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal' \
    --exclude_sensors=WV,WV1,PD \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --input_space_scale="10.0GSD" \
    --window_space_scale="10.0GSD" \
    --output_space_scale="10.0GSD" \
    --chip_dims=196,196 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=16 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --time_steps=7 \
    --channels="$CHANNELS" \
    --saliency_weights="auto" \
    --class_weights="auto" \
    --time_sampling=uniform-soft5-soft4-contiguous \
    --time_kernel="(-1y,-2m,-1w,0,1w,2m,1y)" \
    --time_span=None \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --weight_dilate=10 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=True \
    --normalize_inputs=16384 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=False \
    --mask_samecolor_method=None \
    --observable_threshold=0.0 \
    --quality_threshold=0.0 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --devices="0," \
    --init="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_tune_3GSD_allheads/lightning_logs/version_0/package-interupt/package_epoch29_step30029.pt" \
    --limit_val_batches=0.25

    #--init="$DVC_EXPT_DPATH/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch159_step163840.pt" \
    #--init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt  \
    #--balance_areas=True \


export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
SELECT_VIDEOS='(.name | contains("_R"))'
echo "SELECT_VIDEOS = $SELECT_VIDEOS"
echo "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 4
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    time_steps             : 7
    exclude_sensors        : 'WV,WV1,PD'
    chip_dims              : 256,256
    fixed_resolution       : 8.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 8
    max_epoch_length       : 16384 
    time_steps             : 7
    channels               : '$CHANNELS'
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1y,-2m,-1w,0,1w,2m,1y)'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto 
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
optimizer: 
    class_path: torch.optim.AdamW
    init_args:
        lr           : 1e-4
        amsgrad      : true
        weight_decay : 1e-6
trainer:
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu 
    devices             : 0,
    #devices              : 0,1
    #strategy             : ddp 
    limit_val_batches    : 0.25
    num_sanity_val_steps : 0 
    max_epochs           : 360
initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/lightning_logs/version_0/package-interupt/package_epoch0_step277.pt
    
    #/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_tune_3GSD_allheads/lightning_logs/version_0/package-interupt/package_epoch29_step30029.pt
" > ~/code/watch/watch/tasks/fusion/experiments/crall/config.yaml
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config ~/code/watch/watch/tasks/fusion/experiments/crall/config.yaml


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V3_singlehead
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
SELECT_VIDEOS='(.name | contains("_R"))'
echo "SELECT_VIDEOS = $SELECT_VIDEOS"
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 4
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    time_steps             : 7
    exclude_sensors        : 'WV,WV1,PD'
    chip_dims              : 256,256
    fixed_resolution       : 8.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 8
    max_epoch_length       : 16384 
    time_steps             : 7
    channels               : '$CHANNELS'
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1y,-2m,-1w,0,1w,2m,1y)'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto 
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
optimizer: 
    class_path: torch.optim.AdamW
    init_args:
        lr           : 1e-4
        amsgrad      : true
        weight_decay : 1e-6
trainer:
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu 
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp 
    limit_val_batches    : 0.25
    num_sanity_val_steps : 0 
    max_epochs           : 360
initializer:
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_only_tune_3GSD_at_10GSD_L8_S2_V2/lightning_logs/version_0/package-interupt/package_epoch0_step277.pt
"

python -m watch.tasks.fusion fit --optimizer.help=AdamW

    #num_draw=4 
    #draw_interval=5min 


#/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Drop6/packages/Drop6_BAS_2022_12_10GSD_BGRN_V12/Drop6_BAS_2022_12_10GSD_BGRN_V12_v0_epoch159_step163840.pt

#--time_span=3m-6m-1m \
# /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V12/lightning_logs/version_4/packages/package_epoch160_step163840.pt


#16384 / 12301858
#65536
#65536 / 8388608
#12301858
#3507

    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT4/lightning_logs/version_0/package-interupt/package_epoch11_step36914.pt

    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT2/lightning_logs/version_0/package-interupt/package_epoch14_step40965.pt
    #--init=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/lightning_logs/version_8/package-interupt/package_epoch120_step328801.pt


    #--init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_BAS_2022_12_15GSD_BGRN_V10/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305.pt
    #--init="/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_2022_12_10GSD_BGRN_V11_CONT1/lightning_logs/version_3/package-interupt/package_epoch0_step255.pt"
    #--sqlview=sqlite \
    #--init="$DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_0/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305.pt"
#--init="$DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_0/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305.pt"
#smartwatch model_stats "$DVC_EXPT_DPATH/training/Ooo/joncrall/Drop4-BAS/runs/Drop4_BAS_2022_12_15GSD_BGRN_V10/lightning_logs/version_0/checkpoints/Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305.pt"



# YARDRAT
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split1.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split1.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_JOINT_Split1_5GSD_V13
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --class_loss='dicefocal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --input_space_scale="5GSD" \
    --window_space_scale="5GSD" \
    --output_space_scale="5GSD" \
    --chip_dims=196,196 \
    --neg_to_pos_ratio=1.0 \
    --batch_size=14 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=2 \
    --time_steps=6 \
    --channels="$CHANNELS" \
    --saliency_weights="1:1" \
    --class_weights="auto" \
    --time_sampling=uniform-soft3-contiguous-hardish3+distribute \
    --time_span=3m-6m-1m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --num_draw=4 \
    --weight_dilate=5 \
    --positive_change_weight=1 \
    --negative_change_weight=0.01 \
    --use_centered_positives=True \
    --normalize_inputs=16384 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --mask_low_quality=True \
    --mask_samecolor_method=None \
    --observable_threshold=0.6 \
    --num_sanity_val_steps=0 \
    --max_epoch_length=16384 \
    --min_spacetime_weight=0.6 \
    --accelerator=gpu \
    --balance_areas=False \
    --devices="0," \
    --init="$DVC_EXPT_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt  \
    --limit_val_batches=0.25
    #--sqlview=sqlite \
    

export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_scratch_10GSD_split2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
SELECT_VIDEOS='(.name | contains("_R"))'
echo "SELECT_VIDEOS = $SELECT_VIDEOS"
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 4
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    time_steps             : 7
    exclude_sensors        : 'WV,WV1,PD'
    chip_dims              : 256,256
    fixed_resolution       : 10.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 8
    max_epoch_length       : 16384 
    time_steps             : 7
    channels               : '$CHANNELS'
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1y,-2m,-1w,0,1w,2m,1y)'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : True
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    normalize_inputs       : 16384
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto 
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
optimizer: 
    class_path: torch.optim.AdamW
    init_args:
        lr           : 1e-4
        amsgrad      : true
        weight_decay : 1e-6
trainer:
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu 
    devices              : 0,
    #devices              : 0,1
    #strategy             : ddp 
    limit_val_batches    : 0.25
    num_sanity_val_steps : 0 
    max_epochs           : 360
"


#
# Yardrat landcover scratch, split2
PHASE2_DATA_DPATH_SSD=$(geowatch_dvc --tags="phase2_data" --hardware="hdd")
rsync -avp yardrat:data/dvc-repos/smart_data_dvc-ssd/Drop6/*.kwcoco.* "$PHASE2_DATA_DPATH_SSD"/Drop6
rsync -avprPR yardrat:data/dvc-repos/smart_data_dvc-ssd/Drop6/./_assets "$PHASE2_DATA_DPATH_SSD"/Drop6

PHASE2_DATA_DPATH_SSD=$(geowatch_dvc --tags="phase2_data" --hardware="ssd")
cd "$PHASE2_DATA_DPATH_SSD/Drop6"
python -m watch.cli.prepare_splits \
    --base_fpath "combo_imganns-*_L.kwcoco.json" \
    --suffix=fixquant \
    --constructive_mode=True


# On Yardrat
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V3
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
MAX_STEPS=50000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 4
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
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
    weight_decay: 1e-4
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V3 "$DVC_EXPT_DPATH"


# On Yardrat (smaller LR)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V5
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=50000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 4
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
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
    weight_decay: 1e-4
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V5 "$DVC_EXPT_DPATH"


# On Yardrat (longer,higher-decay,balanced,focal)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V6
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=100000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
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
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  balance_areas : True
  max_epoch_length: 16384
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
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
    weight_decay: 1e-3
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"

DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6/runs/ "$DVC_EXPT_DPATH"




# On Toothbrush (batch size change)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V4
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
MAX_STEPS=50000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 9
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
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
    weight_decay: 1e-4
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"


# On Toothbrush (back to smt)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=50000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    num_workers            : 4
    balance_areas          : True
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    time_steps: 5
    chip_dims: 128
    fixed_resolution       : 10.0GSD
    channels               : '$CHANNELS'
    batch_size: 32
    chip_overlap: 0
    dist_weights: 0
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    neg_to_pos_ratio: 0.25
    normalize_inputs       : 16384
    normalize_perframe: false
    resample_invalid_frames: true
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1y,-2w,0,2w,1y)'
    upweight_centers: true
    use_centered_positives : True
    use_grid_positives: true
    verbose: 1
    max_epoch_length: 3200
    mask_low_quality: true
    mask_samecolor_method: null
model:
    class_path: MultimodalTransformer
    init_args:
        saliency_weights       : auto 
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
    weight_decay: 1e-4
    betas:
      - 0.9
      - 0.99
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
    init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/lightning_logs/version_2/package-interupt/package_epoch79_step7900.pt
"

cd /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_raw_10GSD_split2_smt8_cont/lightning_logs/version_2


# On Yardrat (train longer, f16, cos aneal, adamw)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V8
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
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
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: upsample
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-4
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
  precision: 16
"
#rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V5 "$DVC_EXPT_DPATH"


# On Toothbrush (train longer, f16, cos aneal, adamw, VIT)
# BROKEN. WHY?
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_VITB16_1k_landcover_10GSD_split2_V8
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 1
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 672
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: vit_B_16
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
  precision: 16
"
#rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V5 "$DVC_EXPT_DPATH"


# On Toothbrush (train longer, f16, cos aneal, adamw, VIT)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_big_landcover_10GSD_split2_V9
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 1
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 256
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
        dim: 352
        queries_dim: 352
        logits_dim: 352
        latent_dim_head: 512
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
  precision: 16
"
#rsync -avprPR yardrat:data/dvc-repos/smart_expt_dvc/./training/yardrat/jon.crall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V5 "$DVC_EXPT_DPATH"


# On Toothbrush (train longer, f16, cos aneal, adamw, big Heterogeneous)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_big_landcover_10GSD_split2_V10
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  batch_size: 3
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 256
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
        dim: 352
        queries_dim: 96
        logits_dim: 352
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"

__error__='
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/torch/optim/optimizer.py", line 109, in wrapper                                                                                                                 [36/2647]
    return func(*args, **kwargs)                                 
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context                                                                                                               
    return func(*args, **kwargs)                                 
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/torch/optim/adamw.py", line 119, in step                                                                                                                                 
    loss = closure()                                             
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/plugins/precision/precision_plugin.py", line 108, in _wrap_closure                                                                                     
    self._after_closure(model, optimizer, optimizer_idx)         
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/plugins/precision/precision_plugin.py", line 85, in _after_closure                                                                                     
    self._track_grad_norm(trainer)                               
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/plugins/precision/precision_plugin.py", line 135, in _track_grad_norm                                                                                  
    trainer.lightning_module.log_grad_norm(grad_norm_dict)       
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/core/module.py", line 565, in log_grad_norm                                                                                                            
    self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)                                        
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/core/module.py", line 511, in log_dict                                                                                                                 
    self.log(                                                    
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/core/module.py", line 441, in log                                                                                                                      
    results.log(                                                 
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py", line 495, in log                                                                                       
    batch_size = self._extract_batch_size(self[key], batch_size, meta)                                                             
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/logger_connector/result.py", line 430, in _extract_batch_size                                                                       
    batch_size = extract_batch_size(self.batch)                  
  File "/home/joncrall/.pyenv/versions/3.10.10/envs/pyenv3.10.10/lib/python3.10/site-packages/pytorch_lightning/utilities/data.py", line 94, in extract_batch_size                                                                                                     
    raise MisconfigurationException(error_msg)                   
lightning_lite.utilities.exceptions.MisconfigurationException: We could not infer the batch_size from the batch. Either simplify its structure or provide the batch_size as `self.log(..., batch_size=batch_size)`.                                                    
'


# On Toothbrush (train longer, f16, cos aneal, adamw, big Heterogeneous)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_big_landcover_10GSD_split2_V11
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 3
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 256
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
        dim: 352
        queries_dim: 352
        logits_dim: 352
        latent_dim_head: 512
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 10
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"


# On Toothbrush (train longer, f16, cos aneal, adamw, big Heterogeneous)
export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_big_landcover_10GSD_split2_V11_cont
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=100000
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 2
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 256
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
        dim: 352
        queries_dim: 352
        logits_dim: 352
        latent_dim_head: 512
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 10
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
initializer:
  init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/lightning_logs/version_0/checkpoints/epoch=44-step=48015.ckpt
"
#init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/lightning_logs/version_0/package-interupt/package_epoch77_step82159.pt


/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_big_landcover_10GSD_split2_V12/lightning_logs/version_0


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_big_landcover_10GSD_split2_V13
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
MAX_STEPS=500000
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 7
  chip_dims: 128
  fixed_resolution       : 3.3GSD
  channels               : '$CHANNELS'
  batch_size: 2
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-6w,-2w,0,2w,6w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 256
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
        dim: 352
        queries_dim: 352
        logits_dim: 352
        latent_dim_head: 512
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-7
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 10
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"
#init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/lightning_logs/version_0/package-interupt/package_epoch77_step82159.pt
#

export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
#CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
#(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_sits_landcover_10GSD_split2_V13
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-5
MAX_STEPS=500000
WATCH_GRID_WORKERS=4 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 7
  chip_dims: 128
  fixed_resolution       : 3.3GSD
  channels               : '$CHANNELS'
  batch_size: 2
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-6w,-2w,0,2w,6w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 256
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: sits-former
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-7
    betas:
      - 0.9
      - 0.99
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
  log_every_n_steps: 10
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"
#init: /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_big_landcover_10GSD_split2_V11/lightning_logs/version_0/package-interupt/package_epoch77_step82159.pt
#


# On Toothbrush (train longer, f16, cos aneal, adamw, big Heterogeneous)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_big_landcover_10GSD_split2_V17
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 4
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 16384
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 256
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
        dim: 352
        queries_dim: 352
        logits_dim: 352
        latent_dim_head: 512
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-3
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 32
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
  log_every_n_steps: 10
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"


# On Toothbrush (train longer, f16, cos aneal, adamw, big Heterogeneous)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V18
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  batch_size: 2
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 16384
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-3
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 32
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
  log_every_n_steps: 10
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"


# On Toothbrush (cos aneal, adamw, SITS)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_sits_raw_10GSD_split2_V19
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=2e-4
MAX_STEPS=500000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  batch_size: 32
  num_workers            : 6
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 160
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: sits-former
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-4
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 4
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
initializer:
  init: $DVC_EXPT_DPATH/models/pretrained/sits-former/checkpoint.bert.tar
"



export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_sits_raw_eGSD_split2_V21
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
MAX_STEPS=100000
WATCH_GRID_WORKERS=2 python -m watch.tasks.fusion fit --config "
data:
  batch_size: 24
  num_workers            : 6
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 96
  fixed_resolution       : 2.7GSD
  channels               : '$CHANNELS'
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 160
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: sits-former
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 0.3
    saliency_loss: focal
    class_loss: dicefocal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 5e-7
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 4
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        mode: min
        save_top_k: 5
        auto_insert_metric_name: true
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu 
  #devices              : 0,
  devices              : 0,1
  strategy             : ddp 
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
  init: $DVC_EXPT_DPATH/models/pretrained/sits-former/checkpoint.bert.tar
"


# On Yardrat
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_fixquant_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_fixquant_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V22
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
MAX_STEPS=200000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  batch_size: 4
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    class_loss: dicefocal
    decoder: simple_conv
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
    weight_decay: 5e-7
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 8
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
"


export CUDA_VISIBLE_DEVICES=1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_sits_raw_eGSD_split2_V23
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
MAX_STEPS=100000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  batch_size: 8
  num_workers            : 5
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 5.0GSD
  channels               : '$CHANNELS'
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 160
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: sits-former
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 0.3
    saliency_loss: focal
    class_loss: dicefocal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-5
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 16
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
  limit_val_batches: 10
initializer:
  init: $DVC_EXPT_DPATH/models/pretrained/sits-former/checkpoint.bert.tar
"


export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
EXPERIMENT_NAME=Drop6_BAS_sits_raw_eGSD_split2_V24
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-3
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  batch_size: 8
  num_workers            : 5
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  input_space_scale       : 5.0GSD
  window_space_scale      : 5.0GSD
  output_space_scale      : 100.0GSD
  channels               : '$CHANNELS'
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio       : 1.0
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 160
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone: sits-former
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 1.0
    global_saliency_weight: 0.3
    saliency_loss: focal
    class_loss: dicefocal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-5
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 16
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: 2
  limit_val_batches: 10
initializer:
  init: $DVC_EXPT_DPATH/models/pretrained/sits-former/checkpoint.bert.tar
"


# On Toothbrush (train longer, f16, cos aneal, adamw, big Heterogeneous)
export CUDA_VISIBLE_DEVICES=0,1
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_L_fixed_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_L_fixed_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_nohidden_10GSD_split2_V31
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=5e-5
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
seed_everything: 1104562820436
data:
  batch_size              : 6
  num_workers             : 6
  train_dataset           : $TRAIN_FPATH
  vali_dataset            : $VALI_FPATH
  time_steps              : 5
  chip_dims               : 128
  fixed_resolution        : 10.0GSD
  channels                : '$CHANNELS'
  chip_overlap            : 0
  dist_weights            : 0
  min_spacetime_weight    : 0.5
  neg_to_pos_ratio        : 0.25
  normalize_inputs        : 16384
  normalize_perframe      : false
  resample_invalid_frames : true
  temporal_dropout        : 0.5
  time_sampling           : uniform-soft5-soft4-contiguous
  time_kernel             : '(-1y,-2w,0,2w,1y)'
  upweight_centers        : true
  use_centered_positives  : True
  use_grid_positives      : true
  verbose                 : 1
  max_epoch_length        : 16384
  mask_low_quality        : true
  mask_samecolor_method   : null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
    position_encoder:
      class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
      init_args:
        in_dims: 3
        max_freq: 3
        num_freqs: 16
    backbone:
      class_path: watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder
      init_args:
        encoder_depth: 4
        decoder_depth: 0
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
    spatial_scale_base: 1.0
    temporal_scale_base: 1.0
    global_change_weight: 0.0
    global_class_weight: 0.0
    global_saliency_weight: 1.0
    saliency_loss: focal
    decoder: simple_conv
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 5e-7
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 8
  callbacks:
    - class_path: pytorch_lightning.callbacks.ModelCheckpoint
      init_args:
        monitor: val_loss
        mode: min
        save_top_k: 5
        auto_insert_metric_name: true
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : gpu 
  #devices              : 0,
  devices              : 0,1
  strategy             : ddp 
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  enable_model_summary: true
  log_every_n_steps: 10
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  track_grad_norm: -1
  limit_val_batches: 64
  precision: bf16
"


# On Ooo
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop6
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_split2.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_split2.kwcoco.zip
CHANNELS="(L8,S2,PD):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan,(S2):(water|forest|field|impervious|barren|landcover_hidden.0:32)"
EXPERIMENT_NAME=Drop6_BAS_scratch_landcover_10GSD_split2_V24
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
MAX_STEPS=50000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
  num_workers            : 4
  batch_size: 2
  train_dataset          : $TRAIN_FPATH
  vali_dataset           : $VALI_FPATH
  time_steps: 5
  chip_dims: 128
  fixed_resolution       : 10.0GSD
  channels               : '$CHANNELS'
  chip_overlap: 0
  dist_weights: 0
  min_spacetime_weight: 0.5
  neg_to_pos_ratio: 0.25
  normalize_inputs       : 16384
  normalize_perframe: false
  resample_invalid_frames: true
  temporal_dropout       : 0.5
  time_sampling          : uniform-soft5-soft4-contiguous
  time_kernel            : '(-1y,-2w,0,2w,1y)'
  upweight_centers: true
  use_centered_positives : True
  use_grid_positives: true
  verbose: 1
  max_epoch_length: 3200
  mask_low_quality: true
  mask_samecolor_method: null
model:
  class_path: watch.tasks.fusion.methods.HeterogeneousModel
  init_args:
    token_width: 8
    token_dim: 64
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
        dim: 160
        queries_dim: 96
        logits_dim: 64
        latent_dim_head: 256
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
    anneal_strategy: cos
    pct_start: 0.05
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: 1e-6
    betas:
      - 0.9
      - 0.99
trainer:
  accumulate_grad_batches: 8
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
  log_every_n_steps: 5
  logger: true
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  replace_sampler_ddp: true
  limit_val_batches: 10
  track_grad_norm: 2
"
