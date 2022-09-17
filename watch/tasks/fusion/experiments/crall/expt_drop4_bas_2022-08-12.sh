#!/bin/bash
# Register wherever your data lives
smartwatch_dvc add --name=smart_data --path="$HOME"/data/dvc-repos/smart_data_dvc --hardware=hdd --priority=100 --tags=phase2_data
smartwatch_dvc add --name=smart_expt --path="$HOME"/data/dvc-repos/smart_expt_dvc --hardware=hdd --priority=100 --tags=phase2_expt
smartwatch_dvc list

PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
echo "PHASE1_DATA_DPATH = $PHASE1_DATA_DPATH"
echo "PHASE2_DATA_DPATH = $PHASE2_DATA_DPATH"
echo "PHASE2_EXPT_DPATH = $PHASE2_EXPT_DPATH"
INITIAL_STATE_V323="$PHASE1_DATA_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt


export CUDA_VISIBLE_DEVICES=0
PHASE2_DATA_DPATH=$HOME/data/dvc-repos/smart_data_dvc
PHASE2_EXPT_DPATH=$HOME/data/dvc-repos/smart_expt_dvc
#PHASE2_DATA_DPATH=$(smartwatch_dvc)
#PHASE2_DATA_DPATH=$(smartwatch_dvc --hardware="hdd")
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
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware='ssd')
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data")
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
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
    --init="$INITIAL_STATE" \
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
    --max_epoch_length=8048 \
    --resample_invalid_frames=0 \
    --use_cloudmask=0 \
    --max_epochs=640 \
    --patience=340 \
    --decouple_resolution=0 \
    --init=/home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_BGR_10GSD_V015/lightning_logs/version_0/checkpoints/epoch=43-step=2772.ckpt 
