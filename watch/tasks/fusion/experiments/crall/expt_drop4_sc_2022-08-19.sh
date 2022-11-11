#!/bin/bash


export CUDA_VISIBLE_DEVICES=0
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
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
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --num_workers=4 \
    --accelerator="gpu" \
    --devices "0," \
    --batch_size=1 \
    --accumulate_grad_batches=4 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --space_scale="3GSD" \
    --window_space_scale="3GSD" \
    --chip_dims=128 \
    --time_steps=24 \
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
    --use_centered_positives=False \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
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
         --dump "$WORKDIR/configs/drop4_SC_baseline_20220819.yaml"


export CUDA_VISIBLE_DEVICES=3
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data")
INITIAL_STATE_INVAR_V30="$PHASE1_DATA_DPATH"/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_invar_scratch_V030/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1.pt
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$INITIAL_STATE_INVAR_V30
EXPERIMENT_NAME=Drop4_SC_RGB_frominvar30_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_SC_baseline_20220819.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="(WV,PD,S2):blue|green|red" \
    --saliency_loss='dicefocal' \
    --space_scale="5GSD" \
    --window_space_scale="5GSD" \
    --chip_dims=96,96 \
    --time_steps=24 \
    --temporal_dropout=0.0 \
    --batch_size=6 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=8048 \
    --num_workers=2 \
    --max_epochs=160 \
    --patience=160 


export CUDA_VISIBLE_DEVICES=2
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_SC_RGB_scratch_V002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_SC_baseline_20220819.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="(WV,PD,S2):blue|green|red" \
    --saliency_loss='dicefocal' \
    --space_scale="3GSD" \
    --window_space_scale="3GSD" \
    --chip_dims=128,128 \
    --time_steps=12 \
    --temporal_dropout=0.0 \
    --batch_size=16 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=8048 \
    --max_epochs=160 \
    --patience=160 


### --- toothbrush

export CUDA_VISIBLE_DEVICES=1
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data" --hardware="hdd")
INITIAL_STATE_SC_V006="$PHASE1_DATA_DPATH"/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V006/CropDrop3_SC_V006_epoch=71-step=18431.pt
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$INITIAL_STATE_SC_V006
EXPERIMENT_NAME=Drop4_SC_RGB_from_sc006_V003
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_SC_baseline_20220819.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="(WV,PD,S2):blue|green|red" \
    --saliency_loss='dicefocal' \
    --space_scale="6GSD" \
    --window_space_scale="6GSD" \
    --chip_dims=96,96 \
    --time_steps=16 \
    --temporal_dropout=0.1 \
    --batch_size=12 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=8048 \
    --optim=RAdam \
    --num_workers=6 \
    --max_epochs=240 \
    --stream_channels=32 \
    --patience=240 






export CUDA_VISIBLE_DEVICES=1
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data" --hardware="hdd")
INITIAL_STATE_SC_V006="$PHASE1_DATA_DPATH"/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V006/CropDrop3_SC_V006_epoch=71-step=18431.pt
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=/home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_from_sc006_V003/lightning_logs/version_3/checkpoints/epoch=31-step=21472.ckpt
#INITIAL_STATE=/home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_from_sc006_V003_cont/lightning_logs/version_0/checkpoints/epoch=50-step=34221.ckpt
EXPERIMENT_NAME=Drop4_SC_RGB_from_sc006_V003_cont
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_SC_baseline_20220819.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="(WV,PD,S2):blue|green|red" \
    --saliency_loss='dicefocal' \
    --space_scale="6GSD" \
    --window_space_scale="6GSD" \
    --chip_dims=96,96 \
    --time_steps=16 \
    --temporal_dropout=0.1 \
    --batch_size=12 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=8048 \
    --optim=RAdam \
    --num_workers=6 \
    --max_epochs=240 \
    --stream_channels=32 \
    --patience=240 --auto_resume --sqlview=True


export CUDA_VISIBLE_DEVICES=1
PHASE1_DATA_DPATH=$(smartwatch_dvc --tags="phase1_data" --hardware="hdd")
INITIAL_STATE_SC_V006="$PHASE1_DATA_DPATH"/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V006/CropDrop3_SC_V006_epoch=71-step=18431.pt
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=/home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_from_sc006_V003_cont/lightning_logs/version_2/checkpoints/epoch=96-step=65087.ckpt
EXPERIMENT_NAME=Drop4_SC_RGB_from_sc006_V003_cont2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop4_SC_baseline_20220819.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --init="$INITIAL_STATE" \
    --arch_name=smt_it_joint_p8 \
    --channels="(WV,PD,S2):blue|green|red" \
    --saliency_loss='dicefocal' \
    --space_scale="6GSD" \
    --window_space_scale="6GSD" \
    --chip_dims=96,96 \
    --time_steps=16 \
    --temporal_dropout=0.1 \
    --batch_size=12 \
    --accumulate_grad_batches=1 \
    --max_epoch_length=8048 \
    --optim=AdamW \
    --num_workers=5 \
    --max_epochs=240 \
    --stream_channels=32 \
    --patience=240 --sqlview=True --torch_sharing_strategy=file_system



#### YARDRAT
export CUDA_VISIBLE_DEVICES=0
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="hdd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Drop4-SC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=$PHASE2_EXPT_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_invar_scratch_V030/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1.pt
CHANNELS="(S2,WV):blue|green|red"
EXPERIMENT_NAME=Drop4_tune_V30_V1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=8 WATCH_INIT_VERBOSE=100 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --accumulate_grad_batches=3 \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=3e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_sampling=soft2 \
    --time_span=7m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --use_centered_positives=False \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INITIAL_STATE"


#### Toothbrush
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Drop4-SC
TRAIN_FNAME=data_train.kwcoco.json
VALI_FNAME=data_vali.kwcoco.json
TEST_FNAME=data_vali.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
#CHANNELS="blue|green|red,invariants:0:16"
CHANNELS="(S2,WV):blue|green|red"
INITIAL_STATE=$PHASE2_EXPT_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_invar_scratch_V030/CropDrop3_SC_s2wv_invar_scratch_V030_epoch=78-step=53956-v1.pt
EXPERIMENT_NAME=Drop4_tune_V30_V2
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
WATCH_GRID_WORKERS=8 WATCH_INIT_VERBOSE=100 python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --accumulate_grad_batches=3 \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --input_space_scale="5GSD" \
    --window_space_scale="5GSD" \
    --output_space_scale="5GSD" \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=3e-5 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_sampling=soft2 \
    --time_span=7m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --use_centered_positives=False \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INITIAL_STATE"
