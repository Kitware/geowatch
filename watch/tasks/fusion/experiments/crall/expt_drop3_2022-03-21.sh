#!/bin/bash
__notes__="
"

data_splits(){
    DVC_DPATH=$(python -m watch.cli.find_dvc)
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    python -m watch.cli.prepare_splits \
        --base_fpath="$DVC_DPATH/$DATASET_CODE/combo_LM.kwcoco.json" \
        --run=0 --backend=tmux
}


prep_teamfeat_drop2(){
    # Team Features on drop2
    DVC_DPATH=$(python -m watch.cli.find_dvc --hardware="ssd")
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    python -m watch.cli.prepare_teamfeats \
        --base_fpath="$DVC_DPATH/$DATASET_CODE/data.kwcoco.json" \
        --gres=",0" \
        --with_landcover=1 \
        --with_depth=0 \
        --with_materials=1 \
        --with_invariants=0 \
        --do_splits=1 \
        --depth_workers=0 \
        --cache=0 --run=0 --backend=tmux

        #--backend=slurm
    #python -m watch.cli.prepare_splits --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L.kwcoco.json --run=False

}




export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=Drop3_BASELINE_Template
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
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=380 \
    --time_steps=5 \
    --chip_overlap=0.0 \
    --time_sampling=soft+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --decoder=mlp \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=2048 \
    --draw_interval=5m \
    --num_draw=1 \
    --amp_backend=apex \
    --dist_weight=False \
    --use_centered_positives=True \
    --stream_channels=8 \
    --temporal_dropout=0 \
    --init="$INITIAL_STATE" \
    --amp_backend=apex \
    --num_sanity_val_steps=0 \
    --init="noop" \
       --package_fpath="$PACKAGE_FPATH" \
        --train_dataset="$TRAIN_FPATH" \
         --vali_dataset="$VALI_FPATH" \
         --test_dataset="$TEST_FPATH" \
         --num_sanity_val_steps=0 \
         --dump "$WORKDIR/configs/drop3_baseline_20220323.yaml"


# horologic abalate1
# ------------------

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=drop3_abalate1
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_baseline_20220323.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --init="$INITIAL_STATE" \
    --dump "$WORKDIR/configs/drop3_abalate1.yaml"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BOTH_V301
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BOTH_V302
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 


export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BAS_V303
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 


export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_SC_V304
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 


# toothbrush abalate1
# -------------------

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE="noop"
EXPERIMENT_NAME=bells_and_whistles
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_baseline_20220323.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.35 \
    --saliency_loss='dicefocal' \
    --class_loss='focal' \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=4 \
    --chip_size=380 \
    --time_steps=6 \
    --dist_weight=True \
    --time_sampling=hardish3 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=8 \
    --max_epochs=80 \
    --patience=80 \
    --max_epoch_length=2048 \
    --use_centered_positives=True \
    --stream_channels=8 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.1,Site Preparation*2.0" \
    --init="$INITIAL_STATE" \
    --dump "$WORKDIR/configs/bells_and_whistles_teamfeat.yaml"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_bells_mlp_V305
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/bells_and_whistles_teamfeat.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --decoder=mlp 

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_bells_seg_V306
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/bells_and_whistles_teamfeat.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --decoder=segmenter --init=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Aligned-Drop3-TA1-2022-03-10/runs/Drop3_bells_mlp_V305/lightning_logs/version_0/package-interupt/package_epoch0_step511.pt


# namek abalate1
# --------------

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=bells_and_whistles
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_baseline_20220323.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --num_workers=8 \
    --gpus "1" \
    --batch_size=1 \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --neg_to_pos_ratio=0.35 \
    --saliency_loss='dicefocal' \
    --class_loss='focal' \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --channels="$CHANNELS" \
    --accumulate_grad_batches=4 \
    --chip_size=256 \
    --time_steps=5 \
    --dist_weight=True \
    --time_sampling=hardish3 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=8 \
    --max_epochs=80 \
    --patience=80 \
    --max_epoch_length=2048 \
    --use_centered_positives=True \
    --normalize_inputs=10000 \
    --stream_channels=8 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.1,Site Preparation*2.0" \
    --init="$INITIAL_STATE" \
    --dump "$WORKDIR/configs/bells_and_whistles.yaml"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
EXPERIMENT_NAME=Drop3_bells_raw_mlp_V307
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/bells_and_whistles.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --decoder=mlp 

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_bells_raw_seg_V308
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/bells_and_whistles.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --decoder=segmenter 
