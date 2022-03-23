#!/bin/bash
__notes__="
"


prep_teamfeat_drop2(){
    # Team Features on Drop2
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


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop2-Aligned-TA1-2022-02-15
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE


### TODO: CHANGE TO KWCOCO FILES THAT CONTAIN TEAM FEATURES
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=Drop3_BASELINE_V301
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
    --max_epoch_length=none \
    --draw_interval=5m \
    --num_draw=1 \
    --amp_backend=apex \
    --dist_weight=False \
    --use_centered_positives=True \
    --stream_channels=8 \
    --temporal_dropout=0 \
    --init="$INITIAL_STATE"
