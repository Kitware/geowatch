#!/bin/bash

CROPPED_PRE_EVAL_AND_AGG(){

    #################################
    # Repackage and commit new models
    #################################

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="hdd")
    DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval3_sc_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    python -m watch.tasks.fusion.repackage gather_checkpoints \
        --dvc_dpath="$DVC_DPATH" \
        --storage_dpath="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages" \
        --train_dpath="$DVC_DPATH/training/$HOSTNAME/$USER/$DATASET_CODE/runs/*/lightning_logs" \
        --push_jobs=8 --dvc_remote=aws \
        --mode=commit

    #################################
    # Pull new models on eval machine
    #################################

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    git pull
    dvc pull -r horologic -R models/fusion/eval3_sc_candidates/packages

    #################################
    # Run Prediction & Evaluation
    #################################

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="hdd")
    DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval3_sc_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*.pt" \
            --test_dataset="$VALI_FPATH" \
            --skip_existing=True --backend=tmux --run=0 


    #################################
    # Commit Evaluation Results
    #################################
    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="hdd")
    # shellcheck disable=SC2010
    ls -al "$DVC_DPATH"/models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json | grep -v ' \-> '
    (cd "$DVC_DPATH" && dvc add models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json)
    git commit -am "add measures from $HOSTNAME" && git pull && git push
    (cd "$DVC_DPATH" && dvc push -r aws -R models/fusion/eval3_sc_candidates/eval)

    
    #################################
    # Aggregate Results
    #################################
    # Pull all results onto the machine you want to eval on
    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/eval3_sc_candidates/eval

    #####
    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="hdd")
    EXPT_GROUP_CODE=eval3_sc_candidates
    #EXPT_NAME_PAT="*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="*Drop3*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="BOTH_TA1_COMBO_TINY_p2w_raw*"
    MODEL_EPOCH_PAT="*"
    PRED_DSET_PAT="*"
    PRED_CFG_PAT="*"
    MEASURE_GLOBSTR=${DVC_DPATH}/models/fusion/${EXPT_GROUP_CODE}/eval/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/${PRED_CFG_PAT}/eval/curves/measures2.json

    python -m watch.tasks.fusion.gather_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/$EXPT_GROUP_CODE" \
        --dset_group_key="*Drop3*" --show=True \
        --classes_of_interest "Site Preparation" "Active Construction" "Post Construction"
}

# tooshbrush cropped
# ------------------

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=6 \
    --learning_rate=3e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_sampling=hardish3 \
    --time_span=6m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=1m \
    --use_centered_positives=True \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.1,Site Preparation*2.0"  \
    --init=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Aligned-Drop3-TA1-2022-03-10/runs/Drop3_SpotCheck_V319/lightning_logs/version_2/checkpoints/epoch=60-step=124927.ckpt 


# ooo
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_V003
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=9 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_sampling=hardish3 \
    --time_span=6m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --use_centered_positives=True \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.01,Site Preparation*2.0" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_V004
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_sampling=hardish3 \
    --time_span=12m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --use_centered_positives=True \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.01,Site Preparation*2.0" 


# namek

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_V004
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
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
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.01,Site Preparation*2.0" 

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
CHANNELS="red|green|blue|near-ir1|near-ir2|red-edge|yellow"
EXPERIMENT_NAME=CropDrop3_SC_V004
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
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
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.01,Site Preparation*2.0" 


# namek
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE

python -m watch.cli.prepare_splits \
    --base_fpath="$KWCOCO_BUNDLE_DPATH/data.kwcoco.json" \
    --run=0 --backend=serial


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_V005
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --saliency_loss='focal' \
    --class_loss='focal' \
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
    --temporal_dropout=0.5


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_V006
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --accumulate_grad_batches=8 \
    --saliency_loss='focal' \
    --class_loss='focal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
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
    --temporal_dropout=0.5


##### oooo

DVC_DPATH=$(python -m watch.cli.find_dvc)
INIT_STATE_V001=$DVC_DPATH/models/fusion/eval3_sc_candidates/pred/CropDrop3_SC_V001/pred_CropDrop3_SC_V001_epoch=90-step=186367-v1/Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco.pt
(cd "$DVC_DPATH" && dvc pull -r aws smart_watch_dvc/models/fusion/eval3_sc_candidates/pred/CropDrop3_SC_V004/pred_CropDrop3_SC_V004_epoch=36-step=75775/Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco.pt)


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_xver1_V007
INIT_STATE_V001=$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V001/CropDrop3_SC_V001_epoch=90-step=186367-v1.pt
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_sampling=hardish3 \
    --time_span=12m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --use_centered_positives=True \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --stream_channels=32 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V001"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(python -m watch.cli.find_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_xver1_V008
INIT_STATE_V001=$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V001/CropDrop3_SC_V001_epoch=90-step=186367-v1.pt
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=0.0003 \
    --saliency_loss='dicefocal' \
    --class_loss='focal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=3e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_sampling=hardish3 \
    --time_span=7m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --use_centered_positives=True \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --stream_channels=64 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.01,Site Preparation*2.0" \
    --init="$INIT_STATE_V001"
