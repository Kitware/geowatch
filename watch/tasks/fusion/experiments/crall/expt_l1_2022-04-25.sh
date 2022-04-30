#!/bin/bash
#export CUDA_VISIBLE_DEVICES="1"


CROPPED_PRE_EVAL_AND_AGG(){

    #################################
    # 1. Repackage and commit new models
    #################################
    DVC_DPATH=$(smartwatch_dvc --hardware="ssd")
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH"
    git pull

    DATASET_CODE=Aligned-Drop3-L1
    EXPT_GROUP_CODE=Aligned-Drop3-L1
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    python -m watch.tasks.fusion.repackage gather_checkpoints \
        --dvc_dpath="$DVC_DPATH" \
        --storage_dpath="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages" \
        --train_dpath="$DVC_DPATH/training/$HOSTNAME/$USER/$DATASET_CODE/runs/*/lightning_logs" \
        --push_jobs=8 --dvc_remote=aws \
        --mode=commit

    # Note: this can take care of the gather checkpoint step for hard-coded paths
    # TODO: need to generalize a bit more
    python -m watch.tasks.fusion.dvc_sync_manager --push=True --pull=False --dvc_remote=aws

    #################################
    # 2. Pull new models (and existing evals) on eval machine
    #################################

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/Aligned-Drop3-L1/packages
    dvc pull -r aws -R models/fusion/Aligned-Drop3-L1/eval

    # TODO: add pull packages capability
    python -m watch.tasks.fusion.dvc_sync_manager --push=True --pull=False --dvc_remote=aws

    #################################
    # 3. Run Prediction & Evaluation
    #################################

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_CODE=Aligned-Drop3-L1
    EXPT_GROUP_CODE=Aligned-Drop3-L1
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    EXPT_MODEL_GLOBNAME="*"
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json

    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/$EXPT_MODEL_GLOBNAME/*.pt" \
            --test_dataset="$VALI_FPATH" \
            --enable_pred=1 \
            --enable_eval=1 \
            --enable_actclf=1 \
            --enable_actclf_eval=1 \
            --draw_heatmaps=0 \
            --without_alternatives \
            --skip_existing=1 --backend=tmux --run=1


    #################################
    # 4. Commit Evaluation Results
    #################################

    # TODO: Use new sycn machine state script

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    python -m watch.tasks.fusion.dvc_sync_manager --push=True --pull=False --dvc_remote=aws

    # Check for uncommited evaluations
    # shellcheck disable=SC2010
    ls -al models/fusion/Aligned-Drop3-L1/eval/*/*/*/*/eval/curves/measures2.json | grep -v ' \-> '
    # shellcheck disable=SC2010
    ls -al models/fusion/Aligned-Drop3-L1/eval/*/*/*/*/eval/actclf/*/*_eval/scores/merged/summary3.json | grep -v ' \-> '

    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/Aligned-Drop3-L1/eval/*/*/*/*/eval/actclf/*/*_eval/scores/merged/summary3.json
    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/Aligned-Drop3-L1/eval/*/*/*/*/eval/curves/measures2.json

    #dvc add models/fusion/Aligned-Drop3-L1/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 
    git commit -am "add measures from $HOSTNAME" && git pull && git push
    dvc push -r aws -R models/fusion/Aligned-Drop3-L1/eval
    dvc push -r aws -R models/fusion/*/eval

    
    #################################
    # 5. Aggregate Results
    #################################
    # Pull all results onto the machine you want to eval on
    python -m watch.tasks.fusion.dvc_sync_manager \
        --push=False --pull=True --dvc_remote=aws \
        --with_packages=False --with_evals=True
    #####
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    EXPT_GROUP_CODE=Aligned-Drop3-L1
    #EXPT_NAME_PAT="*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="*Drop3*"
    EXPT_NAME_PAT="*"
    MODEL_EPOCH_PAT="*"
    PRED_DSET_PAT="*"
    PRED_CFG_PAT="*"
    MEASURE_GLOBSTR=${DVC_DPATH}/models/fusion/${EXPT_GROUP_CODE}/eval/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/${PRED_CFG_PAT}/eval/curves/measures2.json

    GROUP_KEY="*Drop3*s2_wv*"
    #GROUP_KEY="*Drop3*"

    python -m watch.tasks.fusion.aggregate_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/$EXPT_GROUP_CODE" \
        --dset_group_key="$GROUP_KEY" --show=True \
        --classes_of_interest "Site Preparation" "Active Construction" 
            #"Post Construction"
}

DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

# Point to latest dataset version
DATASET_CODE=Aligned-Drop3-L1
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE

### TODO: CHANGE TO KWCOCO FILES THAT CONTAIN TEAM FEATURES
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json

CHANNELS="blue|green|red|nir|swir16|swir22"

INITIAL_STATE="noop"

EXPERIMENT_NAME=L1_BASELINE_EXPERIMENT_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME


export CUDA_VISIBLE_DEVICES=0
python -m watch.tasks.fusion.fit \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=L1_Template\
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
    --gpus "0" \
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
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --normalize_inputs=1024 \
    --max_epochs=40 \
    --patience=40 \
    --max_epoch_length=none \
    --draw_interval=5000m \
    --num_draw=1 \
    --amp_backend=apex \
    --init="$INITIAL_STATE" \
    --num_sanity_val_steps=0 \
    --dump "$WORKDIR/configs/drop3_l1_baseline_20220425.yaml"



# ooo
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-L1
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=L1_BASELINE_EXPERIMENT_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_l1_baseline_20220425.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --num_workers=4 \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-L1
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=L1_BASELINE_EXPERIMENT_V002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_l1_baseline_20220425.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --max_epoch_length=2048 \
    --max_epochs=160 \
    --num_workers=4 \
    --init="$INITIAL_STATE" 


## Namek
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-L1
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=L1_BASELINE_EXPERIMENT_V003
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_l1_baseline_20220425.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --max_epoch_length=2048 \
    --max_epochs=160 \
    --num_workers=6 \
    --gpus="1" \
    --temporal_dropout=0.5 \
    --init="$INITIAL_STATE" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-L1
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
INITIAL_STATE="noop"
EXPERIMENT_NAME=L1_BASELINE_EXPERIMENT_V004
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_l1_baseline_20220425.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --max_epoch_length=2048 \
    --max_epochs=160 \
    --num_workers=6 \
    --gpus="1" \
    --time_sampling=hardish3 \
    --temporal_dropout=0.5 \
    --stream_channels=16 \
    --use_centered_positives=True \
    --init="$INITIAL_STATE" 
