#!/bin/bash

CROPPED_PRE_EVAL_AND_AGG(){

    #################################
    # 1. Repackage and commit new models
    #################################

    DVC_DPATH=$(smartwatch_dvc --hardware="ssd")
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH"
    git pull

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
    # 2. Pull new models (and existing evals) on eval machine
    #################################

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/eval3_sc_candidates/packages
    dvc pull -r aws -R models/fusion/eval3_sc_candidates/eval

    #################################
    # 3. Run Prediction & Evaluation
    #################################

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval3_sc_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE

    EXPT_MODEL_GLOBNAME="CropDrop3_SC_s2wv_tf_*V02*"

    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_D_wv_vali.kwcoco.json
    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json

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
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")

    # Check for uncommited evaluations
    # shellcheck disable=SC2010
    ls -al models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json | grep -v ' \-> '
    # shellcheck disable=SC2010
    ls -al models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/actclf/*/*_eval/scores/merged/summary3.json | grep -v ' \-> '

    #dvc unprotect models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 
    #dvc unprotect models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json
    #dvc add models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json

    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/actclf/*/*_eval/scores/merged/summary3.json
    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json

    #dvc add models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 
    git commit -am "add measures from $HOSTNAME" && git pull && git push
    dvc push -r aws -R models/fusion/eval3_sc_candidates/eval
    dvc push -r aws -R models/fusion/*/eval

    
    #################################
    # 5. Aggregate Results
    #################################
    # Pull all results onto the machine you want to eval on
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/eval3_sc_candidates/eval

    #####
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    EXPT_GROUP_CODE=eval3_sc_candidates
    #EXPT_NAME_PAT="*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="*Drop3*"
    EXPT_NAME_PAT="*"
    EXPT_NAME_PAT="*tf*"
    #EXPT_NAME_PAT="BOTH_TA1_COMBO_TINY_p2w_raw*"
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


special_evaluation(){
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    source "$HOME"/local/init/utils.sh
    #smartwatch model_info models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_tf_xver7_V013/CropDrop3_SC_s2wv_tf_xver7_V013_epoch=0-step=2047-v1.pt


    #models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V001/CropDrop3_SC_V001_epoch=55-step=114687-v1.pt
    writeto models/fusion/eval3_sc_candidates/models_of_interest.txt "
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V001/CropDrop3_SC_V001_epoch=1-step=4095-v1.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V001/CropDrop3_SC_V001_epoch=20-step=43007-v1.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V001/CropDrop3_SC_V001_epoch=90-step=186367-v1.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V003/CropDrop3_SC_V003_epoch=17-step=36863-v1.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V003/CropDrop3_SC_V003_epoch=30-step=63487.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V004/CropDrop3_SC_V004_epoch=100-step=206847.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V004/CropDrop3_SC_V004_epoch=11-step=24575-v2.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V005/CropDrop3_SC_V005_epoch=1-step=4095.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V006/CropDrop3_SC_V006_epoch=13-step=3583-v1.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V006/CropDrop3_SC_V006_epoch=71-step=18431.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_raw_xver7_V012/CropDrop3_SC_s2wv_raw_xver7_V012_epoch=0-step=2047-v1.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_wvonly_D_V011/CropDrop3_SC_wvonly_D_V011_epoch=129-step=266239.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_wvonly_D_V011/CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=14-step=30719.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=17-step=36863.pt
        models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V008/CropDrop3_SC_xver1_V008_epoch=26-step=55295-v1.pt
    "

    MODEL_GLOBSTR="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*.pt"
    MODEL_GLOBSTR="$DVC_DPATH"/models/fusion/eval3_sc_candidates/models_of_interest.txt

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval3_sc_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_wv_vali.kwcoco.json
    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_D_wv_vali.kwcoco.json
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1,2,3,4,5,6,7,8" \
            --model_globstr="$MODEL_GLOBSTR" \
            --test_dataset="$VALI_FPATH" \
            --enable_pred=0 \
            --enable_eval=0 \
            --enable_track=0 \
            --enable_iarpa_eval=0 \
            --enable_actclf=1 \
            --enable_actclf_eval=1 \
            --draw_heatmaps=1 \
            --draw_curves=1 \
            --pred_workers=4 \
            --chip_overlap=0.3 \
            --tta_time=0 \
            --tta_fliprot=0 \
            --hack_sc_grid=1 \
            --skip_existing=1 --backend=tmux --run=0
}


prep_features(){
    export CUDA_VISIBLE_DEVICES=1
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")

    echo "DVC_DPATH = $DVC_DPATH"
    BASE_DPATH="$DVC_DPATH/Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json"
    python -m watch.cli.prepare_teamfeats \
        --base_fpath="$BASE_DPATH" \
        --dvc_dpath="$DVC_DPATH" \
        --gres="0,1" \
        --with_landcover=1 \
        --with_depth=1 \
        --with_materials=1 \
        --with_invariants=1 \
        --do_splits=1 \
        --depth_workers=0 \
        --cache=1 --backend=tmux --run=0

    # Or rsync features

    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./_assets ooo:data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10
    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./combo* ooo:data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10

    rsync -avprRP --compress "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./_assets horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10 
    rsync -avprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./combo* horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10

    rsync -avprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./combo_DLM_s2_wv_vali.kwcoco.json horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10
    rsync -avprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./combo_DLM_*.kwcoco.json horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10


    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./_assets horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10
    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./combo* horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10
    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./dzyne* horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10
    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc/Cropped-Drop3-TA1-2022-03-10/./rutgers* horologic:data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10


    # Move to ssd on horologic
    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10/./_assets "$HOME"/data/dvc-repos/smart_watch_dvc-ssd/Cropped-Drop3-TA1-2022-03-10
    rsync -azvprRP "$HOME"/data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop3-TA1-2022-03-10/./combo* "$HOME"/data/dvc-repos/smart_watch_dvc-ssd/Cropped-Drop3-TA1-2022-03-10

}

# tooshbrush cropped
# ------------------

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE

python -m watch.cli.prepare_splits \
    --base_fpath="$KWCOCO_BUNDLE_DPATH/data.kwcoco.json" \
    --run=0 --backend=serial


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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

DVC_DPATH=$(smartwatch_dvc)
INIT_STATE_V001=$DVC_DPATH/models/fusion/eval3_sc_candidates/pred/CropDrop3_SC_V001/pred_CropDrop3_SC_V001_epoch=90-step=186367-v1/Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco.pt
(cd "$DVC_DPATH" && dvc pull -r aws smart_watch_dvc/models/fusion/eval3_sc_candidates/pred/CropDrop3_SC_V004/pred_CropDrop3_SC_V004_epoch=36-step=75775/Cropped-Drop3-TA1-2022-03-10_data_wv_vali.kwcoco.pt)


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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


# tooshbrush cropped + Depth WV only
# ----------------------------------

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_D_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_D_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_D_wv_vali.kwcoco.json
CHANNELS="red|green|blue|depth"
EXPERIMENT_NAME=CropDrop3_SC_wvonly_D_V009
INIT_STATE_V003=$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V003/CropDrop3_SC_V003_epoch=30-step=63487.pt
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
    --normalize_inputs=1536 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V003" \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.01,Site Preparation*2.0" 


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
smartwatch stats "$VALI_FPATH"
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="red|green|blue|depth,red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
EXPERIMENT_NAME=CropDrop3_SC_wvonly_D_V010
INIT_STATE_V003=$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V003/CropDrop3_SC_V003_epoch=30-step=63487.pt
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
    --normalize_inputs=1536 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V003" \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.01,Site Preparation*2.0" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
smartwatch stats "$VALI_FPATH"
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="red|green|blue|near-ir1|near-ir2|depth,red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
EXPERIMENT_NAME=CropDrop3_SC_wvonly_D_V011
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
    --normalize_inputs=1536 \
    --stream_channels=24 \
    --temporal_dropout=0.5 \
    --init=noop 

# tooshbrush cropped + Depth WV only (2022-04-12)
# -----------------------------------------------

DVC_DPATH=$(smartwatch_dvc)
INIT_STATE_V011="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_wvonly_D_V011/CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935.pt"

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
smartwatch stats "$VALI_FPATH"
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1,blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_raw_xver7_V012
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --normalize_inputs=2048 \
    --stream_channels=32 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V007"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
smartwatch stats "$VALI_FPATH"
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_xver7_V013
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --normalize_inputs=2048 \
    --stream_channels=32 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V007"

# ooo


#INIT_STATE_V011="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_wvonly_D_V011/CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935.pt"
export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_xver11_V013
INIT_STATE_V011="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_wvonly_D_V011/CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935.pt"
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
    --time_steps=11 \
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
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V011"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DL_s2_wv_vali.kwcoco.json
smartwatch stats "$VALI_FPATH"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_xver11_V014
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --num_workers=2 \
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
    --normalize_inputs=2048 \
    --stream_channels=24 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V011"


# namek


#INIT_STATE_V011="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_wvonly_D_V011/CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935.pt"
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
ls "$KWCOCO_BUNDLE_DPATH"
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
smartwatch stats "$VALI_FPATH"
CHANNELS="blue|green|red|near-ir1,blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_raw_xver11_V015
INIT_STATE_V011="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_wvonly_D_V011/CropDrop3_SC_wvonly_D_V011_epoch=81-step=167935.pt"
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
    --normalize_inputs=2048 \
    --stream_channels=32 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V011"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
ls "$KWCOCO_BUNDLE_DPATH"
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
smartwatch stats "$VALI_FPATH"
CHANNELS="blue|green|red|near-ir1,blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_raw_xver7_V016
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --normalize_inputs=2048 \
    --stream_channels=32 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V007"


# tooshbrush cropped + Depth WV only (2022-04-13)
# -----------------------------------------------

DVC_DPATH=$(smartwatch_dvc)
export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1,blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_raw_xver12_V018
INIT_STATE_V012="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_raw_xver7_V012/CropDrop3_SC_s2wv_raw_xver7_V012_epoch=19-step=40959-v1.pt"
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
    --accumulate_grad_batches=8 \
    --global_saliency_weight=0.00 \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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
    --normalize_inputs=2048 \
    --stream_channels=32 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.5,No Activity*0.001,Post Construction*0.01,Site Preparation*3.0" \
    --init="$INIT_STATE_V012"


# namek RGB continue
# -----------------------------------------------
export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_s2_wv_vali.kwcoco.json
CHANNELS="red|green|blue"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_rgb_xver6_V019
INIT_STATE_V006="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_V006/CropDrop3_SC_V006_epoch=71-step=18431.pt"
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
    --class_loss='dicefocal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V006"


##### toothbrush 2022-04-17

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_scratch_V020
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --dist_weights=False \
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
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_scratch_V021
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --num_workers=8 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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


##### horologic 2022-04-17

export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_scratch_V022
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --accumulate_grad_batches=1 \
    --saliency_loss='focal' \
    --class_loss='focal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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


export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_scratch_V023
INIT_STATE_V007="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_xver1_V007/CropDrop3_SC_xver1_V007_epoch=5-step=12287.pt"
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
    --accumulate_grad_batches=1 \
    --saliency_loss='focal' \
    --class_loss='focal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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


##### toothbrush 2022-04-19 --continue

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_cont_V024
INIT_STATE_V020="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_tf_scratch_V021/CropDrop3_SC_s2wv_tf_scratch_V021_epoch=10-step=2815.pt"
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
    --learning_rate=8e-4 \
    --num_workers=6 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
    --time_sampling=soft2 \
    --time_span=7m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=20min \
    --use_centered_positives=False \
    --num_draw=8 \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V020"


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field,matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_cont_V025
INIT_STATE_V021="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_tf_scratch_V020/CropDrop3_SC_s2wv_tf_scratch_V020_epoch=5-step=1535.pt"
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
    --class_loss='dicefocal' \
    --chip_size=256 \
    --time_steps=5 \
    --learning_rate=1e-3 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
    --time_sampling=soft2 \
    --time_span=7m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --max_epoch_length=4096 \
    --use_centered_positives=False \
    --num_draw=8 \
    --normalize_inputs=1024 \
    --multimodal_reduce=mean \
    --stream_channels=24 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V021"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
#CHANNELS="WV:red|green|blue|depth,S2:red|green|blue|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_cont2_V026
INIT_STATE_V024="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_tf_cont_V024/CropDrop3_SC_s2wv_tf_cont_V024_epoch=4-step=1279.pt"
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
    --learning_rate=3e-3 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
    --time_sampling=soft2 \
    --time_span=7m \
    --channels="$CHANNELS" \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5min \
    --use_centered_positives=False \
    --num_draw=4 \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --multimodal_reduce=mean \
    --init="$INIT_STATE_V024"



##### toothbrush 2022-04-25 --continue
export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
INIT_STATE_V024="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_tf_cont_V024/CropDrop3_SC_s2wv_tf_cont_V024_epoch=4-step=1279.pt"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_cont24_V027
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
    --accumulate_grad_batches=1 \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=6 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V024"


##### toothbrush 2022-04-26 --continue fixed sampler
export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
INIT_STATE_V024="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_tf_cont_V024/CropDrop3_SC_s2wv_tf_cont_V024_epoch=4-step=1279.pt"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_cont24_V028
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
    --accumulate_grad_batches=1 \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=1e-4 \
    --num_workers=6 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V024"


##### toothbrush 2022-04-26 --continue fixed sampler
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DLM_s2_wv_vali.kwcoco.json
INIT_STATE_V024="$DVC_DPATH/models/fusion/eval3_sc_candidates/packages/CropDrop3_SC_s2wv_tf_cont_V024/CropDrop3_SC_s2wv_tf_cont_V024_epoch=4-step=1279.pt"
CHANNELS="blue|green|red|near-ir1|depth,blue|green|red|nir|swir16|swir22|forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field|matseg_0|matseg_1|matseg_2|matseg_3|mat_up5:64"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_tf_cont24_V029
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
    --accumulate_grad_batches=3 \
    --saliency_loss='focal' \
    --class_loss='dicefocal' \
    --chip_size=256 \
    --time_steps=12 \
    --learning_rate=3e-4 \
    --num_workers=6 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=False \
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
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INIT_STATE_V024"



~/code/watch/scripts/special_reroot.py combo_DILM_s2_wv_*.kwcoco.json

##### horologic 2022-04-27 invariants
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Cropped-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_s2_wv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_s2_wv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_DILM_s2_wv_vali.kwcoco.json
CHANNELS="blue|green|red,invariants:0:16"
EXPERIMENT_NAME=CropDrop3_SC_s2wv_invar_scratch_V030
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
true || \
    smartwatch stats "$VALI_FPATH"
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
    --init="noop"
