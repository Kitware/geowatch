#!/bin/bash
__notes__="

SeeAlso:
    ../../../../../scripts/prepare_drop3.sh


"

gather-checkpoints-repackage(){

    #################################
    # Repackage and commit new models
    #################################
    DVC_DPATH=$(smartwatch_dvc)
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval4_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    python -m watch.tasks.fusion.repackage gather_checkpoints \
        --dvc_dpath="$DVC_DPATH" \
        --storage_dpath="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages" \
        --train_dpath="$DVC_DPATH/training/$HOSTNAME/$USER/$DATASET_CODE/runs/*/lightning_logs" \
        --push_jobs=8 \
        --mode=commit
}


schedule-prediction-and-evlauation(){

    DVC_DPATH=$(smartwatch_dvc)
    cd "$DVC_DPATH" 
    git pull
    #################################
    # Pull new models on eval machine
    #################################

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/eval4_candidates/packages
    dvc pull -r aws -R models/fusion/eval4_candidates/eval

    #################################
    # Run Prediction & Evaluation
    #################################
    # TODO: 
    # - [X] Argument for test time augmentation.
    # - [ ] Argument general predict parameter grid
    # - [ ] Can a task request that slurm only schedule it on a specific GPU?
    # Note: change backend to tmux if slurm is not installed
    DVC_DPATH=$(smartwatch_dvc)
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval4_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
    # The gpus flag does not work for the slurm backend. (Help wanted)
    TMUX_GPUS="0,1"
    #TMUX_GPUS="1,"
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*V3*.pt" \
            --test_dataset="$VALI_FPATH" \
            --run=1 --skip_existing=True --backend=tmux

    TMUX_GPUS="0,1,2,3"
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*xfer*V3*.pt" \
            --test_dataset="$VALI_FPATH" \
            --run=1 --skip_existing=True --backend=tmux

    TMUX_GPUS="0,1,2,3"
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*scratch*V3*.pt" \
            --test_dataset="$VALI_FPATH" \
            --run=1 --skip_existing=True --backend=tmux

    # Iarpa BAS metrics only on existing predictions
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*V3*.pt" \
            --test_dataset="$VALI_FPATH" \
            --skip_existing=True \
            --enable_pred=0 \
            --enable_eval=0 \
            --enable_iarpa_eval=0 \
            --backend=tmux --run=1 

    #################################
    # Commit Evaluation Results
    #################################
    # Be sure to DVC add the eval results after!
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    cd "$DVC_DPATH" 
    # Check for 
    ls -al models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json
    ls -al models/fusion/eval4_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 

    # Check for uncommited evaluations
    # shellcheck disable=SC2010
    ls -al models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json | grep -v ' \-> '
    # shellcheck disable=SC2010
    ls -al models/fusion/eval4_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json | grep -v ' \-> '

    #du -shL models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json | sort -h
    dvc add models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json

    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json
    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/eval4_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json

    git commit -am "add eval from $HOSTNAME"
    git push
    dvc push -r aws -R models/fusion/eval4_candidates/eval

    # For IARPA metrics
    dvc unprotect models/fusion/eval4_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 
    dvc add models/fusion/eval4_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 
    git commit -am "add iarpa eval from $HOSTNAME"
    git push 
    dvc push -r aws -R models/fusion/eval4_candidates/eval

    #dvc push -r local_store -R models/fusion/eval4_candidates/eval
}


aggregate-results(){


    #################################
    # Aggregate Results
    #################################
    # On other machines
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")

    DVC_DPATH=$(smartwatch_dvc)
    cd "$DVC_DPATH" 
    git pull
    #dvc checkout aws models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json.dvc
    #DVC_DPATH=$(smartwatch_dvc)
    #cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json.dvc
    #dvc pull -r aws -R models/fusion/eval4_candidates/eval

    #DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    EXPT_GROUP_CODE=eval4_candidates
    #EXPT_NAME_PAT="*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="*Drop3*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="BOTH_TA1_COMBO_TINY_p2w_raw*"
    MODEL_EPOCH_PAT="*"
    PRED_DSET_PAT="*"
    PRED_CFG_PAT="*"
    MEASURE_GLOBSTR=${DVC_DPATH}/models/fusion/${EXPT_GROUP_CODE}/eval/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/${PRED_CFG_PAT}/eval/curves/measures2.json

    python -m watch.tasks.fusion.aggregate_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/$EXPT_GROUP_CODE" \
        --dset_group_key="*Drop3*combo_LM_nowv_vali*" \
        --classes_of_interest "Site Preparation" "Active Construction" \
        --io_workers=10 --show=True
        #\
        #--embed=True --force-iarpa


    DVC_DPATH=$(smartwatch_dvc)
    cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json.dvc
    EXPT_GROUP_CODE=eval4_candidates
    EXPT_NAME_PAT="*"
    MODEL_EPOCH_PAT="*"
    PRED_DSET_PAT="*"
    PRED_CFG_PAT="*"
    MEASURE_GLOBSTR=${DVC_DPATH}/models/fusion/${EXPT_GROUP_CODE}/eval/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/${PRED_CFG_PAT}/eval/curves/measures2.json
    python -m watch.tasks.fusion.aggregate_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/$EXPT_GROUP_CODE" \
        --dset_group_key="*Drop3*combo_LM_nowv_vali*" \
        --classes_of_interest "Site Preparation" "Active Construction" \
        --io_workers=10 --show=True
}


schedule-prediction-and-evaluate-team-models(){
    # For Uncropped
    DVC_DPATH=$(smartwatch_dvc)
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    EXPT_GROUP_CODE=eval4_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/DZYNE*/*.pt" \
            --test_dataset="$VALI_FPATH" \
            --run=0 --skip_existing=True --backend=serial
}

recovery_eval(){
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval4_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
    TMUX_GPUS="0,1,2,3"

    #--model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/models_of_interest-2.txt" \

    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/models_of_interest.txt" \
            --test_dataset="$VALI_FPATH" \
            --enable_pred=1 \
            --enable_eval=redo \
            --enable_track=0 \
            --enable_iarpa_eval=0 \
            --chip_overlap=0.3 \
            --tta_time=0 \
            --tta_fliprot=0 \
            --bas_thresh=0.1 \
            --draw_heatmaps=1 --draw_curves=1 \
            --skip_existing=1 --backend=tmux --run=0

    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/models_of_interest.txt" \
            --test_dataset="$VALI_FPATH" \
            --enable_pred=1 \
            --enable_eval=0 \
            --enable_track=1 \
            --enable_iarpa_eval=0 \
            --skip_existing=True --backend=tmux --run=0


    #models/fusion/eval4_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=8-step=47069.pt
    #models/fusion/eval4_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
    #models/fusion/eval4_candidates/packages/Drop3_SpotCheck_V313/Drop3_SpotCheck_V313_epoch=34-step=71679.pt
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval4_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
    writeto "$DVC_DPATH/models/fusion/eval4_candidates/models_of_interest-2.txt" "
        models/fusion/eval4_candidates/packages/Drop3_SpotCheck_V319/Drop3_SpotCheck_V319_epoch=29-step=61439-v2.pt
    "

    ls "$DVC_DPATH"/models/fusion/$EXPT_GROUP_CODE/pred/*/*Drop3*
    MODEL_GLOBSTR=$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/*/*.pt
    #MODEL_GLOBSTR="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/models_of_interest-2.txt" 

    TMUX_GPUS="0,1,2,3,4,5,6"
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$MODEL_GLOBSTR" \
            --test_dataset="$VALI_FPATH" \
            --enable_pred=0 \
            --enable_eval=0 \
            --enable_track=1 \
            --enable_iarpa_eval=1 \
            --chip_overlap=0.3 \
            --tta_time=0 \
            --tta_fliprot=0 \
            --bas_thresh=0.1 --hack_bas_grid=0 \
            --skip_existing=1 --backend=tmux --run=0

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    EXPT_GROUP_CODE=eval4_candidates
    #MEASURE_GLOBSTR=$DVC_DPATH/models/fusion/eval4_candidates/eval/BASELINE_EXPERIMENT_V001/pred_BASELINE_EXPERIMENT_V001_epoch=11-step=62759/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/eval/curves/measures2.json
    EXPT_GROUP_CODE=eval4_candidates
    #EXPT_NAME_PAT="*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="*Drop3*"
    EXPT_NAME_PAT="*"
    #EXPT_NAME_PAT="BOTH_TA1_COMBO_TINY_p2w_raw*"
    MODEL_EPOCH_PAT="*"
    MODEL_EPOCH_PAT="*V319_epoch=29*"
    PRED_DSET_PAT="*"
    PRED_CFG_PAT="*"
    MEASURE_GLOBSTR=${DVC_DPATH}/models/fusion/${EXPT_GROUP_CODE}/eval/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/${PRED_CFG_PAT}/eval/curves/measures2.json
    ls "$MEASURE_GLOBSTR"

    python -m watch.tasks.fusion.aggregate_results \
        --measure_globstr="$MEASURE_GLOBSTR" \
        --out_dpath="$DVC_DPATH/agg_results/$EXPT_GROUP_CODE" \
        --dset_group_key="*Drop3*combo_LM_nowv_vali*" --show=0 \
        --io_workers=10 --show=False  \
        --classes_of_interest "Site Preparation" "Active Construction" --force-iarpa 

    # -----------

    TMUX_GPUS="0,"
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/models_of_interest-2.txt" \
            --test_dataset="$VALI_FPATH" \
            --enable_pred=0 \
            --enable_eval=0 \
            --enable_track=1 \
            --enable_iarpa_eval=1 \
            --chip_overlap=0.3 \
            --tta_time=0,1,2,3 \
            --tta_fliprot=0 \
            --bas_thresh=0.1,0.2 \
            --skip_existing=True --backend=tmux --run=1


    TMUX_GPUS="0,1,2,3,4,5,6,7,8"
    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/models_of_interest.txt" \
            --test_dataset="$VALI_FPATH" \
            --enable_pred=0 \
            --enable_eval=0 \
            --enable_track=1 \
            --enable_iarpa_eval=1 \
            --bas_thresh=0.2 \
            --skip_existing=True --backend=tmux --run=1

        #    \
        #--embed=True
}

fix-bad-commit(){

pyblock "

import glob
eval_fpaths = list(glob.glob('models/fusion/eval4_candidates/eval/*/*/*/*/eval/curves/measures2.json'))
fixme = []
for eval_fpath in eval_fpaths:
    eval_fpath = ub.Path(eval_fpath)
    eval_dvc_fpath = eval_fpath.augment(tail='.dvc')
    if eval_dvc_fpath.exists():
        text = eval_dvc_fpath.read_text()
        if '=====' in text:
            fixme.append(eval_fpath)
            print(text)

from watch.utils.simple_dvc import SimpleDVC
dvc = SimpleDVC('.')
dvc.unprotect(fixme)

for p in fixme:
    p.augment(tail='.dvc').delete()

"

}


singleton_commands(){

    DVC_DPATH=$(smartwatch_dvc)
    MODEL_FPATH=$DVC_DPATH/models/fusion/eval4_candidates/packages/Drop3_bells_mlp_V305/Drop3_bells_mlp_V305_epoch=5-step=3071-v1.pt
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json

    #PRED_FPATH=$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/eval4_candidates/pred/Drop3_bells_mlp_V305/pred_Drop3_bells_mlp_V305_epoch=5-step=3071-v1/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json

    python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
            --gpus="$TMUX_GPUS" \
            --model_globstr="$MODEL_FPATH" \
            --test_dataset="$VALI_FPATH" \
            --skip_existing=0 \
            --enable_pred=0 \
            --enable_eval=1 \
            --enable_eval=1 \
            --enable_track=redo \
            --enable_iarpa_eval=redo \
            --backend=serial --run=0


    # Find all models that have predictions
    DVC_DPATH=$(smartwatch_dvc)
    cd "$DVC_DPATH"
    ls models/fusion/eval4_candidates/pred/*/*/*/*/pred.kwcoco.json
}




export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
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
    --accumulate_grad_batches=4 \
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
    --dist_weights=True \
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



INITIAL_STATE_BASELINE="$DVC_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="(S2,L8):blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop4_BAS_30m_Retrain_V001
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
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
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
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
    --init="$INITIAL_STATE_BASELINE"


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc --hardware=hdd)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop4-2022-07-28-c20-TA1-S2-L8-ACC
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json
CHANNELS="(S2,L8):blue|green|red"
EXPERIMENT_NAME=Drop4_BAS_30m_RGB_Retrain_V002
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
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
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
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
    --init="$INITIAL_STATE_BASELINE"
