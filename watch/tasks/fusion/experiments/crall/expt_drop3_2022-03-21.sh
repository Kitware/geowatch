#!/bin/bash
__notes__="

SeeAlso:
    ../../../../../scripts/prepare_drop3.sh


"

data_splits(){
    DVC_DPATH=$(smartwatch_dvc)
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    python -m watch.cli.prepare_splits \
        --base_fpath="$DVC_DPATH/$DATASET_CODE/combo_LM.kwcoco.json" \
        --run=0 --backend=tmux
}


prep_teamfeat_drop3(){
    # Team Features on drop2
    #DVC_DPATH=$(smartwatch_dvc --hardware="ssd")
    DVC_DPATH=$(smartwatch_dvc)
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    python -m watch.cli.prepare_teamfeats \
        --base_fpath="$DVC_DPATH/$DATASET_CODE/data.kwcoco.json" \
        --gres="2,3" \
        --with_landcover=1 \
        --with_depth=0 \
        --with_materials=1 \
        --with_invariants=1 \
        --do_splits=1 \
        --depth_workers=0 \
        --cache=1 --run=1 --backend=tmux
        #--backend=slurm
        #python -m watch.cli.prepare_splits --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-01/combo_L.kwcoco.json --run=False
}


gather-checkpoints-repackage(){

    #################################
    # Repackage and commit new models
    #################################
    DVC_DPATH=$(smartwatch_dvc)
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval3_candidates
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
    dvc pull -r aws -R models/fusion/eval3_candidates/packages
    dvc pull -r aws -R models/fusion/eval3_candidates/eval

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
    EXPT_GROUP_CODE=eval3_candidates
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
    ls -al models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json
    ls -al models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 

    # Check for uncommited evaluations
    # shellcheck disable=SC2010
    ls -al models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json | grep -v ' \-> '
    # shellcheck disable=SC2010
    ls -al models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json | grep -v ' \-> '

    #du -shL models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json | sort -h
    dvc add models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json

    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json
    python -c "import sys, pathlib, watch.utils.simple_dvc; watch.utils.simple_dvc.SimpleDVC().add([p for p in sys.argv[1:] if not pathlib.Path(p).is_symlink()])" models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json

    git commit -am "add eval from $HOSTNAME"
    git push
    dvc push -r aws -R models/fusion/eval3_candidates/eval

    # For IARPA metrics
    dvc unprotect models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 
    dvc add models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json 
    git commit -am "add iarpa eval from $HOSTNAME"
    git push 
    dvc push -r aws -R models/fusion/eval3_candidates/eval

    #dvc push -r local_store -R models/fusion/eval3_candidates/eval
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
    #dvc checkout aws models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json.dvc
    #DVC_DPATH=$(smartwatch_dvc)
    #cd "$DVC_DPATH" 
    git pull
    dvc pull -r aws -R models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json.dvc
    #dvc pull -r aws -R models/fusion/eval3_candidates/eval

    #DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    EXPT_GROUP_CODE=eval3_candidates
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
    dvc pull -r aws -R models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json.dvc
    EXPT_GROUP_CODE=eval3_candidates
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
    EXPT_GROUP_CODE=eval3_candidates
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
    EXPT_GROUP_CODE=eval3_candidates
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


    #models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=8-step=47069.pt
    #models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
    #models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V313/Drop3_SpotCheck_V313_epoch=34-step=71679.pt
    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
    EXPT_GROUP_CODE=eval3_candidates
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    #VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
    writeto "$DVC_DPATH/models/fusion/eval3_candidates/models_of_interest-2.txt" "
        models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V319/Drop3_SpotCheck_V319_epoch=29-step=61439-v2.pt
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
    EXPT_GROUP_CODE=eval3_candidates
    #MEASURE_GLOBSTR=$DVC_DPATH/models/fusion/eval3_candidates/eval/BASELINE_EXPERIMENT_V001/pred_BASELINE_EXPERIMENT_V001_epoch=11-step=62759/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/eval/curves/measures2.json
    EXPT_GROUP_CODE=eval3_candidates
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
eval_fpaths = list(glob.glob('models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json'))
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
    MODEL_FPATH=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_bells_mlp_V305/Drop3_bells_mlp_V305_epoch=5-step=3071-v1.pt
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json

    #PRED_FPATH=$HOME/data/dvc-repos/smart_watch_dvc/models/fusion/eval3_candidates/pred/Drop3_bells_mlp_V305/pred_Drop3_bells_mlp_V305_epoch=5-step=3071-v1/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json

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
    ls models/fusion/eval3_candidates/pred/*/*/*/*/pred.kwcoco.json
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
    --dist_weights=False \
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
    --dist_weights=True \
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
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
    --dist_weights=True \
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
DVC_DPATH=$(smartwatch_dvc)
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
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_bells_raw_seg_V308
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/bells_and_whistles.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --decoder=segmenter 

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
EXPERIMENT_NAME=Drop3_bells_raw_mlp_V307-a
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/bells_and_whistles.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --decoder=mlp 

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_bells_raw_seg_V308-a
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/bells_and_whistles.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --decoder=segmenter 

# horologic abalate1 - v2
# -----------------------


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BAS_V304-a
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BAS_V304-b
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 

export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BAS_V304-c
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 

export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BAS_V304-d
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 


# namek 
# -----------------------

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_BASELINE_BAS_V309
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --max_epochs=120 \
    --patience=120 



# yardrat 
# -------

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SEARCH_BAS_V310
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --max_epochs=80 \
    --patience=80 \
    --dist_weights=True \
    --time_steps=5 \
    --time_sampling=soft2 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=200m \
    --num_draw=0 \
    --max_epoch_length=10000 \
    --normalize_inputs=10000 \
    --stream_channels=8 \
    --temporal_dropout=0.5 


# namek 
# -----------------------

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SEARCH_BAS_V311
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --max_epochs=80 \
    --patience=80 \
    --dist_weights=True \
    --time_steps=5 \
    --time_sampling=hardish3 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=200m \
    --num_draw=0 \
    --max_epoch_length=10000 \
    --normalize_inputs=10000 \
    --stream_channels=8 \
    --temporal_dropout=0.5 

# yardrat 
# -------

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SEARCH_BAS_V312
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --max_epochs=80 \
    --patience=80 \
    --dist_weights=True \
    --time_steps=5 \
    --time_sampling=soft2 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=200m \
    --num_draw=0 \
    --max_epoch_length=10000 \
    --normalize_inputs=10000 \
    --stream_channels=8 \
    --temporal_dropout=0.5 


# tooshbrush spotcheck
# --------------------
export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SpotCheck_V313
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --max_epochs=80 \
    --patience=80 \
    --num_workers=4 \
    --dist_weights=True \
    --time_steps=7 \
    --time_sampling=hardish3 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=1m \
    --num_draw=8 \
    --stream_channels=8 \
    --temporal_dropout=0.5 \
    --normalize_inputs=2048


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SpotCheck_V314
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --num_workers=4 \
    --max_epochs=80 \
    --patience=80 \
    --dist_weights=True \
    --time_steps=7 \
    --time_sampling=hardish3 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=1m \
    --num_draw=8 \
    --stream_channels=8 \
    --temporal_dropout=0.5 



# horologic abalate2
# ------------------
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BOTH_V315
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BOTH_V316
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --normalize_inputs=2000 


export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_BAS_V317
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --normalize_inputs=2048


export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
EXPERIMENT_NAME=Drop3_BASELINE_SC_V318
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --normalize_inputs=4096


# tooshbrush spotcheck2
# --------------------
export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SpotCheck_V319
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=True \
    --time_steps=6 \
    --channels="$CHANNELS" \
    --time_sampling=hardish3 \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=8 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.1,Site Preparation*2.0" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red,nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SpotCheck_V321
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --global_change_weight=0.00 \
    --global_class_weight=0.01 \
    --global_saliency_weight=1.00 \
    --learning_rate=3e-4 \
    --num_workers=4 \
    --max_epochs=160 \
    --patience=160 \
    --dist_weights=True \
    --time_steps=6 \
    --time_sampling=hardish3 \
    --channels="$CHANNELS" \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_n12 \
    --decoder=mlp \
    --draw_interval=5m \
    --use_centered_positives=True \
    --num_draw=8 \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --modulate_class_weights="positive*0,negative*0,background*1.0,No Activity*0.0,Post Construction*0.1,Site Preparation*2.0" 


INITIAL_STATE_BASELINE="$DVC_DPATH"/models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=20-step=109829-v1.pt

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22"
EXPERIMENT_NAME=Drop3_SpotCheck_V323
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


# horologic 2022-04-05
# --------------------


INITIAL_STATE_BASELINE="$DVC_DPATH"/models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=20-step=109829-v1.pt
INITIAL_STATE_V323="$DVC_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
INITIAL_STATE_V323="$DVC_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
EXPERIMENT_NAME=Drop3_TeamFeats_LM_xfer323_V324
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
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
    --num_workers=0 \
    --dist_weights=True \
    --chip_size=256 \
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
    --init="$INITIAL_STATE_V323" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
EXPERIMENT_NAME=Drop3_TeamFeats_LM_xfer323_V325
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
INITIAL_STATE_V323="$DVC_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --init="$INITIAL_STATE" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.00 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --accumulate_grad_batches=4 \
    --max_epochs=160 \
    --chip_size=288 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=0 \
    --time_steps=9 \
    --channels="$CHANNELS" \
    --time_sampling=soft2+distribute \
    --time_span=6m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --multimodal_reduce=mean \
    --temporal_dropout=0.5 \
    --init="$INITIAL_STATE_V323" 


export CUDA_VISIBLE_DEVICES=2
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
EXPERIMENT_NAME=Drop3_TeamFeats_LM_scratch_V326
INITIAL_STATE_BASELINE="$DVC_DPATH"/models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=20-step=109829-v1.pt
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
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
    --chip_size=288 \
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
    --use_centered_positives=True \
    --normalize_inputs=8 \
    --stream_channels=16 \
    --temporal_dropout=0.5 \
    --init="$INITIAL_STATE_BASELINE" 

export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
INITIAL_STATE_BASELINE="$DVC_DPATH"/models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=20-step=109829-v1.pt
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
EXPERIMENT_NAME=Drop3_TeamFeats_LM_xfer1_v328
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
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
    --num_workers=0 \
    --dist_weights=True \
    --chip_size=224 \
    --time_steps=15 \
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
    --stream_channels=64 \
    --temporal_dropout=0.5 \
    --init="$INITIAL_STATE_BASELINE" 



# toothbrush 2022-04-05
# --------------------


DVC_DPATH=$(smartwatch_dvc)

export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
EXPERIMENT_NAME=Drop3_TeamFeats_LM_xfer1_V327
INITIAL_STATE_BASELINE="$DVC_DPATH"/models/fusion/eval3_candidates/packages/BASELINE_EXPERIMENT_V001/BASELINE_EXPERIMENT_V001_epoch=20-step=109829-v1.pt
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --init="$INITIAL_STATE" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.001 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --accumulate_grad_batches=4 \
    --neg_to_pos_ratio=0.2 \
    --max_epochs=160 \
    --patience=160 \
    --chip_size=256 \
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
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --multimodal_reduce=mean \
    --temporal_dropout=0.2 \
    --init="$INITIAL_STATE_BASELINE" 


export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red|nir|swir16|swir22,matseg_0|matseg_1|matseg_2|matseg_3,forest|brush|bare_ground|built_up|cropland|wetland|water|snow_or_ice_field"
INITIAL_STATE=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
EXPERIMENT_NAME=Drop3_TeamFeats_LM_xfer323_V327
INITIAL_STATE_V323="$DVC_DPATH"/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --init="$INITIAL_STATE" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.00 \
    --global_class_weight=0.003 \
    --global_saliency_weight=1.00 \
    --learning_rate=4e-4 \
    --weight_decay=1e-5 \
    --accumulate_grad_batches=4 \
    --max_epoch_length=4096 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=True \
    --chip_size=288 \
    --time_steps=10 \
    --channels="$CHANNELS" \
    --time_sampling=soft2+distribute \
    --time_span=3m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --accumulate_grad_batches=4 \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=32 \
    --multimodal_reduce=mean \
    --temporal_dropout=0.2 \
    --init="$INITIAL_STATE_V323" 

# next is 329

######
# End of Phase I Evaluations
######

export CUDA_VISIBLE_DEVICES=0
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red"
EXPERIMENT_NAME=Drop3_Simplify_S2_RGB_V330
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.0 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --max_epoch_length=4096 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --chip_size=380 \
    --time_steps=5 \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --channels="$CHANNELS" \
    --exclude_sensors "L8" \
    --time_sampling=soft2+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --multimodal_reduce=mean \
    --temporal_dropout=0.2 \
    --init="noop" 


export CUDA_VISIBLE_DEVICES=1
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red"
EXPERIMENT_NAME=Drop3_Simplify_S2_RGB_time_V331
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.0 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --max_epoch_length=4096 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --chip_size=32 \
    --time_steps=100 \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --channels="$CHANNELS" \
    --exclude_sensors "L8" \
    --time_sampling=soft2+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --multimodal_reduce=mean \
    --temporal_dropout=0.2 \
    --init="noop" 

export CUDA_VISIBLE_DEVICES=3
DVC_DPATH=$(smartwatch_dvc)
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_train.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_LM_nowv_vali.kwcoco.json
CHANNELS="blue|green|red"
EXPERIMENT_NAME=Drop3_Simplify_S2_L8_RGB_V332
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion.fit \
    --config="$WORKDIR/configs/drop3_abalate1.yaml" \
    --default_root_dir="$DEFAULT_ROOT_DIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="$CHANNELS" \
    --class_loss='focal' \
    --saliency_loss='dicefocal' \
    --global_change_weight=0.0 \
    --global_class_weight=0.0 \
    --global_saliency_weight=1.00 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --max_epoch_length=4096 \
    --max_epochs=160 \
    --patience=160 \
    --num_workers=4 \
    --dist_weights=False \
    --chip_size=380 \
    --time_steps=5 \
    --batch_size=1 \
    --accumulate_grad_batches=1 \
    --channels="$CHANNELS" \
    --time_sampling=soft2+distribute \
    --time_span=7m \
    --tokenizer=linconv \
    --optimizer=AdamW \
    --arch_name=smt_it_stm_p8 \
    --decoder=mlp \
    --draw_interval=5m \
    --num_draw=4 \
    --use_centered_positives=True \
    --normalize_inputs=2048 \
    --stream_channels=16 \
    --multimodal_reduce=mean \
    --temporal_dropout=0.2 \
    --init="noop" 
