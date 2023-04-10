#!/bin/bash

# RUN-TIME NOTES: 
# 1. Only DSET_DPATH, EXPERIMENT_NAME, TRAIN_FPATH, TEST_FPATH, and VALI_FPATH should be changed
# 2. EXPERIMENT_NAME should be changed between experiments
# 2a. Experiments with the same name will be overwritten by the latest run
# 3. Experiments are saved to the output directory under a subdirectory referencing the EXPERIMENT_NAME
# 3. An experiment consists of modifying one of the aforementioned variables in note #1, or the hyper-parameters in step 3a
# 4. Script was designed to run inside a docker container
# 5. User is expected to define the $DVC_DPATH environment variable that points to the smart_watch_dvc repo
# 6. User is expected to mount the smart_watch_dvc repo pointed to by $DVC_DPATH
# 7. User is expected to mount an output directory to /output/

# 1 - check for kwcoco file with depth channel
export DSET_DPATH=Cropped-Drop3-TA1-2022-03-10
export KWCOCO_BUNDLE_DPATH=/output/$DSET_DPATH
export DATA_KWCOCO_FPATH=$DVC_DPATH/$DSET_DPATH/data.kwcoco.json
export DEPTH_KWCOCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_depth.kwcoco.json
export DEPTH_WEIGHTS_FPATH=$DVC_DPATH/models/depth/weights_v1.pt

if [ ! -f "$DEPTH_KWCOCO_FPATH" ]; then 
    # need to generate kwcoco with depth channel; run depth predictor 
    python3 -m watch.tasks.depth.predict \
		--deployed="$DEPTH_WEIGHTS_FPATH" \
        --dataset="$DATA_KWCOCO_FPATH" \
        --output="$DEPTH_KWCOCO_FPATH" \
        --data_workers=8 \
        --window_size=2048
fi

# 2 - check for splitted kwcoco file with depth channel
export TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_depth_train.kwcoco.json
export VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_depth_vali.kwcoco.json
export TEST_FPATH=$KWCOCO_BUNDLE_DPATH/data_depth_vali.kwcoco.json

if [[ ! -f "$TRAIN_FPATH" || ! -f "$VALI_FPATH" || ! -f "$TEST_FPATH" ]]; then
    # split depth kwcoco file into training and vali datasets
    (cd "$KWCOCO_BUNDLE_DPATH"; python3 -m watch.cli.prepare_splits --base_fpath="$DEPTH_KWCOCO_FPATH")
fi

# 3 - run fusion experiment
# 3a - run fusion train
export EXPERIMENT_NAME=DZYNE_DEPTH_SC_V0
export WORKDIR=/output/$DSET_DPATH/experiments/$EXPERIMENT_NAME
export FINAL_PACKAGE_DPATH=$WORKDIR
export FINAL_PACKAGE_FPATH=$FINAL_PACKAGE_DPATH/final_package.pt   

if [ ! -f "$FINAL_PACKAGE_FPATH" ]; then 
python3 -m watch.tasks.fusion.fit \
    --default_root_dir="$WORKDIR" \
    --name=$EXPERIMENT_NAME \
    --train_dataset="$TRAIN_FPATH" \
    --vali_dataset="$VALI_FPATH" \
    --test_dataset="$TEST_FPATH" \
    --channels="red|green|blue|depth" \
    --global_class_weight=1.00 \
    --global_saliency_weight=1.00 \
    --global_change_weight=0.00 \
    --neg_to_pos_ratio=0.25 \
    --saliency_loss='dicefocal' \
    --class_loss='dicefocal' \
    --num_workers=4 \
    --gpus "1" \
    --batch_size=4 \
    --accumulate_grad_batches=1 \
    --learning_rate=1e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --attention_impl=exact \
    --chip_size=384 \
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
    --init="noop"
fi

# 3b - compute metrics 
if [ -f "$FINAL_PACKAGE_FPATH" ]; then 
export MODEL_GLOBSTR=$FINAL_PACKAGE_FPATH                                  

python3 -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
    --gpus="0" \
    --model_globstr="$MODEL_GLOBSTR" \
    --test_dataset="$VALI_FPATH" \
    --annotations_dpath="/smart/Test_and_Eval_framework/annotations" \
    --workdir="$WORKDIR" \
    --enable_pred=1 \
    --enable_eval=0 \
    --enable_track=0 \
    --enable_actclf=1 \
    --enable_actclf_eval=1 \
    --draw_heatmaps=1 \
    --enable_iarpa_eval=1 \
    --skip_existing=0 --backend=serial --run=1
fi
