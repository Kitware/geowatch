__doc__="

Research question: 

What works better? Training on Onera then Drop1 or training on Onera and Drop1
Jointly?


References:
    [Onera2018] https://arxiv.org/pdf/1810.08468.pdf
    [Chen2020] - A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection - https://www.mdpi.com/2072-4292/12/10/1662

"
prep_dvc_data(){

    # Setup environ
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    #DVC_REMOTE=aws
    DVC_REMOTE=horologic

    # Grab data for training
    mkdir -p $DVC_DPATH/training/$HOSTNAME/$USER
    cd $DVC_DPATH
    dvc pull -r $DVC_REMOTE --recursive extern/onera_2018

    dvc pull -r aws --recursive extern/onera_2018
}

prep_validation_set(){
    # Split out a validation dataset from the training data
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    kwcoco stats $DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json
    python -m watch stats $DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

    python -m watch.cli.coco_modify_channels --normalize=True \
        --src $DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
        --dst $DVC_DPATH/extern/onera_2018/onera_train_norm.kwcoco.json

    python -m watch.cli.coco_modify_channels --normalize=True \
        --src $DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
        --dst $DVC_DPATH/extern/onera_2018/onera_test_norm.kwcoco.json

    python -m watch stats $DVC_DPATH/extern/onera_2018/onera_train_norm.kwcoco.json

    # Add ta2 features 

    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/extern/onera_2018
    UKY_S2_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/_partial_uky_pred_S2.kwcoco.json
    UKY_L8_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/_partial_uky_pred_L8.kwcoco.json
    UKY_INVARIANTS_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/uky_invariants.kwcoco.json
    RUTGERS_MATERIAL_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/rutgers_material_seg.kwcoco.json
    DZYNE_LANDCOVER_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/landcover.kwcoco.json
    UKY_S2_MODEL_FPATH=${UKY_L8_MODEL_FPATH:-$DVC_DPATH/models/uky_invariants/sort_augment_overlap/S2_drop1-S2-L8-aligned-old.0.ckpt}
    UKY_L8_MODEL_FPATH=${UKY_L8_MODEL_FPATH:-$DVC_DPATH/models/uky_invariants/sort_augment_overlap/L8_drop1-S2-L8-aligned-old.0.ckpt}
    RUTGERS_MATERIAL_MODEL_FPATH="$DVC_DPATH/models/rutgers/experiments_epoch_30_loss_0.05691597167379317_valmIoU_0.5694727912477856_time_2021-08-07-09:01:01.pth"
    DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/dzyne/todo.pt"


    # Predict with UKY Invariants (one model for S2 and L8)
    python -m watch.tasks.invariants.predict \
        --sensor S2 \
        --input_kwcoco $DVC_DPATH/extern/onera_2018/onera_train_norm.kwcoco.json \
        --output_kwcoco $DVC_DPATH/extern/onera_2018/onera_train_uky_inv.kwcoco.json \
        --ckpt_path $UKY_S2_MODEL_FPATH

    python -m watch.tasks.invariants.predict \
        --sensor S2 \
        --input_kwcoco $DVC_DPATH/extern/onera_2018/onera_test_norm.kwcoco.json \
        --output_kwcoco $DVC_DPATH/extern/onera_2018/onera_test_uky_inv.kwcoco.json \
        --ckpt_path $UKY_S2_MODEL_FPATH

    # Combine features
    python -m watch.cli.coco_combine_features \
        --src \
            $DVC_DPATH/extern/onera_2018/onera_train_norm.kwcoco.json \
            $DVC_DPATH/extern/onera_2018/onera_train_uky_inv.kwcoco.json \
        --dst \
           $DVC_DPATH/extern/onera_2018/onera_train_combo.kwcoco.json 

    python -m watch.cli.coco_combine_features \
        --src \
            $DVC_DPATH/extern/onera_2018/onera_test_norm.kwcoco.json \
            $DVC_DPATH/extern/onera_2018/onera_test_uky_inv.kwcoco.json \
        --dst \
           $DVC_DPATH/extern/onera_2018/onera_test_combo.kwcoco.json 


    # Make a "validation" dataset
    kwcoco subset --select_videos ".id <= 1" \
        --src $DVC_DPATH/extern/onera_2018/onera_train_combo.kwcoco.json \
        --dst $DVC_DPATH/extern/onera_2018/onera_vali_combo.kwcoco.json

    # Make a "learn" dataset
    kwcoco subset --select_videos ".id >= 2" \
        --src $DVC_DPATH/extern/onera_2018/onera_train_combo.kwcoco.json \
        --dst $DVC_DPATH/extern/onera_2018/onera_learn_combo.kwcoco.json

    # Verify the split looks good
    kwcoco stats \
        $DVC_DPATH/extern/onera_2018/onera_learn_combo.kwcoco.json \
        $DVC_DPATH/extern/onera_2018/onera_vali_combo.kwcoco.json \
        $DVC_DPATH/extern/onera_2018/onera_test_combo.kwcoco.json

    python -m watch stats \
        $DVC_DPATH/extern/onera_2018/onera_learn_combo.kwcoco.json 
    
}


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_learn_combo.kwcoco.json 
VALI_FPATH=$DVC_DPATH/extern/onera_2018/onera_vali_combo.kwcoco.json 
TEST_FPATH=$DVC_DPATH/extern/onera_2018/onera_test_combo.kwcoco.json 

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

#ARCH=smt_it_stm_s12
ARCH=smt_it_joint_p8

#CHANNELS="B05|B06|B07|B08|B8A"
#EXPERIMENT_NAME=DirectCD_${ARCH}_vnir_v6

# Set B8 early so it is visualized
#CHANNELS="B01|B05|B08|B11|B06|B07|B8A|B09|B10|B12|B02|B03|B04"
#CHANNELS="coastal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A"

CHANNELS="coastal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A|inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|inv_shared9|inv_shared10|inv_shared11|inv_shared12|inv_shared13|inv_shared14|inv_shared15|inv_shared16|inv_shared17|inv_shared18|inv_shared19|inv_shared20|inv_shared21|inv_shared22|inv_shared23|inv_shared24|inv_shared25|inv_shared26|inv_shared27|inv_shared28|inv_shared29|inv_shared30|inv_shared31|inv_shared32|inv_shared33|inv_shared34|inv_shared35|inv_shared36|inv_shared37|inv_shared38|inv_shared39|inv_shared40|inv_shared41|inv_shared42|inv_shared43|inv_shared44|inv_shared45|inv_shared46|inv_shared47|inv_shared48|inv_shared49|inv_shared50|inv_shared51|inv_shared52|inv_shared53|inv_shared54|inv_shared55|inv_shared56|inv_shared57|inv_shared58|inv_shared59|inv_shared60|inv_shared61|inv_shared62|inv_shared63|inv_shared64"

#EXPERIMENT_NAME=DirectCD_${ARCH}_allchan_v8
EXPERIMENT_NAME=DirectCD_${ARCH}_combo_v10
DATASET_CODE=Onera

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 


kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

# Write train and prediction configs
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --time_steps=2 \
    --chip_size=96 \
    --batch_size=1 \
    --accumulate_grad_batches=32 \
    --num_workers=12 \
    --max_epochs=2000 \
    --chip_overlap=0.5 \
    --neg_to_pos_ratio=2.0 \
    --patience=2000 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --window_size=8 \
    --attention_impl=performer \
    --dump=$TRAIN_CONFIG_FPATH  

python -m watch.tasks.fusion.predict \
    --gpus=0 \
    --write_preds=True \
    --write_probs=False \
    --dump=$PRED_CONFIG_FPATH

## TODO: predict and eval steps should be called after training.
# But perhaps it should be a different invocation of the fit script?
# So the simple route is still available?

# Execute train -> predict -> evaluate
python -m watch.tasks.fusion.fit \
           --config=$TRAIN_CONFIG_FPATH \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH && \
python -m watch.tasks.fusion.predict \
        --config=$PRED_CONFIG_FPATH \
        --test_dataset=$TEST_FPATH \
       --package_fpath=$PACKAGE_FPATH \
        --pred_dataset=$PRED_FPATH && \
python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_FPATH \
        --pred_dataset=$PRED_FPATH \
          --eval_dpath=$EVAL_DPATH






