
#### Training

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

ARCH=smt_it_stm_s12
ARCH=smt_it_stm_p8

# 142 features

#CHANNELS="coastal|green|inv_sort1"
CHANNELS="coastal|blue|green|red|nir|swir16|cirrus|inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|inv_shared9|inv_shared10|inv_shared11|inv_shared12|inv_shared13|inv_shared14|inv_shared15|inv_shared16|inv_shared17|inv_shared18|inv_shared19|inv_shared20|inv_shared21|inv_shared22|inv_shared23|inv_shared24|inv_shared25|inv_shared26|inv_shared27|inv_shared28|inv_shared29|inv_shared30|inv_shared31|inv_shared32|inv_shared33|inv_shared34|inv_shared35|inv_shared36|inv_shared37|inv_shared38|inv_shared39|inv_shared40|inv_shared41|inv_shared42|inv_shared43|inv_shared44|inv_shared45|inv_shared46|inv_shared47|inv_shared48|inv_shared49|inv_shared50|inv_shared51|inv_shared52|inv_shared53|inv_shared54|inv_shared55|inv_shared56|inv_shared57|inv_shared58|inv_shared59|inv_shared60|inv_shared61|inv_shared62|inv_shared63|inv_shared64|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|matseg_9|matseg_10|matseg_11|matseg_12|matseg_13|matseg_14|matseg_15|matseg_16|matseg_17|matseg_18|matseg_19"
EXPERIMENT_NAME=DirectCD_${ARCH}_teamfeat_v09
DATASET_CODE=Drop1_TeamFeats_V4

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

python -m watch stats $TRAIN_FPATH

kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH


# Write train and prediction configs
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --time_steps=3 \
    --chip_size=96 \
    --batch_size=1 \
    --accumulate_grad_batches=16 \
    --num_workers=6 \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --learning_rate=1e-4 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --window_size=8 \
    --dump=$TRAIN_CONFIG_FPATH 

python -m watch.tasks.fusion.predict \
    --gpus=1 \
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
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=1

python -m watch.tasks.fusion.predict \
        --config=$PRED_CONFIG_FPATH \
        --test_dataset=$TEST_FPATH \
       --package_fpath=$PACKAGE_FPATH \
        --pred_dataset=$PRED_FPATH && \
python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_FPATH \
        --pred_dataset=$PRED_FPATH \
          --eval_dpath=$EVAL_DPATH






#### Preprocessing

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
SUBDATA_PATH=$DVC_DPATH/drop1-S2-L8-aligned


# Ensure "Video Space" is 10 GSD
python -m watch.cli.coco_add_watch_fields \
    --src $SUBDATA_PATH/data.kwcoco.json \
    --dst $SUBDATA_PATH/data_gsd10.kwcoco.json \
    --target_gsd 10


# Split out train and validation data (TODO: add test when we can)
kwcoco subset --src $SUBDATA_PATH/data_gsd10.kwcoco.json \
        --dst $SUBDATA_PATH/vali_gsd10.kwcoco.json \
        --select_videos '.name | startswith("KR_")'

kwcoco subset --src $SUBDATA_PATH/data_gsd10.kwcoco.json \
        --dst $SUBDATA_PATH/train_gsd10.kwcoco.json \
        --select_videos '.name | startswith("KR_") | not'

# Print stats
kwcoco stats \
    $SUBDATA_PATH/train_gsd10.kwcoco.json \
    $SUBDATA_PATH/vali_gsd10.kwcoco.json 

python -m watch watch_coco_stats \
    $SUBDATA_PATH/train_gsd10.kwcoco.json 

#### Training

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
SUBDATA_PATH=$DVC_DPATH/drop1-S2-L8-aligned

TRAIN_FPATH=$SUBDATA_PATH/train_gsd10.kwcoco.json
VALI_FPATH=$SUBDATA_PATH/vali_gsd10.kwcoco.json
TEST_FPATH=$SUBDATA_PATH/vali_gsd10.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

#ARCH=smt_it_stm_p8
ARCH=smt_it_stm_s12
CHANNELS="coastal|blue|green|red|nir|pan|swir16"
EXPERIMENT_NAME=DirectCD_${ARCH}_teamfeat_v9
DATASET_CODE=Drop1_TeamFeats_V3

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
    --time_steps=3 \
    --chip_size=96 \
    --batch_size=1 \
    --accumulate_grad_batches=16 \
    --num_workers=6 \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --learning_rate=1e-4 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --window_size=8 \
    --dump=$TRAIN_CONFIG_FPATH 

python -m watch.tasks.fusion.predict \
    --gpus=1 \
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

