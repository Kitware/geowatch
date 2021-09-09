__doc__="

Research question: 

What works better? Training on Onera then Drop1 or training on Onera and Drop1
Jointly?
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

    # Make a "validation" dataset
    kwcoco subset --select_videos ".id <= 1" \
        --src $DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
        --dst $DVC_DPATH/extern/onera_2018/onera_vali.kwcoco.json

    # Make a "learn" dataset
    kwcoco subset --select_videos ".id >= 2" \
        --src $DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
        --dst $DVC_DPATH/extern/onera_2018/onera_learn.kwcoco.json

    # Verify the split looks good
    kwcoco stats \
        $DVC_DPATH/extern/onera_2018/onera_learn.kwcoco.json \
        $DVC_DPATH/extern/onera_2018/onera_vali.kwcoco.json

    python -m watch stats \
        $DVC_DPATH/extern/onera_2018/onera_learn.kwcoco.json 
}


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_learn.kwcoco.json 
VALI_FPATH=$DVC_DPATH/extern/onera_2018/onera_vali.kwcoco.json 
TEST_FPATH=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json 

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

#ARCH=smt_it_stm_s12
ARCH=smt_it_joint_p8

#CHANNELS="B05|B06|B07|B08|B8A"
#EXPERIMENT_NAME=DirectCD_${ARCH}_vnir_v6

# Set B8 early so it is visualized
CHANNELS="B01|B05|B08|B11|B06|B07|B8A|B09|B10|B12|B02|B03|B04"
EXPERIMENT_NAME=DirectCD_${ARCH}_allchan_v7
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
    --batch_size=2 \
    --accumulate_grad_batches=32 \
    --num_workers=6 \
    --max_epochs=2000 \
    --chip_overlap=0.5 \
    --neg_to_pos_ratio=2.0 \
    --patience=2000 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --window_size=8 \
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






