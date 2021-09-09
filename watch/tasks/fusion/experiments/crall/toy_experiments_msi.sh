__doc__="""
This demonstrates an end-to-end pipeline on multispectral toydata

This walks through the entire process of fit -> predict -> evaluate and the
output if you run this should end with something like
"""

# Generate toy datasets
DATA_DPATH=$HOME/data/work/toy_change
TRAIN_FPATH=$DATA_DPATH/vidshapes_msi_train/data.kwcoco.json
VALI_FPATH=$DATA_DPATH/vidshapes_msi_vali/data.kwcoco.json
TEST_FPATH=$DATA_DPATH/vidshapes_msi_test/data.kwcoco.json 

mkdir -p $DATA_DPATH
kwcoco toydata vidshapes8-frames5-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_train
kwcoco toydata vidshapes4-frames5-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_vali
kwcoco toydata vidshapes2-frames6-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_test


# Print stats
python -m kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH
python -m watch watch_coco_stats $TRAIN_FPATH 


python -m watch.cli.coco_visualize_videos \
    --src $DATA_DPATH/vidshapes_msi_train/data.kwcoco.json \
    --viz_dpath=$DATA_DPATH/vidshapes_msi_train


ARCH=smt_it_stm_p8
CHANNELS="B8|B1|B11|B8a"
EXPERIMENT_NAME=ToyFusion_${ARCH}_v001
DATASET_NAME=ToyDataMSI

# Specify the expected input / output files
WORKDIR=$DATA_DPATH/training/$HOSTNAME/$USER
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_NAME/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/predict_$EXPERIMENT_NAME.yml 


# Configure training hyperparameters
python -m watch.tasks.fusion.fit \
    --channels="$CHANNELS" \
    --method=MultimodalTransformer \
    --arch_name=$ARCH \
    --window_size=8 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=1 \
    --max_epochs=2 \
    --max_steps=100 \
    --gpus=1 \
    --accumulate_grad_batches=1 \
    --dump=$TRAIN_CONFIG_FPATH


# Configure prediction hyperparams
python -m watch.tasks.fusion.predict \
    --gpus=1 \
    --write_preds=True \
    --write_probs=False \
    --dump=$PRED_CONFIG_FPATH


# Fit 
python -m watch.tasks.fusion.fit \
           --config=$TRAIN_CONFIG_FPATH \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH

# Predict 
python -m watch.tasks.fusion.predict \
        --config=$PRED_CONFIG_FPATH \
        --test_dataset=$TEST_FPATH \
       --package_fpath=$PACKAGE_FPATH \
        --pred_dataset=$PRED_FPATH

# Evaluate 
python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_FPATH \
        --pred_dataset=$PRED_FPATH \
          --eval_dpath=$EVAL_DPATH
