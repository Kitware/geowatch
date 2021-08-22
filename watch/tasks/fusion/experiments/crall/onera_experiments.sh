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
    kwcoco subset --select_videos ".id > 2" \
        --src $DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
        --dst $DVC_DPATH/extern/onera_2018/onera_learn.kwcoco.json

    # Verify the split looks good
    kwcoco stats \
        $DVC_DPATH/extern/onera_2018/onera_learn.kwcoco.json \
        $DVC_DPATH/extern/onera_2018/onera_vali.kwcoco.json
}


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_learn.kwcoco.json 
VALI_FPATH=$DVC_DPATH/extern/onera_2018/onera_vali.kwcoco.json 
TEST_FPATH=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json 

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

ARCH=smt_it_stm_s12
CHANNELS="B05|B06|B07|B08|B8A"
EXPERIMENT_NAME=DirectCD_${ARCH}_vnir_v4
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
    --method="MultimodalTransformerDirectCD" \
    --arch_name=$ARCH \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=8 \
    --accumulate_grad_batches=8 \
    --num_workers=6 \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --learning_rate=3e-4 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --window_size=8 \
    --dump=$TRAIN_CONFIG_FPATH 

python -m watch.tasks.fusion.predict \
    --gpus=1 \
    --write_preds=True \
    --write_probs=False \
    --dump=$PRED_CONFIG_FPATH

# Execute train -> predict -> evaluate
python -m watch.tasks.fusion.fit \
           --config=$TRAIN_CONFIG_FPATH \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH 

## TODO: these steps should be called after training
python -m watch.tasks.fusion.predict \
        --config=$PRED_CONFIG_FPATH \
        --test_dataset=$TEST_FPATH \
       --package_fpath=$PACKAGE_FPATH \
        --pred_dataset=$PRED_FPATH 

python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_FPATH \
        --pred_dataset=$PRED_FPATH \
          --eval_dpath=$EVAL_DPATH






# OLDER:

# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v2 \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v2/final_package.pt \
    --method=MultimodalTransformerDirectCD \
    --arch_name=smt_it_stm_s12 \
    --window_size=8 \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --terminate_on_nan=True \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=2 \
    --gpus=1 \
    --accumulate_grad_batches=8 \
    --num_workers=12

python -m watch.tasks.fusion.predict \
    --test_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --gpus 1 \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v2/final_package.pt \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v2/pred/pred.kwcoco.json


# NOTES: This does not handle the first frame predictions well.
python -m watch.tasks.fusion.evaluate \
    --true_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v2/pred/pred.kwcoco.json
    --eval_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v2/pred/eval




# 1080ti
# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v1 \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v1/final_package.pt \
    --method=MultimodalTransformerDirectCD \
    --arch_name=smt_it_stm_s12 \
    --window_size=8 \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.12 \
    --terminate_on_nan=True \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=2 \
    --gpus=1 \
    --accumulate_grad_batches=8 \
    --num_workers=8 


python -m watch.tasks.fusion.predict \
    --test_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v1/final_package.pt \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v1/pred/pred.kwcoco.json \
    --gpus 1 


python -m watch.tasks.fusion.evaluate \
    --true_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v1/pred/pred.kwcoco.json \
    --eval_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v1/pred/eval



AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --workdir=$DVC_DPATH/training/$HOSTNAME/$USER \
    --method=MultimodalTransformerDotProdCD \
    --arch_name=smt_it_stm_s12 \
    --window_size=8 \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --terminate_on_nan=True \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=2 \
    --gpus=1 \
    --accumulate_grad_batches=8 \
    --num_workers=6 




# Reproduce the VNIR experiment


# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=0 \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v3 \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v3/deploy_Onera_DotProd_smt_it_stm_s12_vnir_v1.pt \
    --method=MultimodalTransformerDotProdCD \
    --channels="B05|B06|B07|B08|B8A" \
    --arch_name=smt_it_stm_s12 \
    --window_size=8 \
    --patience=400 \
    --max_epochs=1000 \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.12 \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=4 \
    --gpus=1 \
    --accumulate_grad_batches=8 --auto_lr_find=True \
    --num_workers=8 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=0 \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v4 \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v4/deploy_Onera_DotProd_smt_it_stm_s12_vnir_v1.pt \
    --method=MultimodalTransformerDotProdCD \
    --channels="B05|B06|B07|B08|B8A" \
    --arch_name=smt_it_stm_s12 \
    --window_size=8 \
    --patience=400 \
    --max_epochs=1000 \
    --learning_rate=3.3-04 \
    --weight_decay=1e-4 \
    --dropout=0.12 \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=4 \
    --gpus=1 \
    --accumulate_grad_batches=8 --auto_lr_find=False \
    --num_workers=8 2>/dev/null

python -m watch.tasks.fusion.predict \
    --test_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v3/deploy_Onera_DotProd_smt_it_stm_s12_vnir_v1.pt \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v3/pred/pred.kwcoco.json \
    --gpus 1 

python -m watch.tasks.fusion.evaluate \
    --true_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v3/pred/pred.kwcoco.json \
    --eval_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v3/pred/eval 


# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
python -m watch.tasks.fusion.fit  --help | grep tune -C 10
CUDA_VISIBLE_DEVICES=1 \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v6 \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DotProd_smt_it_stm_s12_vnir_11GB_v6/deploy_Direct_smt_it_stm_s12_vnir_11GB_v5.pt \
    --method=MultimodalTransformerDirectCD \
    --channels="B05|B06|B07|B08|B8A" \
    --arch_name=smt_it_stm_s12 \
    --window_size=8 \
    --patience=400 \
    --max_epochs=1000 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --dropout=0.10 \
    --terminate_on_nan=True \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=3 \
    --gpus=1 \
    --accumulate_grad_batches=8 --auto_lr_find=False \
    --num_workers=8 2>/dev/null 
