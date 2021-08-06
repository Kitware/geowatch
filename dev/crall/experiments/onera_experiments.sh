
prep_dvc_data(){
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    DVC_REMOTE=horologic
    mkdir -p $DVC_DPATH/$USER/training
    cd $DVC_DPATH
    dvc pull -r $DVC_REMOTE --recursive extern/onera_2018
}

# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --workdir=$DVC_DPATH/$USER/training \
    --method=MultimodalTransformerDotProdCD \
    --model_name=smt_it_stm_s12 \
    --window_size=8 \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --terminate_on_nan=True \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --num_workers=12 \


# NOTES: at 15/901 and 59/901
