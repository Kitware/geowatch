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

# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --workdir=$DVC_DPATH/training/$HOSTNAME/$USER \
    --method=MultimodalTransformerDirectCD \
    --model_name=smt_it_stm_s12 \
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
    --package_fpath=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/MultimodalTransformerDirectCD-bd29d1074f926b3a/lightning_logs/version_6/packages/package_epoch16_step663.pt \
    --pred_dataset=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/MultimodalTransformerDirectCD-bd29d1074f926b3a/lightning_logs/version_6/packages/pred/pred.kwcoco.json  # [**pred_hyperparams]


python -m watch.tasks.fusion.evaluate \
    --true_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --pred_dataset=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/MultimodalTransformerDirectCD-bd29d1074f926b3a/lightning_logs/version_6/packages/pred/pred.kwcoco.json  \
    --eval_dpath=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/MultimodalTransformerDirectCD-bd29d1074f926b3a/lightning_logs/version_6/packages/pred/eval  



/home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc/training/yardrat/jon.crall/MultimodalTransformerDirectCD-21150fb65110ebd1/lightning_logs/version_3/checkpoints/epoch=236-step=9242.ckpt



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
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Onera/DirectCD_smt_it_stm_s12_v1/package.pt \
    --method=MultimodalTransformerDirectCD \
    --model_name=smt_it_stm_s12 \
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
    --gpus 1 \
    --package_fpath=/home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc/training/yardrat/jon.crall/MultimodalTransformerDirectCD-21150fb65110ebd1/lightning_logs/version_3/package.pt \
    --pred_dataset=/home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc/training/yardrat/jon.crall/MultimodalTransformerDirectCD-21150fb65110ebd1/lightning_logs/version_3/pred/pred.kwcoco.json  # [**pred_hyperparams]


python -m watch.tasks.fusion.evaluate \
    --true_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --pred_dataset=/home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc/training/yardrat/jon.crall/MultimodalTransformerDirectCD-21150fb65110ebd1/lightning_logs/version_3/pred/pred.kwcoco.json \
    --eval_dpath=/home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc/training/yardrat/jon.crall/MultimodalTransformerDirectCD-21150fb65110ebd1/lightning_logs/version_3/pred/eval  


AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json \
    --vali_dataset=$DVC_DPATH/extern/onera_2018/onera_test.kwcoco.json \
    --workdir=$DVC_DPATH/training/$HOSTNAME/$USER \
    --method=MultimodalTransformerDotProdCD \
    --model_name=smt_it_stm_s12 \
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
