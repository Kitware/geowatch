__doc__="""
This demonstrates a end-to-end pipeline on toydata
"""

DATA_DPATH=$HOME/data/work/toy_change
mkdir -p $DATA_DPATH
cd $DATA_DPATH

# Generate toy datasets
kwcoco toydata vidshapes8-frames5-multispectral --bundle_dpath $DATA_DPATH/vidshapes_train
kwcoco toydata vidshapes4-frames5-multispectral --bundle_dpath $DATA_DPATH/vidshapes_vali
kwcoco toydata vidshapes2-frames6-multispectral --bundle_dpath $DATA_DPATH/vidshapes_test


# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
#CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \

DATA_DPATH=$HOME/data/work/toy_change
python -m watch watch_coco_stats $DATA_DPATH/vidshapes_train

DATA_DPATH=$HOME/data/work/toy_change
python -m watch.tasks.fusion.fit \
    --train_dataset=$DATA_DPATH/vidshapes_train/data.kwcoco.json \
    --vali_dataset=$DATA_DPATH/vidshapes_vali/data.kwcoco.json \
    --test_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --workdir=$DATA_DPATH/fit/ \
    --package_fpath=$DATA_DPATH/toy_model.pt \
    --channels="B8|B1|B11|B8a" \
    --method=MultimodalTransformerDirectCD \
    --model_name=smt_it_stm_s12 \
    --window_size=8 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=1 \
    --max_epochs=3 \
    --max_steps=100 \
    --gpus=1 \
    --accumulate_grad_batches=4 \
    --num_workers=2 2>/dev/null

python -m watch.tasks.fusion.predict \
    --package_fpath=$DATA_DPATH/toy_model.pt \
    --test_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --pred_dataset=$DATA_DPATH/vidshapes_test/pred/pred.kwcoco.json

python -m watch.tasks.fusion.evaluate \
    --true_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --pred_dataset=$DATA_DPATH/vidshapes_test/pred/pred.kwcoco.json \
    --eval_dpath=$DATA_DPATH/vidshapes_test/pred/eval  # [**eval_hyperparams]
