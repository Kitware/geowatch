__doc__="""
This demonstrates a end-to-end pipeline on toydata
"""

DATA_DPATH=$HOME/data/work/toy_change
mkdir -p $DATA_DPATH
#cd $DATA_DPATH

# Generate toy datasets
kwcoco toydata vidshapes8-multispectral --bundle_dpath $DATA_DPATH/vidshapes_train
kwcoco toydata vidshapes4-multispectral --bundle_dpath $DATA_DPATH/vidshapes_vali
kwcoco toydata vidshapes2-multispectral --bundle_dpath $DATA_DPATH/vidshapes_test


# TRAINING COMMANDS
AUTO_DEVICE=$(python -c "import netharn; print(netharn.XPU.coerce('auto').device.index)")
echo "AUTO_DEVICE = $AUTO_DEVICE"
#CUDA_VISIBLE_DEVICES=$AUTO_DEVICE \

python -m watch.tasks.fusion.fit \
    --train_dataset=$DATA_DPATH/vidshapes_train/data.kwcoco.json \
    --vali_dataset=$DATA_DPATH/vidshapes_vali/data.kwcoco.json \
    --test_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --workdir=$DATA_DPATH/fit/ \
    --package_fpath=$DATA_DPATH/toy_model.pt \
    --method=MultimodalTransformerDirectCD \
    --model_name=smt_it_stm_s12 \
    --window_size=8 \
    --learning_rate=3e-4 \
    --weight_decay=1e-5 \
    --dropout=0.1 \
    --time_steps=2 \
    --chip_size=128 \
    --batch_size=1 \
    --max_epochs=1 \
    --max_steps=1 \
    --auto_select_gpus=False \
    --gpus=None \
    --accumulate_grad_batches=8 \
    --num_workers=2

python -m watch.tasks.fusion.predict \
    --package_fpath=$DATA_DPATH/toy_model.pt \
    --test_dataset=$DATA_DPATH/vidshapes_test/data.kwcoco.json \
    --results_dir=$DATA_DPATH/results

# TODO
#python -m watch.tasks.fusion.evaluate \
