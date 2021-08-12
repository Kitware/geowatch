# Takes ~18GB on a 3090
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_SUBPATH=$DVC_DPATH/drop1_S2_aligned_c1
CUDA_VISIBLE_DEVICES=0 \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_SUBPATH/train_data.kwcoco.json \
    --vali_dataset=$DVC_SUBPATH/vali_data.kwcoco.json \
    --time_steps=8 \
    --channels="coastal|blue|green|red|nir|swir16|swir22" \
    --chip_size=192 \
    --method="MultimodalTransformerDotProdCD" \
    --model_name=smt_it_stm_p8 \
    --batch_size=2 \
    --accumulate_grad_batches=8 \
    --num_workers=12 \
    --gpus=1 2>/dev/null


# Takes ~14GB on a 3090
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_SUBPATH=$DVC_DPATH/drop1_S2_aligned_c1
CUDA_VISIBLE_DEVICES=0 \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_SUBPATH/train_data.kwcoco.json \
    --vali_dataset=$DVC_SUBPATH/vali_data.kwcoco.json \
    --time_steps=7 \
    --channels="coastal|blue|green|red|nir|swir16|swir22" \
    --chip_size=192 \
    --chip_overlap=0.66 \
    --time_overlap=0.3 \
    --method="MultimodalTransformerDotProdCD" \
    --model_name=smt_it_stm_small \
    --batch_size=4 \
    --accumulate_grad_batches=4 \
    --num_workers=12 \
    --gpus=1

2>/dev/null

# Can run on a 1080ti
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
DVC_SUBPATH=$DVC_DPATH/drop1-S2-L8-LS-aligned-v2
CUDA_VISIBLE_DEVICES=1 \
python -m watch.tasks.fusion.fit \
    --train_dataset=$DVC_SUBPATH/train_data.kwcoco.json \
    --vali_dataset=$DVC_SUBPATH/vali_data.kwcoco.json \
    --time_steps=7 \
    --channels="coastal|blue|green|red|nir|swir16|swir22" \
    --chip_size=192 \
    --method="MultimodalTransformerDirectCD" \
    --model_name=smt_it_stm_p8 \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --num_workers=12 \
    --gpus=1 2>/dev/null

