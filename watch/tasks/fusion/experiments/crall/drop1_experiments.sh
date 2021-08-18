# Takes ~18GB on a 3090



# Build the experiment configs
CUDA_VISIBLE_DEVICES=0 

mkdir -p $DVC_DPATH/training/$HOSTNAME/$USER/Drop1/configs


python -m watch.cli.coco_add_watch_fields \
    --src $DVC_DPATH/drop1-S2-L8-aligned-c1/train_data.kwcoco.json \
    --dst $DVC_DPATH/drop1-S2-L8-aligned-c1/train_gsd10_data.kwcoco.json \
    --target_gsd 10

python -m watch.cli.coco_add_watch_fields \
    --src $DVC_DPATH/drop1-S2-L8-aligned-c1/vali_data.kwcoco.json \
    --dst $DVC_DPATH/drop1-S2-L8-aligned-c1/vali_gsd10_data.kwcoco.json \
    --target_gsd 10

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
mkdir -p $DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/configs
python -m watch.tasks.fusion.fit \
    --channels="coastal|blue|green|red|nir|swir16|swir22" \
    --method="MultimodalTransformerDirectCD" \
    --model_name=smt_it_stm_p8 \
    --time_steps=8 \
    --chip_size=128 \
    --batch_size=2 \
    --accumulate_grad_batches=8 \
    --num_workers=4 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --window_size=8 \
    --train_dataset=$DVC_DPATH/drop1-S2-L8-aligned-c1/train_gsd10_data.kwcoco.json \
    --vali_dataset=$DVC_DPATH/drop1-S2-L8-aligned-c1/vali_gsd10_data.kwcoco.json \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/runs/DirectCD_smt_it_stm_s12_v3 \
       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/runs/DirectCD_smt_it_stm_s12_v3/final_package.pt \
                --dump=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/configs/DirectCD_smt_it_stm_s12_v3.yml 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=1 \
python -m watch.tasks.fusion.fit \
              --config=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/configs/DirectCD_smt_it_stm_s12_v3.yml \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/DirectCD_smt_it_stm_s12_v3 \
       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/DirectCD_smt_it_stm_s12_v3/final_package.pt 

python -m watch.tasks.fusion.predict 
        --test_dataset=$DVC_DPATH/drop1-S2-L8-aligned-c1/vali_data.kwcoco.json \
       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1/DirectCD_smt_it_stm_s12_v3/final_package.pt \
        --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1/DirectCD_smt_it_stm_s12_v3/pred.kwcoco.json

python -m watch.tasks.fusion.evaluate 
        --true_dataset=$DVC_DPATH/drop1-S2-L8-aligned-c1/vali_data.kwcoco.json \
        --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1/DirectCD_smt_it_stm_s12_v3/pred.kwcoco.json
          --eval_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1/DirectCD_smt_it_stm_s12_v3/eval


# TODO Create configs for the base set of experiments 

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

