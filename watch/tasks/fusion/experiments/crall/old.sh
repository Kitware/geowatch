# OLDER Drop1:
# Takes ~18GB on a 3090

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
mkdir -p $DVC_DPATH/training/$HOSTNAME/$USER/Drop1_S2_L8_GSD10/configs
python -m watch.tasks.fusion.fit \
    --channels="coastal|blue|green|red|nir|swir16|swir22" \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
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
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
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
    --arch_name=smt_it_stm_small \
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
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --num_workers=12 \
    --gpus=1 2>/dev/null



#####  TEAMFEATS V1  #####


CHANNEL_SPEC="inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|B05|B07|swir16|B09|nir|coastal|B06|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|inv_shared9|inv_shared10|inv_shared11|inv_shared12|inv_shared13|inv_shared14|inv_shared15|inv_shared16|inv_shared17|inv_shared18|inv_shared19|inv_shared20|inv_shared21|inv_shared22|inv_shared23|inv_shared24|inv_shared25|inv_shared26|inv_shared27|inv_shared28|inv_shared29|inv_shared30|inv_shared31|inv_shared32|inv_shared33|inv_shared34|inv_shared35|inv_shared36|inv_shared37|inv_shared38|inv_shared39|inv_shared40|inv_shared41|inv_shared42|inv_shared43|inv_shared44|inv_shared45|inv_shared46|inv_shared47|inv_shared48|inv_shared49|inv_shared50|inv_shared51|inv_shared52|inv_shared53|inv_shared54|inv_shared55|inv_shared56|inv_shared57|inv_shared58|inv_shared59|inv_shared60|inv_shared61|inv_shared62|inv_shared63|inv_shared64|blue|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|red|cirrus|swir22|B8A|green|r|g|b"

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
python -m watch.tasks.fusion.fit \
    --channels="$CHANNEL_SPEC" \
    --method="MultimodalTransformer" \
    --arch_name=smt_it_stm_p8 \
    --time_steps=4 \
    --chip_size=96 \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --num_workers=4 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --window_size=8 \
    --train_dataset=$DVC_DPATH/drop1-S2-aligned-c1-old/train_data_teamfeats.kwcoco.json \
    --vali_dataset=$DVC_DPATH/drop1-S2-aligned-c1-old/vali_data_teamfeats.kwcoco.json \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/runs/DirectCD_smt_it_stm_s12_v3 \
       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/runs/DirectCD_smt_it_stm_s12_v3/final_package.pt \
                --dump=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/configs/DirectCD_smt_it_stm_s12_v3.yml 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
CUDA_VISIBLE_DEVICES=1 \
python -m watch.tasks.fusion.fit \
              --config=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/configs/DirectCD_smt_it_stm_s12_v3.yml \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/DirectCD_smt_it_stm_s12_v3 \
       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/DirectCD_smt_it_stm_s12_v3/final_package.pt 

python -m watch.tasks.fusion.predict 
        --test_dataset=$DVC_DPATH/drop1-S2-L8-aligned-c1/vali_data.kwcoco.json \
       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/DirectCD_smt_it_stm_s12_v3/final_package.pt \
        --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/DirectCD_smt_it_stm_s12_v3/pred.kwcoco.json

python -m watch.tasks.fusion.evaluate 
        --true_dataset=$DVC_DPATH/drop1-S2-L8-aligned-c1/vali_data.kwcoco.json \
        --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/DirectCD_smt_it_stm_s12_v3/pred.kwcoco.json
          --eval_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V1/DirectCD_smt_it_stm_s12_v3/eval


#####  TEAMFEATS V2  #####


#CHANNEL_SPEC="blue|inv_sort1|inv_sort2|inv_sort3|inv_sort4|inv_sort5|inv_sort6|inv_sort7|inv_sort8|B05|B07|swir16|B09|nir|coastal|B06|inv_overlap1|inv_overlap2|inv_overlap3|inv_overlap4|inv_overlap5|inv_overlap6|inv_overlap7|inv_overlap8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8|inv_shared9|inv_shared10|inv_shared11|inv_shared12|inv_shared13|inv_shared14|inv_shared15|inv_shared16|inv_shared17|inv_shared18|inv_shared19|inv_shared20|inv_shared21|inv_shared22|inv_shared23|inv_shared24|inv_shared25|inv_shared26|inv_shared27|inv_shared28|inv_shared29|inv_shared30|inv_shared31|inv_shared32|inv_shared33|inv_shared34|inv_shared35|inv_shared36|inv_shared37|inv_shared38|inv_shared39|inv_shared40|inv_shared41|inv_shared42|inv_shared43|inv_shared44|inv_shared45|inv_shared46|inv_shared47|inv_shared48|inv_shared49|inv_shared50|inv_shared51|inv_shared52|inv_shared53|inv_shared54|inv_shared55|inv_shared56|inv_shared57|inv_shared58|inv_shared59|inv_shared60|inv_shared61|inv_shared62|inv_shared63|inv_shared64|inv_augment1|inv_augment2|inv_augment3|inv_augment4|inv_augment5|inv_augment6|inv_augment7|inv_augment8|red|cirrus|swir22|B8A|green"

#CHANNEL_SPEC="coastal|green"

#DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
#python -m watch.tasks.fusion.fit \
#    --channels="$CHANNEL_SPEC" \
#    --method="MultimodalTransformer" \
#    --arch_name=smt_it_stm_p8 \
#    --time_steps=4 \
#    --chip_size=96 \
#    --batch_size=1 \
#    --accumulate_grad_batches=8 \
#    --num_workers=4 \
#    --gpus=1  \
#    --learning_rate=1e-3 \
#    --weight_decay=1e-4 \
#    --dropout=0.1 \
#    --window_size=8 \
#    --dump=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/configs/DirectCD_smt_it_stm_s12_v4.yml 

#cat $DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/configs/DirectCD_smt_it_stm_s12_v4.yml 

#DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
#python -m watch.tasks.fusion.fit \
#              --config=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/configs/DirectCD_smt_it_stm_s12_v4.yml \
#    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/runs/DirectCD_smt_it_stm_s12_v4 \
#       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/runs/DirectCD_smt_it_stm_s12_v4/final_package.pt  \
#        --train_dataset=$DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_train.kwcoco.json \
#        --vali_dataset=$DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_vali.kwcoco.json 
#        #--test_dataset=$DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_vali.kwcoco.json \

#python -m watch.tasks.fusion.predict 
#        --test_dataset=$DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_vali.kwcoco.json \
#       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/DirectCD_smt_it_stm_s12_v4/final_package.pt \
#        --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/DirectCD_smt_it_stm_s12_v4/pred.kwcoco.json

#python -m watch.tasks.fusion.evaluate 
#        --true_dataset=$DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_vali.kwcoco.json \
#        --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/DirectCD_smt_it_stm_s12_v4/pred.kwcoco.json
#          --eval_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/DirectCD_smt_it_stm_s12_v4/eval


#python -m watch.cli.coco_add_watch_fields \
#    --src $DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_train.kwcoco.json \
#    --dst $DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_train.kwcoco.json \
#    --target_gsd 10

#python -m watch.cli.coco_add_watch_fields \
#    --src $DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_vali.kwcoco.json \
#    --dst $DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_vali.kwcoco.json \
#    --target_gsd 10

#DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
#python -m watch.tasks.fusion.fit \
#    --method="MultimodalTransformer" \
#    --arch_name=smt_it_stm_p8 \
#    --time_steps=4 \
#    --channels="green" \
#    --chip_size=96 \
#    --batch_size=1 \
#    --accumulate_grad_batches=8 \
#    --num_workers=4 \
#    --gpus=1  \
#    --learning_rate=1e-3 \
#    --weight_decay=1e-4 \
#    --dropout=0.1 \
#    --window_size=8 \
#    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/runs/DirectCD_smt_it_stm_s12_v4 \
#       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/Drop1_TeamFeats_V2/runs/DirectCD_smt_it_stm_s12_v4/final_package.pt  \
#        --train_dataset=$DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_train.kwcoco.json \
#        --vali_dataset=$DVC_DPATH/drop1-S2-L8-aligned-old/data_gsd10_vali.kwcoco.json 




# OLDER ONERA:

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
    --method=MultimodalTransformer \
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
    --method=MultimodalTransformer \
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
    --method=MultimodalTransformer \
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
