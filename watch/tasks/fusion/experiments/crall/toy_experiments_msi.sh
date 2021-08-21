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
cd $DATA_DPATH
kwcoco toydata vidshapes8-frames5-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_train
kwcoco toydata vidshapes4-frames5-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_vali
kwcoco toydata vidshapes2-frames6-multispectral --bundle_dpath $DATA_DPATH/vidshapes_msi_test


# Print stats
python -m kwcoco stats $TRAIN_FPATH 
python -m watch watch_coco_stats $TRAIN_FPATH 


# Configure training hyperparameters
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
python -m watch.tasks.fusion.fit \
    --channels="B8|B1|B11|B8a" \
    --method=MultimodalTransformerDirectCD \
    --arch_name=smt_it_stm_p8 \
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
    --auto_select_gpus=True \
    --accumulate_grad_batches=1 \
    --dump=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/configs/DirectCD_smt_it_smt_p8_vnir_v1.yml 

# Fit 
python -m watch.tasks.fusion.fit \
     --config=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/configs/DirectCD_smt_it_smt_p8_vnir_v1.yml \
     --train_dataset=$TRAIN_FPATH \
     --vali_dataset=$VALI_FPATH \
     --test_dataset=$TEST_FPATH \
    --default_root_dir=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/runs/DirectCD_smt_it_smt_p8_vnir_v1 \
       --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/runs/DirectCD_smt_it_smt_p8_vnir_v1/final_package.pt 


# Predict 
python -m watch.tasks.fusion.predict \
    --test_dataset=$TEST_FPATH \
    --package_fpath=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/runs/DirectCD_smt_it_smt_p8_vnir_v1/final_package.pt  \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/runs/DirectCD_smt_it_smt_p8_vnir_v1/pred/pred.kwcoco.json


# Evaluate 
python -m watch.tasks.fusion.evaluate \
    --true_dataset=$TEST_FPATH \
    --pred_dataset=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/runs/DirectCD_smt_it_smt_p8_vnir_v1/pred/pred.kwcoco.json
      --eval_dpath=$DVC_DPATH/training/$HOSTNAME/$USER/ToyMSI/runs/DirectCD_smt_it_smt_p8_vnir_v1/pred/eval 
