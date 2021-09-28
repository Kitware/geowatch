#### Simple classifier on raw bands

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned


# Ensure "Video Space" is 10 GSD
#python -m watch.cli.coco_add_watch_fields \
#    --src $KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
#    --dst $KWCOCO_BUNDLE_DPATH/data_gsd10.kwcoco.json \
#    --target_gsd 10


basic_left_right_split(){
    # Optional: visualize the combo data before and after propogation
    KWCOCO_BUNDLE_DPATH=$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned
    python -m watch.cli.coco_visualize_videos \
        --src $KWCOCO_BUNDLE_DPATH/data.kwcoco.json --space=video --num_workers=6 \
        --viz_dpath $KWCOCO_BUNDLE_DPATH/_viz_base_data \
        --channels "red|green|blue"

    # This is just the basic data for the teams
    LEFT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_left.kwcoco.json
    RIGHT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_right.kwcoco.json

    python -m watch.cli.coco_spatial_crop \
            --src $KWCOCO_BUNDLE_DPATH/data.kwcoco.json --dst $LEFT_COCO_FPATH \
            --suffix=_left

    python -m watch.cli.coco_spatial_crop \
            --src $KWCOCO_BUNDLE_DPATH/data.kwcoco.json --dst $RIGHT_COCO_FPATH \
            --suffix=_right
    

}


#jq ".videos[] | .name"  $KWCOCO_BUNDLE_DPATH/data_gsd10.kwcoco.json

## Split out train and validation data (TODO: add test when we can)
#kwcoco subset --src $KWCOCO_BUNDLE_DPATH/data_gsd10.kwcoco.json \
#        --dst $KWCOCO_BUNDLE_DPATH/vali_gsd10.kwcoco.json \
#        --select_videos '.name | startswith("KR_Pyeongchang_R02")'

#kwcoco subset --src $KWCOCO_BUNDLE_DPATH/data_gsd10.kwcoco.json \
#        --dst $KWCOCO_BUNDLE_DPATH/train_gsd10.kwcoco.json \
#        --select_videos '.name | startswith("KR_Pyeongchang_R02") | not'


#### Training

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned

LEFT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_left.kwcoco.json
RIGHT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_right.kwcoco.json

TRAIN_FPATH=$RIGHT_COCO_FPATH
VALI_FPATH=$LEFT_COCO_FPATH
TEST_FPATH=$LEFT_COCO_FPATH

# Print stats
kwcoco stats \
    $TRAIN_FPATH \
    $VALI_FPATH 

python -m watch watch_coco_stats \
    $TRAIN_FPATH

__note__="
k = ('smt_it_joint_p8', 'performer')
           max_mem_alloc_str                                                                         
M                        3        5        7        11       13       32       64       128       256
num_params            280079   280081   280083   280087   280089   280108   280140   280204    280332
S   T                                                                                                
32  2                0.01 GB  0.02 GB  0.03 GB  0.04 GB  0.04 GB  0.10 GB  0.19 GB  0.39 GB   0.77 GB
64  2                0.04 GB  0.07 GB  0.09 GB  0.13 GB  0.16 GB  0.39 GB  0.77 GB  1.53 GB   3.06 GB
96  2                0.08 GB  0.14 GB  0.19 GB  0.31 GB  0.36 GB  0.87 GB  1.70 GB  3.40 GB   6.79 GB
128 2                0.15 GB  0.25 GB  0.35 GB  0.52 GB  0.65 GB  1.53 GB  3.06 GB  6.03 GB  12.05 GB


k = ('smt_it_joint_n12', 'performer')
           max_mem_alloc_str                                                                         
M                        3        5        7        11       13       32       64       128       256
num_params            413199   413201   413203   413207   413209   413228   413260   413324    413452
S   T                                                                                                
32  2                0.02 GB  0.03 GB  0.04 GB  0.05 GB  0.06 GB  0.15 GB  0.29 GB  0.57 GB   1.14 GB
64  2                0.06 GB  0.10 GB  0.13 GB  0.20 GB  0.24 GB  0.57 GB  1.14 GB  2.27 GB   4.54 GB
96  2                0.12 GB  0.21 GB  0.28 GB  0.46 GB  0.53 GB  1.29 GB  2.52 GB  5.04 GB  10.08 GB
128 2                0.22 GB  0.37 GB  0.51 GB  0.78 GB  0.96 GB  2.27 GB  4.54 GB  8.95 GB  17.89 GB
"


#export CUDA_VISIBLE_DEVICES="1"
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

ARCH=smt_it_joint_p8
#ARCH=smt_it_joint_n12
BATCH_SIZE=2
CHIP_SIZE=128
TIME_STEPS=2
#ARCH=smt_it_stm_s12
CHANNELS="coastal|blue|green|red|nir|swir16"

EXPERIMENT_NAME=DirectCD_${ARCH}_raw7common_v5
DATASET_NAME=Drop1RawLeftRight

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_NAME/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/predict_$EXPERIMENT_NAME.yml 

kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

python -m watch.tasks.fusion.predict \
    --gpus=1 \
    --write_preds=True \
    --write_probs=False \
    --dump=$PRED_CONFIG_FPATH

# Write train and prediction configs
python -m watch.tasks.fusion.fit \
    --method="MultimodalTransformer" \
    --arch_name=${ARCH} \
    --channels=${CHANNELS} \
    --time_steps=$TIME_STEPS \
    --chip_size=$CHIP_SIZE \
    --batch_size=$BATCH_SIZE \
    --accumulate_grad_batches=32 \
    --num_workers=8 \
    --max_lookahead=1000000 \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --attention_impl=performer \
    --learning_rate=1e-3 \
    --weight_decay=1.2e-4 \
    --dropout=0.11 \
    --window_size=8 \
    --window_overlap=0.5 \
    --global_class_weight=0.0 \
    --neg_to_pos_ratio=0.2 \
    --global_change_weight=1.0 \
    --diff_inputs=True \
    --torch_sharing_strategy=default \
    --torch_start_method=default \
    --num_sanity_val_steps=0 \
    --dump=$TRAIN_CONFIG_FPATH 

## TODO: predict and eval steps should be called after training.
# But perhaps it should be a different invocation of the fit script?
# So the simple route is still available?

# Execute train -> predict -> evaluate
python -m watch.tasks.fusion.fit \
           --config=$TRAIN_CONFIG_FPATH \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH && \
python -m watch.tasks.fusion.predict \
        --config=$PRED_CONFIG_FPATH \
        --test_dataset=$TEST_FPATH \
       --package_fpath=$PACKAGE_FPATH \
        --pred_dataset=$PRED_FPATH && \
python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_FPATH \
        --pred_dataset=$PRED_FPATH \
          --eval_dpath=$EVAL_DPATH




# Tune from previous
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned

LEFT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_left.kwcoco.json
RIGHT_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_right.kwcoco.json

TRAIN_FPATH=$RIGHT_COCO_FPATH
VALI_FPATH=$LEFT_COCO_FPATH
TEST_FPATH=$LEFT_COCO_FPATH
ARCH=smt_it_joint_p8
BATCH_SIZE=2
CHIP_SIZE=128
TIME_STEPS=2
CHANNELS="coastal|blue|green|red|nir|swir16"

EXPERIMENT_NAME=DirectCD_${ARCH}_raw7common_v5_tune
DATASET_NAME=Drop1RawLeftRight

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_NAME/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/predict_$EXPERIMENT_NAME.yml 

# Write train and prediction configs
python -m watch.tasks.fusion.fit \
    --method="MultimodalTransformer" \
    --arch_name=${ARCH} \
    --channels=${CHANNELS} \
    --time_steps=$TIME_STEPS \
    --chip_size=$CHIP_SIZE \
    --batch_size=$BATCH_SIZE \
    --accumulate_grad_batches=32 \
    --num_workers=8 \
    --max_lookahead=1000000 \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --attention_impl=performer \
    --learning_rate=1e-3 \
    --weight_decay=1.2e-4 \
    --dropout=0.11 \
    --window_size=8 \
    --window_overlap=0.5 \
    --global_class_weight=0.0 \
    --neg_to_pos_ratio=0.2 \
    --global_change_weight=1.0 \
    --negative_change_weight=0.05 \
    --diff_inputs=False \
    --torch_sharing_strategy=default \
    --torch_start_method=default \
    --num_sanity_val_steps=0 \
    --init=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawLeftRight/runs/DirectCD_smt_it_joint_p8_raw7common_v4/lightning_logs/version_2/checkpoints/epoch=2-step=1241-v3.ckpt \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH 

--init=$HOME/remote/yardrat/data/dvc-repos/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawLeftRight/runs/DirectCD_smt_it_joint_p8_raw7common_v4/lightning_logs/version_1/package-interupt/package_epoch38_step26579.pt \
--init=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawLeftRight/runs/DirectCD_smt_it_joint_p8_raw7common_v4/lightning_logs/version_2/checkpoints/epoch=2-step=1241-v3.ckpt \




# Tune from previous
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

kwcoco stats $TRAIN_FPATH $VALI_FPATH

ARCH=smt_it_joint_p8
BATCH_SIZE=2
CHIP_SIZE=128
TIME_STEPS=2
CHANNELS="coastal|blue|green|red|nir|swir16"

EXPERIMENT_NAME=DirectCD_${ARCH}_raw9common_v5_tune_from_onera
DATASET_NAME=Drop1RawHoldout

DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_NAME/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_NAME/configs/predict_$EXPERIMENT_NAME.yml 

# Write train and prediction configs
python -m watch.tasks.fusion.fit \
    --method="MultimodalTransformer" \
    --arch_name=${ARCH} \
    --channels=${CHANNELS} \
    --time_steps=$TIME_STEPS \
    --chip_size=$CHIP_SIZE \
    --batch_size=$BATCH_SIZE \
    --accumulate_grad_batches=32 \
    --num_workers=8 \
    --max_lookahead=1000000 \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --attention_impl=performer \
    --learning_rate=1e-3 \
    --weight_decay=1.2e-4 \
    --dropout=0.11 \
    --window_size=8 \
    --window_overlap=0.5 \
    --global_class_weight=0.0 \
    --neg_to_pos_ratio=1.0 \
    --global_change_weight=1.0 \
    --negative_change_weight=0.1 \
    --diff_inputs=False \
    --torch_sharing_strategy=default \
    --torch_start_method=default \
    --num_sanity_val_steps=0 \
    --init=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Onera/runs/DirectCD_smt_it_joint_p8_combo_v10/lightning_logs/version_1/checkpoints/epoch=233-step=5381.ckpt \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH 

--init=$HOME/remote/yardrat/data/dvc-repos/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawLeftRight/runs/DirectCD_smt_it_joint_p8_raw7common_v4/lightning_logs/version_1/package-interupt/package_epoch38_step26579.pt \
--init=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawLeftRight/runs/DirectCD_smt_it_joint_p8_raw7common_v4/lightning_logs/version_2/checkpoints/epoch=2-step=1241-v3.ckpt \



EXPERIMENT_NAME=DirectCD_${ARCH}_raw9common_v6_tune_from_onera
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_NAME/runs/$EXPERIMENT_NAME
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval
# Write train and prediction configs
python -m watch.tasks.fusion.fit \
    --method="MultimodalTransformer" \
    --arch_name=${ARCH} \
    --channels=${CHANNELS} \
    --time_steps=$TIME_STEPS \
    --chip_size=$CHIP_SIZE \
    --batch_size=$BATCH_SIZE \
    --accumulate_grad_batches=16 \
    --num_workers=8 \
    --max_lookahead=1000000 \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --attention_impl=performer \
    --learning_rate=3e-3 \
    --weight_decay=1e-4 \
    --dropout=0.1 \
    --window_size=8 \
    --window_overlap=0.9 \
    --global_class_weight=0.000 \
    --neg_to_pos_ratio=1.0 \
    --global_change_weight=1.0 \
    --negative_change_weight=0.05 \
    --diff_inputs=False \
    --torch_sharing_strategy=default \
    --torch_start_method=default \
    --num_sanity_val_steps=0 \
    --init=$HOME/remote/yardrat/smart_watch_dvc/training/yardrat/jon.crall/Drop1RawHoldout/runs/DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera/lightning_logs/version_1/checkpoints/epoch=0-step=715-v4.ckpt \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH 


DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned

TRAIN_DATASET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
VALI_DATASET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
# Run on the explicit kwcoco files
python -m netharn.examples.segmentation \
    --name=check_train_rgb_v2 \
    --train_dataset=$TRAIN_DATASET \
    --vali_dataset=$VALI_DATASET \
    --channels="red|green|blue|nir|swir22" \
    --input_overlap=0.5 \
    --input_dims=256,256 \
    --batch_size=32 \
    --arch=psp \
    --optim=AdamW \
    --lr=1e-6 \
    --max_epoch=500 \
    --patience=500 \
    --decay=1e-8 \
    --workers=14 --xpu=0 --reset 
