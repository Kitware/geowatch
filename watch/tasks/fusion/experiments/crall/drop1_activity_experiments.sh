
#### Training

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
#TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=Drop1_Raw_Holdout
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval


#### 
# Config 2021-10-01-A
# Consumes 6.7GB
####

#CHANNELS="blue|green|red|nir|swir16|coastal"
CHANNELS="blue|green|red|nir|swir16|swir22"
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=ActivityClf_${ARCH}_raw_v019
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

#python -m watch stats $TRAIN_FPATH 
#kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

# Write train and prediction configs
CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.86 \
    --time_steps=5 \
    --time_sampling=hard+distribute \
    --batch_size=6 \
    --accumulate_grad_batches=10 \
    --num_workers=14 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --class_loss='cce' \
    --diff_inputs=False \
    --max_epochs=400 \
    --patience=400 \
    --gpus=1  \
    --learning_rate=1e-2 \
    --weight_decay=1e-5 \
    --num_draw=6 \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  




#### 
# Config 2021-10-01-A
# Consumes 18.4 GB, but then jumps to > 24GB on validation, unsure what causes this.
# Reduce batch size from 6 to 4 to mitigate
# Reduces to 13.2GB on init
# Crashed again, reduce batch size from 4 to 2 to mitigate
# Its at 7.778GB -> 7.8
# .... ok, something messed up. reducting temporal extent
# Oh wow, it spiked to 20.0GB, crazy. Is there a leak, or bug?
####

#CHANNELS="blue|green|red|nir|swir16|coastal"
CHANNELS="blue|green|red|nir|swir16|swir22"
ARCH=smt_it_joint_n12
EXPERIMENT_NAME=ActivityClf_${ARCH}_raw_v021
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

#python -m watch.tasks.fusion.fit --help | grep arch | grep joint

CUDA_VISIBLE_DEVICES="0" python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.86 \
    --time_steps=11 \
    --time_sampling=hard+distribute \
    --batch_size=2 \
    --accumulate_grad_batches=16 \
    --num_workers=14 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --class_loss='cce' \
    --diff_inputs=False \
    --max_epochs=220 \
    --patience=220 \
    --gpus=1  \
    --learning_rate=1e-2 \
    --weight_decay=1e-5 \
    --num_draw=6 \
    --draw_interval="30s" \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  


CUDA_VISIBLE_DEVICES="0" python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.86 \
    --time_steps=11 \
    --time_sampling=hard+distribute \
    --batch_size=2 \
    --accumulate_grad_batches=16 \
    --num_workers=14 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --class_loss='cce' \
    --diff_inputs=False \
    --max_epochs=220 \
    --patience=220 \
    --gpus=1  \
    --init=/home/joncrall/data/dvc-repos/smart_watch_dvc/training/toothbrush/joncrall/Drop1_Raw_Holdout/runs/ActivityClf_smt_it_joint_n12_raw_v021/lightning_logs/version_2/package-interupt/package_epoch12_step8965.pt \
    --learning_rate=1e-3 \
    --weight_decay=1e-5 \
    --num_draw=6 \
    --draw_interval="1m" \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  


#### 
# Config 2021-10-04
# Its at 7.778GB -> 7.8
# Oh wow, it spiked to 20.0GB, crazy. Is there a leak, or bug?
####


#### Training

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
#TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=Drop1_Raw_Holdout
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval


#### 
# Config 2021-10-01-A
# Consumes 6.7GB
####

CHANNELS="ASI|AF|BSI|blue|green|red|nir|swir16|swir22"
#ARCH=smt_it_joint_p8
ARCH=deit
EXPERIMENT_NAME=ActivityClf_${ARCH}_raw_v029
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

#python -m watch.tasks.fusion.fit --help | grep arch | grep joint

export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=64 \
    --chip_overlap=0.0 \
    --time_steps=6 \
    --time_sampling=hardish+distribute \
    --batch_size=1 \
    --accumulate_grad_batches=8 \
    --num_workers=10 \
    --attention_impl=performer \
    --tokenizer=dwcnn \
    --token_norm=group \
    --neg_to_pos_ratio=0.0 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --class_loss='focal' \
    --diff_inputs=False \
    --max_epochs=20 \
    --patience=20 \
    --gpus=1  \
    --learning_rate=1e0 \
    --weight_decay=1e-6 \
    --num_draw=8 \
    --draw_interval="1m" \
    --dropout=0.03 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  



#### 
# Config 2021-10-08
#### Training

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=Drop1_Raw_Holdout
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

#CHANNELS="blue|green|red|nir|swir16|coastal"
CHANNELS="blue|green|red|nir|swir16|swir22"
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=ActivityClf_${ARCH}_temporal_augment_v027
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

#python -m watch stats $TRAIN_FPATH 
#kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

# Write train and prediction configs
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.2 \
    --time_steps=5 \
    --time_sampling=soft+distribute+pairwise \
    --batch_size=2 \
    --accumulate_grad_batches=16 \
    --num_workers=14 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --class_loss='focal' \
    --diff_inputs=False \
    --max_epochs=300 \
    --patience=300 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-6 \
    --num_draw=6 \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  



DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=Drop1_Raw_Holdout
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval

#CHANNELS="blue|green|red|nir|swir16|coastal"
CHANNELS="blue|green|red|nir|swir16|swir22"
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=ActivityClf_${ARCH}_temporal_augment_v027
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

#python -m watch stats $TRAIN_FPATH 
#kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

# Write train and prediction configs
export CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.8 \
    --time_steps=7 \
    --time_sampling=soft+distribute+pairwise \
    --batch_size=2 \
    --accumulate_grad_batches=16 \
    --num_workers=14 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --class_loss='focal' \
    --diff_inputs=False \
    --max_epochs=300 \
    --patience=300 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-6 \
    --num_draw=6 \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  





#### New Features 

DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json
#TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=Drop1_Gen2_Features
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval


#### 
# Config 2021-10-01-A
# Consumes 6.7GB
####



#CHANNELS="blue|green|red|nir|swir16|coastal"
#CHANNELS="ASI|EVI|blue|green|red|nir|swir16|swir22|AF|SDF|VDF|BSI|MBI|matseg_0|matseg_1|matseg_2|matseg_3|matseg_4|matseg_5|matseg_6|matseg_7|matseg_8|inv_shared1|inv_shared2|inv_shared3|inv_shared4|inv_shared5|inv_shared6|inv_shared7|inv_shared8"
CHANNELS="ASI|inv_shared1|EVI|blue|green|red|nir|swir16|swir22|AF|SDF|VDF|BSI|MBI"
CHANNELS="ASI|inv_shared1|EVI|blue|green|red|nir|swir16|swir22|AF|SDF|VDF|BSI|MBI"
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=ActivityClf_${ARCH}_raw_v019
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

#python -m watch stats $TRAIN_FPATH 
#kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

# Write train and prediction configs
CUDA_VISIBLE_DEVICES="0"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.86 \
    --time_steps=5 \
    --time_sampling=hard+distribute \
    --batch_size=6 \
    --accumulate_grad_batches=10 \
    --num_workers=14 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=1.0 \
    --global_change_weight=0.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --class_loss='cce' \
    --diff_inputs=False \
    --max_epochs=200 \
    --patience=200 \
    --gpus=1  \
    --learning_rate=1e-2 \
    --weight_decay=1e-5 \
    --num_draw=6 \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0  


#### 
# Config 2021-10-11
# Consumes 6.7GB
####



DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/combo_train_data.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
#TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
#VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json
#TEST_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=Drop1_Gen2_Features
PACKAGE_FPATH=$DEFAULT_ROOT_DIR/final_package.pt 
PRED_FPATH=$DEFAULT_ROOT_DIR/pred/pred.kwcoco.json
EVAL_DPATH=$DEFAULT_ROOT_DIR/pred/eval
CHANNELS="ASI|AF|BSI|blue|green|red|nir|swir16|swir22"
ARCH=smt_it_joint_p8
EXPERIMENT_NAME=ActivityClf_${ARCH}_raw_v030
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TRAIN_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/train_$EXPERIMENT_NAME.yml 
PRED_CONFIG_FPATH=$WORKDIR/$DATASET_CODE/configs/predict_$EXPERIMENT_NAME.yml 

#python -m watch stats $TRAIN_FPATH 
#kwcoco stats $TRAIN_FPATH $VALI_FPATH $TEST_FPATH

# Write train and prediction configs
export CUDA_VISIBLE_DEVICES="1"
python -m watch.tasks.fusion.fit \
    --channels=${CHANNELS} \
    --method="MultimodalTransformer" \
    --arch_name=$ARCH \
    --chip_size=96 \
    --chip_overlap=0.0 \
    --time_steps=5 \
    --time_sampling=soft+pairwise+distribute \
    --batch_size=4 \
    --accumulate_grad_batches=4 \
    --num_workers=14 \
    --attention_impl=performer \
    --neg_to_pos_ratio=0.5 \
    --global_class_weight=0.0 \
    --global_change_weight=0.0 \
    --global_saliency_weight=1.0 \
    --negative_change_weight=0.05 \
    --change_loss='dicefocal' \
    --saliency_loss='focal' \
    --class_loss='cce' \
    --diff_inputs=False \
    --max_epochs=200 \
    --patience=200 \
    --gpus=1  \
    --learning_rate=1e-3 \
    --weight_decay=1e-7 \
    --num_draw=6 \
    --dropout=0.1 \
    --window_size=8 \
    --default_root_dir=$DEFAULT_ROOT_DIR \
       --package_fpath=$PACKAGE_FPATH \
        --train_dataset=$TRAIN_FPATH \
         --vali_dataset=$VALI_FPATH \
         --test_dataset=$TEST_FPATH \
         --num_sanity_val_steps=0 
