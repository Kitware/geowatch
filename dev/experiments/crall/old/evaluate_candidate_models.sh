__doc__="
Look in these paths for models:
/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_Raw_Holdout/runs/ActivityClf_smt_it_joint_p8_temporal_augment_v027/lightning_logs/version_2
/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_Raw_Holdout/runs/ActivityClf_smt_it_joint_p8_temporal_augment_v027/lightning_logs/version_0/packages/package_epoch290_step89697.pt


"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

#PACKAGE_FPATH=$DVC_DPATH/models/fusion/activity/package_ActivityClf_smt_it_joint_n12_raw_v021_epoch12_step8965.pt
#PACKAGE_FPATH=$DVC_DPATH/models/fusion/activity/package_ActivityClf_smt_it_joint_n12_raw_v021_epoch12_step8965.pt


PACKAGE_FPATH=$DVC_DPATH/models/fusion/bas/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001_epoch=21-step=11593.pt
PACKAGE_FPATH=$DVC_DPATH/models/fusion/bas/Saliency_smt_it_joint_p8_raw_v001/Saliency_smt_it_joint_p8_raw_v001_epoch=41-step=22133.pt


CAND_PACKAGES1=$(ls --color=never $DVC_DPATH/models/fusion/bas/Saliency_smt_it_joint_p8_raw_v001/*.pt)
CAND_PACKAGES2=$(ls --color=never $DVC_DPATH/models/fusion/bas/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/*.pt)

CAND_PACKAGES=("${CAND_PACKAGES2[@]}" "${CAND_PACKAGES1[@]}")



DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 
ls $DVC_DPATH/models/fusion/unevaluated-activity-2021-11-12

#KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned
#TEST_DATASET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-L1
TEST_DATASET=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json

echo "CAND_PACKAGES = $CAND_PACKAGES"
for PACKAGE_FPATH in ${CAND_PACKAGES[@]}; do



    SUGGESTIONS="$(python -m watch.tasks.fusion.organize suggest_paths \
        --package_fpath=$PACKAGE_FPATH \
        --test_dataset=$TEST_DATASET)"
    PRED_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
    EVAL_DPATH="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"

    echo "
    TEST_DATASET = $TEST_DATASET
    PACKAGE_FPATH = $PACKAGE_FPATH
    PRED_DATASET = $PRED_DATASET
    "

#CUDA_VISIBLE_DEVICES=1 \
#    python -m watch.tasks.fusion.predict \
#    --gpus=0 \
#    --write_preds=True \
#    --write_probs=True \
#    --write_change=False \
#    --write_saliency=False \
#    --write_class=True \
#    --test_dataset=$TEST_DATASET \
#   --package_fpath=$PACKAGE_FPATH \
#    --pred_dataset=$PRED_DATASET \
#    --time_sampling="soft+dilated" \
#    --chip_overlap=0 \
#    --num_workers=14

#    # hidden flag: --debug-timesample



#####
#DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

#PACKAGE_FPATH=$DVC_DPATH/models/fusion/bas/Saliency_smt_it_joint_p8_raw_v001/Saliency_smt_it_joint_p8_raw_v001_epoch=145-step=76941.pt
#KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned
#TEST_DATASET=$KWCOCO_BUNDLE_DPATH/combo_vali_data.kwcoco.json
#SUGGESTIONS="$(python -m watch.tasks.fusion.organize suggest_paths \
#    --package_fpath=$PACKAGE_FPATH \
#    --test_dataset=$TEST_DATASET)"
#PRED_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
#EVAL_DATASET="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"

#kwcoco validate $TEST_DATASET


CUDA_VISIBLE_DEVICES=0 \
    python -m watch.tasks.fusion.predict \
    --gpus=0 \
    --write_preds=False \
    --write_probs=True \
    --write_change=False \
    --write_saliency=auto \
    --write_class=auto \
    --test_dataset=$TEST_DATASET \
   --package_fpath=$PACKAGE_FPATH \
    --pred_dataset=$PRED_DATASET \
    --num_workers=16 \
    --batch_size=32 


# Evaluate 
python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_DATASET \
        --pred_dataset=$PRED_DATASET \
          --eval_dpath="$EVAL_DPATH"
    
#--profile
#--time_sampling="hard+dilated" \
#--chip_overlap=0 \

done



#PACKAGE_FPATH=$DVC_DPATH/models/fusion/bas/Saliency_smt_it_joint_p8_raw_v001/Saliency_smt_it_joint_p8_raw_v001_epoch=41-step=22133.pt
