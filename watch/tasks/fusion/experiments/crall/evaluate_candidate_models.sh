__doc__="
Look in these paths for models:
/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_Raw_Holdout/runs/ActivityClf_smt_it_joint_p8_temporal_augment_v027/lightning_logs/version_2
/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_Raw_Holdout/runs/ActivityClf_smt_it_joint_p8_temporal_augment_v027/lightning_logs/version_0/packages/package_epoch290_step89697.pt


"
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc 

PACKAGE_FPATH=$DVC_DPATH/models/fusion/activity/package_ActivityClf_smt_it_joint_n12_raw_v021_epoch12_step8965.pt
KWCOCO_BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned
TEST_DATASET=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json
SUGGESTIONS="$(python -m watch.tasks.fusion.organize suggest_paths \
    --package_fpath=$PACKAGE_FPATH \
    --test_dataset=$TEST_DATASET)"
PRED_DATASET="$(echo "$SUGGESTIONS" | jq -r .pred_dataset)"
EVAL_DATASET="$(echo "$SUGGESTIONS" | jq -r .eval_dpath)"


CUDA_VISIBLE_DEVICES=1 \
    python -m watch.tasks.fusion.predict \
    --gpus=0 \
    --write_preds=True \
    --write_probs=True \
    --write_change=False \
    --write_saliency=False \
    --write_class=True \
    --test_dataset=$TEST_DATASET \
   --package_fpath=$PACKAGE_FPATH \
    --pred_dataset=$PRED_DATASET \
    --time_sampling="soft+dilated" \
    --chip_overlap=0 \
    --num_workers=14

    # hidden flag: --debug-timesample
