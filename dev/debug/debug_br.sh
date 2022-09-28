DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")

DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC

TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_vali_KR_R001.kwcoco.json


TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_BR_R002.kwcoco.json
if [ ! -f "$TEST_DATASET" ]; then
    SRC_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
    kwcoco subset "$SRC_DATASET" "$TEST_DATASET" --select_videos '.name | test("BR_R002")'
fi

TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_BR_R002_small.kwcoco.json
if [ ! -f "$TEST_DATASET" ]; then
    SRC_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
    kwcoco subset "$SRC_DATASET" "$TEST_DATASET" --select_videos '.name | test("BR_R002")' \
        --select_images '.frame_index < 250 and .frame_index > 150' 
fi


PACKAGE_FPATH=$DVC_EXPT_DPATH/models/fusion/$DATASET_CODE/packages/Drop4_BAS_Continue_15GSD_BGR_V004/Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584.pt.pt

TEST_DATASET="./cropped_kwcoco_for_bas.json"

python -m watch.tasks.fusion.predict \
    --devices 0 \
    --write_preds False --write_probs True --with_change False --with_saliency True --with_class False  \
    --test_dataset "$TEST_DATASET" \
    --package_fpath "$PACKAGE_FPATH" \
    --pred_dataset "./tmp_br2/pred.kwcoco.json" \
    --num_workers 4 \
    --set_cover_algo approx \
    --batch_size 8 \
    --tta_time 1 \
    --tta_time 1 \
    --tta_fliprot 0 \
    --chip_overlap 0.3 

    #\
    #--input_space_scale=15GSD \
    #--output_space_scale=15GSD \
    #--window_space_scale=15GSD 



DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data_BR_R002_small.kwcoco.json
PACKAGE_FPATH=$DVC_EXPT_DPATH/models/fusion/$DATASET_CODE/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
TMP_DPATH=$DATA_DVC_DPATH/_tmp
mkdir -p "$OUT_DPATH"

python -m watch.tasks.fusion.predict \
    --devices 0 \
    --write_preds False --write_probs True --with_change False --with_saliency True --with_class False  \
    --test_dataset "$TEST_DATASET" \
    --package_fpath "$PACKAGE_FPATH" \
    --pred_dataset "$TMP_DPATH/br2_testv1/pred.kwcoco.json" \
    --num_workers 4 \
    --set_cover_algo approx \
    --track_emissions=False \
    --batch_size 1 \
    --tta_time 0 \
    --tta_fliprot 0 \
    --chip_overlap 0.3 


smartwatch visualize "$TMP_DPATH/br2_testv1/pred.kwcoco.json" \
    --channels='red|green|blue,salient' --workers=4 --draw_anns=True
