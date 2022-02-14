__doc__='
Script demonstrating how to split a base dataset into train / validation,
specifically for Drop1-Aligned-TA1-2022-01 in this case
'

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/Drop1-Algined-TA1-2022-01}
BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json

TRAIN_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/train_data.kwcoco.json
VALI_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/vali_data.kwcoco.json

# Split out train and validation data 
# (TODO: add test when we get enough data)
kwcoco subset --src $BASE_COCO_FPATH \
        --dst $TRAIN_COCO_FPATH \
        --select_videos '.name | startswith("KR_")'

kwcoco subset --src $BASE_COCO_FPATH \
        --dst $VALI_COCO_FPATH \
        --select_videos '.name | startswith("KR_") | not'


# Show basic kwcoco stats
kwcoco stats --src $BASE_COCO_FPATH $TRAIN_COCO_FPATH $VALI_COCO_FPATH

# Show watch-specific kwcoco stats
python -m watch stats $TRAIN_COCO_FPATH
python -m watch stats $VALI_COCO_FPATH

# Split out WV
kwcoco subset --src $TRAIN_COCO_FPATH \
        --dst $KWCOCO_BUNDLE_DPATH/train_data_wv.kwcoco.json \
        --select_images '.sensor_coarse == "WV"'

kwcoco subset --src $TRAIN_COCO_FPATH \
        --dst $KWCOCO_BUNDLE_DPATH/train_data_nowv.kwcoco.json \
        --select_images '.sensor_coarse == "WV" | not'

kwcoco subset --src $VALI_COCO_FPATH \
        --dst $KWCOCO_BUNDLE_DPATH/vali_data_wv.kwcoco.json \
        --select_images '.sensor_coarse == "WV"'

kwcoco subset --src $VALI_COCO_FPATH \
        --dst $KWCOCO_BUNDLE_DPATH/vali_data_nowv.kwcoco.json \
        --select_images '.sensor_coarse == "WV" | not'
