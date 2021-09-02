__doc__='
Script demonstrating how to split a base dataset into train / validation,
specifically for drop1-S2-L8-aligned in this case
'

DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
KWCOCO_BUNDLE_DPATH=${KWCOCO_BUNDLE_DPATH:-$DVC_DPATH/drop1-S2-L8-aligned}
BASE_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json

TRAIN_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_train.kwcoco.json
VALI_COCO_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali.kwcoco.json

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
