__doc__="""
This script describes how to run the IARPA evaluation metrics.

Prerequisites:

0. Access to https://smartgitlab.com, including registering an SSH key for 
cloning at https://smartgitlab.com/-/profile/keys

1. A local copy of the WATCH DVC repo,
https://gitlab.kitware.com/smart/smart_watch_dvc, with the dataset
'drop1-S2-L8-aligned' checked-out. This script is specialized to work on that
dataset, though in principle any predictions that overlap the program regions
can be evaluated.
Also ensure that smart_watch_dvc/annotations/ is populated. This is a submodule
containing the ground-truth site models https://smartgitlab.com/TE/annotations,
which are needed for evaluation.
If it exists, ensure it is up to date with:
$ git pull --recurse-submodules
Or if it does not exist, create it with:
$ git submodule update --init --recursive --remote

2. A local copy of the evaluation framework at 
https://smartgitlab.com/TE/metrics-and-test-framework. This script is to be run
in its root directory.
Install the requirements in a virtual environment of your choice, such as
with conda:
$ conda env create -n te; conda activate te; pip install -r requirements.txt

3. Run the 'predict.py' script in your WATCH submodule to produce a kwcoco
file of predictions, then run the commented-out block below in the WATCH
virtual environment to produce .geojson site models.

4. Then switch to your new virtual environment, and run the rest of the script.
Evaluation metrics are produced one region at a time, so you will have to
change REGION and rerun to evaluate multiple regions.

"""
# Set your DVC_DPATH (path to smart_watch_dvc) here, or before calling this
# script: $ export DVC_DPATH=...
# must be absolute path
if [ -z ${DVC_DPATH} ]; then
    DVC_DPATH=~/smart/data/smart_watch_dvc;
fi

TILE=52SDG

REGION=KR_Pyeongchang_R01
#REGION=KR_Pyeongchang_R02

DSET_PATH=$DVC_DPATH/drop1-S2-L8-aligned

__="""
# run this in the WATCH virtual environment
# replacing the kwcoco file name as appropriate.

rm $DSET_PATH/site_models/*.geojson
python -m watch.cli.kwcoco_to_geojson \
    --in_file $DSET_PATH/pred.kwcoco_timeagg_v1.json \
    --out_dir $DSET_PATH/site_models

# and the rest of this script in the T&E virtual environment
"""

IMAGE_PATH=$REGION
rm -r $IMAGE_PATH
mkdir $IMAGE_PATH
ln -s $DSET_PATH/$REGION/*/affine_warp/*/*blue.tif $IMAGE_PATH

GT_PATH=gt
rm -r $GT_PATH
mkdir $GT_PATH
ln -s $DVC_DPATH/annotations/site_models/*$REGION\_*.geojson $GT_PATH

SITES_PATH=sites
rm -r $SITES_PATH
mkdir $SITES_PATH
ln -s $DSET_PATH/site_models/*$REGION\_*.geojson $SITES_PATH

OUT_PATH=out-$REGION
rm -r $OUT_PATH
mkdir $OUT_PATH

# this is supposed to be a cache, but it is currently bugged on T&E's end
rm -r pickled

python run_evaluation.py \
    --roi $REGION \
    --gt_path  $GT_PATH \
    --sm_path $SITES_PATH \
    --output_dir $OUT_PATH \
    --image_path $IMAGE_PATH \
    --rm_path $DVC_DPATH/annotations/region_models/$TILE\_$REGION.geojson
