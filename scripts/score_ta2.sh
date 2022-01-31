__doc__="""
This script is a sample run of the IARPA evaluation metrics.
The intended audience is teams producing TA-2 features (Rutgers, UKy, DZYNE)
to create scored predictions from those features to evaluate their impact.

Before running this script, you will need:

0. Access to https://smartgitlab.com, including registering an SSH key for 
cloning at https://smartgitlab.com/-/profile/keys

1. A local copy of the WATCH DVC repo,
https://gitlab.kitware.com/smart/smart_watch_dvc

Ensure that the appropriate dataset is checked out, for example:
$ dvc pull -R drop1-S2-L8-aligned

Also ensure that the annotations are up to date. This is a submodule pointing to
https://smartgitlab.com/TE/annotations

If smart_watch_dvc/annotations/ exists:
$ git pull --recurse-submodules
Otherwise:
$ git submodule update --init --recursive --remote

2. A local copy of the evaluation framework, 
https://smartgitlab.com/TE/metrics-and-test-framework

Install its requirements in a virtual environment of your choice, such as
with conda:
$ cd metrics-and-test-framework
$ conda env create -n te; conda activate te; pip install -r requirements.txt

You will need its local path and a command to activate the virtualenv.

3. You also need a trained model package to produce BAS and/or SC results.
Usage pathways of varying complexity can be evaluated:
  - Run BAS on a region
  - Run SC on a region
  - Run BAS and SC on a region
  - Run BAS to propose potential sites in a region, then run SC on each site
  - Run BAS to propose potential sites in a region, add VHR imagery,
    then run SC on each site
  - Run BAS to propose potential sites in a region, crop to each potential site,
    then run SC on each site
  - Run BAS to propose potential sites in a region, add VHR imagery,
    crop to each potential site, then run SC on each site
Where the final pathway is the full WATCH system.

Depending on your features, you may need to train only a BAS or SC model.
      Size              | Sensors
BAS | Region (~10x10km) | LS,S2
SC  | Site (~1x1km)     | LS,S2,VHR
(VHR = very high resolution = WorldView, Planet)

3a. To train a fusion model with just your features:

python -m watch.tasks.fusion.fit \
    --train_dataset=dataset_with_my_features.kwcoco.json \
    --channels=<my|channel|names>
for example invocations, see:
watch/tasks/fusion/experiments/crall/expt_drop1_2021-11-17.sh

3b. Alternatively, roll your own model, as long as it produces predictions in the
format of:
python -m watch.tasks.mytask.predict

Predictions for BAS are probability heatmaps of 'saliency' (not 'change') or
polygons bounding salient areas.

Predictions for SC are probability heatmaps or polygons of the program
categories:
'No Activity', 'Site Preparation', 'Active Construction', 'Post Construction'

The machinery that cares about this BAS or SC data formatting is in the args:

python -m watch.cli.kwcoco_to_geojson \
    --default_track_fn ('saliency_heatmaps' | 'saliency_polys' |
                        'class_heatmaps' | 'class_polys' |
                        BAS_HEATMAP_CHANNEL_CODE)
or
    --track_fn (subclass of watch.tasks.tracking.util.TrackFunction)

see --help for more info.
"""

# --- EDIT THESE VARIABLES ---
if [ -z ${DVC_DPATH} ]; then
    export DVC_DPATH=~/smart/data/smart_watch_dvc;
fi

# DSET_DPATH=$DVC_DPATH/Drop1-Aligned-L1
DSET_DPATH=$DVC_DPATH/Drop1-Aligned-L1-2022-01

METRICS_DPATH=~/smart/metrics-and-test-framework

# workaround to get conda commands running in a subprocess
# https://github.com/conda/conda/issues/7980#issuecomment-901423711
METRICS_VENV_CMD="eval \"\$(conda shell.`basename -- \$SHELL` hook)\" && conda activate te"
# ----------------------------


#
# Choose data to run on (here, KR for validation)
# Not always needed; use the existing train/vali spit where available
#

_="
REGIONS_VALI=(
    KR_R001
    KR_R002
)
REGIONS_TRAIN=(
    US_R001
    BR_R001
    BR_R002
    LT_R001
    BH_R001
    NZ_R001
)
REGIONS=("${REGIONS_VALI[@]}")
# REGIONS=("${REGIONS_TRAIN[@]}")


# DSET_FPATH=$DSET_DPATH/train_data.kwcoco.json
DSET_FPATH=$DSET_DPATH/vali_data.kwcoco.json

if ![ -e "$DSET_FPATH" ]; then
    for REGION in "${REGIONS[@]}"; do
        REGION_PATHS+=($DSET_DPATH/$REGION/subdata.kwcoco.json)
    done
    kwcoco union --src $REGION_PATHS --dst $DSET_FPATH --absolute 1
fi
"
DSET_FPATH=$DSET_DPATH/vali_data_nowv.kwcoco.json


#
# Choose models to evaluate and predict on the regions with them
#


BAS_PACKAGE_FPATH=$DVC_DPATH/models/fusion/bas/Saliency_smt_it_joint_p8_raw_v001/Saliency_smt_it_joint_p8_raw_v001_epoch\=145-step\=76941.pt

# SC_PACKAGE_FPATH=$DVC_DPATH/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_weighted_raw_v39/SC_smt_it_stm_p8_newanns_weighted_raw_v39_epoch\=59-step\=2568779.pt
SC_PACKAGE_FPATH=$DVC_DPATH/models/fusion/SC-20201117/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v30/SC_smt_it_stm_p8_newanns_cs64_t5_perframe_rgb_v30_epoch\=3-step\=171251.pt
# SC_PACKAGE_FPATH=$DVC_DPATH/models/fusion/activity/package_ActivityClf_smt_it_joint_n12_raw_v021_epoch17_step13283.pt


# creates bas.kwcoco.json
python -m watch.tasks.fusion.predict \
    --package_fpath=$BAS_PACKAGE_FPATH \
    --test_dataset=$DSET_FPATH \
    --pred_dataset=$DSET_DPATH/bas.kwcoco.json \
    --gpus 1 \
    --write_preds 0 \
    --write_probs 1

# creates sc.kwcoco.json
python -m watch.tasks.fusion.predict \
    --package_fpath=$SC_PACKAGE_FPATH \
    --test_dataset=$DSET_FPATH \
    --pred_dataset=$DSET_DPATH/sc.kwcoco.json \
    --gpus 1 \
    --write_preds 0 \
    --write_probs 1 \
    --chip_overlap 0


# the following sections are independent of each other


#
# DEMO: run BAS on a region
# 

rm -r $DSET_DPATH/sites
rm -r $DSET_DPATH/scores
python -m watch.cli.kwcoco_to_geojson \
    $DSET_DPATH/bas.kwcoco.json \
    --default_track_fn saliency_heatmaps \
    score  -- \
        --metrics_dpath $METRICS_DPATH \
        --virtualenv_cmd $METRICS_VENV_CMD \
        --out_dir $DSET_DPATH/scores/ \
        --merge # optional: merge results across regions


#
# DEMO: run SC on a region
# 

rm -r $DSET_DPATH/sites
rm -r $DSET_DPATH/scores
python -m watch.cli.kwcoco_to_geojson \
    $DSET_DPATH/sc.kwcoco.json \
    --default_track_fn class_heatmaps \
    --track_kwargs "{\"use_boundary_annots\": false}" \
    score  -- \
        --metrics_dpath $METRICS_DPATH \
        --virtualenv_cmd $METRICS_VENV_CMD \
        --out_dir $DSET_DPATH/scores/


#
# DEMO: run BAS and SC on a region
# 

rm -r $DSET_DPATH/sites
rm -r $DSET_DPATH/scores
python -m watch.cli.kwcoco_to_geojson \
    $DSET_DPATH/bas.kwcoco.json \
    --track_fn watch.tasks.tracking.from_heatmap.TimeAggregatedHybrid \
    --track_kwargs "{\"coco_dset_sc\": \"$DSET_DPATH/sc.kwcoco.json\"}" \
    score -- \
        --metrics_dpath $METRICS_DPATH \
        --virtualenv_cmd $METRICS_VENV_CMD \
        --out_dir $DSET_DPATH/scores/


#
# DEMO: Run BAS to propose potential sites in a region, then run SC on each site
# 


# creates regions/[regions].geojson
rm -r $DSET_DPATH/regions
python -m watch.cli.kwcoco_to_geojson \
    $DSET_DPATH/bas.kwcoco.json \
    --default_track_fn saliency_heatmaps \
    --bas_mode

# creates sites/[sites].geojson and scores/*
rm -r $DSET_DPATH/sites
rm -r $DSET_DPATH/scores
for REGION_FILE in $DSET_DPATH/regions/*geojson; do
    python -m watch.cli.kwcoco_to_geojson \
        $DSET_DPATH/sc.kwcoco.json \
        --default_track_fn class_heatmaps \
        --site_summary $REGION_FILE \
        # optional: behaves like TimeAggregatedHybrid
        # --track_kwargs "{\"boundaries_as\": \"polys\"}" \
        score -- \
            --metrics_dpath $METRICS_DPATH \
            --virtualenv_cmd $METRICS_VENV_CMD \
            --out_dir $DSET_DPATH/scores/
done


#
# DEMO: ceiling analysis: set GT sites as the proposed sites, 
# then run SC on each site
# (will break visualizations)
# 


# creates sites/[sites].geojson and scores/*
rm -r $DSET_DPATH/sites
rm -r $DSET_DPATH/scores
for REGION_FILE in $DVC_DPATH/annotations/region_models/KR_*.geojson; do
    python -m watch.cli.kwcoco_to_geojson \
        $DSET_DPATH/sc.kwcoco.json \
        --default_track_fn class_heatmaps \
        --site_summary $REGION_FILE \
        score -- \
            --metrics_dpath $METRICS_DPATH \
            --virtualenv_cmd $METRICS_VENV_CMD \
            --out_dir $DSET_DPATH/scores/ \
done

