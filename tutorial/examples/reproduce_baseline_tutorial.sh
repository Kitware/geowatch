#!/bin/bash
__doc__='
This is a companion script to ./feature_fusion_tutorial.sh which walks through
the steps of computing your features, training a fusion model, and comparing it
to a baseline result.

In contrast this script simply illustrates how to reproduce the baseline
result.

This tutorial assumes you have:

    1. Setup the project DVC repo

    2. Have registered the location of your DVC repo with geowatch_dvc.  
    
    3. Have pulled the appropriate dataset (in this case Drop4)
       and have unzipped the annotations.

    3. Have the IARPA metrics code installed:

        # Clone this repo and pip install it to your watch environment
        https://gitlab.kitware.com/smart/metrics-and-test-framework


This tutorial will cover:

    1. Running the evaluation script on a single model.
'


# Determine the paths to your SMART data and experiment repositories.
DATA_DVC_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
EXPT_DVC_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)


echo "
EXPT_DVC_DPATH=$EXPT_DVC_DPATH
DATA_DVC_DPATH=$DATA_DVC_DPATH
"


# The baseline model is checked into the experiment DVC repo.  This is the
# model we used in the November delievery. You may need to pull it from DVC if
# you haven't already.
BASELINE_PACKAGE_FPATH="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt


# NOTE:
# the feature_fusion_tutorial curently just runs the baseline on
# ``data_vali.kwcoco.json`` but here we run that file through ``split_videos``
# first which breaks it up into a kwcoco file per region. We then run the
# evaluation on each region separately. We will likely want to adopt this
# strategy for running evaluations so we can compare results at a more 
# granular level.
python -m geowatch.cli.split_videos "$DATA_DVC_DPATH"/Drop4-BAS/data_vali.kwcoco.json


# The BAS-only evalution pipeline can be executed as follows.  The
# ``--root_dpath`` specifies where the output will be written.


# NOTE: In this exmaple we have commented out KR_R002 and US_R007 to allow for
# getting the results of the pipeline fairly quickly (~15 minutes). 
python -m geowatch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $BASELINE_PACKAGE_FPATH
            bas_pxl.channels:
                - 'auto'
            bas_pxl.test_dataset:
                - $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R001.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R002.kwcoco.json
                #- $DATA_DVC_DPATH/Drop4-BAS/data_vali_US_R007.kwcoco.json
            bas_pxl.chip_dims: auto
            bas_pxl.chip_overlap: 0.3
            bas_pxl.window_space_scale: auto
            bas_pxl.output_space_scale: auto
            bas_pxl.input_space_scale: auto
            bas_pxl.time_span: auto
            bas_pxl.time_sampling: auto
            bas_poly.moving_window_size: null
            bas_poly.thresh:
                - 0.1
            bas_pxl.enabled: 1
            bas_poly.enabled: 1
            bas_poly_eval.enabled: 1
            bas_pxl_eval.enabled: 1
            bas_poly_viz.enabled: 1
    " \
    --root_dpath="$EXPT_DVC_DPATH/_evaluations" \
    --devices="0," --queue_size=1 \
    --backend=tmux --queue_name "baseline-queue" \
    --pipeline=bas \
    --run=0


### NOTE:
# The above script assumes that your bashrc activates the appropriate
# virtualenv by default. If this is not the case you will need to specify an
# additional argument to `watch.mlops.schedule_evaluation`. Namely:
# ``--virtualenv_cmd``. For instance if you have a conda environment named
# "watch", you would add ``--virtualenv_cmd="watch"`` to the command.

__evaldoc__='
To inspect the results nativate to ``$EXPT_DVC_DPATH/_evaluations`` In there
there will be a folder "eval/flat/base_poly_eval". This corresponds to the "BAS
Polygon Evaluation Node" in the BAS Pipeline.  Inside that folder there will be
another folder with the same name and a hash suffix. This represents a specific
node configuration.  When I ran this I got: `bas_poly_eval_id_ace8cc06`.  Path
differences on different machines may cause this hash to be different.

Inside this folder is

    * invoke.sh - a script to reproduce the run of that specific node
    * job_config.json - the configuration of this node at the pipeline level
    * summary.csv - a high level summary of the T&E Results for this run.
    * poly_eval.json - a summarized run of high level T&E scores as well as a compute history.

There will also be a folder for each region (in this case just KR_R001)
containing detailed T&E evaluation outputs and visualizations.


In poly_eval.json running:

    jq ".best_bas_rows.data[0]" poly_eval.json

Should result in scores simlar to:

    {
      "region_id": "KR_R001",
      "rho": 0.5,
      "tau": 0.2,
      "tp sites": 7,
      "tp exact": 0,
      "tp under": 7,
      "tp under (IoU)": 0,
      "tp under (IoT)": 7,
      "tp over": 0,
      "fp sites": 6,
      "fp area": 2.168,
      "ffpa": 0.0663,
      "proposal area": 2.9851,
      "fpa": 0.0913,
      "fn sites": 2,
      "truth annotations": 25,
      "truth sites": 9,
      "proposed annotations": 6,
      "proposed sites": 13,
      "total sites": 15,
      "truth slices": 341,
      "proposed slices": 2301,
      "precision": 0.5385,
      "recall (PD)": 0.7778,
      "F1": 0.6364,
      "spatial FAR": 0.1836,
      "temporal FAR": 0.0021,
      "images FAR": 0.0026
    }

'
