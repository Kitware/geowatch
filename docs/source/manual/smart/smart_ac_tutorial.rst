SMART Activity Characterization Tutorial
========================================


This document is a tutorial on Activity Characterization (AC) in the context of the SMART project.
It goes over:

1. Access AC data via DVC.
2. Modifying the data / predicting features on the data
3. Train a model geowatch.tasks.fusion
4. Evaluate a model with geowatch.mlops



1. Access AC Data
-----------------

Due to `an issue <https://discuss.dvc.org/t/dvc-says-everything-is-up-to-date-when-it-is-not/1717>`_ that I don't fully understand, the pre-clustered and pre-cropped Drop7 AC training data is in a different DVC repo.


Navigate to where you would like to store it and grab the DVC repo.

.. code:: bash

   git clone git@gitlab.kitware.com:smart/smart_drop7.git

To ensure commands in this tutorial are runnable, be sure to register this new
repo with geowatch using the "drop7_data" tag. (Important, I'm assuming you
have not changed directories after you ran git clone, make sure the path is
correctly set in the following command. Also change the hardware or name params
to your liking, the only thing that matters is that the tag is exactly
"drop7_data" and the path is correct).

.. code:: bash

   geowatch_dvc add drop7_data_ssd --path="$(pwd)/smart_drop7" --tags drop7_data --hardware ssd


Now that you have that setup, pull the data

.. code:: bash

   AC_DATA_DVC_DPATH=$(geowatch_dvc --tags drop7_data)
   # Make sure this prints the expected path to the repo, otherwise the rest of
   # the tutorial will not work.
   echo "AC_DATA_DVC_DPATH=$AC_DATA_DVC_DPATH"

   # Navigate to the DVC repo
   cd $AC_DATA_DVC_DPATH

   # Run DVC pull on Drop7-Cropped2GSD to grab the cropped raw bands.
   # (in the future I may add precomputed team features here)
   dvc pull -r aws -R Drop7-Cropped2GSD
   dvc pull -r toothbrush_ssd -R Drop7-Cropped2GSD

2. Modify AC Data
-----------------

The raw bands AC data is training-ready as is, but you may want to compute team
features on it, or update the annotations in some way.


The following is a loose (untested) way of accomplishing this. Using
prepare_teamfeats will requires that your feature is registered with it (which
hopefully it is).

.. code:: bash

    AC_DATA_DVC_DPATH=$(geowatch_dvc --tags drop7_data)

    export CUDA_VISIBLE_DEVICES="0,1"
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
    BUNDLE_DPATH=$AC_DATA_DVC_DPATH/Drop6-MeanYear10GSD-V2
    python -m geowatch.cli.queue_cli.prepare_teamfeats \
        --base_fpath "$AC_DATA_DVC_DPATH"/imganns-*[0-9].kwcoco.zip \
        --expt_dvc_dpath="$DVC_EXPT_DPATH" \
        --with_landcover=1 \
        --with_invariants2=1 \
        --with_sam=1 \
        --with_materials=0 \
        --with_depth=0 \
        --with_cold=0 \
        --skip_existing=1 \
        --assets_dname=teamfeats \
        --gres=0, --tmux_workers=1 --backend=tmux --run=0


Alternatively, we can write a bash script that loops over regions, and submits
jobs to cmd-queue which can then be inspected before being executed. You can
get pretty fancy here.

TODO: show example of actually doing a feature computation here.

.. code:: bash

    REGION_IDS=(KR_R001 KR_R002 AE_R001 PE_R001 US_R007 BH_R001 BR_R001 BR_R002 BR_R004 BR_R005 CH_R001 LT_R001 NZ_R001 US_C010 US_C011 US_C012 US_C016 US_R001 US_R004 US_R005 US_R006)

    # Grab the regular DVC repo to get acces to the truth
    TRUTH_DVC_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')

    # Create a new queue
    python -m cmd_queue new "modify_ac_queue"

    for REGION_ID in "${REGION_IDS[@]}"; do

        python -m cmd_queue submit --jobname="feature-$REGION_ID" -- modify_ac_queue \
            ... THE COMMAND TO COMPUTE YOUR FEATURE ...

        python -m cmd_queue submit --jobname="reproject-$REGION_ID" --depends="feature-$REGION_ID" -- modify_ac_queue \
            geowatch reproject_annotations \
                --src "$DST_BUNDLE_DPATH/$REGION_ID/$REGION_ID.kwcoco.zip" \
                --dst "$DST_BUNDLE_DPATH/$REGION_ID/imgannots-$REGION_ID.kwcoco.zip" \
                --io_workers="avail/2" \
                --region_models="$TRUTH_DVC_DPATH/annotations/drop6_hard_v1/region_models/${REGION_ID}.geojson" \
                --site_models="$TRUTH_DVC_DPATH/annotations/drop6_hard_v1/site_models/${REGION_ID}_*.geojson"

    done

    # Show the generated script
    python -m cmd_queue show "modify_ac_queue"

    # Execute the generated script
    python -m cmd_queue run --workers=8 "modify_ac_queue"


Lastly, after you update per-region kwcoco files you will need to write new
kwcoco train/validation splits that use these updated files (because the ones
that exist in the repo only reference raw bands).

.. code:: bash

    # TODO:
    # * Modify the suffix depending on the team feats
    # * Modify the base fpath to be correct.
    python -m geowatch.cli.queue_cli.prepare_splits \
        --base_fpath "$AC_DATA_DVC_DPATHVC_DATA_DPATH"/Drop7-Cropped2GSD/*/imgannots-*.kwcoco.zip \
        --dst_dpath "$AC_DATA_DVC_DPATH"/Drop7-Cropped2GSD \
        --suffix=rawbands --run=1 --workers=2


Note: see ../../scripts/prepare_drop7.sh for details on how this dataset was
initially computed.


3. Train an AC Model
--------------------

The following is a training run that I recently ran, and I have no idea if its
params are good or not, but it provides an example of how to train an AC model


Be sure to grab a pretrained model to start from:

.. code:: bash

    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
    python -m geowatch.utils.simple_dvc request \
        "$DVC_EXPT_DPATH"/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch336_step28982.pt


.. code:: bash

    export CUDA_VISIBLE_DEVICES=1
    DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
    echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
    WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
    DATASET_CODE=Drop7-Cropped2GSD
    KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
    TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
    VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
    CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV,WV1):pan"
    EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V11
    DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
    TARGET_LR=1e-4
    WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
    echo "WEIGHT_DECAY = $WEIGHT_DECAY"
    MAX_STEPS=80000
    WATCH_GRID_WORKERS=0 python -m geowatch.tasks.fusion fit --config "
    data:
        select_videos          : $SELECT_VIDEOS
        num_workers            : 5
        train_dataset          : $TRAIN_FPATH
        vali_dataset           : $VALI_FPATH
        window_dims            : '224,224'
        time_steps             : 9
        time_sampling          : soft4
        time_kernel            : '(-1.08y,-1y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1y,1.08y)'
        window_resolution     : 2.0GSD
        input_resolution      : 2.0GSD
        output_resolution     : 2.0GSD
        neg_to_pos_ratio       : 1.0
        batch_size             : 2
        normalize_perframe     : false
        normalize_peritem      : 'blue|green|red|nir|pan'
        max_epoch_length       : 1000000
        channels               : '$CHANNELS'
        min_spacetime_weight   : 0.6
        temporal_dropout       : 0.5
        mask_low_quality       : False
        mask_samecolor_method  : None
        observable_threshold   : 0.1
        quality_threshold      : 0.0
        weight_dilate          : 10
        use_centered_positives : True
        use_grid_positives     : False
        use_grid_negatives     : False
        normalize_inputs       : 1024
        balance_areas          : True
    model:
        class_path: MultimodalTransformer
        init_args:
            #saliency_weights      : '1:1'
            #class_weights         : auto
            tokenizer              : linconv
            arch_name              : smt_it_stm_p16
            decoder                : mlp
            positive_change_weight : 1
            negative_change_weight : 0.01
            stream_channels        : 16
            class_loss             : 'dicefocal'
            saliency_loss          : 'focal'
            saliency_head_hidden   : 6
            change_head_hidden     : 6
            class_head_hidden      : 6
            global_change_weight   : 0.00
            global_class_weight    : 1.00
            global_saliency_weight : 0.00001
            multimodal_reduce      : learned_linear
    optimizer:
        class_path: torch.optim.AdamW
        init_args:
            lr           : $TARGET_LR
            weight_decay : $WEIGHT_DECAY
            betas:
                - 0.85
                - 0.998
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.OneCycleLR
      init_args:
        max_lr: $TARGET_LR
        total_steps: $MAX_STEPS
        anneal_strategy: cos
        pct_start: 0.3
        div_factor: 10
        final_div_factor: 10000
        cycle_momentum: false
    trainer:
        accumulate_grad_batches: 48
        default_root_dir     : $DEFAULT_ROOT_DIR
        accelerator          : gpu
        devices              : 0,
        limit_val_batches    : 256
        limit_train_batches  : 2048
        num_sanity_val_steps : 0
        max_epochs           : 560
        callbacks:
            - class_path: pytorch_lightning.callbacks.ModelCheckpoint
              init_args:
                  monitor: val_loss
                  mode: min
                  save_top_k: 5
                  filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
                  save_last: true

    torch_globals:
        float32_matmul_precision: auto

    initializer:
        init: $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V08/Drop7-Cropped2GSD_SC_bgrn_split6_V08_epoch336_step28982.pt
    "


4. Evaluate an AC Model with MLOps
----------------------------------


The following code runs an AC-only mlops evaluation using the ground truth
polygons as a proxy for the polygons that come out of BAS. This provides a
consistent way to compare models, but a full evaluation of BAS+SV+AC is needed
for final evaluation (TODO, add this).

The following command only runs over KR1 and KR2, add more regions as necessary.

This also includes 3 existing baseline SC models (which you will need to pull
from the dvc expt repo) to compare your model against. Put the path to your
packaged model in the grid and adjust parameters as desired.

.. code:: bash

    python -m geowatch.mlops.manager "list" --dataset_codes Drop7-Cropped2GSD

    HIRES_DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware=auto)
    TRUTH_DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

    kwcoco stats \
        $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip \
        $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip \
        $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/CH_R001/CH_R001.kwcoco.zip

    geowatch stats $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip
    geowatch stats $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip
    geowatch stats $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/CH_R001/CH_R001.kwcoco.zip

    python -m geowatch.mlops.schedule_evaluation --params="
        matrix:
            ########################
            ## AC/SC PIXEL PARAMS ##
            ########################

            sc_pxl.test_dataset:
              - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip
              - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip
              - $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/CH_R001/CH_R001.kwcoco.zip

            sc_pxl.package_fpath:
                - $DVC_EXPT_DPATH/models/fusion/Drop4-SC/packages/Drop4_tune_V30_8GSD_V3/Drop4_tune_V30_8GSD_V3_epoch=2-step=17334.pt.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V07/Drop7-Cropped2GSD_SC_bgrn_split6_V07_epoch73_step6364.pt
                #- $DVC_EXPT_DPATH/models/fusion/Drop7-Cropped2GSD/packages/Drop7-Cropped2GSD_SC_bgrn_split6_V11/Drop7-Cropped2GSD_SC_bgrn_split6_V11_epoch444_step19135.pt

            sc_pxl.tta_fliprot: 0.0
            sc_pxl.tta_time: 0.0
            sc_pxl.chip_overlap: 0.3
            #sc_pxl.input_space_scale: 2GSD
            #sc_pxl.window_space_scale: 2GSD
            #sc_pxl.output_space_scale: 2GSD
            #sc_pxl.time_span: 6m
            #sc_pxl.time_sampling: auto
            #sc_pxl.time_steps: 12
            #sc_pxl.chip_dims: auto
            sc_pxl.set_cover_algo: null
            sc_pxl.resample_invalid_frames: 3
            sc_pxl.observable_threshold: 0.0
            sc_pxl.mask_low_quality: true
            sc_pxl.drop_unused_frames: true
            sc_pxl.num_workers: 12
            sc_pxl.batch_size: 1
            sc_pxl.write_workers: 0

            ########################
            ## AC/SC POLY PARAMS  ##
            ########################

            sc_poly.thresh: 0.07
            sc_poly.boundaries_as: polys
            #sc_poly.resolution: 2GSD
            sc_poly.min_area_square_meters: 7200

            #############################
            ## AC/SC POLY EVAL PARAMS  ##
            #############################

            sc_poly_eval.true_site_dpath: $TRUTH_DVC_DATA_DPATH/annotations/drop6/site_models
            sc_poly_eval.true_region_dpath: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models

            ##################################
            ## HIGH LEVEL PIPELINE CONTROLS ##
            ##################################
            sc_pxl.enabled: 1
            sc_pxl_eval.enabled: 1
            sc_poly.enabled: 1
            sc_poly_eval.enabled: 1
            sc_poly_viz.enabled: 0

        submatrices:
            - sc_pxl.test_dataset: $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip
              sc_poly.site_summary: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models/KR_R001.geojson
            - sc_pxl.test_dataset: $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip
              sc_poly.site_summary: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models/KR_R002.geojson
            - sc_pxl.test_dataset: $HIRES_DVC_DATA_DPATH/Drop7-Cropped2GSD/CH_R001/CH_R001.kwcoco.zip
              sc_poly.site_summary: $TRUTH_DVC_DATA_DPATH/annotations/drop6/region_models/CH_R001.geojson
        " \
        --pipeline=sc \
        --root_dpath="$DVC_EXPT_DPATH/_demo_ac_eval" \
        --queue_name "_demo_ac_eval" \
        --devices="0,1" \
        --backend=tmux --tmux_workers=6 \
        --cache=1 --skip_existing=1 --run=1


After mlops evaluation completes you can inspect your results with mlops
aggregate to produce reports and gain insight.

.. code:: bash

    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)
    python -m geowatch.mlops.aggregate \
        --pipeline=sc \
        --target "
            - $DVC_EXPT_DPATH/_demo_ac_eval
        " \
        --output_dpath="$DVC_EXPT_DPATH/_demo_ac_eval/aggregate" \
        --resource_report=0 \
        --eval_nodes="
            - sc_poly_eval
        " \
        --plot_params="
            enabled: 0
            stats_ranking: 0
            min_variations: 1
            params_of_interest:
                - params.sc_poly.thresh
        " \
        --stdout_report="
            top_k: 13
            per_group: 1
            macro_analysis: 0
            analyze: 0
            print_models: True
            reference_region: final
            concise: 0
            show_csv: 0
        "

        #\
        #--rois="KR_R002,NZ_R001,CH_R001,KR_R001"
