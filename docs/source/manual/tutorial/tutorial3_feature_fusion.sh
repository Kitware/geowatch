#!/bin/bash
__doc__='
This shell script serves as an executable example for how to train and evaluate
a fusion model on SMART project data.


This tutorial assumes you have:

    1. Setup the project DVC repo

    2. Have registered the location of your DVC repo with geowatch_dvc.

    4. Have pulled the appropriate dataset (in this case Drop4)
       and have unzipped the annotations.

    3. Have a script that predicts features you would like to test.

    5. Have the IARPA metrics code installed:

        # Clone this repo and pip install it to your watch environment
        https://gitlab.kitware.com/smart/metrics-and-test-framework

See these docs for details:
    ../docs/getting_started_dvc.rst
    ../docs/access_dvc_repos.rst
    ../docs/using_geowatch_dvc.rst

This tutorial will cover:

    1. Predicting your features.
    2. Training a fusion model with your features.
    3. Packaging your fusion model checkpoints.
    4. Evaluating your fusion model against the baseline.
'

DATA_DVC_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
EXPT_DVC_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

echo "
EXPT_DVC_DPATH=$EXPT_DVC_DPATH
DATA_DVC_DPATH=$DATA_DVC_DPATH
"


__doc_compute_feature__='

Your predict command must specify:

    1. the path to the input kwcoco file

    2. the path to the output kwcoco file which will contain your features
       (Avoid requiring that other output paths are specified. Use default
        paths that are relative to the directory of the output kwcoco file)

    3. the path to your model(s)

    4. any other CLI parameters to configure details of feature prediction.

You will have to specify the exact details for your features, but as an example we
provide a script that will work to predict invariant features if your machine has
enough resources (you need over 100GB of RAM as of 2022-12-21; we would like to fix
this in the future).

You will also need to ensure the referenced model is pulled from the experiments DVC repo.
'
compute_features(){
    # A bash function that runs invariant prediction on a kwcoco file.
    SRC_KWCOCO_FPATH=$1
    DST_KWCOCO_FPATH=$1
    python -m geowatch.tasks.invariants.predict \
        --input_kwcoco="$SRC_KWCOCO_FPATH" \
        --output_kwcoco="$DST_KWCOCO_FPATH" \
        --pretext_package_path="$EXPT_DVC_DPATH"/models/uky/uky_invariants_2022_12_05/TA1_pretext_model/pretext_package.pt \
        --input_space_scale=30GSD \
        --window_space_scale=30GSD \
        --patch_size=256 \
        --do_pca 0 \
        --patch_overlap=0.0 \
        --num_workers="2" \
        --write_workers 2 \
        --tasks before_after pretext
}

# Compute your features on the train and validation dataset
compute_features \
    "$DATA_DVC_DPATH"/Drop4-BAS/data_train.kwcoco.json
    "$DATA_DVC_DPATH"/Drop4-BAS/data_train_invariants.kwcoco.json

compute_features \
    "$DATA_DVC_DPATH"/Drop4-BAS/data_vali.kwcoco.json
    "$DATA_DVC_DPATH"/Drop4-BAS/data_vali_invariants.kwcoco.json

# After your model predicts the outputs, you should be able to use the
# geowatch visualize tool to inspect your features. The specific channels you
# select will depend on the output of your predict script.
python -m geowatch visualize "$DATA_DVC_DPATH"/Drop4-BAS/data_vali_invariants.kwcoco.json \
    --channels "invariants.5:8,invariants.8:11,invariants.14:17" --stack=only --workers=avail --animate=True \
    --draw_anns=False


# shellcheck disable=SC2016
__doc_data_splits__='

Because only some of the regions actually need 100GB to compute the invariants,
it is possible to split the train and validation kwcoco files into a single
kwcoco file per-video and run the compute_features function on the output
individually.

.. code:: bash

    python -m geowatch.cli.split_videos \
        --src "$DATA_DVC_DPATH/Drop4-BAS/data_train.kwcoco.json" \
              "$DATA_DVC_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
        --dst_dpath "$DATA_DVC_DPATH/Drop4-BAS/"


In fact, if your feature prediction script is registered with the
prepare_teamfeats tool, then you can schedule prediction to run on all of them
individually. You can specify a pattern as the input to this tool.

.. code:: bash

    python -m geowatch.cli.queue_cli.prepare_teamfeats \
        --base_fpath \
            "$DATA_DVC_DPATH/Drop4-BAS/data_train_*.kwcoco.json" \
            "$DATA_DVC_DPATH/Drop4-BAS/data_vali_*.kwcoco.json" \
        --expt_dpath="$EXPT_DVC_DPATH" \
        --with_landcover=0 \
        --with_materials=0 \
        --with_invariants=0 \
        --with_invariants2=1 \
        --with_depth=0 \
        --gres=0, --workers=1 --backend=tmux --run=1

You can then union any custom set of regions into a train and validation kwcoco
file for the subsequent steps.

.. code:: bash

    DATA_DVC_DPATH=$(geowatch_dvc --tags=phase2_data --hardware=auto)
    EXPT_DVC_DPATH=$(geowatch_dvc --tags=phase2_expt --hardware=auto)

    kwcoco union \
        --src $DATA_DVC_DPATH/Drop4-BAS/*_train_*_uky_invariants*.kwcoco.json \
        --dst $DATA_DVC_DPATH/Drop4-BAS/combo_train_I2.kwcoco.json

    kwcoco union \
        --src $DATA_DVC_DPATH/Drop4-BAS/*_vali_*_uky_invariants*.kwcoco.json \
        --dst $DATA_DVC_DPATH/Drop4-BAS/combo_vali_I2.kwcoco.json

We recognize that this is currently a pain-point, but we hope that the existing
tools make it somewhat easier to solve or work around problems, and we hope
that our tooling improves to make this even easier in the future.
'


__doc_run_fusion__='
Now that we have a train and validation kwcoco dataset that contain our
computed features we can train or fine-tune a fusion model.

The following is a set of baseline settings that you should start with.  We
also encourage you to try other hyperparameter settings to maximize the
effectiveness of your features. But you should at least train once with this
configuration as a baseline.
'


# Set according to your hardware requirements
# TODO: expose the unused GPU script and use that.
export CUDA_VISIBLE_DEVICES=0

DATA_DVC_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='auto')
EXPT_DVC_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')

DATASET_CODE=Drop4-BAS
KWCOCO_BUNDLE_DPATH=$DATA_DVC_DPATH/$DATASET_CODE

# You should specify a unique name for your experiment.
# This name will be the default in reports generated by the watch mlops
EXPERIMENT_NAME=Drop4_BAS_my_feature_experiment_$(date --iso-8601)

# These are the paths to the kwcoco files that should contain your features
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_invariants.kwcoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_invariants.kwcoco.json

# The pretrained state should be checked out of DVC.  This is the best BAS
# model as of 2022-12-21, we will partially initialize a subset of the network
# with these weights.
PRETRAINED_STATE="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt

# You can use the model_stats command to inspect details about any fusion model.
geowatch model_stats "$PRETRAINED_STATE"

# shellcheck disable=SC2016
__doc_channel_conf__='
When training a fusion model, you must specify a channel configuration.
By default we recommend imputing your features as a separate "stream" in
addition to the original six raw bands.

Remember, early fused channels are separated with a pipe (|) and late fused
channel groups are separated with a comma.  This means in the sensorchan
configuration, separate your channels from the raw channels with a comma.  E.g.

    blue|green|red|nir|swir16|swir22,invariants.0:17


By default each channel assumes it exists in each sensor. You can specify
which channels belong to what sensors by prefixing a group. For instance:

    (S2,L8):(blue|green|red|nir|swir16|swir22),(S2):(invariants.0:17)

The above uses S2 and L8 raw bands, but only adds the invariants from
Sentinel-2 images.

You may try early fusing your features with the RGB channels, or any more
complex input channel scheme, but you must train the simple late fused network
as a baseline.
'

CHANNELS="(S2,L8):(blue|green|red|nir|swir16|swir22),(S2,L8):(invariants.0:17)"

# We recommend this training directory layout to differentiate
# training runs on different machines / from different people.
WORKDIR=$EXPT_DVC_DPATH/training/$HOSTNAME/$USER
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

MAX_STEPS=10000
TARGET_LR=5e-5
python -m geowatch.tasks.fusion fit --config "
    data:
        num_workers             : 3
        train_dataset           : $TRAIN_FPATH
        vali_dataset            : $VALI_FPATH
        channels                : '$CHANNELS'
        time_steps              : 5
        chip_dims               : '224,224'
        batch_size              : 2
        window_space_scale      : 10GSD
        input_space_scale       : 10GSD
        output_space_scale      : 10GSD
        dist_weights            : false
        neg_to_pos_ratio        : 0.1
        time_sampling           : soft2-contiguous-hardish3
        time_span               : '3m-6m-1y'
        use_centered_positives  : true
        normalize_inputs        : 128
        temporal_dropout        : 0.5
        resample_invalid_frames : 1
        quality_threshold       : 0.8
    model:
        class_path: MultimodalTransformer
        init_args:
            name                   : $EXPERIMENT_NAME
            arch_name              : smt_it_stm_p8
            tokenizer              : linconv
            decoder                : mlp
            stream_channels        : 16
            saliency_weights       : 1:70
            class_loss             : focal
            saliency_loss          : focal
            global_change_weight   : 0.00
            global_class_weight    : 0.00
            global_saliency_weight : 1.00
    lr_scheduler:
      class_path: torch.optim.lr_scheduler.OneCycleLR
      init_args:
        max_lr: $TARGET_LR
        total_steps: $MAX_STEPS
        anneal_strategy: linear
        pct_start: 0.05
    optimizer:
      class_path: torch.optim.Adam
      init_args:
        lr: $TARGET_LR
        weight_decay: 1e-3
        betas:
          - 0.9
          - 0.99
    trainer:
      accumulate_grad_batches: 8
      default_root_dir     : $DEFAULT_ROOT_DIR
      accelerator          : gpu
      devices              : 0,
      #devices             : 0,1
      #strategy            : ddp
      check_val_every_n_epoch: 1
      enable_checkpointing: true
      enable_model_summary: true
      log_every_n_steps: 5
      logger: true
      max_steps: $MAX_STEPS
      num_sanity_val_steps: 0
      replace_sampler_ddp: true
      track_grad_norm: 2
    initializer:
        init: $PRETRAINED_STATE
"


# The result of training will output a list of checkpoints in the lightning
# output directory
ls "$DEFAULT_ROOT_DIR"/lightning_logs/*/checkpoints/*.ckpt

# To use them we need to ensure they are packaged.

# Let's assume we have a checkpoint, This command should grab one of them, you
# should be more selective in the one(s) you choose.
CHECKPOINT_FPATH=$(for i in "$DEFAULT_ROOT_DIR"/lightning_logs/*/checkpoints/*.ckpt; do printf '%s\n' "$i"; break; done)
echo "CHECKPOINT_FPATH = $CHECKPOINT_FPATH"

# repackage it as such: (This command may change in the future to make this
# easier / more robust, but it should work in this context)
python -m geowatch.mlops.repackager "$CHECKPOINT_FPATH"

# That should have written a .pt package with a similar name.  To make this
# bash script work, we will just glob for a package and assume its the one we
# want.
PACKAGE_FPATH=$(for i in "$DEFAULT_ROOT_DIR"/lightning_logs/*/checkpoints/*.pt; do printf '%s\n' "$i"; break; done)
echo "PACKAGE_FPATH = $PACKAGE_FPATH"


__doc_eval__='
Now we have a trained packaged model that is aware of your team features.  The
goal is to use it to demonstrate an improvement in the IAPRA scores.  This can
be done using the mlops framework. You can specify multiple values for an
option to grid search over the Cartesian product of all settings.  You should
at the least include your model and the baseline model to determine if your
features are driving an improvement in the scores.
'

DATA_DVC_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
EXPT_DVC_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware=auto)

BASELINE_PACKAGE_FPATH="$EXPT_DVC_DPATH"/models/fusion/Drop4-BAS/packages/Drop4_TuneV323_BAS_30GSD_BGRNSH_V2/package_epoch0_step41.pt.pt
geowatch model_stats "$BASELINE_PACKAGE_FPATH"

# NOTE:
# The schedule evaluation script originally ran on a single coco file that
# contains all of the validation regions. A more stable way to run the system
# involves splitting the larger validation dataset into a single kwcoco file
# per region, and then running it on all regions separately.
python -m geowatch.cli.split_videos "$DATA_DVC_DPATH"/Drop4-BAS/data_vali_invariants.kwcoco.json

python -m geowatch.mlops.schedule_evaluation \
    --params="
        matrix:
            bas_pxl.package_fpath:
                - $PACKAGE_FPATH
                - $BASELINE_PACKAGE_FPATH
            bas_pxl.channels:
                - 'auto'
            bas_pxl.test_dataset:
                - $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R001_uky_invariants.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/data_vali_KR_R002_uky_invariants.kwcoco.json
                - $DATA_DVC_DPATH/Drop4-BAS/data_vali_US_R007_uky_invariants.kwcoco.json
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
    --backend=tmux --queue_name "demo-queue" \
    --pipeline=bas \
    --run=1

### NOTE:
# The above script assumes that your bashrc activates the appropriate
# virtualenv by default. If this is not the case you will need to specify an
# additional argument to `watch.mlops.schedule_evaluation`. Namely:
# ``--virtualenv_cmd``. For instance if you have a conda environment named
# "watch", you would add ``--virtualenv_cmd="watch"`` to the command.


__doc_mlops__='
This script will run through the entire BAS pipeline and output results in the
"root_dpath".


The names of the outputs are chosen based on a hash of the configuration, which
enables us to reuse existing results. Symlinks are setup such that it is clear
what previous steps a specific result relied on.

The important part is that there will be a folder for each pipeline node.
The bas_pxl_eval node is the IARPA evaluation results and that stores the
metrics we are interested in.

For now you should manually inspect those results, but in the future
the mlops framework will contain a way to aggregate and analyze results
automatically.
'
