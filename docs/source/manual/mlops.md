# Using the MLOps tool to run GeoWATCH

NOTE: THE PATH FORMAT IS IN HIGH FLUX. DOCS MAY BE OUTDATED

MLOps is a wrapper around the cmd_queue library that provides a single entrypoint for all steps of the GeoWATCH TA-2 pipeline, starting with a model checkpoint and the program data:

1. Predict BAS on low-res data (S2/L8)
2. Pixel evaluation metrics on BAS
3. Convert to polygons in IARPA site model geojson format
4. Score on IARPA metrics [using a heuristic for SC phases]
5. Visualize BAS
6. Create cropped SC dataset from BAS predictions with high-res data (S2/WV/PD)
7. Predict SC on high-res data
2. Pixel evaluation metrics on SC
3. Convert to polygons in IARPA site model geojson format (again)
4. Score on IARPA metrics (again)
5. Visualize SC

You can invoke any one of these things through `python -m geowatch...` without MLOps, but it handles all the plumbing of feeding jobs' input/output to each other and spawning them to use all your available compute resources.


## using DVC and geowatch_dvc


(Note: `geowatch_dvc` will be replaced by `sdvc` from the `simple_dvc` package.)

`geowatch_dvc` is an alias for `python -m geowatch.cli.find_dvc`.
MLOps uses it to manage paths to your DVC repos.

To get started:
`geowatch_dvc list`

Do you see a data and expt repo with exists=True? If not,

a) you have an existing local DVC checkout, but it's not listed

```bash
geowatch_dvc add --name=my_data_repo --path=path/to/smart_data_dvc --hardware=hdd --priority=100 --tags=phase2_data
geowatch_dvc add --name=my_expt_repo --path=path/to/smart_expt_dvc --hardware=hdd --priority=100 --tags=phase2_expt
```

b) you don't have an existing local DVC checkout

clone the source repos:
https://gitlab.kitware.com/smart/smart_data_dvc
https://gitlab.kitware.com/smart/smart_expt_dvc

and put them in the recommended places.

these can also be overridden by environment variables; do so at your own risk

if this worked, you can:
```bash
cd $(geowatch_dvc --tags="phase2_expt")
```
cd doesn't work when the terminal is too narrow for 1 line, because it'll prettyprint with linebreaks


### sidebar: setting up a shared DVC cache on a shared machine

Multiple users can share a DVC cache so they don't have to duplicate the data!
```bash
dvc config cache.shared group
dvc config cache.type symlink
dvc cache dir /data/dvc-caches/smart_watch_dvc  # make sure everyone can read/write here
dvc checkout
```

## using mlops to checkout a model

`geowatch mlops` is an alias for `python -m geowatch.cli.mlops_cli` or `python -m geowatch.mlops.expt_manager`. 
The entrypoint in code is `watch/mlops/expt_manager.py`.

Let's choose a model and do stuff with it!
```bash
geowatch mlops --help

MODEL="Drop4_BAS_Retrain_V002"

geowatch mlops "pull packages" --model_pattern="${MODEL}*"
geowatch mlops "pull evals" --model_pattern="${MODEL}*"
geowatch mlops "status"
python -m geowatch.mlops.expt_manager "list" --model_pattern="${MODEL_OF_INTEREST}*"  # for more info
python -m geowatch.mlops.expt_manager "report" --model_pattern="${MODEL_OF_INTEREST}*"  # for more info
```
The "volatile" table shows model predictions, which are not versioned for space reasons, but are an intermediate product of running pixel and polygon evaluations. The "versioned" table shows models and evaluations which should now exist in your local DVC repo.

the "versioned" table in more detail:
  - type - "pkg_fpath" (model) or "eval_*"
  - has_raw - the file exists
  - has_dvc - the `.dvc` file exists
  - needs_pull - the `.dvc` file is not backed by data on this machine
  - needs_push - the `.dvc` file is not backed by data on the remote

TODO does "staging" still exist?

models on disk live in `$(geowatch_dvc --tags="phase2_expt")/models/fusion/${DATASET}/packages`.

predictions on disk live in `$(geowatch_dvc --tags="phase2_expt")/models/fusion/${DATASET}/pred`.

evals on disk live in `$(geowatch_dvc --tags="phase2_expt")/models/fusion/${DATASET}/eval`.

## using mlops to run an evaluation

geowatch mlops "evaluate" should work as an alias but doesn't right now

if you haven't used this TEST_DATASET yet, remember to unzip its splits.zip first

```bash
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
DATA_DVC_DPATH=$(geowatch_dvc --tags="phase2_data")
DVC_EXPT_DPATH=$(geowatch_dvc --tags="phase2_expt")
TEST_DATASET=$DATA_DVC_DPATH/$DATASET_CODE/data.kwcoco.json
python -m geowatch.mlops.schedule_evaluation \
    --params="
        matrix:
            trk.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/packages/Drop4_BAS_Retrain_V002/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt
            trk.pxl.data.test_dataset:
                - $TEST_DATASET
            trk.pxl.data.window_scale_space: 15GSD
            trk.pxl.data.time_sampling:
                - "contiguous"
            trk.pxl.data.input_scale_space:
                - "15GSD"
            trk.poly.thresh:
                - 0.07
                - 0.1
                - 0.12
                - 0.175
            crop.src:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc/online_v1/kwcoco_for_sc_fielded.json
            crop.regions:
                - trk.poly.output
            act.pxl.data.test_dataset:
                - /home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/crop/online_v1_kwcoco_for_sc_fielded/trk_poly_id_0408400f/crop_f64d5b9a/crop_id_59ed6e1b/crop.kwcoco.json
            act.pxl.data.input_scale_space:
                - 3GSD
            act.pxl.data.time_steps:
                - 3
            act.pxl.data.chip_overlap:
                - 0.35
            act.poly.thresh:
                - 0.01
            act.poly.use_viterbi:
                - 0
            act.pxl.model:
                - $DVC_EXPT_DPATH/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/packages/Drop4_SC_RGB_scratch_V002/Drop4_SC_RGB_scratch_V002_epoch=99-step=50300-v1.pt.pt
        include:
            - act.pxl.data.chip_dims: 256,256
              act.pxl.data.window_scale_space: 3GSD
              act.pxl.data.input_scale_space: 3GSD
              act.pxl.data.output_scale_space: 3GSD
    " \
    --enable_pred_trk_pxl=1 \
    --enable_pred_trk_poly=1 \
    --enable_eval_trk_pxl=1 \
    --enable_eval_trk_poly=1 \
    --enable_crop=0 \
    --enable_pred_act_pxl=0 \
    --enable_pred_act_poly=0 \
    --enable_eval_act_pxl=0 \
    --enable_eval_act_poly=0 \
    --enable_viz_pred_trk_poly=1 \
    --enable_viz_pred_act_poly=0 \
    --enable_links=1 \
    --devices="0,1" --queue_size=2 \
    --queue_name='secondary-eval' \
    --backend=serial --skip_existing=0 \
    --run=0

```
the enable pred, eval, crop, and viz flags are the various stages of the GeoWATCH TA2 system. They are all enabled by default. Any params for a step with `--enable_*=0` will be ignored, and duplicate params will be grid-searched over.

The params are namespaced for convenience
  - trk.pxl - BAS predict and pixel evaluate
  - trk.poly - BAS tracking and IARPA poly evaluate
  - crop - create SC dataset
  - act.pxl - SC predict and pixel evaluate
  - act.poly - SC tracking and IARPA poly evaluate

First, dry-run with run=0, and if it looks ok, set run=1. --backend=tmux is also highly recommended instead of serial, it makes the running queue much easier to debug (backend="slurm" is also available if your machine supports it). 
`--virtualenv_cmd="conda activate watch"`
or something to that effect is also needed if your bashrc does not automatically drop you into the watch virtualenv.

## debugging/examining a running mlops invocation

This assumes you're using --backend=tmux.

check on a running queue with
```bash
tmux a
<C-b> s # switch between sessions
<C-b> d # quit back to main window
```

You'll see each session start with the command `source /long/path/_cmd_queue_schedule/${QUEUE_NAME}_etc/etc/etc.sh`. The paths will also be printed to the main window for convenience, along with the commands to kill any leftover tmux sessions or drop into them.
These sourced scripts define what is being run in each tmux session ("job"). In serial mode, they will all run in the main window.

When each job finishes, the result will be cached and its dependencies will start running.


### sidebar: help, my predict step is hanging!
This might happen due to data workers not being cleaned up; you'll see "return dset.fpath = ..." but then the predict job will never finish. Kill the python process, open the sourced script for that job and manually run the steps under "# Signal this worker is complete". You may have to kill and rerun the whole mlops pipeline, but it will pick up where it left off.

## What does an mlops invocation run?

You can dig through the sourced scripts to piece together what a full GeoWATCH TA2 run looks like, minus the boilerplate and error handling. (There are plenty of examples in the mlops source code, but here's how to tell exactly what you were running.) For example, the pipeline above turns into:

```bash
# prediction (omitted)
# python -m geowatch.tasks.fusion.predict ...

# tracking
ROOT="/home/local/KHQ/matthew.bernstein/data/dvc-repos/smart_expt_dvc-ssd/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/pred/trk/Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC_data.kwcoco"
PXL_EXPT="${ROOT}/trk_pxl_b788335d"
TRK_EXPT="trk_poly_ca4372e1"
python -m geowatch.cli.run_tracker \
        "${TRK_ROOT}/pred.kwcoco.json" \
        --default_track_fn saliency_heatmaps \
        --track_kwargs '{"thresh": 0.12, "moving_window_size": null, "polygon_fn": "heatmaps_to_polys"}' \
        --clear_annots \
        --out_sites_dir "${TRK_ROOT}/${TRK_EXPT}/sites" \
        --out_site_summaries_dir "${TRK_ROOT}/${TRK_EXPT}/site-summaries" \
        --out_sites_fpath "${TRK_ROOT}/${TRK_EXPT}/site_tracks_manifest.json" \
        --out_site_summaries_fpath "${TRK_ROOT}/${TRK_EXPT}/site_summary_tracks_manifest.json" \
        --out_kwcoco "${TRK_ROOT}/${TRK_EXPT}/tracks.kwcoco.json"

# pxl eval
python -m geowatch.tasks.fusion.evaluate \
        --true_dataset=/home/local/KHQ/matthew.bernstein/data/dvc-repos/smart_data_dvc-ssd/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/data.kwcoco.json \
        --pred_dataset="${TRK_ROOT}/pred.kwcoco.json" \
        --eval_dpath="${TRK_ROOT}"
        --score_space=video \
        --draw_curves=True \
        --draw_heatmaps=True \
        --viz_thresh=0.2 \
        --workers=2

# iarpa poly eval
python -m geowatch.cli.run_metrics_framework \
        --merge=True \
        --name "Drop4_BAS_Retrain_V002_epoch=31-step=16384.pt.pt-trk_pxl_b788335d-${TRK_EXPT}" \
        --true_site_dpath "/home/local/KHQ/matthew.bernstein/data/dvc-repos/smart_data_dvc-ssd/annotations/site_models" \
        --true_region_dpath "/home/local/KHQ/matthew.bernstein/data/dvc-repos/smart_data_dvc-ssd/annotations/region_models" \
        --pred_sites "${TRK_ROOT}/${TRK_EXPT}/site_tracks_manifest.json" \
        --tmp_dir "${TRK_ROOT}/${TRK_EXPT}/_tmp" \
        --out_dir "${TRK_ROOT}/${TRK_EXPT}" \
        --merge_fpath "${TRK_ROOT}/${TRK_EXPT}/merged/summary2.json" \

# viz
geowatch visualize \
        "${TRK_ROOT}/${TRK_EXPT}/tracks.kwcoco.json" \
        --channels="red|green|blue,salient" \
        --stack=only \
        --workers=avail/2 \
        --workers=avail/2 \
        --extra_header="\ntrk_pxl_b788335d-${TRK_EXPT}" \
        --animate=True
```
