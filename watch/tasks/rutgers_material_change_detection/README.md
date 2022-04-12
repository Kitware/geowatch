Model can be found in DVC at:

BAS model:

```
smart_watch_dvc/models/rutgers/rutgers_bas_model_v0.pth.tar
```

SC model:

```
smart_watch_dvc/models/rutgers/rutgers_sc_model_v4.pth.tar
```

To generate BAS heatmap predictions run:

```
export PATH_TO_MODEL=/SSD1TB/purri/smart_watch/saved_models/iarpa_drop2/total_bin_change/early_fusion/0001/best_model.pth.tar
export PATH_TO_KWCOCO_DIR=/data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-01/  
export PATH_TO_OUTPUT_KWCOCO_FILE=/data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-01/results.kwcoco.json
python -m watch/tasks/rutgers_material_change_detection/predict.py \
  $PATH_TO_MODEL $PATH_TO_KWCOCO_DIR $PATH_TO_OUTPUT_KWCOCO_FILE
```

To generate SC heatmap predictions run:

```
export PATH_TO_MODEL=/data4/datasets/smart_watch_dvc/models/rutgers/rutgers_sc_model_v4.pth.tar
export PATH_TO_KWCOCO_DIR=/data4/datasets/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/
export PRED_MODEL_NAME=material_heatmap_pred
python -m watch/tasks/rutgers_material_change_detection/predict_sc.py \
  $PATH_TO_MODEL $PATH_TO_KWCOCO_DIR $PRED_MODEL_NAME
```

Commandline optional arguments:
`device` - The type of hardware to process data with model. Set to "device" by default.
`n_workers` - Number of CPU processes to load data into model. Set to 4 by default.
`batch_size` - Number of examples combined that go into the model in forward pass. Set to 8 by default.
`stride` - The amount a window will be moved over the original image. By default set to the models trained stride (usually 25).
`heatmap_pred_channel_names` - Overwrite the name of heatmap predictions in the produced kwcoco file. Set to `None` by default.
