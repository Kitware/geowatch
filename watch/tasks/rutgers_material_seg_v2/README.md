# Rutgers Material Segmentation

## How to generate the material predictions:
```
python ./watch/tasks/rutgers_material_seg_v2/predict.py <path_to_checkpoint> <path_to_config_path> <input_kwcoco_path> <path_to_save_kwcoco_predictions>
```

## Most up to date model and config file:
Model path: `smart_expt_dvc/models/rutgers/mat_seg_12_14_22/model-epoch=04-valid_f1_score=0.964.ckpt`  <br />
Config path: `smart_expt_dvc/models/rutgers/mat_seg_12_14_22/config.yaml`