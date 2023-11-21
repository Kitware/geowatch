# Rutgers Material Segmentation

## Additional packages to install:

- `pip install pycm`

## How to generate the material predictions:

```python
python ./geowatch/tasks/rutgers_material_seg_v2/predict.py  --kwcoco_fpath <input_kwcoco_path> --model_fpath <model_fpath> --config_fpath <config_fpath> --output_kwcoco_fpath <output_kwcoco_fpath> 
```

Example:

```python
python ./geowatch/tasks/rutgers_material_seg_v2/predict.py /data4/datasets/dvc-repos/smart_data_dvc/Drop6-MeanYear10GSD-V2/data_vali_I2L_split6.kwcoco.zip /home/purri/research/smart_watch_mat_seg/matseg_exps/2023-05-22/21-51-37/checkpoints/model-epoch=35-train_F1Score=0.32025.ckpt
```

## Most up to date model and config file:

Model path: `$EXPT_DVC_DPATH/smart_expt_dvc/models/rutgers/mat_seg_12_14_22/model-epoch=04-valid_f1_score=0.964.ckpt`  <br/>
Config path: `$EXPT_DVC_DPATH/smart_expt_dvc/models/rutgers/mat_seg_12_14_22/config.yaml`