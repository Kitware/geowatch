UKy temporal ordering prediction.
====

UKy code for predicting the arrow of time of image pairs using a Siamese Network with UNet/UNet blur backbone. Code for training on drop0_aligned is located in `fit.py` and evaluating on `drop0_aligned` is located in `predict.py`. In addition, scripts for setting up datasets and training on SpaceNet 7 are located in the `spacenet` folder. Running these scripts should be done through Python's `-m` flag and the `watch` module. For instance to run `predict.py` you would run `python -m watch.tasks.uky_temporal_prediction`

Conda Environment
----
A minimal conda environment to run our code can be found in `conda_env.yml`. To install and activate the environment, run `conda env create -f conda_env.yml` followed by `conda activate uky_temporal_prediction`. For full "watch" integration, the `conda_env.yml` in the root of the watch repository should be used.

Project Data
----
Training a UNet or UNet blur model on the temporal prediction task using project data can be accomplished using `fit.py`. The sensor, number of channels, identities of the train/validation video sequences should be specified. The script uses the [Pytorch Lightning](https://www.pytorchlightning.ai/) library. Models will be stored as Lightning checkpoints by default in the `logs/` folder. 

Examples: 

```bash
# Set this to your path to your watch DVC or data directory
WATCH_DATA_DPATH=/u/eag-d1/data/watch
WATCH_DATA_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

# Print stats about the videos in the dataset
python -m watch.scripts.watch_coco_stats $WATCH_DATA_DPATH/drop0_aligned/data.kwcoco.json

python -m watch.tasks.uky_temporal_prediction.fit \
    --max_epochs 100 --sensor S2 --train_video 5 --val_video 4 --in_channels 3 \
    --train_dataset $WATCH_DATA_DPATH/drop0_aligned/data.kwcoco.json \
    --val_dataset $WATCH_DATA_DPATH/drop0_aligned/data.kwcoco.json


python -m watch.tasks.uky_temporal_prediction.fit \
    --max_epochs 100 --sensor LC --train_video 2 --val_video 4 --in_channels 1 \
    --train_dataset $WATCH_DATA_DPATH/drop0_aligned/data.kwcoco.json \
    --val_dataset $WATCH_DATA_DPATH/drop0_aligned/data.kwcoco.json
```

To predict on drop0 data use `predict.py`. Arguments include lightning checkpoint and specify the sensor and number of channels corresponding to the trained model. The script also loads a kwcoco file and outputs another kwcoco file (these can be the same file). The output kwcoco file will include a path to features as an entry in the dictionary for each image. Features are stored in `args.output_folder` as .pt files. The script can accept a list of desired image ids to run on. If none are specified, the script will run on all available images and skip images that come from non-matching sensors.

Example: 

```

WATCH_DATA_DPATH=/localdisk0/SCRATCH/watch/smart_watch_dvc
WATCH_DATA_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

# List all checkpoints from the previous training step
find logs -iname "*.ckpt"

# Choose one of them
CHECKPOINT_FPATH=$(find logs -iname "*.ckpt" | tail -n1)
echo "CHECKPOINT_FPATH = $CHECKPOINT_FPATH"

python -m watch.tasks.uky_temporal_prediction.predict \
    --sensor LC \
    --dataset $WATCH_DATA_DPATH/drop0_aligned/data.kwcoco.json \
    --data_folder $WATCH_DATA_DPATH/drop0_aligned/ \
    --output_kwcoco $WATCH_DATA_DPATH/drop0_aligned/data_uky_time_sort_features.kwcoco.json \
    --output_folder $WATCH_DATA_DPATH/drop0_aligned/uky_time_sort_features/ \
    --checkpoint $CHECKPOINT_FPATH
```


Notes:
- Because of the small size of the datasets, we do not expect models trained only on drop0_aligned to provide transferable features
- Due to differences in sizes of images, we have been unable to run these scripts on WV imagery in drop0_aligned
- Other training hyperparameters can also be adjusted through command line arguments in `fit.py`

SpaceNet 7
----
To prepare to train on SpaceNet 7: run 

```
python -m watch.tasks.uky_temporal_prediction.spacenet.data.create_splits --data_dir /path/to/SpaceNet/7/train

python -m watch.tasks.uky_temporal_prediction.spacenet.data.splits_unmasked.create_splits  --data_dir /path/to/SpaceNet/7/train

```

Train a model on SpaceNet 7 using `python time_sort_S7.py`. The module relies on the [Pytorch Lightning](https://www.pytorchlightning.ai/) library. Checkpoints will be stored by default in `./logs/` folder. Trained checkpoints can then be loaded into `predict.py` to evaluate on the before/after task using S2 RGB imagery from `drop0_aligned`.

