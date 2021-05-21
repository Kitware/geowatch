UKy code for predicting the arrow of time of image pairs using a Siamese Network with UNet/UNet blur backbone. Code for training on SpaceNet 7 as well as drop0_aligned is included. Code for loading a checkpoint, extracting, and storing features from drop0_aligned imagery is also included.

To prepare to train on SpaceNet 7: run `python spacenet/data/create_splits.py  --data_dir /path/to/SpaceNet/7/train` and `python spacenet/data/splits_unmasked/create_splits.py  --data_dir /path/to/SpaceNet/7/train`.

Train a model on SpaceNet 7 using `time_sort_S7.py`. The module relies on the [Pytorch Lightning](https://www.pytorchlightning.ai/) library. Checkpoints will be stored by default in `./logs/` folder.

Loaded checkpoints can be used to extract features from drop0_aligned data using `extract_features.py`. Using a model trained on SpaceNet 7 will only allow use of Sentinel-2 RGB imagery. The input to the function includes a path the dvc repo, the data.kwoco.json file and an output kwoco.json file (which may overwrite the original). The output will include paths to features, stored as .pt files, for each image specified at run time (or all images matching the chosen sensor if none are specified).

In addition, models can be trained on the arrow of time prediction task using any available sensor in drop0_aligned using the script `time_sort_drop0.py`. Models are trained on the chosen "video" sequences from drop0_aligned. Note that you will have to now manually choose the number of in_channels corresponding to the data type.
