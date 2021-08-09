This folder contains code for training and generating features using a multi-headed architecture on a set of different feature tasks. The tasks are pixel-level temporal sorting, associating aligned image patches across time, and associating augmented images with each other. Features for each task can be generated and stored using `predict.py`. In addition, the general purpose features can be stored. Code for training a model is found in `fit.py`. Subsets of these tasks for feature generating and training can be specified. In addition, the feature dimension for each task can be specified during training.

Ex: `python -m watch.tasks.invariants.fit.py --train_dataset path/to/train_dataset.kwcoco.json --save_dir logs`

Ex: `python -m watch.tasks.invariants.predict.py --input_kwcoco path/to/kwcoco_for_feature_generation.kwcoco.json --output_kwcoco path/to/store_updated_features.kwcoco.json --ckpt_path path/to/pytorch-lightning_checkpoint.ckpt`

The code has been tested and features have been generated and shared for drop1_S2_aligned_c1 and drop1-S2-L8-LS-aligned-c1. Argparse defaults correspond to the hyperparameters used to generate the features we have shared via dvc.