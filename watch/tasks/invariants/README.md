## Update 1/2022

Features are now generated with a multitemporal UNet with attention architecture. `predict.py` generates PCA reduced features from the given high dimensional feature maps generated from pretext tasks. Pretext features, before/after heatmaps, and segmentation heatmaps are generated in within the same script and turned on and off with flags. Features are generated on shared bands between S2 and L8.

With example below, the `output_kwcoco` of predict.py generates an 8 dimension feature map, of which one dimension is a segmentation heatmap and one is before/after heatmap. As currently set up, the resulting `invariants.kwcoco.json` file reroots the original image paths out of the uky_invariants folder, so that the original image bands can be loaded alongside the invariant features without disturbing or copying the original file structure.

Ex: 

```
SMART_WATCH_DVC=$(WATCH_HACK_IMPORT_ORDER="" python -m watch.cli.find_dvc)
echo "SMART_WATCH_DVC='${SMART_WATCH_DVC}'"
ls "$SMART_WATCH_DVC"

(cd $SMART_WATCH_DVC && dvc pull -r aws --recursive models/uky)

# export CUDA_VISIBLE_DEVICES=2
python -m watch.tasks.invariants.predict \
    --input_kwcoco $SMART_WATCH_DVC/Drop1-Aligned-L1-2022-01/data.kwcoco.json \
    --output_kwcoco $SMART_WATCH_DVC/uky_invariants/Drop1-Aligned-L1-2022-01/invariants.kwcoco.json \
    --pretext_ckpt_path $SMART_WATCH_DVC/models/uky/uky_invariants_2022_01/pretext/pretext.ckpt \
    --segmentation_ckpt $SMART_WATCH_DVC/models/uky/uky_invariants_2022_01/segmentation/segmentation.ckpt \
    --do_pca 1 \
    --num_dim 8 \
    --num_workers 0 \
    --tasks all
```

If --tasks is set to "invariants", then all 8 dimensions will come from pca reduced pretext features.

## Update: 11/2021

There are now separate `fit_segment.py` and `predict_segment.py` scripts following the same syntax for producing binary segmentation heatmaps using a UNet architecture with attention blocks after each down convolution. Additional arguments are `--num_attention_layers` (from 0 to 4) and `--positional_encoding`, to include an encoding based on date of capture.

This folder contains code for training and generating features using a multi-headed architecture on a set of different pretext-learning tasks. The tasks are pixel-level temporal sorting, associating aligned image patches across time, and associating augmented images with each other. Features for each task can be generated and stored using `predict.py`. In addition, the general purpose features corresponding the output of the backbone network can be stored. Each feature map is a pixel-level map corresponding to the input image. Code for training a model is found in `fit.py`. Subsets of these tasks for feature generating and training can be specified. In addition, the feature dimension for each task can be specified during training. By default, general feature maps are 64-dimensional and tasks specific features are 8-dimensional.

Ex: 

```
python -m watch.tasks.invariants.fit 
    --train_dataset path/to/train_dataset.kwcoco.json 
    --save_dir logs
```

Ex: 

```
python -m watch.tasks.invariants.predict \
    --input_kwcoco path/to/kwcoco_for_feature_generation.kwcoco.json \
    --output_kwcoco path/to/store_updated_features.kwcoco.json \
    --ckpt_path path/to/pytorch-lightning_checkpoint.ckpt
```

Currently a separate model must be trained for each different input sensor (S2, L8, WV). If a kwcoco dataset contains multiple sensors, different checkpoints must be used to generate features. The output dataset can be updated in-place to add features for a the different sensors.

Future functionality: We are working on integrating the cost-sensitive channel selection loss to reduce the generated features from the full dimension to include only a subset that is most impactful for downstream tasks. This functionality will be available soon.

## Updates 10/1/21: 

The main updates to `fit` and `predict` are the use of updated loss functions in the arrow-of-time prediction tasks, which we have found to provide a significant benefit in downstream change detection on the Onera dataset. Current checkpoints have been trained on drop1-S2-L8-aligned. An important step for future work is to include more data into the training pipeline, which we are in the process of gathering.
