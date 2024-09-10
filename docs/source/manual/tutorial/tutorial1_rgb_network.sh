#!/bin/bash
__doc__="

Basic ToyData Pipeline Tutorial
===============================

This demonstrates an end-to-end pipeline on RGB toydata

The tutorial generates its own training data so it can be run with minimal
effort to test that all core components of the system are working.

This walks through the entire process of fit -> predict -> evaluate to
train a fusion model on RGB data.

RunMe:
    export ACCELERATOR=gpu
    source ~/code/geowatch/docs/source/manual/tutorial/tutorial1_rgb_network.sh
"

# Default to GPU if you have nvidia-smi, otherwise cpu
if type nvidia-smi; then
    export ACCELERATOR="${ACCELERATOR:-gpu}"
else
    export ACCELERATOR="${ACCELERATOR:-cpu}"
fi


# For those windows folks:
if [[ "$(uname -a)" == "MINGW"* ]]; then
    echo "detected windows with mingw"
    export HOME=$USERPROFILE
    export USER=$USERNAME
fi


# This tutorial will generate its own training data. Change these paths to
# wherever you would like the data to go (or use the defaults).  In general
# experiments will have a "data" DVC directory where the raw data lives, and an
# "experiment" DVC directory where you will train your model and store the
# results of prediction and evaluation.

# In this example we are not using any DVC directories, but we will use DVC in
# the variable names to be consistent with future tutorials.

DVC_DATA_DPATH=$HOME/data/dvc-repos/toy_data_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/toy_expt_dvc

mkdir -p "$DVC_DATA_DPATH"
mkdir -p "$DVC_EXPT_DPATH"

echo "
Generate Toy Data
-----------------

Now that we know where the data and our intermediate files will go, lets
generate the data we will use to train and evaluate with.

The kwcoco package comes with a commandline utility called 'kwcoco toydata' to
accomplish this.
"

# Define the names of the kwcoco files to generate
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_train/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_vali/data.kwcoco.json
TEST_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_test/data.kwcoco.json

# Generate toy datasets using the "kwcoco toydata" tool
kwcoco toydata vidshapes2-frames10-amazon --dst "$TRAIN_FPATH"
kwcoco toydata vidshapes4-frames10-amazon --dst "$VALI_FPATH"
kwcoco toydata vidshapes2-frames6-amazon --dst "$TEST_FPATH"


echo "
Inspect Generated Kwcoco Files
------------------------------

Now that we have generated the kwcoco files, lets get used to the 'geowatch'
and 'kwcoco' command tooling to insepct the content of the files.

Printing statistics is a good first step. The kwcoco stats are for basic
image-level statistics, whereas the geowatch stats will give information
relevant to the geowatch project, i.e. about videos, sensors, and channels.
"
# First try the kwcoco stats (can pass multiple files)
kwcoco stats "$TRAIN_FPATH" "$VALI_FPATH" "$TEST_FPATH"

# Next try the geowatch stats
geowatch stats "$TRAIN_FPATH"


echo "

Another important CLI tool is 'geowatch visualize' which can be used to
visually inspect the contents of a kwcoco file. It does this by simply dumping
image files to disk.  This is most useful when the underlying dataset has data
outside of the visual range, but it will work on 'regular' rgb data too!

Running visualize by default will write images for all channels in the exiting
'kwcoco bundle' (i.e. the directory that contains the kwcoco json file) with a
hash corresponding to the state of the kwcoco file. It will also output all the
channels by default. Use 'geowatch visualize --help' for a list of additional
options.

Some useful options are:

    * '--channels' to view only specific channels
    * '--animate' create animated gifs from the sequence
    * '--viz_dpath' specify a custom output directory
"

# Try visualizing the path to the training kwcoco file
geowatch visualize "$TRAIN_FPATH" --viz_dpath="$DVC_EXPT_DPATH/_viz_toyrgb" --animate=False


echo "

Training, Prediction, and Evaluation
------------------------------------

Now that we are more comfortable with kwcoco files, lets get into the simplest
and most direct way of training a fusion model. This is done by simply calling
'geowatch.tasks.fusion' as the main module. We will specify:

Data arguments:

* paths to the training and validation kwcoco files
* what channels we want to early / late fuse (given by a kwcoco sensorchan spec)
* information about the input chip size and temporal window

Model arguments:

* the underlying architecture

Other arguments:

* learning rate schedulers
* optimizers
* training strategies

In this tutorial - for compatability - we will use 'cpu' as our lightning
accelerator. See the lightning docs for other available trainer settings:
https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#devices

We will also specify a work directory that will be similar to directories used
when real geowatch models are trained.
"

# For convinience we remind the user these variables need to be populated
# even if they already ran the above steps.
if [[ "$(uname -a)" == "MINGW"* ]]; then
    echo "detected windows with mingw"
    export HOME=$USERPROFILE
    export USER=$USERNAME
fi
if type nvidia-smi; then
    export ACCELERATOR="${ACCELERATOR:-gpu}"
else
    export ACCELERATOR="${ACCELERATOR:-cpu}"
fi

DVC_DATA_DPATH=$HOME/data/dvc-repos/toy_data_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/toy_expt_dvc
TRAIN_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_train/data.kwcoco.json
VALI_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_vali/data.kwcoco.json
TEST_FPATH=$DVC_DATA_DPATH/vidshapes_rgb_test/data.kwcoco.json


WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
EXPERIMENT_NAME=ToyRGB_Demo_V001
DATASET_CODE=ToyRGB
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
MAX_STEPS=32
TARGET_LR=3e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 1e-2)")
python -m geowatch.tasks.fusion fit --config "
data:
    num_workers          : 0
    train_dataset        : $TRAIN_FPATH
    vali_dataset         : $VALI_FPATH
    channels             : 'r|g|b'
    time_steps           : 5
    chip_dims            : 128
    batch_size           : 3
model:
    class_path: MultimodalTransformer
    init_args:
        name        : $EXPERIMENT_NAME
        arch_name   : smt_it_stm_p8
        global_box_weight: 0
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: $TARGET_LR
    weight_decay: $WEIGHT_DECAY
trainer:
  default_root_dir     : $DEFAULT_ROOT_DIR
  accelerator          : $ACCELERATOR
  devices              : 1
  max_steps: $MAX_STEPS
  num_sanity_val_steps: 0
  limit_val_batches    : 2
  limit_train_batches  : 4
"

# For more options with this particular model see:
# python -m geowatch.tasks.fusion fit --model.help=MultimodalTransformer


echo '
The training code will output useful information in the "DEFAULT_ROOT_DIR".

This will include

   * A set of checkpoints that score the best on validation metrics in $DEFAULT_ROOT_DIR/lightning_logs/*/checkpoints
   * A monitor directory containing visualizations of train and validation batches in
       `$DEFAULT_ROOT_DIR/lightning_logs/*/monitor/train/batches` and
       `$DEFAULT_ROOT_DIR/lightning_logs/*/monitor/validate/batches`
   * Image files containing visualized tensorboard curves in $DEFAULT_ROOT_DIR/lightning_logs/*/monitor/tensorboard
       (you can start a tensorboard server if you want to)

In this examples we also specify a "package_fpath" for convenience.
Typically you will want to repackage multiple checkpoints as torch models (see
tutorial TODO), but for this first example, "package_fpath" provides an easy way to
tell the training to always package up the final checkpoint in a serialized model.

Torch models are great because they combine the weights with the code needed to
execute the forward pass of the network.  They can also store metadata about
how the model was trained, which is critical for performing robust analysis on
large numbers of models.

We provide a CLI tool to summarize the info contained in a torch model via
"geowatch torch_model_stats". Lets try that on the model we just built.
'

PACKAGE_FPATH="$DEFAULT_ROOT_DIR"/final_package.pt

# Find a package if training did not complete
PACKAGE_FPATH=$(python -c "if 1:
    import pathlib
    default_root = pathlib.Path(r'$DEFAULT_ROOT_DIR')
    pkg = default_root / 'final_package.pt'
    if pkg.exists():
        print(pkg)
    else:
        cand = sorted(default_root.glob('package-interupt/*.pt'))
        assert len(cand)
        print(cand[-1])
")
echo "$PACKAGE_FPATH"

geowatch torch_model_stats "$PACKAGE_FPATH" --stem_stats=True

# NOTE: There are other model weights available in the
# $DEFAULT_ROOT_DIR/*/*/checkpoints directory that can be converted into
# packages using the geowatch.mlops.repackager script. The final package may not
# be the best model.


echo '
You can see the model knows what kwcoco model it was trained with as well as
what sensors and channels it is applicable to.

Furthermore it knows (1) the names of the classes that correspond to its final
classification layer and (2) the estimated mean / std of the data. These are
two pieces of information that are hardly ever properly bundled with models
distributed by the ML community, and that is something that needs to change.

The model itself knows how to subtract the mean / std, so the dataloader should
never have to. Our fusion training code also knows how to estimate this for
each new dataset, so you never have to hard code it.
'


echo '
Now that we have an understanding of what metadata the model contains, we can
start to appreciate the dead simplicity of predicting with it.

To use a model to predict on an unseed kwcoco dataset (in this case the toy
test set) we simply call the "geowatch.tasks.fusion.predict" script and pass it:

   * the kwcoco file of the dataset to predict on
   * the path to the model we want to predict with
   * the name of the output kwcoco file that will contain the predictions

All necessary metadata you would normally have to (redundantly) specify in
other frameworks is inferred by programmatically reading the model. You also
have the option to overwrite prediction parameters. See --help for details, but
for now lets just run with the defaults that match how the model was trained.

Note that the test dataset contains groundtruth annotations. All annotations
are stripped and ignored during prediction.
'

# Note: --format=png is only necessary on windows, due to issues that can
# happen with the GDAL geotiff drivers. On other systems it is recommended to
# remove this option.

# Predict
python -m geowatch.tasks.fusion.predict \
    --test_dataset="$TEST_FPATH" \
    --package_fpath="$PACKAGE_FPATH"  \
    --pred_dataset="$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json \
    --format="png"  \
    --accelerator="$ACCELERATOR" \
    --devices=1

echo '
The output of the predictions is just another kwcoco file, but it augments the
input images with new channels corresponding to predicted heatmaps. We can use
the "geowatch stats" command to inspect what these new channels are.
'

# Inspect the channels in the prediction file
geowatch stats "$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json


echo '
Running this command you can see that images now have a channels "salient",
which corresponds to the BAS saliency task, and "star", "eff", and "superstar"
which correspond to the classification head (for SC), and lastly the "change"
channel, which is from the change head.

Because these are just rasters, we can visualize them using "geowatch
visualize"
'

# Visualize the channels in the prediction file
geowatch visualize "$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json --stack=True


echo '
The last step in this basic tutorial is to measure how good our model is.
We can do this with pixelwise metrics.

This is done by using "watch.tasks.fusion.evaluate" as the main module, and
its arguments are:

    * The true kwcoco data with groundtruth annotations (i.e. the test dataset)
    * The pred kwcoco data that we predicted earlier
    * An output path for results
'

# Evaluate
python -m geowatch.tasks.fusion.evaluate \
    --true_dataset="$TEST_FPATH" \
    --pred_dataset="$DVC_EXPT_DPATH"/predictions/pred.kwcoco.json \
      --eval_dpath="$DVC_EXPT_DPATH"/predictions/eval

echo '
This will output summary pixelwise metrics to the terminal but more detailed reports
will be written to the eval_dpath.

This will include ROC curves, PR curves, and threshold curves drawn as png
images.  These curves are also stored as serialized json files so they can be
reparsed an replotted or used to compute different metric visualizations.

It will also include visualizations of heatmaps overlaid with the truth, with
areas of confusion higlighted.


This concludes the basic RGB tutorial.

The next tutorial lives in tutorial2_msi_network.sh and will use a different
variant of kwcoco generated data that more closely matches the watch problem by
simulating different sensors with different channels.
'


LAST_TRAIN_DIR=$(python -c "
import pathlib
logs_dpath = pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs')
print(sorted(logs_dpath.glob('*/monitor/train'))[-1])
")
echo "LAST_TRAIN_DIR = $LAST_TRAIN_DIR"


python -c "if 1:
    import rich
    rich.print('''Training    : [link=$LAST_TRAIN_DIR]'$LAST_TRAIN_DIR'[/link]''')
    rich.print('''Predictions : [link=$DEFAULT_ROOT_DIR]'$DEFAULT_ROOT_DIR/predictions'[/link]''')
    rich.print('''Evaluation  : [link=$DEFAULT_ROOT_DIR/predictions/eval]'$DEFAULT_ROOT_DIR/predictions/eval'[/link]''')
"
