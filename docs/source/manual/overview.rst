GeoWATCH Overview
=================

In 2020 Kitware won a contract to compete in the
`IARPA SMART <https://www.iarpa.gov/research-programs/smart>`_ program.
The SMART program challenged 5 teams to design algorithms for utilizing
satellite imagery from multiple sensors (e.g., Sentinel2, Landsat-8, Worldview,
Planetscope) to search over multiple large areas of space and time (e.g., 1000
square kilometers over 8 years) for events of interest. The driving use case
was heavy construction (e.g., construction of buildings more than 8000 square
kilometers). The program also evaluated performers on other tasks such as
detecting transient events (e.g. burning man, renaissance fairs) encouraged
teams to look for solutions that were generalizable.

The input to the problem is a SpatioTemporal Asset Catalog (STAC) catalog that
indexes the images available for use. The goal is to ingest data from a region
of interest, and predict geojson "sites summaries": polygons with a start and
end date that should contain the "event of interest". Additionally, each "site"
is sub-classified into phases of the event (i.e. for heavy construction the
phases are site-preparation and active construction).

Kitware, and team members from University of Connecticut, Washington
University, Rutgers, and DZYNE Technologies developed GeoWATCH as its
solution. The initial pitch was that each subteam would develop a semantically
meaningful raster feature, and then Kitware would develop a "fusion" module
that combined these features together, producing a heatmap for every (selected)
input image in the sequence.  The heatmaps are trained to be bright when an
event occurs and dark otherwise. We extract polygons from these heatmaps using
a "tracker module" to produce the final geojson output.

KWCOCO Data Interchange
-----------------------

To facilitate data interchange between the teams in a machine-learning friendly
way, we expanded the
`kwcoco <https://gitlab.kitware.com/computer-vision/kwcoco>`_ module with additional
features to support large multispectral satellite images.
The kwcoco module itself, uses the MS-COCO format as a starting point. We
augmented each "image" dictionary to be able to reference one or more "assets",
which each correspond to a file on disk. Each asset contains a channel code as
well as a transform (usually affine) that warps the pixels in the asset into an
aligned "image view" or "image space". Similarly, multiple images can be
registered to a "video", and another transform can be specified that aligns the
images in a "video view" or "video space". More details on warping and space (which we are going to rename views) can be found in
`the KWCOCO Spaces document
<https://gitlab.kitware.com/computer-vision/kwcoco/-/blob/40386202aa34ce9cf5b48fd3b93cd5e9a2fc0db0/docs/source/concepts/warping_and_spaces.rst>`_.
The KWCOCO spec isn't specifically for geospatial data, but it allows user-defined keys, so in the geowatch tooling, we define
`kwcoco extensions <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/utils/kwcoco_extensions.py?ref_type=heads>`_ that populate
each image with fields like "sensor_coarse", and uses geotiff metadata to infer the transforms.

To create the initial base KWCOCO for a region, we use
`a pipeline <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/cli/prepare_ta2_dataset.py>`_ that
pulls the STAC catalog that points to processed large image tiles,
creates a virtual uncropped kwcoco dataset that points to the large image tiles
(using the GDAL vitual filesystem), and then crops out (othorectifing if
necessary) a set of images that are
`aligned <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/cli/coco_align.py?ref_type=heads>`_
up to an affine transform. The idea is that we are going to try and delay as
much resampling until the very last minute so we can leverage fused affine
transforms. In production, datasets are constructed on the fly, but for
development, we gather kwcoco datasets for each region and push them to members
of the development team using `DVC <https://dvc.org/>`_.

Semantically Rich Features
--------------------------

Because the KWCOCO format allows an image to reference multiple assets, we
define an API such that the input to each module is a KWCOCO file, and the
output is a new KWCOCO file.  In this way we can compute a kwcoco where each
image points to both its original sensed assets as well as the semantically
rich features.  Some of the features our team members contributed generate
rasters for: landcover, depth, `COLD <https://github.com/GERSL/pycold>`_,
materials features, SAM-features, MAE-features, and time-invariant features.

For example, if we wanted to enrich a kwcoco file
with an additional feature (e.g. features from SegmentAnything), we would run a
command like:

.. code:: bash

    python -m geowatch.tasks.sam.predict \
        --input_kwcoco "/path/to/input.kwcoco.zip" \
        --output_kwcoco "/path/to/output.kwcoco.zip" \
        --weights_fpath "/path/to/models/sam/sam_vit_h_4b8939.pth"


Training Pipeline
-----------------

Given a kwcoco file with (or without) enriched features, we can now train a
"fusion" model. Our training backend is implemented with a customized
`LightningCLI <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/tasks/fusion/fit_lightning.py>`_.
The main components are the
`KWCocoDataModule <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/tasks/fusion/datamodules/kwcoco_datamodule.py>`_, the `KWCocoDataset <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/tasks/fusion/datamodules/kwcoco_dataset.py>`_, and the `MultimodalTransformer <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/tasks/fusion/methods/channelwise_transformer.py>`_.
To train a fusion model you specify:

* A path to an input kwcoco file

* The size of the spatial window for each batch

* A "time kernel", which defines how images will be sampled over time to construct a batch

* A sensor-channel code, which defines which sensors / channels will be fed to the network as input

A typical configuration will use a ``196 x 196 @ 2m GSD`` spatial window (note
that the sampling resolution is specified here, and our dataloader will
efficiently resample data at that resolution), and 7 time samples spread over 2
years. While this is the basic case, we can do more complex sampling where each
image is sampled in its native resolution based on a common spatial window, but
we are still working on making that work better. However, in order to ensure we
can handle this case we define a very particular output of our dataloader where
we *do not* collate batch items.
Instead it is up to the network to determine how to tokenize structure data in
the forward pass.  Our main network network processes batch items 1 at a time
because each batch item may contain a different number of tokens. Tokens are
enriched with positional encodings that specify relative space/time location,
which sensor the image is from, and which channel the modality is from.

To sample data efficiently, we abstract the sampling of data via the
`ndsampler <https://gitlab.kitware.com/computer-vision/ndsampler/>`_ API, which uses
`delayed-image <https://gitlab.kitware.com/computer-vision/delayed_image>`_ under the hood to
translate the "virtual coordinate system" visible to the developer to the
correct locations in the larger images.  As long as the images are COGs this is
reasonably fast. For annotation sampling, ndsampler builds an rtree to lookup
annotations that overlap a target region.

When training a model, we check if the hash of the kwcoco file has been seen
before. If not, then we compute and cache dataset statistics like mean/std for
each modality and class frequencies. Based on the configuration of the
KWCocoDataset, the sample grid can be spatially regular, or centered on
specific annotations. Similarly, the "time kernel" controls the time steps
selected for each spatial location. This defines a large flat list of grid
"targets", which we organize in a tree in order to balance sampling of data
(e.g. positive / negative regions).

Our Lightning trainer is fairly standard with a few extensions. First, we have
sophisticated batch item visualization that provides the developer with real
time feedback on qualitative performance. An example of this visualization for a training run that
used the following sensor/channel configuration:

.. code::

    (L8,S2):(blue|green|red|nir),
    (WV):(blue|green|red),
    (WV,WV1):pan,
    (S2):(water|forest|field|impervious|barren|landcover_hidden.0:32,invariants:16),
    (L8):(COLD.0:36),(L8,S2,WV,WV1):(sam.0:64)

.. image:: https://data.kitware.com/api/v1/file/65a6deefd5d9e43895a66459/download?contentDisposition=inline

This image gives a summary of much of the information provided in the batch
including: truth heatmaps, truth bounding boxes, per-task pixelwise weights,
and selected bands from the underlying imagery. Also notable in the above data
is some of the images have checkerboard patterns. This represents NODATA
regions. These are maintained as nans in the tensors all the way up to the
network forward pass, at which point we subtract the mean and divide by the
std, and then zero the nans, which means that the nan values are always imputed
as the mean of the datasets.

In the above sensorchan spec, the pipe separated channels early fused channels,
for each frame all of these channels are stacked into a single tensor that is
passed through a sensor-specific ConvNet to normalize the number of channels
(we literally maintain a dictionary that maps a sensorchan code to a specific stem).
Then we tokenize these channel-normalized features, add positional encodings,
stack them, and send them through the transformer. At the end we pool
activations from timesteps that have multiple sensors and pass them to task
specific heads, which produce heatmaps aligned to the inputs (although in the
future we plan on adding a decoder to ask for predictions at unobserved times).
Given the outputs, the network computes the loss and then lightning does its
thing.

A rough illustration of the network looks like this:

.. image:: https://data.kitware.com/api/v1/file/65a6eb1ed5d9e43895a6645c/download

Additional interesting training capabilities we have is a partial
implementation of
`loss-of-plasticity <https://www.reddit.com/r/MachineLearning/comments/164qc8c/r_loss_of_plasticity_in_deep_continual_learning/>`_.
We also have the ability to initialize a network from another one that is
similar, but may have different numbers of layers / heads / stems, using
`partial-weight-loading <https://devpost.com/software/torchliberator-partial-weight-loading>`_,
which maps weights from one network to another by finding a maximal subtree
isomorphism.
This has been critical to continue training our networks over a long time and
changing the feature configurations. We have observed that after models are
improved by training on semantically rich features, we can drop those features
and retrain a new network that retains some of the old performance. In other
words, the heavyweight features seem to be "instilled" into the network.

Prediction Pipeline
-------------------

After a model is trained, we use torch.package to build a model bundle that
contains its training configuration, model code, and weights. The idea is that
we should be able to pass this model to our prediction script, and have all
train time configurations (e.g. batch sampling) inferred by the predict script
as defaults.

The predict script itself will run a model over a sliding window and stitch the
heatmaps back into a larger raster as illustrated:

.. image:: https://data.kitware.com/api/v1/file/656fd3a8dfc0e5d60cffa244/download



Software Testing
----------------

GeoWATCH places a much larger emphasis on testing than the average research
repository. To enable testing we've developed "kwcoco toydata", which can
produce demo kwcoco dataset for object detection / tracking / segmentation /
classification problems. It can generate dummy MSI imagery and has several
knobs that can be configured. A sample RGB visualization looks like this:

.. image:: https://i.imgur.com/LNBkckz.gif


For GeoWATCH itself, we sometimes need geo-referenced data and not just image
data, and for this geowatch
`extends kwcoco demodata <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/main/geowatch/demo/smart_kwcoco_demodata.py?ref_type=heads>`_ to add these additional fields.


Additionally, many other data structures defined in geowatch and other
supporting libraries come equipped with a ``.random()`` or ``.demo()`` classmethod
to help create instances of them on the fly for testing.

While there are some unit tests, most of the testing is done via doctests and
run with `xdoctest <https://github.com/Erotemic/xdoctest>`_.


MLOps
-----

To evaluate our systems over a parameter grid, we've written an
`mlops <https://gitlab.kitware.com/computer-vision/geowatch/-/tree/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/mlops>`_
system to define prediction pipelines, and run them over a grid of parameters
(using a github actions-like YAML configuration).

The
`basic pipeline structure <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/mlops/pipeline_nodes.py>`_
has the user define the paths a process is expected to take as inputs and
produce as outputs. Outputs of one process can be connected as inputs of
another without the user needing to manually specify them.  Only unconnected
inputs need to be given or non-default configuration variables must be
specified. The user specifies the relative name for each output file, but the
mlops system chooses the directory the outputs will be written to.
It does this using a hashed directory structure, which lets it determine if a
process has completed or not and causes changes in pipeline configurations to
only cause new results to be recomputed. To make navigation of this directory
structure easier, each node's output folder is equipped with a symlinks to its
predecessor nodes that it depends on as well as its successor nodes that depend
on it.

The system assumes that all processes are invokable as a bash script (i.e.
there is a CLI for each operation a user might want), which is a key design
decision. This allows the mlops system to only be concerned about generating
the right bash invocations to run a pipeline. In each output node we write an
"invoke.sh" script which provides the bash invocation used to compute the nodes
results. This has been instrumental when debugging.

The bash-script assumption also means that we can abstract how a pipeline or
DAG is run. We do this via the
`cmd_queue <https://gitlab.kitware.com/computer-vision/cmd_queue>`_ module. To
use this module the user creates a queue and then submits job as a bash command
in the form of a string as well as references to the jobs that it depends on.
The actual execution of the jobs is abstracted by one of three (perhaps soon to
be four) backends:

1. The serial backend where all commands are topologically sorted and run one
   by one in the current terminal. This is great for debugging and stability,
   but does not leverage any parallelism.

2. The SLURM backend, which uses the SLURM CLI to submit all jobs into a SLURM queue.
   This is a very powerful way of submitting jobs, but SLURM is heavyweight and
   can be difficult to setup correctly. Thus we have implemented a third backend

3. The TMUX backend. This is a lightweight custom backend which distributes
   jobs that can run in parallel across multiple TMUX sessions. This also lets
   a user attach to the sessions to watch multiple jobs simultaneously.
   It just statically runs a set sequence of jobs, so it doesn't maximize
   CPU usage like a more dynamic scheduler, but its often good enough.


Relationship to RasterVision
----------------------------

This section was written on 2024-01-16.

Our dataloader automatically computes mean/std of input dataset as well as
class frequency. This seems similar to the "ANALYZE" step in RasterVision.
Something GeoWATCH does not yet do is allow the user to specify the mean/std or
frequency statistics so training is not forced to compute those.

Our virtual sample grid seems to corresponds to "CHIP" in the RasterVision
pipeline. Raster visions direct sampling seems to correspond to what we can do
with ndsampler. We are going to run some tests further compare them and see
which if one is faster than the other. GeoWATCH doesn't have the ability to
pre-chip data, but if you can afford the preprocessing it will likely be
faster than sampling directly from COGs, although it does limit the translation
augmentation that can be done by the dataloader.

For the "TRAIN" step it seems like both frameworks settled on Lightning, so porting
our `callbacks <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/utils/lightning_ext/callbacks?ref_type=heads>`_
for use in RasterVision shouldn't be too hard.

Something that is nice about like about how geowatch invokes LightningCLI is
that it can specify the entire config inline in bash.
Our `tutorial 1 <//gitlab.kitware.com/computer-vision/geowatch/-/blob/main/docs/source/manual/tutorial/tutorial1_rgb_network.sh?ref_type=heads#L163>`_ shows an example of this.
This requires a `small hack <https://gitlab.kitware.com/computer-vision/geowatch/-/blame/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/tasks/fusion/fit_lightning.py#L430>`_
to make it work.

RasterVision uses pydantic for configuration, whereas we use
(what is a less popular but more flexible tool)
`scriptconfig <https://gitlab.kitware.com/utils/scriptconfig>`_.
This also requires some
`monkeypatches <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/utils/lightning_ext/lightning_cli_ext.py>`_
on top of jsonargparse to make it work, but my hope is that I can upstream some
of those changes so pydantic and scriptconfig based configs can both be used.


For PREDICT it seems both frameworks have similar strategies of incrementally
stitching together heatmap predictions from batches. For vector outputs such as
bounding boxes, the main GeoWATCH fusion tool doesn't work with it yet, but it
is in development and it will work similarly to our
`implementation of a DINO box predictor <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/tasks/dino_detector/predict.py>`_,
where detections are accumulated and non-max suppressed. Note that our
implementation of non-max suppression and other efficient annotation data
structures are powered by a standalone library
`kwimage <https://gitlab.kitware.com/computer-vision/kwimage>`_.
Something we've strived for in building these tools is to modularize them into
separate Python modules with fewer dependencies, so it is easier to re-use or
re-purpose them in other libraries.


For EVAL, we have object detection and pixelwise segmentation metrics, as well
as official metrics code which was provided to us by IARPA. Currently object
detection metrics live in kwcoco, and the plan is to port the pixelwise
segmentation metrics there as well. A good deal of work has gone into making
them efficient, so it will be interesting to compare implementations.

For BUNDLE, it looks like both frameworks again have similar solutions.
I'm glad others have realized how important this is. We use torch.package to bundle the code and the weights.
One tweak we needed to make is to include a
`package header <https://gitlab.kitware.com/computer-vision/geowatch/-/blob/11da8ebfa94c2723d9649429331844e60d1bc7d6/geowatch/tasks/fusion/methods/watch_module_mixins.py#L790>`_ so the predict script knows the name of the module that is packaged.
