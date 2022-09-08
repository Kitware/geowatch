Important WATCH Scripts
-----------------------

The SMART WATCH module comes with a command line interface (CLI). This can be invoked
via ``python -m watch --help`` (note: if the module has been pip installed
``python -m watch`` can be replaced with ``smartwatch`` for primary CLI commands).

More information can be found in the `watch cli docs <docs/watch_cli.rst>`_.

In these examples we use the ``smartwatch`` invocation to be concise, but you
can simply replace them with ``python -m smartwatch`` if your shell does not
support the entrypoint.


The following is a list of the primary CLI commands:

* ``smartwatch find_dvc --help`` - Helper to return the path the the WATCH DVC Repo (if it is a known location)

* ``smartwatch stats --help`` - Print statistics about a kwcoco file with a focus on sensor / channel frequency and region information.

* ``smartwatch coco_intensity_histograms --help`` - Show per-band / per-sensor histograms of pixel intensities. This is useful for acessing the harmonization between sensors. 

* ``smartwatch coco_visualize_videos --help`` - Visualize a video sequence with and without annotations. This can also create an animation of arbitrary feature channels.

* ``smartwatch coco_align_geotiffs --help`` - Crop a set of unstructured kwcoco file (that registers a set of geotiffs) into a TA-2 ready kwcoco file containing cropped video sequences corresponding to each region in a specified set of regions files.

* ``smartwatch project_annotations --help`` - Project annotations from raw site/region models onto the pixel space of a kwcoco file. This also propogates these annotations in time as needed.

* ``smartwatch kwcoco_to_geojson --help`` - Transform "saliency" or "class" heatmaps into tracked geojson site models, and optionally score these with IARPA metrics.


Using ``--help`` shows the top level modal CLI:


.. code:: 

        usage: smartwatch [-h] [--version] {command}
                          ...

        The SMART WATCH CLI

        positional arguments:
            coco_add_watch_fields (add_fields)
                                Updates image transforms in a kwcoco json file to align all videos to a
            coco_align_geotiffs (align)
                                Create a dataset of aligned temporal sequences around objects of interest
            coco_extract_geo_bounds
                                Extract bounds of geojson tiffs (in a kwcoco file) into a regions file
            geotiffs_to_kwcoco  Create a kwcoco manifest of a set of on-disk geotiffs
            watch_coco_stats (stats)
                                Print watch-relevant information about a kwcoco dataset
            merge_region_models
                                Combine the specific features from multiple region files into a single one.
            project_annotations (project)
                                Projects annotations from geospace onto a kwcoco dataset and optionally
            coco_show_auxiliary
                                Visualize kwcoco auxiliary channels to spot-inspect if they are aligned
            coco_visualize_videos (visualize)
                                Visualizes annotations on kwcoco video frames on each band
            coco_intensity_histograms (intensity_histograms)
                                Updates image transforms in a kwcoco json file to align all videos to a
            find_dvc            Command line helper to find the path to the watch DVC repo
            kwcoco_to_geojson   opaque sub command
            run_metrics_framework
                                opaque sub command
            torch_model_stats (model_info)
                                Print stats about a torch model.


        optional arguments:
          -h, --help            show this help message and exit
          --version             show version number and exit (default: False)
   


Other important commands that are not exposed via the main CLI are:

* ``python -m watch.tasks.fusion.fit --help`` - Train a TA2 fusion model.
  
* ``python -m watch.tasks.fusion.predict --help`` - Predict using a pretrained TA2 fusion model on a target dataset.

* ``python -m watch.tasks.fusion.evaluate --help`` - Measure pixel-level quality metrics between a prediction and truth kwcoco file.


Note to developers: if an important script exists and is not listed here,
please submit an MR.

New Python command line scripts can be added under the ``watch/cli`` directory.
New tools can be registered with the ``watch-cli`` tool in the
``watch/cli/__main__.py`` file, or invoked explicitly via ``python -m
watch.cli.<script-name>``.

Scripts that don’t quite belong in the WATCH Python module itself
(e.g. due to a lack of general purpose use, or lack of polish) can be
added to the ``scripts`` or ``dev`` directory. Generally, the ``scripts``
directory is for data processing and ``dev`` is for scripts related to
repository maintenence. 
  


Summary of WATCH Scripts
------------------------


The following document summarizes some of the scripts in the smartwatch CLI.


Main Commands / Scripts
~~~~~~~~~~~~~~~~~~~~~~~

watch_coco_stats - Very useful. Stats about bands / videos in a kwcoco file.

coco_visualize_videos - Very useful. Renders bands and annotations to images or animated gifs. Lots of options. Should be ported to kwcoco proper eventaully.

torch_model_stats - Very useful. Human readable metadata report for a trained torch package. (i.e. what bands / sensors / datasets was it trained on). 

coco_intensity_histograms - Reasonably useful. Makes histograms to visualize and compare channel intensity across sensors / videos.

find_dvc - This is "smartwatch_dvc". This helps register / recall paths to DVC repos based on tags to help allow scripts to be written in a magic agnostic way.


Dataset Preparation / Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prepare_ta2_dataset - The cmdqueue script that does the entire STAC -> Finalized kwcoco "DropX" dataset. This is how we make new drops.

stac_search - Step 1 in "prepare_ta2_dataset". How we search stac to find images. Produces an "inputs" file.

baseline_framework_ingress - Step 2 in "prepare_ta2_dataset". Creates a catalog from results of a STAC query.

ta1_stac_to_kwcoco - Step 3 in "prepare_ta2_datset". Very useful. The main stac to kwcoco conversion. Given a stac catalog makes a kwcoco file that references the virtual gdal images. Might need a rename.

coco_add_watch_fields - Step 3 in "prepare_ta2_dataset. Helper to add special fields (e.g. geodata) to an existing kwcoco file from geotiff metata.

coco_align_geotiffs - Step 4 in "prepare_ta2_dataset". The big cropping script that creates the main videos. Could be better.

project_annotations - Step 5 in "prepare_ta2_dataset". Projects site models onto a kwcoco set and adds the them as kwcoco annotations.

prepare_splits - Runs after "prepare_ta2_dataset" to finalize train/valiation splits. Computes predefined train / validation splits on main kwcoco files.

Production / Prediction / Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Note: new watch.mlops stuff will go in this category.

* TODO: The watch.<task>.predict scripts should be exposed here.

* TODO: The watch.<task>.evaluate scripts should be exposed here.

prepare_teamfeats - The cmdqueue team feature computation script. Computes team features on an existing raw kwcoco dataset. Part of evaluation.

kwcoco_to_geojson - This is the tracking / activity classification pipeline. A rename would be good.

run_metrics_framework - Executes IARPA metrics

coco_average_features - Takes the average of specified bands. The idea is this is used to ensemble the output of multiple predictions from different models.

coco_combine_features - Takes two kwcoco files with complementary feature bands (i.e. materials and landcover team features) and combines them to a single one. Might need a rename to concatenate assets?

gifify - Helper script that should be moved elsewhere.

crop_sites_to_regions - Crops site models to remove ones outside region models. Used at the end of the production pipeline.


Secondary Scripts
~~~~~~~~~~~~~~~~~

coco_crop_tracks - Crops an existing kwcoco to per-track videos. Originally designed to move from BAS to SC, but it might not be useful anymore. Not quite sure.

animate_visualizations - Helper to make animated gifs from visualize videos. Should be folded into visualize_videos

coco_shard - The idea is to split kwcoco files into multiple smaller ones. Not really used.

coco_remove_empty_images - helper to find images with no data in a kwcoco file and remove them

coco_reformat_channels - helps quantize data to uint16 if any underlying image data is float32, this is a fixit script for old results that didnt quantize predictions. Might still be useful.

geotiffs_to_kwcoco - Make a kwcoco from unordered geotiffs collections.

merge_region_models - merges a multiple geojson file into a single one. Probably not needed, but still used in one demo.


DevOps Scripts
~~~~~~~~~~~~~~

baseline_framework_egress - Mainly used for TA-1 to upload STAC and assets to S3

baseline_framework_kwcoco_egress - TA-2 tool for downloading STACified KWCOCO manifest and data (very simple script as it just assumes there's a single STAC item to pull down that's the full KWCOCO manifest and directory of crops etc.)

baseline_framework_kwcoco_ingress - Useful for both TA-1 and TA-2 to download STAC and assets from S3 or optionally replacing S3 asset links with /vsis3/ links



The "smartwatch_dvc" command
----------------------------

We provide a utility to help manage data paths called "smartwatch_dvc".  It
comes preconfigured with common paths for core-developer machines You can see
what paths are available by using the "list" command

.. code:: bash

    smartwatch_dvc list

which outputs something like this:


.. code::

                   name hardware         tags                                                               path  exists
    0    drop4_expt_ssd      ssd  phase2_expt                            /root/data/dvc-repos/smart_expt_dvc-ssd   False
    1    drop4_data_ssd      ssd  phase2_data                            /root/data/dvc-repos/smart_data_dvc-ssd   False
    2    drop4_expt_hdd      hdd  phase2_expt                                /root/data/dvc-repos/smart_expt_dvc   False
    3    drop4_data_hdd      hdd  phase2_data                                /root/data/dvc-repos/smart_data_dvc   False


To see full help use `smartwatch_dvc --help`

.. code:: bash

    usage: FindDVCConfig 

    Command line helper to find the path to the watch DVC repo

    positional arguments:
      command               can be find, set, add, list, or remove
      name                  specify a name to query or store or remove

    options:
      -h, --help            show this help message and exit
      --command COMMAND     can be find, set, add, list, or remove (default: find)
      --name NAME           specify a name to query or store or remove (default: None)
      --hardware HARDWARE   Specify hdd, ssd, etc..., Setable and getable property (default: None)
      --priority PRIORITY   Higher is more likely. Setable and getable property (default: None)
      --tags TAGS           User note. Setable and queryable property (default: None)
      --path PATH           The path to the dvc repo. Setable and queryable property (default: None)
      --verbose VERBOSE     verbosity mode (default: 1)
      --must_exist MUST_EXIST
                            if True, filter to only directories that exist. Defaults to false except on "find", which is True. (default: auto)
      --config CONFIG       special scriptconfig option that accepts the path to a on-disk configuration file, and loads that into this 'FindDVCConfig' object. (default: None)
      --dump DUMP           If specified, dump this config to disk. (default: None)
      --dumps               If specified, dump this config stdout (default: False)


