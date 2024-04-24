Important GeoWATCH Scripts
--------------------------

The GeoWATCH module comes with a command line interface (CLI). This can be
invoked via ``geowatch --help`` (note: alternatively you can invoke the module
directly via python using ``python -m geowatch``).


In these examples we use the ``geowatch`` invocation to be concise, but you
can simply replace them with ``python -m geowatch`` if your shell does not
support the entrypoint.


The following is a list of the primary CLI commands:

* ``geowatch find_dvc --help`` - Helper to return the path the the GeoWATCH DVC Repo (if it is a known location)

* ``geowatch stats --help`` - Print statistics about a kwcoco file with a focus on sensor / channel frequency and region information.

* ``geowatch coco_spectra --help`` - Show per-band / per-sensor histograms of pixel intensities. This is useful for acessing the harmonization between sensors.

* ``geowatch coco_visualize_videos --help`` - Visualize a video sequence with and without annotations. This can also create an animation of arbitrary feature channels.

* ``geowatch coco_align_geotiffs --help`` - Crop a set of unstructured kwcoco file (that registers a set of geotiffs) into a TA-2 ready kwcoco file containing cropped video sequences corresponding to each region in a specified set of regions files.

* ``geowatch reproject_annotations --help`` - Project annotations from raw site/region models onto the pixel space of a kwcoco file. This also propogates these annotations in time as needed.

* ``geowatch kwcoco_to_geojson --help`` - Transform "saliency" or "class" heatmaps into tracked geojson site models, and optionally score these with IARPA metrics.


Using ``--help`` shows the top level modal CLI:


.. code::

    usage: geowatch [-h] [--version]
                    {stats,site_stats,site_validate,model_stats,spectra,draw_region,visualize,dvc,add_fields,align,clean_geotiffs,average_features,time_combine,reproject,run_tracker,iarpa_eval,crop_sitemodels,remove_bad_images,fit,predict,finish_install,schedule,manager,aggregate,repackage}
                    ...

    🌐🌐🌐 The GeoWATCH CLI 🌐🌐🌐

    An open source research and production environment for image and video
    segmentation and detection with geospatial awareness.

    Developed by Kitware. Funded by the IARPA SMART challenge.

    Version: 0.17.0

    options:
      -h, --help            show this help message and exit
      --version             show version number and exit (default: False)

    commands:
      {stats,site_stats,site_validate,model_stats,spectra,draw_region,visualize,dvc,add_fields,align,clean_geotiffs,average_features,time_combine,reproject,run_tracker,iarpa_eval,crop_sitemodels,remove_bad_images,fit,predict,finish_install,schedule,manager,aggregate,repackage}
                            specify a command to run
        stats (watch_coco_stats)
                            Print geowatch-relevant information about a kwcoco dataset.
        site_stats (geojson_stats, geomodel_stats, geojson_site_stats)
                            Compute statistics about geojson sites.
        site_validate (validate_sites, validate_annotation_schemas)
                            Validate the site / region model schemas
        model_stats (model_info, torch_model_stats)
                            Print stats about a torch model.
        spectra (intensity_histograms, coco_spectra)
                            Plot the spectrum of band intensities in a kwcoco file.
        draw_region         Ignore:
        visualize (coco_visualize_videos)
                            Visualizes annotations on kwcoco video frames on each band
        dvc (dvcdir, find_dvc)
                            Find the path to a registered DVC repo.
        add_fields (coco_add_watch_fields)
                            Updates kwcoco image transforms and sets video space to a target GSD.
        align (coco_align, coco_align_geotiff, coco_align_geotiffs)
                            Create a dataset of aligned temporal sequences around objects of interest
        clean_geotiffs (coco_clean_geotiffs)
                            Clean geotiff files inplace by masking bad pixels with NODATA.
        average_features (ensemble, coco_average_features)
                            Average multiple kwcoco files - i.e. ensemble heatmap predictions.
        time_combine (coco_time_combine)
                            Averages kwcoco images over a sliding temporal window in a video.
        reproject (project, reproject_annotations)
                            Warp annotations from geospace onto kwcoco pixel space.
        run_tracker         Convert KWCOCO to IARPA GeoJSON
        iarpa_eval (run_metrics_framework)
                            Score IARPA site model GeoJSON files using IARPA's metrics-and-test-framework
        crop_sitemodels (crop_sites_to_regions)
                            Crops site models to the bounds of a region model.
        remove_bad_images (coco_remove_bad_images)
                            Remove coco images that are mostly nodata.
        fit (fusion_fit)    Does not work from geowatch CLI yet. See help.
        predict (fusion_predict)
                            Does not work from geowatch CLI yet. See help.
        finish_install      Finish the install of geowatch.
        schedule (mlops_schedule, schedule_evaluation)
                            Driver for GeoWATCH mlops evaluation scheduling
        manager (mlops_manager)
                            Manage trained models in the GeoWATCH experiment DVC repo.
        aggregate (mlops_aggregate)
                            Aggregates results from multiple DAG evaluations.
        repackage (repackager)
                            Convert a raw torch checkpoint into a torch package.



As a researcher / developer / user the most important commands for you to know are:

* ``geowatch stats <kwcoco_file>`` - Get geowatch-relevant statistics about data in a kwcoco file

* ``geowatch visualize <kwcoco_file>`` - Visualize the image / videos / annotations in a kwcoco file.

* ``geowatch spectra <kwcoco_file>`` - Look at the distribution of intensity values per band / per sensor in a kwcoco file.

* ``geowatch model_stats <fusion_model_file>`` - Get stats / info about a trained fusion model.

* ``geowatch reproject`` - Reproject CRS84 (geojson) annoations to image space and write to a kwcoco file.

* ``geowatch align`` - Crop a kwcoco dataset based on CRS84 (geojson) regions.

* ``geowatch clean_geotiff`` - Heuristic to detect large regions of black pixels and edit them to NODATA in the geotiff.

* ``geowatch geotiffs_to_kwcoco`` - Create a kwcoco file from a set of on-disk geotiffs.

* ``geowatch_dvc`` - Helper to register / retreive your DVC paths so scripts can be written agnostic to filesystem layouts. See `docs <data/using_geowatch_dvc.rst>`_ for more details.



Other important commands that are not exposed via the main CLI are:

* ``python -m geowatch.tasks.fusion.fit --help`` - Train a TA2 fusion model.

* ``python -m geowatch.tasks.fusion.predict --help`` - Predict using a pretrained TA2 fusion model on a target dataset.

* ``python -m geowatch.tasks.fusion.evaluate --help`` - Measure pixel-level quality metrics between a prediction and truth kwcoco file.


Note to developers: if an important script exists and is not listed here,
please submit an MR.

New Python command line scripts can be added under the ``geowatch/cli`` directory.
New tools can be registered with the ``geowatch`` tool in the
``geowatch/cli/__main__.py`` file, or invoked explicitly via ``python -m
geowatch.cli.<script-name>``.

Scripts that don’t quite belong in the GeoWATCH Python module itself
(e.g. due to a lack of general purpose use, or lack of polish) can be
added to the ``scripts`` or ``dev`` directory. Generally, the ``scripts``
directory is for data processing and ``dev`` is for scripts related to
repository maintenence.



Summary of GeoWATCH Scripts
---------------------------


The following document summarizes some of the scripts in the geowatch CLI.


Main Commands / Scripts
~~~~~~~~~~~~~~~~~~~~~~~

watch_coco_stats - Very useful. Stats about bands / videos in a kwcoco file.

coco_visualize_videos - Very useful. Renders bands and annotations to images or animated gifs. Lots of options. Should be ported to kwcoco proper eventaully.

torch_model_stats - Very useful. Human readable metadata report for a trained torch package. (i.e. what bands / sensors / datasets was it trained on).

coco_spectra - Reasonably useful. Makes histograms to visualize and compare channel intensity across sensors / videos.

find_dvc - This is "geowatch_dvc". This helps register / recall paths to DVC repos based on tags to help allow scripts to be written in a magic agnostic way.


Dataset Preparation / Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

prepare_ta2_dataset - The cmdqueue script that does the entire STAC -> Finalized kwcoco "DropX" dataset. This is how we make new drops.

stac_search - Step 1 in "prepare_ta2_dataset". How we search stac to find images. Produces an "inputs" file.

baseline_framework_ingress - Step 2 in "prepare_ta2_dataset". Creates a catalog from results of a STAC query.

stac_to_kwcoco - Step 3 in "prepare_ta2_datset". Very useful. The main stac to kwcoco conversion. Given a stac catalog makes a kwcoco file that references the virtual gdal images. Might need a rename.

coco_add_watch_fields - Step 3 in "prepare_ta2_dataset. Helper to add special fields (e.g. geodata) to an existing kwcoco file from geotiff metata.

coco_align_geotiffs - Step 4 in "prepare_ta2_dataset". The big cropping script that creates the main videos. Could be better.

reproject_annotations - Step 5 in "prepare_ta2_dataset". Projects site models onto a kwcoco set and adds the them as kwcoco annotations.

prepare_splits - Runs after "prepare_ta2_dataset" to finalize train/valiation splits. Computes predefined train / validation splits on main kwcoco files.

Production / Prediction / Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Note: new geowatch.mlops stuff will go in this category.

* TODO: The geowatch.<task>.predict scripts should be exposed here.

* TODO: The geowatch.<task>.evaluate scripts should be exposed here.

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

coco_bad_empty_images - helper to find images with no data in a kwcoco file and remove them

coco_reformat_channels - helps quantize data to uint16 if any underlying image data is float32, this is a fixit script for old results that didnt quantize predictions. Might still be useful.

geotiffs_to_kwcoco - Make a kwcoco from unordered geotiffs collections.

merge_region_models - merges a multiple geojson file into a single one. Probably not needed, but still used in one demo.


The "geowatch_dvc" command
--------------------------

We provide a utility to help manage data paths called "geowatch_dvc".  It
comes preconfigured with common paths for core-developer machines You can see
what paths are available by using the "list" command

.. code:: bash

    geowatch_dvc list

which outputs something like this:


.. code::

                   name hardware         tags                                                               path  exists
    0    drop4_expt_ssd      ssd  phase2_expt                            /root/data/dvc-repos/smart_expt_dvc-ssd   False
    1    drop4_data_ssd      ssd  phase2_data                            /root/data/dvc-repos/smart_data_dvc-ssd   False
    2    drop4_expt_hdd      hdd  phase2_expt                                /root/data/dvc-repos/smart_expt_dvc   False
    3    drop4_data_hdd      hdd  phase2_data                                /root/data/dvc-repos/smart_data_dvc   False


To see full help use `geowatch_dvc --help`

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


