FAQ: STAC
=========

Question
--------

Does geowatch integrate with STAC?


Answer
------

Yes, we use STAC.

In fact STAC is the basis for all of our geospatial-first data interchange.
Kitware is a performer in the IARPA SMART program, which provides all data via
STAC endpoints. We started using it in early 2021, and we've seen it improve
over the past 3 years. Kitware's entry for this challenge is powered by our
geowatch software.

We have a process to convert STAC to KWCOCO, which puts the geospatial data
into a pixel-first context. The end-to-end pipeline is outlined here:
geowatch/cli/queue_cli/prepare_ta2_dataset.py where the input is a geojson file
indicating the spacetime area of interest. The output is a corresponding kwcoco
dataset.

The above just defines the pipeline. The information flow is as follows:

* Run geowatch.cli.stac_search to identify matching STAC entries.

* Run geowatch.cli.baseline_framework_ingress  which converts query results to a STAC catalog.

* Run geowatch.cli.stac_to_kwcoco which does the conversion to kwcoco (note: no data is pulled here, we just make a kwcoco file that points at the remote URLs for each relevant asset)

* Run geowatch.cli.coco_align which uses GDAL to download crops / overviews of the underlying assets. This is what puts the data on disk and lets the ML engineers play with it. Everything in the geowatch system is KWCOCO from this point on, but I do see a future where predicted heatmaps added as new raster bands to the KWCOCO file are then indexed via STAC and pushed back up to some storage location.

Once inside of a kwcoco file we have a fairly sophisticated dataloader that
allows users to request data in arbitrary resolutions. This logic lives in
geowatch.tasks.fusion.datamodules.kwcoco_dataset. It's powered by the
delayed-image library.
