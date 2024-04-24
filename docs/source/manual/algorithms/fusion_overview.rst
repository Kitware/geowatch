
GeoWATCH System
---------------

Kitware's GeoWATCH Python module is deep-learning segmentation tool for single
or multi-frame data of a region over time with multiple sensors, modalities,
and resolutions.


How it works:
* A kwcoco dataset registers images which can have different combinations of bands / resolutions
* If the images are geo-registered, GeoWATCH can crop and stitch images into an aligned sequence that matches the bounds of a geojson file.
* User specifies groups of channels
* Each combination of sensor / channel group to fit a segmentation model
* Each unique combination gets its own input "stem" that normalizes the number of channels.
* The output of each foot is divided into a grid of "tokens".
* Each token is given a positional encoding describing its spatial position, temporal position, sensor, and mode-group.
* Tokens are stacked over all modes and time steps.
* This flat input of tokens is fed to an arbitrary transformer backbone.
* A simple MLP decoder or (in the future) Segmenter-style decoder produces an output heatmap for each category at each space/time position.
* A kwcoco dataset is output that contains the new predictions as polygons or additional heatmaps.

Comments:

* The flattening of tokens is the key to heterogeneous processing. The
  coordinates are numerically encoded in the tokens themselves, this is in
  contrast to convolutional networks, which rely on the locality of the data in
  a tensor.

* The heterogeneous processing enables the network to fuse information from
  multiple sensors
