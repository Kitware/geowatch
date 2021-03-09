# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.0.1 - Unreleased


#### Added 

* Add function `watch.utils.util_norm.normalize_intensity` for making raw geotiff data viewable.

* Add function `watch.utils.util_girder.grabdata_girder` for downloading and caching data from Girder. (This is used to download and cache the GTop30 DEM dataset). 

* Add function `watch.gis.geotiff.geotiff_filepath_info` to heuristically recognize known file name conventions and parse out relevant metadata. 

* Add function `watch.gis.geotiff.geotiff_header_info` to parse out information related to number of bands, possible sensor candidates, etc...

* Add function `watch.gis.geotiff.geotiff_metadata` which uses all functions in `watch.gis.geotiff` to return all available metadata for a geotiff. 

* Add `scripts/geojson_to_kwcoco.py` - The script that converts IARPA's geojson file to kwcoco files. 

* Add `scripts/coco_chip_regions.py` - This is the script used to crop the images using a pixel grid and sample positives and negatives.

* Add `scripts/coco_align_geotiffs.py` - the script to (1) Find spatial ROI clusters (2) extract and orthorectify that spatial ROI from all overlapping images and (3) write the dataset out where each time sequence is a video in a kwcoco json file.



#### Changed

* Modified `watch.gis.spatial_reference.RPCTransform` to make better use of DEM information. The default elevation is now GTop30 instead of Open-Elevation.

* Modified `watch.gis.geotiff.geotiff_crs_info` to return more information including what convention for the CRS `axis_mapping` (which has been one of the biggest pain points).

* Updated requirements in `conda_env.yml` to reflect dependencies in this and upcoming MRs.


#### Fixed

* Minor problems in `watch.gis.geotiff.geotiff_crs_info`

* Transforms in `watch.gis.spatial_reference.RPCTransform` were previously not using elevation data correctly.


## Version 0.0.0 - Unreleased

* Undocumented initial structure