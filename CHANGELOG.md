# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.3.5 - Target 2022-09-30


### Removed
* Removed original fusion schedule evaluation code.
* Removed old CLI scripts. 

### Fixed
* Fixed various bugs where `project_annotations`, and `coco_align_geotiffs`
  would fail when a video was empty.
* fusion predict now writes nodata correctly and georeferences predictions.


## Version 0.3.4 - Finalized 2022-08-31

### Added
* New "space_scale" parameter to Dataset such that a specific DATA GSD can be given on the fly
* New samecolor heuristic to remove invalid regions in data sampling.
* Can now force nodata values in prep ta2 and align script.
* Perceiver backbone
* New "window_space_scale" parameter to Dataset that allows a WINDOW GSD to be given on the fly.
* New "watch.mlops" module for phase2 dvc operations.

### Changed
* Data loader now returns the size it would like the output predicted at via `target_dims`.
* Data loader now supports fully heterogeneous sampling.
* The nir08 sensor for L8 is now aliased to nir for consistency.
* Add support for lightning 1.7
* Changed defaults of ignore_dilate to 0
* Improved the `smartwatch_dvc` registry system.
* Using scriptconfig and key/value arguments for `run_metrics_framework`
* The coco align geotiff now populates the `valid_region` for each video based
  on the query region in video space. The visualize video script will respect this.

### Removed
* Removed support for `score_args` from kwcoco-to-geojson


### Fixed
* Issue where augmentation might shave a pixel off of the sample box causing an
  unexpected shape.
* Fixed issue with negative was being marked as salient
* Issue where initializer was not working correctly


## Version 0.3.3 - Unreleased

### Changed
* Updates to handle new "Drop4" datasets
* Fixes to sensorchan integration with the model and dataset (training with new settings is now verified)
* Fixes to handling of nan data
* Fixes to lightning modules
* Fixed issue where learning_rate was not respected by some optimizers (RADam, but AdamW seemed ok)
* Fixed issue where weight_decay was not respected by AdamW.
* One-off scripts for fixing models.
* Better handling of gdal warp / translate / mosaic failures.
* Moved monkey patches to their own module.
* Other minor fixes.

## Version 0.3.2 - Unreleased

Many undocumented changes


## Version 0.0.1 - Unreleased


#### Added 

* Add function `watch.demo.landsat_demodata.grab_landsat_product` to download / cache a landsat product for demo / test purposes

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
