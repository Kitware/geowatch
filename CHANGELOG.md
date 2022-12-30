# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.3.8 - Target 2022-02-xx

### Added
* Initial scripts for new teamfeatures
* New MLOPs-V3 DAG definitions
* New QA band handling to support different QA encodings
* New endpoints for ACC-2 data.
* Merge in Nov21 Pseudolive branch
* Add `quality_threshold` argument to the kwcoco video dataset, deprecate `use_cloudmask`.
* Add `observable_threshold` which is similar to but not exactly the same as quality threshold
* Add uniform time sampling and better mutli-sampler support
* KWcocoVideoDataLoader now has initial support for a bounding box task.

### Fixed
* Issue in visualize where frames might be ordered incorrectly.
* Issue when input scale was native and output scale was a fixed GSD
* Patch torchmetrics for old models
* Fix corner-case crash in SC tracker.
* Quality mask filtering was broken and is now using correct data.
* The spectra script now correctly dequantizes kwcoco images.
* Issue in KWcocoVideoDataLoader where change labels were not computed correctly if the class task was not requested.

### Changed
* Consolidate monkey patches into a single subpackage.
* Refactor `util_globals`
* New arguments to fusion.predict to filter inputs / outputs.
* Cleaned old code that was ported to kwimage
* Faster samevalue region histogram approximation in data loader


## Version 0.3.7 - Target 2022-11-21

### Added
* min / max world area threshold in tracker

### Fixed
* The `coco_align_geotiffs` will now skip images if it contains no requested bands.
* `coco_align_geotiffs` now ensures image frames are in the correct temporal order.
* Fixed issue in fusion.fit where weight transfer did not work.

### Changed
* Update requirements to support Python 3.11
* Improved the reported information in `smartwatch stats` (i.e. `watch.cli.watch_coco_stats`)
* The invariants and fusion module now reuse the same stitching manager code.
* The invariants can now use dynamic fixed GSDs.
* Improved determinism in the time sampler.
* Can now control the kwcoco datamodule grid cache via the CLI.
* Minor cleanup to the tracker CLI.
* Tracker speed optimizations 
* Tracker now has better nodata handling.


## Version 0.3.6 - Finalized 2022-11-01

### Fixed
* Bug where `window_space_overlap` was always zero at predict time.
* Bug in tracker with multipolygons
* Tracker no longer crashes on nans, and replaces them with zeros.
* Bug in SC tracking when multiple videos were present. 
* Fixed several issues in tracker where it output non-json-serializable data.
* Bug in dataloader for heterogeneous GSD annotations 
* Issue in giffify where images were stretched in videos.

### Added
* Experimental new mlops pipeline that runs bas-to-sc end-to-end or independently.
* Ability to stack different channels in `coco_visualize_videos`
* Common function for handling the loading of multiple geojson files `util_gis.coerce_geojson_datas`
* New SC prediction quality visualization
* Micro and Macro SC averages are now reported
* New heterogeneous model can now be trained with lightning CLI.


### Changed
* Modified color defaults `coco_visualize_videos`
* Moved `kwcoco_to_geojson` to scriptconfig and reworked CLI arguments
* Improved ProcessContext metadata tracking
* Increased default timeouts in `coco_align_geotiffs`
* `prepare_ta2_dataset` can now work on regions without site annotations
* Bumped minimum required Python to 3.9


## Version 0.3.5 - Finalized 2022-09-29

### Added
* dataloader can now specify `output_space_scale` as native or in GSD. (requires ndsampler 0.7.1)
* Added `--stack` argument to visualize cli
* Add `util_kwimage.Box` and other candidate kwimage tools


### Changed
* Large changes in `watch.mlops`
* Reorganized dev folder to reduce clutter
* More docs in docs folder
* Can pass hard threshold to evaluate


### Removed
* Removed original fusion schedule evaluation code.
* Removed old CLI scripts. 
* Old scripts in fusion

### Fixed
* Fixed various bugs where `project_annotations`, and `coco_align_geotiffs`
  would fail when a video was empty.
* fusion predict now writes nodata correctly and georeferences predictions.
* Fixed issue where `chip_dims` was not set correctly at predict time.
* Fixed model GSD is not respected by fusion.predict
* Several params were not inferred from the package at test time. Including `time_sampling` (now fixed).
* Issue in `delayed_image` where bottom right of the image was getting cropped off in some circumstances.
* Some config settings were not resolved in predict.
* Issue in tracker where it was not able to handle nans


### Changed
* Speedup in dataloading by doing the samecolor check in a downsampled image.
* Changed main name of data loader parameter from `space_scale` to `input_space_scale`. Old alias still exists.


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
* Fixed issue where learning_rate was not respected by some optimizers (RAdam, but AdamW seemed ok)
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
