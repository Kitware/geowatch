# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.

## Version 0.3.10 - Target 2023-01-xx

### Added
* `smartwatch align` new argument `--minimum_size`: to ensure all regions have enough pixels
* New `time_kernel` method for temporal sampling in `KWcocoVideoDataLoader`
* New visualizations in `geojson_site_stats`

### Changed
* Renamed `project_annotations` to `reproject_annotations`
* The `reproject_annotations` script no longer requires images to exist on disk.
* The mlops schedule evaluation now writes to .kwcoco.zip files, which does break existing caches
* `clean_geotiffs` now sets NODATA if it is unset and can be given a set of bands to never clean.
* `coco_add_watch_fields` now uses correct `resolution` field instead of `target_gsd`
* `KWcocoVideoDataLoader.compute_dataset_stats` now estimates instance-level class frequency data.
* `KWcocoVideoDataLoader.compute_dataset_stats` will now try to estimate stats for all sensors/channels if the main pass does not sample them
* New aliases for `KWcocoVideoDataLoader` parameters
* Added `utils.coerce_crs`.

### Fixed
* issue in reproject annots when frames had no annotations
* `KWcocoVideoDataLoader` no longer throws an error if it fails to load a QA band.
* Removed `tensorboard` and `tensorflow` dependencies
* Issue in `warp_annot_segmentations_from_geos`
* Issue in `polygon_distance_transform`.


### Misc
* Drop6 scripts in the dev folder
* DVC cache surgery and improved `simple_dvc`
* FFVC POC
* Reworked `utils` to use `lazy_loader`.
* New `util_kwplot` constructs

## Version 0.3.9 - Target 2023-01-31

### Added
* Add new CLI tool: `smartwatch coco_clean_geotiffs` to fix NODATA values directly in geotiffs
* New `util_prog.py` with experimental `ProgressManager` class to generalize ProgIter and rich.
* Add `smartwatch visualize` option `--resolution` to specify the resolution to output the visualization at (e.g. "10GSD")
* Add `smartwatch visualize` option `--role_order` for showing different annotation roles on different channel stacks.
* Experimental new logic in `smartwatch project` for keyframe propagation / interpolation.
* Added `polygon_simplify_tolerance` to tracking.
* Add `--resolution` parameter to the tracker.
* Add `--mask_samecolor_method` and default to `histogram`, will later change to False.

### Fixed
* Switched scriptconfig objects to use "data=kwargs" rather than "default=kwargs" to avoid a copy issue.
* `find_dvc_dpath` no longer is impacted by environment variables.
* Fixed interpolation artifacts in quality mask sampling.


### Changed
* `smartwatch stats` now outputs a histogram over sensor / time range
* In the spacetime grid builder `window_dims` now always corresponds to the spatial window dimensions and `time_dims` is given for time.
* KWCocoDataloader exposes `window_resolution`, `input_resolution`, `output_resolution` as aliases for the `*_space_scale` arguments and will become the main ids in the future.


### Deprecated
* `add_watch_fields` no longer populates `wgs84_corners`, `wgs84_crs_info`,
  `utm_corners`, and `utm_crs_info`, which are redundant with `geos_corners`.


## Version 0.3.8 - Target 2022-12-31

### Added
* Initial scripts for new teamfeatures
* New MLOPs-V3 DAG definitions
* New QA band handling to support different QA encodings
* New endpoints for ACC-2 data.
* Merge in Nov21 Pseudolive branch
* Add uniform time sampling and better multi-sampler support
* KWcocoVideoDataLoader now has initial support for a bounding box task.
* Port code for generating random region / site models into `metrics_demo`.
* Add new CLI tool: `smartwatch split_videos` to break a kwcoco file into one file per video.
* Add MultimodalTransformer option `multimodal_reduce=linear` for learned per-frame mode reductions.
* Add KWcocoVideoDataLoader option `normalize_perframe` for robust per-batch-item normalization for specified channels.
* Add KWcocoVideoDataLoader option `resample_invalid_frames` specifying the number of tries to resample bad frames.
* Add KWcocoVideoDataLoader option `force_bad_frames` to help visualize bad frame resampling.
* Add KWcocoVideoDataLoader option `mask_low_quality` to force cloud pixels to be nan.
* Add KWcocoVideoDataLoader option `quality_threshold` to filter frames based on quality, deprecates `use_cloudmask`.
* Add KWcocoVideoDataLoader option `observable_threshold` to filter frames based on nan content.
* New `MultiscaleMask` class to make tracking accumulating unobservable regions easier in KWcocoVideoDataLoader
* New `RobustParameterDict` class for sensor/channel specific parameters
* Add predict option `drop_unused_frames` that removes frames with no predictions from the output kwcoco

### Fixed
* Issue in visualize where frames might be ordered incorrectly.
* Issue when input scale was native and output scale was a fixed GSD
* Patch torchmetrics for old models
* Fix corner-case crash in SC tracker.
* Quality mask filtering was broken and is now using correct data.
* The `smartwatch spectra` script now correctly dequantizes kwcoco images.
* Issue in KWcocoVideoDataLoader where change labels were not computed correctly if the class task was not requested.
* Fixed `smartwatch reproject_annotations` for new regions / sites from T&E
* Fix bugs in `util_kwimage.colorize_label_image`
* Fix bugs in `util_kwimage.find_samecolor_region`

### Changed
* Consolidate monkey patches into a single subpackage.
* Refactor `util_globals`
* New arguments to fusion.predict to filter inputs / outputs.
* Cleaned old code that was ported to kwimage
* Faster samevalue region histogram approximation in data loader
* Renamed `coco_intensity_histograms.py` to `coco_spectra.py`
* Ported relevant code from `netharn` to reduce the dependency on it.
* Moved MLOps-V2 code into an `old` submodule.
* The KWcocoVideoDataLoader now defaults to transferring nan values from the
  red channel to all other channels in a frame. Governed by experimental
  `PROPOGATE_NAN_BANDS` hidden option.
* Tensorboard visualization now adds a third smoothed line (at 0.95 smoothing) by default.


### Documentation
* More docs on `smartwatch find_dvc` in the docstring and `docs/using_smartwatch_dvc.rst`
* Added `examples/feature_fusion_tutorial.sh` describing how to train/evaluate
  a fusion model with team features.



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
* Fixed various bugs where `reproject_annotations`, and `coco_align_geotiffs`
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
