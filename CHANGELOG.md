# Changelog

This changelog follows the specifications detailed in: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html), although we have not yet reached a `1.0.0` release.


## Version 0.19.0 - Unreleased

### Added
* Add `DataFrame.argextrema`
* Add suboptimize to `Aggregator.report_best`
* Added experimental way to specify metric metadata in a mlop ProcessNode via the `_default_metrics2` method, which will likely change in the future.

### Changed
* mlops manager package list command now sorts the package name in a reasonable way. 
* refactored utilities to reduce inter-dependencies and allow for easier porting

### Fixed
* Bug in prediction error reporting
* Issue in visualize if video names were not strings
* Issue with video names that contain invalid path characters.

## Version 0.18.4 - Released 2024-12-18

### Changed

* Vendored a fork of jsonargparse in geowatch-tpl to maintain compatibility with scriptconfig
* Changed instances of `__config__` to `__cli__`
* KWCocoVideoDataset will now respect the `"sensor"` field in an image in addition to `"sensor_coarse"` (which still takes priority)
* Added aliases of `GEOWATCH_PREIMPORT`, `GEOWATCH_HACK_IMPORT_ORDER`, `GEOWATCH_GRID_WORKERS`, `GEOWATCH_LOOSE_CLI` for the old `GEO`-less versions.
* Respect the annotation color property in coco visualize videos

### Modified
* More documentation with an mlops minimal example.

### Added
* Added detectron2 as a tpl submodule
* mlops manager can now accept `expt_pattern`
* initial experimental ability to draw track trails in coco visualize videos

### Fixed
* Support latest DETR in transformers library 4.46.2


## Version 0.18.3 - Released 2024-09-10

### Added
* Add top-level matrices argument to param_grid.
* Experimental progiter-based progress backend for lightning to work around slurm issues.
* Added initial limited accelerator option to fusion.predict to move towards faster prediction

### Fixed
* Fix issue in MultimodalModel when input sensor is None.
* Able to unpin jsonargparse and lightning with new versions

### Changed
* More information is now saved when packaging checkpoints.
* Prevent the loading of some external torch modules unless they are used via module properties.
* `geowatch.utils.util_gdal` now forwards the user to `kwgis.utils.util_gdal`.

## Version 0.18.2 - Released 2024-08-22

### Added
* Ability to change cold feature frequency
* Ability to output hidden features from the multimodal transformer in fusion.predict.
* Add the concept of a `"primary_out_key"` to mlops ProcessNode
* A ProcessNode can now specify a `load_result` function to utilize mlops 
* Add `override_meanstd` to fusion.predict

### Changed
* Moved balanced samplers to their own module.
* MLops now puts slurm output in the node directory
* Refactored / cleaned up code in fusion.predict
* Big speedup in `geowatch spectra`
* `geowatch spectra` now shows stats estimates as it is computing
* `geowatch spectra` can now be given a directory to intermediate dump cache file.

### Removed
* Removed several vendored utilities in favor of ones in other packages (`process_context`)

### Fixed
* Added missing top level functions to ArrayAPI
* normalize_peritem can now be specified as True on the command line without casting to a string.
* normalize_peritem no longer breaks computing cached dataset stats
* Added workaround to load 
* Handle better printf escapes in mlops for job config and invoke 
* Issue where images only datasets using dummy videos would keep prediction heatmaps in memory indefinitely
* Bug in visualize where max dim was not respected


## Version 0.18.1 - Released 2024-06-25

### Added
* Trainer now remembers the initial weights if requested.
* Option to `KWCocoVideoDatasetConfig` `(balance_options=sequential_without_replacement)` to iterate sequentially through the dataset without replacement.

### Fixed
* Fix box head visualizations
* Fix dataset issue when `upweight_centers`=False
* Fix issue in computing class stats when due to predictable classes.

### Changed
* Better training restart invocation docs
* The kwcoco dataset no longer uses video names as proxys for domains.

### Removed
* The old "watch" module is no longer packaged in the pypi install. The
  "geowatch" module now must be used. The mirror module will be removed from
  the repo in the next minor version increment.

## Version 0.18.0 - Released 2024-06-18

### Added
* New CLI tool: `coco_add_ignore_buffer`, for adding an uncertainty region around each annotation.
* Dataloader can now produce frame-level classification labels.

### Changed
* `geowatch.cli.reproject_annotations` now attempts to fix simple invalid geometries.
* `geowatch.cli.geojson_site_stats` now reports information about invalid geometries.
* `geowatch.cli.coco_add_watch_fields` is now more robust to forbidden permission errors.
* `geowatch.cli.smartflow.run_bas_datagen` now outputs more debugging information.
* Dataloader now returns a `class HeterogeneousBatchItem(dict)` to facilitate evolution of the dataloader.
* Reorganized `SpacetimeGridBuilder` to move it in an object oriented direction and attempt to clean it up.

### Fixed
* DataLoader with `dist_weights=True` no longer has distance transforms weights
  of annotations clobber each other when they overlap.
* Re-enable the "negative-classes" concept in the `spacetime_grid_builder`.


## Version 0.17.1 - Released 2024-05-31

### Added
* Initial `BatchItem` class to help better organize connections between dataloaders and networks.
* Confusion analysis is now run in per-stage metrics.
* New task: `poly_from_point` that contains various methods to transform
  point-based annotations into polygon-based ones using heuristics.
* Added `dynamic_fixed_resolution` as an experimental kwcocodataset and tracker option

### Changed
* Baseline tensorboard callback now also does the stacking automatically.

### Fixed
* `geowatch.cli.draw_region` now correctly draws timeline information and does a better job at distinguishing them. 
* Fixed issue where DinoSV would output the same decision for every site.
* Fixed issue in GSD calculation in the grid builder when unitless scale factors are given.

### Removed
* Removed dependency on netharn. All relevant code has been vendored into helper modules.
* Removed hardcoded `gewatch_dvc` paths


## Version 0.16.2 - Released 2024-04-15

### Added
* Function `geowatch.tasks.fusion.utils.load_model_header` to read header info from a torch package
* Aggregator can now output a varied parameter report.
* Added `loads`, `load`, and `dump` method to RegionModel and SiteModel
* Added `coerce_multiple_paths` classmethod to RegionModel and SiteModel
* Experimental code to let the user specify custom pipelines in `geowatch.mlops.schedule_evaluation`.
* QA encoding can now be specified for different data sources in coco-align
* coco-align now writes text files to indicate which images errors occurred on.

### Changed
* Can now configure COLD teamfeat via the commandline.
* Can now configure saliency weights as a YAML mapping
* Updated coverage minimum version, which decreases CI time from >1 hour to 10-30 minutes.
* Allow "kit" as annotator in geomodel schemas
* `coco_align` no longer errors and simply writes an empty kwcoco file if the input regions are empty by default. Old behavior can be enabled with 
* Updated wrapper scoring code to support latest IARPA metrics.
* Moved `prepare-ta2-dataset`, `prepare-splits`, and `prepare-teamfeats` to `cli/queue_cli`
* The tracker now warns the user if it might overload memory

### Fixed
* Cluster sites no longer includes system-rejected sites
* Fix issue in cluster-sites where incorrect start/end dates could be written

### Removed
* Remove visualize option from coco-align.

## Version 0.15.1 - Released 2024-03-04

### Added
* Add experimental CLI for looking up recent checkpoints.
* Add experimental CLI for looking up recommending train size adjustments.
* Add experimental CLI for looking up disk hardware information.
* Add more helper scripts for tensorboard callbacks
* New train-time sample balancing backend.

### Changed
* Remove developer dependency on `fire`.
* Disable shrinking timeouts in `util_gdal`
* Added keys to specify input keys to smartflow CLI scripts 

### Fixed

* Added workaround for issue in STAC to kwcoco, where channel names would be non-unique.
* Issue in `geowatch visualize` where requested channels were not respected.
* Issue in smartflow when SC predicts no site models.
* Fusion.predict now correctly writes out the training config, mlops handles the fixed format. 


## Version 0.14.4 - Released 2023-02-06

### Added

* `demo_kwcoco_multisensor` can now generate heatmaps with any given chanelspec

* Tracker can now re-estimate polygons with the new "crall-leotta" algorithm.


### Changed

* Sphinx documentation now includes all manual and auto-generated docs.

* Tracker no longer splits / merges the kwcoco file and instead passes a `video_id` reference.

* Removed a sizable amount of unused code.


### Fixed

* Docker scripts now work with the new repo name.



## Version 0.13.3 - Released 2023-12-xx


### Added

* Experimental Lightning Telemetry callback


### Changed

* The code has moved from the "watch" module to the "geowatch" module. This has
  the potential to be a breaking change, but transition should be reasonably
  smooth. A dummy "watch" model is still exposed, but will be removed in the
  future.

* Can now set dpath in demo data

* Moved CI dashboard to public repo

* requirements files are now bundled with the module itself.

* `geowatch finish_install` now correctly handles the `--strict` option and uses 

### Fixed

* Fixed issue in dataloader where centered positives requested the top-right corners, and not the actual centers.

* Fixed finish-install issue not working due to eager imports

* Fixed TPL packaging in the pypi wheel 

* Issue in dataloader where truth masks were drawn incorrectly for partially out of bounds samples. 

* Add workaround for failed balance areas in dataloader.


## Version 0.12.1 - Released 2023-11-21


### Added

* Initial support for ndsampler backends (version >0.7.6)

* Support for v3 polygon extraction / merging to allow temporally disjoint polygons at the same location !540

* Support WV and S2 landcover at AC time in smartflow

* New polygon extraction algorithm based on volumetric connected components 

* Ability to temporally pad BAS site start / end dates.

### Changed

* Smartflow egress writes to its outbucket are now transactional.

* Smartflow egress now cleans its output location before writing to it.

* SMARTTrainer now writes helpful scripts to visualize batches as an independent process (works around ddp issue) 

* Reorganized mlops analysis plots for more user control and stability

* Batch visualizations now have the option of showing summary text information

* KWCocoVideoDataset now handles the case where datasets contain only images and no videos

### Fixed

* Fixed handling of nans in `result_analysis.varied_values`

* confusion analysis now attempts to work around small errors and produce some sort of visualization for each case.


## Version 0.11.1 - Released 2023-11-xx

### Added
* mlops can now make symlinks to results nodes by region and param-hashids 

* Ability to draw batches at predict time for debugging.

* Add `channel_dropout` to KWCocoVideoDataset

* Added environment variables (e.g. `PROCESS_CONTEXT_DISABLE_ALL_TELEMETRY`) and parameters in ProcessContext to disable telemetry

### Changed
* Batch outputs weights are new members of a frame item and are factored into
  stitching weights in fusion.predict. These are also combined with old weights
  by default.

* Changed the semantics of temporal dropout to be more natural. There is now a
  `temporal_dropout_rate` for the probability a batch item will have temporal
  dropout applied, and `temporal_dropout` itself now is the probability a frame
  is dropped within a given batch.

* Dino and Depth Site Validation now write reasons for its decisions to site
  models and summaries.

* MLops updates to handle results ingested from smartflow

* `KWCocoVideoDataset` now produces one-hot-embedding truth masks for the class task.


### Fixed
* SAM features can now be run on toydata

* Fixed minor issues in utilities

* Improved smartflow stability in datagen nodes.


## Version 0.10.0 - Released 2023-10-01

### Added 

* Tracking now assigns a single score to each site model.
* Added ability to smooth scores in AC tracking
* Added ability to filter sites by score in AC tracking
* Added a fork of torchview and loss-of-plasticity to geowatch-tpl
* Confusion visualization now shows BAS and AC heatmaps if available
* Experimental Generate & Test algorithm for continual backprop.
* Shrink & Perterb Algorithm for regularization.
* Add `fix_backwards_dates` to site / region geomodels `fixup` method.


### Changed
* Add remove-bad-images to smartflow datagen pipeline.
* STAC metadata is now preserved by coco-align.
* Selection of images when picking 1 image per time window in coco-align is now influenced by STAC metadata for `eo:cloud_cover` and `quality_info:contaminated_percentage`
* prepare ta2 dataset now outputs per-region kwcoco files inside their region directories and adds the -rawbands suffix.
* Can now change the name of the predicted salient channel.
* Merged changes for incremental mode into the main branch
* Bumped versions of ndsampler, kwimage, and delayed-image to fix an issue in delayed image sampling.

### Fixed
* coco-align now handles overviews correctly for quality bands.
* fixed issue where dataloader did not relabel post construction to background if all frames post construction.
* cluster-sites now attempts to fix broken site models before it uses them.
* Fix `predicted_phase_transition_date` to be a date instead of a datetime.
* Added workaround for issue when a classifier can predict more than the 4 main categories
* Fix issue where SV output was not passed to AC input in smartflow

### Removed
* Moved unused cli scripts to "dev"


## Version 0.9.3 - Released 2023-08-31

### Fixed
* The `valid_region` property computed in `coco_populate_geo_heuristics` now properly uses image space (before it was in asset space, which usually aligned, but was not guaranteed)
* Fix issue where positive-pending sites were not reprojected correctly when they were missing a start date.

### Changed
* Prepare-ta2-dataset now uses the new v3 mlops pipeline instead of the old v1 pipeline.
* Moved third-party-libraries into a new `geowatch_tpl` module.
* `geowatch geomodel_stats` has an easier to use CLI.
* Upgrade to utilize 2.6.0 metrics
* Training channelwise model now records all loss components separately
* Generalized `cold.transfer_features` script to help project BAS predictions on high res kwcoco files.
* The tracker now uses AC labels to set start / end dates.
* Change the name of `dag_cli` to `smartflow`

### Added
* Confusion analysis script is given new functionality to visualize and categorize cases on a per-site basis.
* Add `DDP_WORKAROUND` environ flag to disable batch plotting / work around issues with distributed training.
* Added site clustering to smartflow

### Removed:
* Removed `separate_region_queues` `separate_align_jobs` options from `prepare-ta2-dataset`, they are now always True.
* Removed old `watch.tasks.fusion.organize` module
* Removed old `watch.tasks.fusion.aggregate_results` module
* Removed old `watch.mlops.old` module, which included the old pipeline
* Removed other old code.


## Version 0.8.2 - Released 2023-07-31

### Added

* Add `watch.utils.util_ffspec` as an alternative to `AWS_S3_Command`, and updated most old usages.
* Add `attention_kwargs` to models for "fixed off-by-one attention"

### Fixed
* Fixed issue with kwcoco directory roots in prepare splits.
* TemporalSampler now is allowed to return fewer timesteps than requested if
  none are available.
* Fixed issue with coco visualize colors in construction activity
  characterization, but a proper general fix is needed for choosing darker colors
  when color information for a category channel is not given.
* Major Bugfix: Issue in tracker where annotations were not read in the requested resolution causing scores of zero.
* Fixed issue where DiceFocalLoss was not passed class weights correctly.
* We no longer require a hacked pycold
* Demo region model timestamps no longer depend on the locale.
* Bumped to kwutil 0.2.3, which fixes a problem with numeric timestamps being parsed in the local time zone instead of UTC.
* Added workaround to time kernel parser for an ambiguous case. Still not totally fixed, but current models now work by default.


### Changed
* Improved `geowatch site_stats` to accept region and/or site models and provide nice statistics.
* Pytorch Package Headers now include extra information like timestamp, and git hashes.
* Refactored smartflow ingress / egress 
* Metrics now remember the git hash of the metrics code in addition to its version.

### Removed
* Removed old fit script. The lightning CLI script is considered stable.


## Version 0.7.5 - Released 2023-06-30

### Added
* Add transfer-feature script to COLD task to port features to time-averaged data
* Add MAE features to prepare teamfeats
* Add material features to prepare teamfeats
* Added `params_of_interest` option to `aggregate`'s `plot_params` config.
* Added `COLD` step in smartflow dags.
* New `simple_dvc` CLI with quality of life improvements.

### Fixed
* Added timeout to gdal subprocess commands to prevent hanging 
* Fixed `geowatch model_stats` not respecting the LightningCLI config
* If the kwcoco file is in a read-only directory, cached hashids no longer raise an error.
* Fixed edge connections in mlops site characterization pipelines 
* Bug in dataloader when `use_centered_annots=False` and `use_grid_negatives=cleared`

### Changed
* MLOPs Nodes can now specify input paths as dictionaries if default input configs are known.
* `coco_align` no longer handles annotations
* MLOps can now handle input nodes with multiple connections (variable length inputs)
* DINO Filter now marks sites as `system_rejected` instead of removing them.
* Depth Filter now marks sites as `system_rejected` instead of removing them.
* Large rework of the scripts in `watch.cli.dag_cli`.
* Removed internal utils in favor of the new `kwutil` module.
* Hacked tensorflow to always use CPU


## Version 0.6.8 - Released 2023-05-22

### Added

* SAM - segment anything features
* DZYNE site validation: `depth_pcd` task
* fusion.predict can now output pngs
* Added `request_rlimit_nofile` to KWCocoVideoDataModule for easier ulimit configuration
* The `fusion.predict` script can now output predictions in cog or png format
* Added transient labels to heuristics
* Add `rescale_nans` param to MultimodalTransformer.
* Add `cooldown` argument to coco align script to specify time between tries.
* Integrated `depth_pcd` into MLOps under the `sv_depth_filter` node.
* Added pandas query language to mlops aggregate
* Added `dvc` as an alias of `find_dvc` in geowatch CLI.

### Changed

* The crop-sites-to-regions script no longer clips polygons if they would become multipolygons.
* The crop-sites-to-regions script can now filter regions by min area.
* Fixed issue in crop-sites-to-regions where it would output an unexpected "id" field.
* STAC roles are now preserved in kwcoco conversion.
* Renamed `ta1_stac_to_kwcoco` to `stac_to_kwcoco`.
* Tweaked dependencies for windows
* The kwcoco video dataset now respects the weight attribute in annotations.
* Reorganized docs
* `coco_align` now uses process context
* Change weights now use geometric mean instead of direct product
* Documentation improvements
* Now using scriptconfig in fusion.predict
* Reworked how submatrices behave in mlops, added submatrices1, submatrices2. Concept might need refinement.
* Update site / region schemas
* More debugging output in `stac_search`


### Fixed
* bug in coco-align where minimum resolution trigger would not be hit if there
  were multiple assets for an item and they were at different resolutions (e.g.
  for quality bands)
* safer no longer uses `temp_file` on windows
* Erroneous assertion errors in reproject and kwcoco-to-geotiffs
* Issue where `DINO_SV` would write region models to the out-site-manifest, now correctly points at site models.
* Bug in coco-align where nodata values were not properly set on data that moved through gdal-merge.
* Fixed minor issue in `stac_search` where regions with no results might get features from previous results.
* Fixed issue where incorrect fields were expected / used in the site / region models 


## Version 0.5.6 - Released 2023-04-30

### Added
* `coco_time_combine` can now ignore seasons / handle median images with less memory
* `use_grid_negatives` as dataset option, which can be set to "cleared" to only use negatives from cleared regions.
* Add `modality_dropout` to kwcoco dataloader
* Add DINO site validation prediction to mlops pipeline
* Add split attention backbones to heterogeneous model.

### Changed

* Added `boundary_region` arg to the tracker to crop all outputs to the region.
* Unpinned lightning for 2.x
* kwcoco draw item `norm_over_time` now defaults to `auto`, which is true if
  `normalize_peritem` or `normalize_perdomain` is on.
* Started transition to a new package name: `geowatch`.
* CI updated to `pyenv:3.11`
* Cleanup `util_yaml`.
* Lots of CLI improvements.
* Faster loading of multiple kwcoco files in geowatch stats, ensemble, combine.
* Improvements to mlops.aggregate
* Updated prepare-ta2-dataset with new cmd-queue and ACC-3 data
* Users can now overwrite mlops output locations.
* DATALOADER CHANGE: We were previously nan-ing all bands if red was nan. We no longer do this because it clobbers pan.
* Phase change prediction now only uses the last observation.

### Fixed

* Fixed gdal warnings
* Visualization issue where annots would not always be rendered

### Removed

* The old fit script will now raise an error. Use the `fit_lightning` script instead.

## Version 0.4.8 - Released 2023-03-31

### Changed
* Lightning Packager callback now saves checkpoints on errors in addition to packages.
* Updated tutorials to LightningCLI
* Renamed `coco_align_geotiff` to `coco_align`
* CocoStitcher now writes quantization metadata to the geotiff itself
* The kwcoco dataset now does extra work to randomize itself in training to bypass seed-everything issues. This can be disabled via the `reseed_fit_random_generators` parameter.
* Prettier progress bars
* `watch.cli.coco_add_watch_fields` how modifies image space to agree with the highest resolution asset.
* Reworked the `geowatch` CLI. Response time is much faster. Added autocomplete
* Refactored `save_package` to use common code between all models
* Improve yaml utils, allow !include tag.
* `watch.mlops.aggregate` CLI tool now has basic functionality.
* `watch.cli.coco_spectra` can now pool results per video / month.
* Improve tracker speed / efficiency
* The tracker `track_kwargs` can now take yaml as well as json.

### Added
* Add SITS-former 
* WU MAE Backbone in Heterogeneous Model
* The `watch.cli.cluster_sites` script now works.
* The `watch.cli.coco_time_combine` script
* Space / transform consistency checks in `kwcoco_extensions`
* Initial prenormalizer support in data loader.
* Site verification logic in the tracker
* Inner window averaging in the tracker
* Poly merge method to BAS tracking

### Fixed
* Bug in `coco_add_watch_fields` where video properties were incorrectly updated if auxiliary and image space was not aligned.
* Workaround issue loading old models due to newer torch.package checks
* Bug in coco dataset grid builder, `use_centered_positives=True` now properly centers annotations.
* Models now define their layers in a consistent order to workaround lightning#17025
* Fixed issue in fusion models where we could not resume from checkpoints with lightning.
* Fixed basic DDP issues; there still seems to be some lingering issues if certain callbacks are enabled.
* Issues in watch.tasks.cold preventing it from running on larger regions
* Fix inconsistency in UTM space estimation.
* Fixes in `cold` task

### Removed
* Old `demo_smart_raw/aligned kwcoco` in favor of `demo_kwcoco_multisensor`.

## Version 0.3.10 - Released 2023-02-28

### Added
* `geowatch align` new argument `--minimum_size`: to ensure all regions have enough pixels
* New `time_kernel` method for temporal sampling in `KWcocoVideoDataLoader`
* New visualizations in `geojson_site_stats`
* Add `submatrices` to MLOPs
* LightningCLI is now fully supported with partial weight loading and tensorboard plots
* Scriptconfig aliases now hook up with LightningCLI
* Added `fixed_resultion` arg to dataloader
* Tracker now ensures at most one site observation per day
* Support for image space stitching in `CocoStitchingManager`
* Pretrained VIT support in heterogeneous model.
* Decoderless support in heterogeneous model.
* Landcover features predictor with sliding window and hidden feature support.
* Add COLD to `prepare_teamfeatures`

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
* Tracker now prefers arguments `min_area_square_meters` and `max_area_square_meters` for size thresholds
* Tracker size thresholds now default to `None` (i.e. off)
* Change the default of `mask_samecolor_method` to None
* Change the default of `time_span` to None
* Better legends and labeling in `aggregate.build_all_param_plots`
* Vendored lightning parse-gpu-devices
* Refactored `CocoStitchingManager` into its own module.

### Fixed
* issue in reproject annots when frames had no annotations
* `KWcocoVideoDataLoader` no longer throws an error if it fails to load a QA band.
* Removed `tensorboard` and `tensorflow` dependencies
* Issue in `warp_annot_segmentations_from_geos`
* Issue in `polygon_distance_transform`.
* Issue in tracker where it would use 30GSD even if there was a default.
* Issue where kwcoco annotations in a track may not be sorted; use `sorted_annots`

### Misc
* Drop6 scripts in the dev folder
* DVC cache surgery and improved `simple_dvc`
* FFVC POC
* Reworked `utils` to use `lazy_loader`.
* New `util_kwplot` constructs

## Version 0.3.9 - Released 2023-01-31

### Added
* Add new CLI tool: `geowatch coco_clean_geotiffs` to fix NODATA values directly in geotiffs
* New `util_prog.py` with experimental `ProgressManager` class to generalize ProgIter and rich.
* Add `geowatch visualize` option `--resolution` to specify the resolution to output the visualization at (e.g. "10GSD")
* Add `geowatch visualize` option `--role_order` for showing different annotation roles on different channel stacks.
* Experimental new logic in `geowatch project` for keyframe propagation / interpolation.
* Added `polygon_simplify_tolerance` to tracking.
* Add `--resolution` parameter to the tracker.
* Add `--mask_samecolor_method` and default to `histogram`, will later change to False.

### Fixed
* Switched scriptconfig objects to use "data=kwargs" rather than "default=kwargs" to avoid a copy issue.
* `find_dvc_dpath` no longer is impacted by environment variables.
* Fixed interpolation artifacts in quality mask sampling.


### Changed
* `geowatch stats` now outputs a histogram over sensor / time range
* In the spacetime grid builder `window_dims` now always corresponds to the spatial window dimensions and `time_dims` is given for time.
* KWCocoDataloader exposes `window_resolution`, `input_resolution`, `output_resolution` as aliases for the `*_space_scale` arguments and will become the main ids in the future.


### Deprecated
* `add_watch_fields` no longer populates `wgs84_corners`, `wgs84_crs_info`,
  `utm_corners`, and `utm_crs_info`, which are redundant with `geos_corners`.


## Version 0.3.8 - Released 2022-12-31

### Added
* Initial scripts for new teamfeatures
* New MLOPs-V3 DAG definitions
* New QA band handling to support different QA encodings
* New endpoints for ACC-2 data.
* Merge in Nov21 Pseudolive branch
* Add uniform time sampling and better multi-sampler support
* KWcocoVideoDataLoader now has initial support for a bounding box task.
* Port code for generating random region / site models into `metrics_demo`.
* Add new CLI tool: `geowatch split_videos` to break a kwcoco file into one file per video.
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
* The `geowatch spectra` script now correctly dequantizes kwcoco images.
* Issue in KWcocoVideoDataLoader where change labels were not computed correctly if the class task was not requested.
* Fixed `geowatch reproject_annotations` for new regions / sites from T&E
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
* More docs on `geowatch find_dvc` in the docstring and `docs/using_geowatch_dvc.rst`
* Added `examples/feature_fusion_tutorial.sh` describing how to train/evaluate
  a fusion model with team features.



## Version 0.3.7 - Released 2022-11-21

### Added
* min / max world area threshold in tracker

### Fixed
* The `coco_align_geotiffs` will now skip images if it contains no requested bands.
* `coco_align_geotiffs` now ensures image frames are in the correct temporal order.
* Fixed issue in fusion.fit where weight transfer did not work.

### Changed
* Update requirements to support Python 3.11
* Improved the reported information in `geowatch stats` (i.e. `watch.cli.watch_coco_stats`)
* The invariants and fusion module now reuse the same stitching manager code.
* The invariants can now use dynamic fixed GSDs.
* Improved determinism in the time sampler.
* Can now control the kwcoco datamodule grid cache via the CLI.
* Minor cleanup to the tracker CLI.
* Tracker speed optimizations 
* Tracker now has better nodata handling.


## Version 0.3.6 - Released 2022-11-01

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


## Version 0.3.5 - Released 2022-09-29

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


## Version 0.3.4 - Released 2022-08-31

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
* Improved the `geowatch_dvc` registry system.
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


## Version 0.3.3 - Released

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

## Version 0.3.2 - Released

Many undocumented changes


## Version 0.0.1 - Released


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


## Version 0.0.0 - Released

* Undocumented initial structure
