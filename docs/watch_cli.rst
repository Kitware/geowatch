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


git mv stac_egress.py ../../dev/deprecated/cli/
git mv collate_ta1_output.py ../../dev/deprecated/cli/
git mv coco_extract_geo_bounds.py ../../dev/deprecated/cli/
git mv add_angle_bands.py ../../dev/deprecated/cli/
git mv add_sites_to_region.py ../../dev/deprecated/cli/
git mv align_crs.py ../../dev/deprecated/cli/
git mv run_brdf.py ../../dev/deprecated/cli/
git mv run_mtra.py ../../dev/deprecated/cli/
git mv mtra_preprocess.py ../../dev/deprecated/cli/
git mv coco_show_auxiliary.py ../../dev/deprecated/cli/
git mv coco_modify_channels.py ../../dev/deprecated/cli/
git mv stac_to_kwcoco.py ../../dev/deprecated/cli/
git mv geotiffs_to_kwcoco.py ../../dev/deprecated/cli/
git mv merge_region_models.py ../../dev/deprecated/cli/


mkdir dev/deprecated/scripts
git mv scripts/run_ta1_collation_streaming.py dev/deprecated/scripts
git mv scripts/run_ta1_collation_for_baseline.py dev/deprecated/scripts

git mv scripts/run_fmask_streaming.py dev/deprecated/scripts
git mv scripts/run_fmask_for_baseline.py dev/deprecated/scripts
git mv scripts/run_bas_fusion_for_baseline.py dev/deprecated/scripts
