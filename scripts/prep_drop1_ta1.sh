#!/bin/bash
__doc__="
Download TA1 processed data from S3 and create a TA-2 ready training dataset
for DVC
"

# user-specified arguments
DVC_DPATH=${DVC_DPATH:-$HOME/data/dvc-repos/smart_watch_dvc}
S3_DPATH=${S3_DPATH:-s3://kitware-smart-watch-data/processed/ta1/drop1/coreg_and_brdf/}


# The paths we will use to download and store uncropped tiles
OUTPUT_DPATH=$DVC_DPATH/TA1-Processed
QUERY_DPATH=$OUTPUT_DPATH/_query/items
INGRESS_DPATH=$OUTPUT_DPATH/ingress
QUERY_FPATH=$OUTPUT_DPATH/_query/query.json
CATALOG_FPATH=$INGRESS_DPATH/catalog.json
UNSTRUCTURED_KWCOCO_FPATH=$OUTPUT_DPATH/data.kwcoco.json


# The file that defines the TA-2 regions to crop to
REGION_FPATH=$DVC_DPATH/drop1/all_regions.geojson


# The folder that will contain the TA2-ready kwcoco dataset
# TODO: is the the right name for the new TA2-dataset?
ALIGNED_KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-TA1-2021-11
ALIGNED_KWCOCO_FPATH=$ALIGNED_KWCOCO_BUNDLE_DPATH/data.kwcoco.json


download_uncropped_data(){
    __doc__="
    Download and prepare the uncropped data
    "

    # Grab the stac items we will query directly from S3 and combine it into a single query json file
    mkdir -p $INGRESS_DPATH
    mkdir -p $QUERY_DPATH
    aws s3 --profile iarpa sync --exclude '*' --include '*.json' $S3_DPATH $QUERY_DPATH
    jq . --indent 0 $QUERY_DPATH/*.json > $QUERY_FPATH


    # Use the watch script to pull down the images as a stac catalog
    python -m watch.cli.baseline_framework_ingress \
        --aws_profile iarpa \
        --jobs 4 \
        --outdir $INGRESS_DPATH $QUERY_FPATH 


    # Convert the stac catalog to kwcoco format
    python -m watch.cli.ta1_stac_to_kwcoco \
        $CATALOG_FPATH \
        --outpath=$UNSTRUCTURED_KWCOCO_FPATH 


    # Preprocess the unstructured kwcoco file to ensure geo-info is in the json
    # (This is optional, but it makes the next step faster)
    python -m watch.cli.coco_add_watch_fields \
        --src $UNSTRUCTURED_KWCOCO_FPATH \
        --dst $UNSTRUCTURED_KWCOCO_FPATH \
        --overwrite=warp
}


crop_to_regions(){
    __doc__="
    Crop the downloaded data to create TA2 training data.
    "
    # Crop the unstructured data into "videos" aligned to each region.
    python -m watch.cli.coco_align_geotiffs \
        --src $UNSTRUCTURED_KWCOCO_FPATH \
        --dst $ALIGNED_KWCOCO_FPATH \
        --regions $REGION_FPATH \
        --max_workers=10 \
        --aux_workers=3 \
        --context_factor=1 \
        --visualize=0 \
        --keep none

    # Project and propogate annotations from the site files in the kwcoco files
    python -m watch.cli.project_annotations \
        --site_models="$DVC_DPATH/drop1/site_models/*.geojson" \
        --src $ALIGNED_KWCOCO_FPATH \
        --dst $ALIGNED_KWCOCO_FPATH
}


unprotect_old_cropped_dvc_data(){
    __doc__="
    If we are updating a DVC repo we need to unprotect any file we could overwrite
    "
    dvc unprotect $ALIGNED_KWCOCO_BUNDLE_DPATH/*/*.json
    dvc unprotect $ALIGNED_KWCOCO_BUNDLE_DPATH/*.json
}


add_new_data_to_dvc(){
    __doc__="
    If we are updating a DVC repo, we need to perform:
        * dvc add
        * git add 
        * git push
        * dvc push 
    "

    # Add the new files to our local DVC cache and 
    # creat the .dvc files for git to track
    dvc add *.kwcoco.json
    dvc add */L8 */S2 */WV
    dvc add */subdata.kwcoco.json

    # Tell git to track the new DVC files
    git add *.kwcoco.json.dvc
    git add */L8.dvc */S2.dvc */WV.dvc
    git add */subdata.kwcoco.json.dvc

    # Push new DVC files to git
    # TODO: what branch should we push to?
    git push origin

    # Push new items from the local cache to the remote AWS cache
    dvc push -r aws --recursive $ALIGNED_KWCOCO_BUNDLE_DPATH

}


visualize_cropped_dataset(){
    __doc__="
    Optional: visualize the results of the cropped dataset
    "
    python -m watch.cli.coco_visualize_videos \
        --src $ALIGNED_KWCOCO_FPATH \
        --space="video" \
        --num_workers=avail \
        --channels="red|green|blue" \
        --viz_dpath=$ALIGNED_KWCOCO_BUNDLE_DPATH/_viz \
        --animate=True
}


prep_drop1_ta1_main(){
    # Execute main steps
    download_uncropped_data

    # Comment out the DVC managment for now.
    # Ensure we have the mechanincal parts correct first.

    #unprotect_old_cropped_dvc_data
    crop_to_regions
    #add_new_data_to_dvc

    #visualize_cropped_dataset
}


# Execute the main script
prep_drop1_ta1_main
