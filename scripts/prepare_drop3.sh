#!/bin/bash
__doc__="
See Also:
    ~/code/watch/watch/cli/prepare_ta2_dataset.py
"



DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)-hdd
cd $DVC_DPATH

DATASET_SUFFIX=Drop3-TA1-2022-03-10 
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part2.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220120/iMERIT_COMBINED.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part1.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/TA-1_PROCESSED_TA-2_SUPERREGIONS_WV_ONLY.unique.input \
    --collated False False False True True \
    --dvc_dpath="$DVC_DPATH" \
    --aws_profile=iarpa \
    --fields_workers=8 \
    --convert_workers=8 \
    --align_workers=26 \
    --cache=0 \
    --serial=True --run=0



# Second-to-last iMERIT drop
# s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part3.unique.input

DATASET_SUFFIX=Drop3-TA1-2022-03-10 
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part3.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part2.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220120/iMERIT_COMBINED.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part1.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/TA-1_PROCESSED_TA-2_SUPERREGIONS_WV_ONLY.unique.input \
    --collated False False False True True \
    --dvc_dpath="$DVC_DPATH" \
    --aws_profile=iarpa \
    --fields_workers=8 \
    --convert_workers=8 \
    --align_workers=26 \
    --cache=0 \
    --serial=True --run=0


DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
DATASET_SUFFIX=Drop3-TA1-2022-03-10 
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part1.unique.input \
    --collated False \
    --dvc_dpath="$DVC_DPATH" \
    --align_workers=4 \
    --aws_profile=iarpa \
    --fields_workers=0 \
    --align_workers=0 \
    --convert_workers=0 \
    --cache=0 \
    --serial=True --run=0



extra_debuging(){
    AWS_DEFAULT_PROFILE=iarpa gdalinfo /vsis3/smart-data-kitware/ta-1/ta1-ls-kit/40/R/BN/2018/02/06/LC08_L1TP_160043_20180206_20200902_02_T1_40RBN_KIT/LC08_L1TP_160043_20180206_20200902_02_T1_40RBN_KIT_B06.tif

    AWS_DEFAULT_PROFILE=iarpa gdalwarp \
        --debug off -t_srs epsg:32640 -overwrite \
            -of COG -co OVERVIEWS=AUTO -co BLOCKSIZE=256 -co COMPRESS=DEFLATE \
            -te 55.03711696746862 24.85608737642 55.19922093144159 24.98780335943784 \
            -te_srs epsg:4326 -multi --config GDAL_CACHEMAX 500 -wm 500 -co NUM_THREADS=2 \
            /vsis3/smart-data-kitware/ta-1/ta1-ls-kit/40/R/BN/2018/02/06/LC08_L1TP_160043_20180206_20200902_02_T1_40RBN_KIT/LC08_L1TP_160043_20180206_20200902_02_T1_40RBN_KIT_B06.tif \
            /tmp/tmpglvmdhdq.tif

    AWS_DEFAULT_PROFILE=iarpa python -m watch.cli.coco_add_watch_fields \
        --src "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-03-07/data.kwcoco.json" \
        --dst "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop2-TA1-2022-03-07/data.kwcoco.json" \
        --workers="min(avail,max(all/2,8))" \
        --enable_video_stats=True \
        --overwrite=warp \
        --enable_valid_region=True \
        --target_gsd=10

    python -m watch visualize \
        --src "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json" \
        "--channels=red|green|blue" \
        --draw_anns=False --workers=avail

    AWS_DEFAULT_PROFILE=iarpa python -m watch.cli.coco_align_geotiffs \
        --src "$HOME/data/dvc-repos/smart_watch_dvc/Uncropped-Drop3-TA1-2022-03-10/data_prepped.kwcoco.json" \
        --dst "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json" \
        --regions "$HOME/data/dvc-repos/smart_watch_dvc/annotations/region_models/*.geojson" \
        --workers=0 \
        --context_factor=1 \
        --geo_preprop=auto \
        --keep=img \
        --visualize=False \
        --debug_valid_regions=False \
        --rpc_align_method affine_warp

    rsync -avrpRP "$HOME/data/dvc-repos/smart_watch_dvc/./Aligned-Drop3-TA1-2022-03-10" "$HOME/data/dvc-repos/smart_watch_dvc-hdd/"
    rsync -avrpRP "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/./US_R004" "$HOME/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop3-TA1-2022-03-10"
    rsync -avrpRP "$HOME/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/US_R004/./WV" "$HOME/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop3-TA1-2022-03-10/US_R004"
    


}


hack_prep(){
    # Move over data we already cropped out

    # We already did some of the ingress, copy it over
    cp ~/data/dvc-repos/smart_watch_dvc-hdd/Uncropped-Drop2-TA1-2022-02-24/data.kwcoco.json \
       ~/data/dvc-repos/smart_watch_dvc/Uncropped-Drop3-TA1-2022-03-10/data_ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input.kwcoco.json

    ls  ~/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop2-TA1-2022-02-24/*_R0*

    cp -r ~/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop2-TA1-2022-02-24/*_R0* \
         ~/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10

    cp -r ~/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop2-TA1-2022-03-07/*_C0* \
         ~/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10

}


hack_fix_empty_imges(){

    cd /home/local/KHQ/jon.crall/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop3-TA1-2022-03-10
    ls -- */WV
    ls -- */L8
    ls -- */S2
    ls -- */*.json

    dvc add -- */WV */L8 */S2 */*.json
    #dvc add data_*nowv*.kwcoco.json

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
    echo "DVC_DPATH='$DVC_DPATH'"

    cd "$DVC_DPATH/"
    git pull  # ensure you are up to date with master on DVC
    cd "$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10"
    dvc pull -- */L8.dvc */S2.dvc
    dvc pull
    #*/*.json

    7z 

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
    python -m watch.cli.prepare_splits \
        --base_fpath="$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json" \
        --run=1 --serial=True

    ls data*.kwcoco.json

    st iMERIT drop

    # s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part3.unique.input

    DATASET_SUFFIX=Drop3-TA1-2022-03-10 
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --s3_fpath \
            s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part3.unique.input \
            s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part2.unique.input \
            s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220120/iMERIT_COMBINED.unique.input \
            s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220314/iMERIT_COMBINED_20220314_part1.unique.input \
            s3://ki
    

    python -m watch.cli.coco_combine_features \
        --src /home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json \
                  /home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/dzyne_landcover.kwcoco.json \
                  /home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/rutgers_material_seg_v3.kwcoco.json \
        --dst /home/joncrall/data/dvc-repos/smart_watch_dvc/Aligned-Drop3-TA1-2022-03-10/combo_LM.kwcoco.json

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="ssd")
    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10/
    python -m watch.cli.prepare_splits \
        --base_fpath="$DVC_DPATH/$DATASET_CODE/data.kwcoco.json" \
        --run=1 --backend=tmux 

    7z a splits_v2.zip data*.kwcoco.json

    git pull
    dvc pull splits_v2.zip.dvc -r aws
    7z x splits_v2.zip
        
    smartwatch visualize combo_LM_nowv.kwcoco.json \
        --channels="matseg_0|matseg_1|matseg_2" \
        --select_images'.sensor_coarse != "WV"' \
        --animate=True --workers=4 \
        --skip_missing=True \
        --select_videos='.name | startswith("AE_C003")'

    smartwatch visualize combo_LM_nowv.kwcoco.json \
        --channels="bare_ground|forest|water" \
        --select_images'.sensor_coarse != "WV"' \
        --animate=True --workers=4 \
        --skip_missing=True \
        --select_videos='.name | startswith("AE_C003")'

    smartwatch visualize combo_LM_nowv.kwcoco.json \
        --channels="red|green|blue" \
        --select_images'.sensor_coarse != "WV"' \
        --animate=True --workers=4 \
        --skip_missing=True \
        --select_videos='.name | startswith("AE_C003")'


}




transfer_features(){
    SSD_DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware=ssd)
    HDD_DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware=hdd)
    echo "SSD_DVC_DPATH = $SSD_DVC_DPATH"
    echo "HDD_DVC_DPATH = $HDD_DVC_DPATH"

    DATASET_CODE=Aligned-Drop3-TA1-2022-03-10
    SRC_BUNDLE_DPATH=$SSD_DVC_DPATH/$DATASET_CODE
    DST_BUNDLE_DPATH=$HDD_DVC_DPATH/$DATASET_CODE
    #du -sh "$SRC_BUNDLE_DPATH"/./_assets
    rsync -avprPR "$SRC_BUNDLE_DPATH"/./_assets "$DST_BUNDLE_DPATH"

    # Ensure everything has relative paths
    jq .images[0] combo_LM.kwcoco.json

    # See special reroot
    #kwcoco reroot combo_LM.kwcoco.json \
    #    combo_LM.rel.kwcoco.json --absolute=False --check=False

    rsync -p "$SRC_BUNDLE_DPATH/combo_LM.rel.kwcoco.json" "$DST_BUNDLE_DPATH/combo_LM.kwcoco.json"
}



prepare_l1_version_of_drop3(){

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
    echo "DVC_DPATH = $DVC_DPATH"
    S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input.l1.mini
    DATASET_SUFFIX=Drop3-L1-MINI

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)
    echo "DVC_DPATH = $DVC_DPATH"
    S3_FPATH=s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input.l1
    DATASET_SUFFIX=Drop3-L1
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix="$DATASET_SUFFIX" \
        --s3_fpath="$S3_FPATH" \
        --dvc_dpath="$DVC_DPATH" \
        --collated=False \
        --requester_pays=True \
        --ignore_duplicates=True \
        --fields_workers=8 \
        --align_workers=8 \
        --convert_workers=8 \
        --align_aux_workers=0 \
        --verbose=3 \
        --debug=False \
        --channels="blue|green|red|nir|swir16|swir22" \
        --cache=False --run=0

}


prepare_wv_crop_from_sites(){

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc)-hdd
    cd "$DVC_DPATH"

    DATASET_SUFFIX=Drop3-TA1-SiteCropsWV-2022-03-30 
    python -m watch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --s3_fpath \
            s3://kitware-smart-watch-data/processed/ta1/TA-1_PROCESSED_TA-2_SUPERREGIONS_WV_ONLY.unique.input \
        --collated True \
        --dvc_dpath="$DVC_DPATH" \
        --aws_profile=iarpa \
        --fields_workers=8 \
        --convert_workers=8 \
        --align_aux_workers=13 \
        --align_workers=26 \
        --channels="blue|green|red|nir|swir16|swir22" \
        --region_globstr="$DVC_DPATH/annotations/site_models/*.geojson" \
        --cache=0 \
        --serial=True --run=0
}


prepare_cropped_from_tracks(){

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="ssd")
    IMGONLY_FPATH="$DVC_DPATH/Cropped-Drop3-TA1-2022-03-10/imgonly_S2_L8_WV.kwcoco.json"
    echo "IMGONLY_FPATH = $IMGONLY_FPATH"
    python -m watch.cli.coco_remove_empty_images \
        --src="$IMGONLY_FPATH" \
        --dst="$IMGONLY_FPATH.tmp" \
        --workers=8 \
        --channels=red|blue \
        --interactive=True \
        --overview=0 

    mv "$IMGONLY_FPATH.tmp" "$IMGONLY_FPATH"

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="ssd")
    BASE_DPATH="$DVC_DPATH/Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json"
    python -m watch project_annotations \
        --src "$IMGONLY_FPATH" \
        --dst "$BASE_DPATH" \
        --site_models="$DVC_DPATH/annotations/site_models/*.geojson" \
        --region_models="$DVC_DPATH/annotations/region_models/*.geojson"

    python -m watch.cli.prepare_splits \
        --base_fpath="$BASE_DPATH" \
        --run=0 --backend=serial

    7z a splits.zip data*.kwcoco.json
    dvc add -- *.zip
    git commit -am "Add splits"
    git push 
    dvc push -r aws splits.zip

    export CUDA_VISIBLE_DEVICES=1
    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="ssd")
    echo "DVC_DPATH = $DVC_DPATH"
    BASE_DPATH="$DVC_DPATH/Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json"
    python -m watch.cli.prepare_teamfeats \
        --base_fpath="$BASE_DPATH" \
        --dvc_dpath="$DVC_DPATH" \
        --gres=",0" \
        --with_landcover=1 \
        --with_depth=1 \
        --with_materials=1 \
        --with_invariants=0 \
        --do_splits=1 \
        --depth_workers=0 \
        --cache=1 --backend=tmux --run=0


}

cropped_with_more_context(){

    DVC_DPATH=$(WATCH_PREIMPORT=0 python -m watch.cli.find_dvc --hardware="hdd")
    echo "$DVC_DPATH"
    INPUT_FPATH=$BASE_DPATH/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json
    NEW_KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Cropped-Drop3-TA1-Context
    #smartwatch stats "$INPUT_FPATH"

    CHANNELS="blue|green|red|nir|swir16|swir22|cloudmask|near-ir1|panchromatic"

    # This takes forever and a half (i.e. 7-12 hours)
    python -m watch.cli.coco_crop_tracks \
        --src="$INPUT_FPATH" \
        --dst="$NEW_KWCOCO_BUNDLE_DPATH/imgonly_S2_WV.kwcoco.json" \
        --exclude_sensors=L8 \
        --channels=$CHANNELS \
        --mode=process --workers=24 \
        --channels="$CHANNELS" \
        --context_factor=1.8

    python -m watch project_annotations \
        --src "$NEW_KWCOCO_BUNDLE_DPATH/imgonly_S2_WV.kwcoco.json" \
        --dst "$NEW_KWCOCO_BUNDLE_DPATH/data.kwcoco.json" \
        --site_models="$DVC_DPATH/annotations/site_models/*.geojson" \
        --region_models="$DVC_DPATH/annotations/region_models/*.geojson"

    export CUDA_VISIBLE_DEVICES=1
    python -m watch.cli.prepare_teamfeats \
        --base_fpath="$NEW_KWCOCO_BUNDLE_DPATH/.kwcoco.json" \
        --dvc_dpath="$DVC_DPATH" \
        --gres="0,1" \
        --with_landcover=0 \
        --with_depth=1 \
        --with_materials=0 \
        --with_invariants=0 \
        --do_splits=0 \
        --depth_workers=0 \
        --cache=1 --run=1 --backend=serial

}
