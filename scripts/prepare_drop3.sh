#!/bin/bash


DVC_DPATH=$(python -m watch.cli.find_dvc)
DATASET_SUFFIX=Drop3-TA1-2022-03-10 
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --s3_fpath \
        s3://kitware-smart-watch-data/processed/ta1/iMERIT_20220120/iMERIT_COMBINED.unique.input \
        s3://kitware-smart-watch-data/processed/ta1/ALL_ANNOTATED_REGIONS_TA-1_PROCESSED_20220222.unique.input \
    --dvc_dpath="$DVC_DPATH" \
    --collated False True \
    --align_workers=4 \
    --serial=True --run=0


extras(){

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
