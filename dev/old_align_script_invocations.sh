

Test:

    There was a bug in KR-WV, run the script only on that region to test if we
    have fixed it.

    kwcoco stats ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    jq .images ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json

    kwcoco subset ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json --gids=1129,1130 --dst ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json

    python -m watch.cli.coco_align_geotiffs \
            --src ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/subtmp.kwcoco.json \
            --dst ~/remote/namek/data/dvc-repos/smart_watch_dvc/drop0_aligned_WV_Fix \
            --rpc_align_method pixel_crop \
            --context_factor=3.5

           # --src ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/data.kwcoco.json \


Notes:

    # Given the output from geojson_to_kwcoco this script extracts
    # orthorectified regions.

    # https://data.kitware.com/#collection/602457272fa25629b95d1718/folder/602c3e9e2fa25629b97e5b5e

    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2 \
            --context_factor=1.5

    # Archive the data and upload to data.kitware.com
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    7z a ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2.zip \
            ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_v2

    stamp=$(date +"%Y-%m-%d")
    # resolve links (7z cant handl)
    rsync -avpL drop0_aligned_v2 drop0_aligned_v2_$stamp
    7z a drop0_aligned_v2_$stamp.zip drop0_aligned_v2_$stamp

    source $HOME/internal/secrets
    cd $HOME/data/dvc-repos/smart_watch_dvc/
    girder-client --api-url https://data.kitware.com/api/v1 upload \
            602c3e9e2fa25629b97e5b5e drop0_aligned_v2_$stamp.zip

    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi \
            --context_factor=1.5


    python -m watch.cli.coco_align_geotiffs \
            --src ~/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json \
            --dst ~/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi_big \
            --context_factor=3.5



Notes:

    # Example invocation to create the full drop1 aligned dataset
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    INPUT_COCO_FPATH=$DVC_DPATH/drop1/data.kwcoco.json
    OUTPUT_COCO_FPATH=$DVC_DPATH/drop1-S2-L8-WV-aligned/data.kwcoco.json
    REGION_FPATH=$DVC_DPATH/drop1/all_regions.geojson
    VIZ_DPATH=$DVC_DPATH/drop1-S2-L8-WV-aligned/_viz_video

    # Quick stats about input datasets
    python -m kwcoco stats $INPUT_COCO_FPATH
    python -m watch stats $INPUT_COCO_FPATH

    # Combine the region models
    python -m watch.cli.merge_region_models \
        --src $DVC_DPATH/drop1/region_models/*.geojson \
        --dst $REGION_FPATH

    python -m watch.cli.coco_add_watch_fields \
        --src $INPUT_COCO_FPATH \
        --dst $INPUT_COCO_FPATH.prepped \
        --workers 16 \
        --target_gsd=10

    # Execute alignment / crop script
    python -m watch.cli.coco_align_geotiffs \
        --src $INPUT_COCO_FPATH.prepped \
        --dst $OUTPUT_COCO_FPATH \
        --regions $REGION_FPATH \
        --rpc_align_method orthorectify \
        --max_workers=10 \
        --aux_workers=2 \
        --context_factor=1 \
        --visualize=False \
        --skip_geo_preprop True \
        --keep img

    python -m watch.cli.coco_visualize_videos \
        --src $OUTPUT_COCO_FPATH \
        --space="video"

    # Make an animated gif for specified bands (use "," to separate)
    python -m watch.cli.animate_visualizations \
            --viz_dpath $VIZ_DPATH \
            --draw_imgs=False \
            --draw_anns=True \
            --channels "red|green|blue"

    # Propagation actually touches the images, so this is necessary
    # Propagate annotations forward in time
    watch-cli propagate_labels \
        --src $OUTPUT_COCO_FPATH \
        --dst $OUTPUT_COCO_FPATH.tmp \
        --ext $DVC_DPATH/drop1/annots.kwcoco.json \
        --viz_dpath None \
        --verbose 1 \
        --validate 1 \
        --crop 1 \
        --max_workers None


    python -m watch.cli.coco_align_geotiffs \

    # Output stats
    python -m kwcoco stats $OUTPUT_COCO_FPATH
    python -m watch stats $OUTPUT_COCO_FPATH
    python -m watch.cli.coco_visualize_videos \
        --src $OUTPUT_COCO_FPATH \
        --space="video"


Ignore:
    # Input Args
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    TA1_KWCOCO_FPATH=$DVC_DPATH/TA1-Processed/data.kwcoco.json
    ALIGNED_KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop1-Aligned-TA1-2021-11
    ALIGNED_KWCOCO_FPATH=$ALIGNED_KWCOCO_BUNDLE_DPATH/data.kwcoco.json

    dvc unprotect $ALIGNED_KWCOCO_BUNDLE_DPATH/*/*.kwcoco.json

    python -m watch.cli.coco_align_geotiffs \
        --src $TA1_KWCOCO_FPATH \
        --dst $ALIGNED_KWCOCO_BUNDLE_DPATH/aligned.kwcoco.json \
        --regions $DVC_DPATH/drop1/region_models/LT_R001.geojson \
        --rpc_align_method orthorectify \
        --max_workers=10 \
        --aux_workers=2 \
        --skip_geo_preprop True \
        --max_frames 1000 \
        --target_gsd=10 --visualize="red|green|blue"

    jq ".images[23].auxiliary[0].parent_file_name" /home/joncrall/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-TA1-2021-11/aligned.kwcoco.json
    jq ".images[24].auxiliary[0].parent_file_name" /home/joncrall/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-TA1-2021-11/aligned.kwcoco.json

    jq ".images[23].id" /home/joncrall/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-TA1-2021-11/aligned.kwcoco.json
    jq ".images[24].id" /home/joncrall/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-TA1-2021-11/aligned.kwcoco.json

    jq ".images[11].id" /home/joncrall/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-TA1-2021-11/aligned.kwcoco.json

    rm -rf $ALIGNED_KWCOCO_BUNDLE_DPATH/_aligned_viz
    python -m watch.cli.coco_visualize_videos \
        --src $ALIGNED_KWCOCO_BUNDLE_DPATH/aligned.kwcoco.json \
        --viz_dpath $ALIGNED_KWCOCO_BUNDLE_DPATH/_aligned_viz \
        --channels "red|green|blue" \
        --num_workers=10 --animate=True
