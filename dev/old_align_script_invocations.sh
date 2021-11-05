

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
