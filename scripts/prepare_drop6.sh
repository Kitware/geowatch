#!/bin/bash

source "$HOME"/code/watch/secrets/secrets

DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="auto")
SENSORS=TA1-S2-L8-WV-PD-ACC-2
DATASET_SUFFIX=Drop6-2022-12-01-c30-$SENSORS
REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/region_models/*.geojson"
SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/site_models/*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=30 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DATA_DVC_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --exclude_channels=pan \
    --requester_pays=False \
    --fields_workers=8 \
    --convert_workers=8 \
    --align_workers=4 \
    --align_aux_workers=0 \
    --max_queue_size=1 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --visualize=0 \
    --target_gsd=10 \
    --cache=1 \
    --verbose=100 \
    --skip_existing=1 \
    --warp_tries=1 \
    --asset_timeout="1hour" \
    --image_timeout="1hour" \
    --backend=tmux --run=1

#--hack_lazy=True

cd "$DATA_DVC_DPATH"
ln -s "$DATASET_SUFFIX" Drop6


codeblock  "

# CHECK SIZES:
sensors = ['WV', 'PD', 'S2', 'L8']
sensor_to_sizes = ub.ddict(list)
for s in sensors:
    for p in list(bundle_dpath.glob('*/' + s)):
        nbytes = ub.cmd(f'du -sL {p}')['out'].split('\t')[0]
        sensor_to_sizes[s].append(nbytes)

sensor_to_bytes = ub.udict(sensor_to_sizes).map_values(lambda x: sum(int(b) * 1024 for b in x))
import xdev as xd
sensor_to_size = sensor_to_bytes.map_values(xd.byte_str)
print('sensor_to_size = {}'.format(ub.repr2(sensor_to_size, nl=1)))

total_size = sum(sensor_to_bytes.values())
print('total = {}'.format(xd.byte_str(total_size)))

"

#add_dvc_data(){
#    # TODO: before doing this, remember to change the bundle name
#    DATA_DVC_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=hdd)
#    cd "$DATA_DVC_DPATH"
#    git add Drop6
#    cd "$DATA_DVC_DPATH/Drop6"
#    python -m watch.cli.prepare_splits data.kwcoco.json --cache=0 --run=1
#    7z a splits.zip data*.kwcoco.json imganns-*.kwcoco.json
#    dvc add -- */L8 */S2 */WV *.zip && dvc push -r horologic -R . && git commit -am "Add Drop6 ACC-2" && git push
#}



#### FIXUP

#COCO_FPATH="$DVC_DATA_DPATH/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/imganns-AE_R001.kwcoco.json"
DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
COCO_FPATH="$DVC_DATA_DPATH/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/data.kwcoco.json"
smartwatch clean_geotiffs \
    --src "$COCO_FPATH" \
    --channels="red|green|blue|nir|swir16|swir22" \
    --prefilter_channels="red" \
    --min_region_size=256 \
    --nodata_value=-9999 \
    --workers="min(4,avail)" \
    --probe_scale=0.25



#### L2 version

source "$HOME"/code/watch/secrets/secrets
### TEST
DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="auto")
SENSORS=L2-S2-L8
DATASET_SUFFIX=Drop6-c10-$SENSORS
REGION_GLOBSTR="$DATA_DVC_DPATH/annotations/drop6/region_models/*.geojson"
SITE_GLOBSTR="$DATA_DVC_DPATH/annotations/drop6/site_models/*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=5 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated False \
    --dvc_dpath="$DATA_DVC_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --include_channels="red|gren|blue|nir|quality" \
    --requester_pays=True \
    --fields_workers=8 \
    --convert_workers=8 \
    --align_workers=4 \
    --align_aux_workers=0 \
    --max_queue_size=4 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --visualize=0 \
    --target_gsd=10 \
    --cache=1 \
    --verbose=100 \
    --skip_existing=1 \
    --warp_tries=1 \
    --asset_timeout="1hour" \
    --image_timeout="1hour" \
    --backend=tmux --run=1

    #--hack_lazy=True


poc_util_grab_array(){
    # This would be nice if I could reliably use my utils... but I cant quite
    # yet.
    mkdir -p "$HOME"/.local/bashutil/
    SCRIPT_FPATH="$HOME"/.local/bashutil/erotemic_utils.sh
    if type ipfs; then
        ipfs get QmZhnyMsQotTWRzUyxpNsMJGC1SqPC2XZVkrNCtyYG37x5 -o "$SCRIPT_FPATH"
    else
        curl https://raw.githubusercontent.com/Erotemic/local/b8015365f5a70417dc665fa2ddfa2c4e8b696841/init/utils.sh > "$SCRIPT_FPATH"
    fi
    # sha256sum should be ca92c9e0cc2f40db93a8261b531a1bfd56db948f29e69c71f9c1949b845b6f71
    source "$HOME"/.local/bashutil/erotemic_utils.sh

    ls_array IMAGE_ZIP_FPATHS "*/*.zip"
    ls_array SPLIT_ZIP_FPATHS "splits.zip"
    bash_array_repr "${IMAGE_ZIP_FPATHS[@]}"
    bash_array_repr "${SPLIT_ZIP_FPATHS[@]}"
    ZIP_FPATHS=( "${IMAGE_ZIP_FPATHS[@]}" "${SPLIT_ZIP_FPATHS[@]}" )
    bash_array_repr "${ZIP_FPATHS[@]}"
}

dvc_add(){
    python -m watch.cli.prepare_splits data.kwcoco.json --cache=0 --run=1

    7z a splits2.zip data*.kwcoco.json img*.kwcoco.json -mx9

    du -shL AE_C002/L8.zip PE_C001/PD.zip US_R001/WV.zip US_R006/L8.zip AE_R001/L8.zip LT_R001/L8.zip US_R006/S2.zip PE_C001/L8.zip PE_R001/PD.zip US_C011/WV.zip KR_R001/S2.zip BR_R002/S2.zip BR_R005/PD.zip AE_R001/S2.zip KR_R001/L8.zip US_C012/S2.zip PE_C001/WV.zip AE_C003/WV.zip BR_R004/S2.zip AE_C003/S2.zip AE_C002/WV.zip | sort -h

    SENSORS=("L8" "S2" "WV" "PD")
    for sensor in "${SENSORS[@]}"; do
        echo " * sensor=$sensor"
        for dpath in */"$sensor"; do
          echo "  * dpath=$dpath"
          7z a "$dpath".zip "$dpath"
        done
    done
    #-v100m
    # 7z a deprecated_archive_2023-01-12.zip deprecated -mx9

    du -sh */*.zip

    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    cd "$DVC_DATA_DPATH"
    ln -s Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2 Drop6

    cd Drop6
    ZIP_FPATHS=(*/*.zip *.zip)
    echo "${ZIP_FPATHS[@]}"

    dvc add "${ZIP_FPATHS[@]}" -vv && \
    dvc push -r aws "${ZIP_FPATHS[@]}" -vv && \
    git commit -am "Add Drop6 ACC-2" &&  \
    git push
}
