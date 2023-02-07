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
    --requester_pays=False \
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
    --cache=0 \
    --verbose=100 \
    --skip_existing=0 \
    --warp_tries=1 \
    --asset_timeout="1hour" \
    --image_timeout="1hour" \
    --force_min_gsd=2.0 \
    --backend=tmux --run=1

    #--exclude_channels=pan \

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


queue_archive_dmj_assets(){
    REGION_ID=$1 
    QUEUE_NAME=$2

    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
    DMJ_DPATH=/home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan
    DST_DPATH=$DATA_DVC_DPATH/Drop6

    DMJ_COCO_FPATH=$DMJ_DPATH/$REGION_ID/kwcoco-dataset/cropped_kwcoco.json
    IMGONLY_FPATH=$DST_DPATH/imgonly-$REGION_ID.kwcoco.json
    IMGANNS_FPATH=$DST_DPATH/imganns-$REGION_ID.kwcoco.zip
    DJM_ASSET_DPATH=$DMJ_DPATH/$REGION_ID/kwcoco-dataset/$REGION_ID
    DST_ASSET_DPATH=$DST_DPATH/$REGION_ID

    # Zip up imagery and write to our directory
    #SENSORS=("L8" "S2")
    SENSORS=("L8" "S2" "WV" "WV1" "PD")
    for sensor in "${SENSORS[@]}"; do
        echo "### sensor=$sensor"
        dpath="$DJM_ASSET_DPATH"/"$sensor"
        if [ -d "$dpath" ]; then
            if [ ! -e "${DST_ASSET_DPATH}/${sensor}.zip" ]; then
                mkdir -p "$DST_ASSET_DPATH"
                cmd_queue --action=submit --name="$QUEUE_NAME" --command="7z a '${DST_ASSET_DPATH}/${sensor}.zip' '$dpath'"
            fi
        fi
    done
}


queue_dmj_reproject(){
    REGION_ID=$1 
    QUEUE_NAME=$2

    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
    DMJ_DPATH=/home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan
    DST_DPATH=$DATA_DVC_DPATH/Drop6

    DMJ_COCO_FPATH=$DMJ_DPATH/$REGION_ID/kwcoco-dataset/cropped_kwcoco.json
    IMGONLY_FPATH=$DST_DPATH/imgonly-$REGION_ID.kwcoco.json
    IMGANNS_FPATH=$DST_DPATH/imganns-$REGION_ID.kwcoco.zip
    DJM_ASSET_DPATH=$DMJ_DPATH/$REGION_ID/kwcoco-dataset/$REGION_ID
    DST_ASSET_DPATH=$DST_DPATH/$REGION_ID

    cmd_queue submit "$QUEUE_NAME" -- \
        python -m watch reproject_annotations \
            --src "$IMGONLY_FPATH" \
            --dst "$IMGANNS_FPATH" \
            --propogate_strategy="SMART" \
            --site_models="$DATA_DVC_DPATH/annotations/drop6/site_models/${REGION_ID}_*" \
            --region_models="$DATA_DVC_DPATH/annotations/drop6/region_models/${REGION_ID}*" 
}


update_from_dmj_constructions(){
    __doc__="

    READY:
        BH_R001  BR_R002  BR_R004  LR_R001  NZ_R001  US_R001  US_R005  KR_R001  KR_R002 BR_R001 
        
        DONE:
        KR_R001  KR_R002 BR_R001

    NEW
        echo AE_C003 PE_C003 QA_C001 SA_C005 US_C000 US_C010 US_C011 US_C012 US_C014
    
    "

    REGION_IDS=(AE_C003 PE_C003 QA_C001 SA_C005 US_C000 US_C010 US_C011 US_C012 US_C014)
    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
    DMJ_DPATH=/home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan
    DST_DPATH=$DATA_DVC_DPATH/Drop6

    QUEUE_NAME=fixup-drop6-zip
    cmd_queue new $QUEUE_NAME
    # Zip the new data from DMJ drive to dst drive
    for REGION_ID in "${REGION_IDS[@]}"; do
        queue_archive_dmj_assets "$REGION_ID" "$QUEUE_NAME"
    done
    cmd_queue show fixup-drop6-zip
    cmd_queue run fixup-drop6-zip

    # Copy over the kwcoco files
    DMJ_COCO_FPATH=$DMJ_DPATH/$REGION_ID/kwcoco-dataset/cropped_kwcoco.json
    QUEUE_NAME=fixup-drop6-copy
    cmd_queue new $QUEUE_NAME
    for REGION_ID in "${REGION_IDS[@]}"; do
        IMGONLY_FPATH=$DST_DPATH/imgonly-$REGION_ID.kwcoco.json
        # Overwrite old kwcoco files with new ones
        cmd_queue submit $QUEUE_NAME "cp '$DMJ_COCO_FPATH' '$IMGONLY_FPATH'"
    done
    cmd_queue show $QUEUE_NAME --backend=serial
    cmd_queue run $QUEUE_NAME --backend=serial

    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
    TODO_ZIPS=( )
    cd $DATA_DVC_DPATH/Drop6
    for REGION_ID in "${REGION_IDS[@]}"; do
        ls_array HAVE_ZIPS "$REGION_ID/*.zip"
        bash_array_repr "${HAVE_ZIPS[@]}"
        TODO_ZIPS+=("${HAVE_ZIPS[@]}")
    done
    bash_array_repr "${TODO_ZIPS[@]}"
    dvc add -vv -- "${TODO_ZIPS[@]}"
    git commit -am "Add updated Drop6 images"
    dvc push -r aws -vv -- "${TODO_ZIPS[@]}"
    #dvc add -- BH_R001/*.zip BR_R002/*.zip BR_R004/*.zip LR_R001/*.zip NZ_R001/*.zip US_R001/*.zip US_R005/*.zip

    QUEUE_NAME=fixup-drop6-reproject
    cmd_queue new $QUEUE_NAME
    for REGION_ID in "${REGION_IDS[@]}"; do
        queue_dmj_reproject "$REGION_ID" "$QUEUE_NAME"
    done
    cmd_queue show $QUEUE_NAME --backend=serial
    cmd_queue run $QUEUE_NAME --backend=tmux --workers=4

    ls /home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_data_dvc/annotations/drop6/region_models/US_C012


    # On horologic
    ls /home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan
    ls /home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan/KR_R001/kwcoco-dataset/
    #smartwatch stats /home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan/KR_R001/kwcoco-dataset/cropped_kwcoco.json
    ls /home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan/KR_R001/kwcoco-dataset/KR_R001/
    ls /home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_data_dvc/Drop6/KR_R001 

    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
    DMJ_DPATH=/home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan
    DST_DPATH=$DATA_DVC_DPATH/Drop6
    ls $DMJ_DPATH


    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
    DMJ_DPATH=/home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan
    DST_DPATH=$DATA_DVC_DPATH/Drop6

    ### DO THIS AT THE VERY END.
    DATA_DVC_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware="hdd")
    DST_DPATH=$DATA_DVC_DPATH/Drop6
    cd $DST_DPATH

    cmd_queue new dvc-add
    ZIP_FPATHS=(*/*.zip)
    cmd_queue submit dvc-add -- 
    echo dvc add "${ZIP_FPATHS[@]}" -vv
    cmd_queue show dvc-add

    for REGION_ID in "${REGION_IDS[@]}"; do
        echo dvc add "$REGION_ID"/*.zip
        #dvc add -- BH_R001/*.zip BR_R002/*.zip BR_R004/*.zip LR_R001/*.zip NZ_R001/*.zip US_R001/*.zip US_R005/*.zip
    done

    dvc add -- */*.zip
    ZIP_DVC_FPATHS=($REGION_ID/*.zip.dvc)

    dvc push -r aws "${ZIP_DVC_FPATHS[@]}" -v

    python ~/code/watch/watch/cli/prepare_splits.py \
        --base_fpath="imganns*.kwcoco.*" \
        --workers=5 \
        --constructive_mode=True --run=1

    rm splits.zip
    7z a splits.zip -mx9 -- *.kwcoco.* 
    dvc add splits.zip
    git commit -am "Update annotations"
    git push 
    dvc push -r aws splits.zip

    __check_overlap__="
    import ubelt as ub
    import xdev

    dpath_new = ub.Path('/home/local/KHQ/jon.crall/data/david.joy/DatasetGeneration2023Jan/KR_R001/kwcoco-dataset/KR_R001/')
    dpath_old = ub.Path('/home/local/KHQ/jon.crall/remote/horologic/data/dvc-repos/smart_data_dvc/Drop6/KR_R001').expand()

    files_new = []
    for r, ds, fs in dpath_new.walk():
        rpath = r.relative_to(dpath_new)
        files_new.extend([rpath / f for f in fs if f.endswith('.tif')])

    files_old = []
    for r, ds, fs in dpath_old.walk():
        rpath = r.relative_to(dpath_old)
        files_old.extend([rpath / f for f in fs if f.endswith('.tif')])

    missing_from_new = set(files_old) - set(files_new)
    missing_from_old = set(files_new) - set(files_old)
    common = set(files_new) & set(files_old)

    grouped_missing_old = ub.udict(ub.group_items(missing_from_old, lambda x: x.parent.name)).map_values(len).sorted_keys()
    grouped_missing_new = ub.udict(ub.group_items(missing_from_new, lambda x: x.parent.name)).map_values(len).sorted_keys()

    normkey_old = [k[:-2] for k in grouped_missing_old.keys()]
    normkey_new = [k[:-2] for k in grouped_missing_new.keys()]

    set(normkey_new) - set(normkey_old)
    set(normkey_old) - set(normkey_new)

    print('grouped_missing_old ' + ub.urepr(grouped_missing_old))
    print('grouped_missing_new ' + ub.urepr(grouped_missing_new))
    print('grouped_missing_old ' + ub.urepr(grouped_missing_old))
    print('grouped_missing_new ' + ub.urepr(grouped_missing_new))

    print(f'{len(common)}')

    flags = []
    for c in common:
        asset1 = dpath_new / c
        asset2 = dpath_old / c

        print(asset1.shrinkuser('~/remote/horologic'))
        print(asset2.shrinkuser('~/remote/horologic'))

        flag = ub.hash_file(asset2) == ub.hash_file(asset1)
        assert flag
        flags.append(flag)
        print(np.mean(flags))
        ...

    "

}
