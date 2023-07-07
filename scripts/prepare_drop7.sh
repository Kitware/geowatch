#!/bin/bash

source "$HOME"/code/watch/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-3
DATASET_SUFFIX=Drop7
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

# Construct the TA2-ready dataset
python -m watch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=20 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DVC_DATA_DPATH" \
    --aws_profile=iarpa \
    --region_globstr="$REGION_GLOBSTR" \
    --site_globstr="$SITE_GLOBSTR" \
    --requester_pays=False \
    --fields_workers=8 \
    --convert_workers=0 \
    --align_workers=4 \
    --align_aux_workers=0 \
    --ignore_duplicates=1 \
    --separate_region_queues=1 \
    --separate_align_jobs=1 \
    --visualize=0 \
    --target_gsd=10 \
    --cache=0 \
    --verbose=100 \
    --skip_existing=0 \
    --force_min_gsd=2.0 \
    --force_nodata=-9999 \
    --hack_lazy=False \
    --backend=tmux \
    --tmux_workers=8 \
    --run=1



ls_array(){
    __doc__='
    Read the results of a glob pattern into an array

    Args:
        arr_name
        glob_pattern

    Example:
        arr_name="myarray"
        glob_pattern="*"
        pass
        bash_array_repr "${array[@]}"
        mkdir -p $HOME/tmp/tests/test_ls_arr
        cd $HOME/tmp/tests/test_ls_arr
        touch "$HOME/tmp/tests/test_ls_arr/path ological files"
        touch "$HOME/tmp/tests/test_ls_arr/are so fun"
        touch "$HOME/tmp/tests/test_ls_arr/foo"
        touch "$HOME/tmp/tests/test_ls_arr/bar"
        touch "$HOME/tmp/tests/test_ls_arr/baz"
        touch "$HOME/tmp/tests/test_ls_arr/biz"
        touch "$HOME/tmp/tests/test_ls_arr/fake_newline\n in fils? YES!"
        python -c "import ubelt; ubelt.Path(\"$HOME/tmp/tests/test_ls_arr/Real newline \n in fname\").expand().touch()"
        python -c "import ubelt; ubelt.Path(\"$HOME/tmp/tests/test_ls_arr/Realnewline\ninfname\").expand().touch()"

        arr_name="myarray"
        glob_pattern="*"
        ls_array "$arr_name" "$glob_pattern"
        bash_array_repr "${array[@]}"

    References:
        .. [1] https://stackoverflow.com/a/18887210/887074
        .. [2] https://stackoverflow.com/questions/14564746/in-bash-how-to-get-the-current-status-of-set-x

    TODO:
        get the echo of shopt off
    '
    local arr_name="$1"
    local glob_pattern="$2"

    local toggle_nullglob=""
    local toggle_noglob=""
    # Can check the "$-" variable to see what current settings are i.e. set -x, set -e
    # Can check "set -o" to get currentenabled options
    # Can check "shopt" to get current enabled options

    if shopt nullglob; then
        # Check if null glob is enabled, if it is, this will be true
        toggle_nullglob=0
    else
        toggle_nullglob=1
    fi
    # Check for -f to see if noglob is enabled
    if [[ -n "${-//[^f]/}" ]]; then
        toggle_noglob=1
    else
        toggle_noglob=0
    fi

    if [[ "$toggle_nullglob" == "1" ]]; then
        # Enable nullglob if it was off
        shopt -s nullglob
    fi
    if [[ "$toggle_noglob" == "1" ]]; then
        # Enable nullglob if it was off
        shopt -s nullglob
    fi

    # The f corresponds to if noglob was set

    # shellcheck disable=SC2206
    array=($glob_pattern)

    if [[ "$toggle_noglob" == "1" ]]; then
        # need to reenable noglob
        set -o noglob  # enable noglob
    fi
    if [[ "$toggle_nullglob" == "1" ]]; then
        # Disable nullglob if it was off to make sure it doesn't interfere with anything later
        shopt -u nullglob
    fi
    # Copy the array into the dynamically named variable
    readarray -t "$arr_name" < <(printf '%s\n' "${array[@]}")
}


move_kwcoco_paths(){
    __doc__="
    "

    python -c "if 1:

    import ubelt as ub
    bundle_dpath = ub.Path('.').absolute()
    all_paths = list(bundle_dpath.glob('*_[C]*'))
    region_dpaths = []
    for p in all_paths:
        if p.is_dir() and str(p.name)[2] == '_':
            region_dpaths.append(p)

    import cmd_queue
    queue = cmd_queue.Queue.create('tmux', size=16)

    for dpath in region_dpaths:
        print(ub.urepr(dpath.ls()))

    for dpath in region_dpaths:
        region_id = dpath.name
        fnames = [
            f'imgonly-{region_id}.kwcoco.zip',
            f'imganns-{region_id}.kwcoco.zip',
        ]
        for fname in fnames:
            old_fpath = bundle_dpath / fname
            new_fpath = dpath / fname
            if old_fpath.exists() and not new_fpath.exists():
                queue.submit(f'kwcoco move {old_fpath} {new_fpath}')

    queue.run()
    "
    #kwcoco move imgonly-AE_R001.kwcoco.zip ./AE_R001
    kwcoco move imganns-AE_C001.kwcoco.zip ./AE_R001

}

add_coco_files(){
    ls -- */*.kwcoco.zip
    dvc add -- */*.kwcoco.zip
    git commit -am "Add Drop7 TnE Region annotations"
    git push
    dvc push -r aws -- */*.kwcoco.zip.dvc
}


remove_empty_kwcoco_files(){
    python -c "if 1:

    import ubelt as ub
    bundle_dpath = ub.Path('.').absolute()
    all_paths = list(bundle_dpath.glob('imganns*zip'))
    region_dpaths = []
    dsets = list(kwcoco.CocoDataset.coerce_multiple(all_paths, workers=8))

    bad_fpaths = []
    for dset in dsets:
        if dset.n_images == 0:
            bad_fpaths += [dset.fpath]

    for f in bad_fpaths:
        ub.Path(f).delete()

    "

}


redo_add_raw_data(){
    # Not sure I like adding each date as its own DVC file. Lets try doing it
    # by sensor again.
    __doc__="
    "
    python -c "if 1:
    import watch
    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')
    bundle_dpath = dvc_data_dpath / 'Aligned-Drop7'
    #tne_dpaths = list(bundle_dpath.glob('[A-Z][A-Z]_R0*'))
    tne_dpaths = list(bundle_dpath.glob('[A-Z][A-Z]_C0*'))

    # Try using DVC at the image level instead?
    old_dvc_paths = []
    for dpath in tne_dpaths:
        if dpath.name != 'BR_R005':
            old_dvc_paths += list(dpath.glob('*/affine_warp/*.dvc'))

    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce(bundle_dpath)
    dvc.remove(old_dvc_paths)

    new_dpaths = []
    for dpath in tne_dpaths:
        if dpath.name != 'BR_R005':
            new_dpaths += [d for d in dpath.glob('*') if d.is_dir() and d.name in {'L8', 'WV1', 'WV', 'S2', 'PD'}]

    dvc.add(new_dpaths)
    dvc.git_commitpush(message='Add images for {bundle_dpath.name}')
    dvc.push(new_dpaths, remote='aws')
    "
}


dvc_add_raw_imgdata(){
    #DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
    #cd $DVC_DATA_DPATH/Aligned-Drop7
    #
    python -c "if 1:
    import watch
    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')
    bundle_dpath = dvc_data_dpath / 'Aligned-Drop7'
    #tne_dpaths = list(bundle_dpath.glob('[A-Z][A-Z]_R0*'))
    tne_dpaths = list(bundle_dpath.glob('[A-Z][A-Z]_C0*'))

    new_dpaths = []
    for dpath in tne_dpaths:
        new_dpaths += [d for d in dpath.glob('*') if d.is_dir() and d.name in {'L8', 'WV1', 'WV', 'S2', 'PD'}]

    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce(bundle_dpath)
    dvc.add(new_dpaths)
    dvc.git_commitpush(message='Add images for {bundle_dpath.name}')
    dvc.push(new_dpaths, remote='aws')
    "
    #declare -a ROI_DPATHS=()
    #ROI_DPATHS+=TNE_REGIONS
    #dvc add -- */PD */WV */S2 && dvc push -r aws -R . && git commit -am "Add Drop7 Raw Images Images" && git push  &&
}


dvc_add_(){
    #DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
    #cd $DVC_DATA_DPATH/Aligned-Drop7
    #
    python -c "if 1:
    import watch
    bundle_name = 'Drop7-MedianNoWinter10GSD'
    dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='hdd')
    bundle_dpath = dvc_data_dpath / bundle_name
    tne_dpaths = list(bundle_dpath.glob('raw_bands/[A-Z][A-Z]_R0*'))
    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce(bundle_dpath)
    dvc.add(tne_dpaths)
    dvc.git_commitpush(message='Add raw bands for {bundle_dpath.name}')
    dvc.push(tne_dpaths, remote='aws')
    "
    #declare -a ROI_DPATHS=()
    #ROI_DPATHS+=TNE_REGIONS
    #dvc add -- */PD */WV */S2 && dvc push -r aws -R . && git commit -am "Add Drop7 Raw Images Images" && git push  &&
}


# COLD FEATURES on Raw Data
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Aligned-Drop7
python -m watch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/*/imganns-AE*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_invariants2=0 \
    --with_sam=0 \
    --with_materials=0 \
    --with_depth=0 \
    --with_cold=1 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, \
    --cold_workermode=process \
    --cold_workers=8 \
    --tmux_workers=16 \
    --backend=tmux --run=0
    #--base_fpath "$BUNDLE_DPATH"/*/imganns-*[0-9].kwcoco.zip \

dvc add Aligned-Drop7/LT_R001/combo_imganns-LT_R001_C.kwcoco.zip

rm -rf Aligned-Drop7/*/reccg/*/cold_feature/tmp
ls Aligned-Drop7/*/reccg/*/cold_feature
dvc add Aligned-Drop7/*/reccg/*/cold_feature
dvc add Aligned-Drop7/*/reccg/*/cold_feature
dvc add Aligned-Drop7/*/combo_imganns-*_C.kwcoco.zip




#python ~/code/watch-smartflow-dags/reproduce_mlops.py imgonly-US_R006.kwcoco.zip
# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/watch/cli/queue_cli/prepare_time_combined_dataset.py \
    --regions=all_tne \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop6_hard_v1/region_models \
    --spatial_tile_size=1024 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=4 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --run=1



# Compute all feature except COLD
export CUDA_VISIBLE_DEVICES="1"
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD
python -m watch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/imganns-*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=1 \
    --with_invariants2=1 \
    --with_sam=1 \
    --with_materials=1 \
    --with_mae=1 \
    --with_depth=0 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, --tmux_workers=8 --backend=tmux --run=1



# Transfer the COLD features from the raw data
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/watch/cli/queue_cli/prepare_cold_transfer.py \
    --src_kwcocos "$DVC_DATA_DPATH/Aligned-Drop7/*/*cold.kwcoco.zip" \
    --dst_kwcocos "$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/*_EI2LMS.kwcoco.zip" \
    --run=0


redo_cold_transfer(){
    # If we already transfered COLD from another dataset, but need to push them
    # onto a new version, do it like this:
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
    python ~/code/watch/watch/cli/queue_cli/prepare_cold_transfer.py \
        --src_kwcocos "$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/*_I2LSC.kwcoco.zip" \
        --dst_kwcocos "$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD/*_EI2LMS.kwcoco.zip" \
        --run=1
}



DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m watch.cli.prepare_splits \
    --base_fpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD/combo_imganns*-*_[RC]*_EI2LMSC.kwcoco.zip \
    --suffix=EI2LMSC \
    --backend=tmux --tmux_workers=6 \
    --run=1


dvc add combo*_EI2LMSC.kwcoco.zip data_*_EI2LMSC_*.kwcoco.zip data.kwcoco.zip


HDD_DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SSD_DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="ssd")

rsync -avprPR "$HDD_DVC_DATA_DPATH"/./Drop7-MedianNoWinter10GSD "$SSD_DVC_DATA_DPATH"


kwcoco validate data.kwcoco.json
geowatch visualize data.kwcoco.json --smart


geowatch stats data_vali_EI2LMSC_split6.kwcoco.zip
geowatch visualize data_vali_EI2LMSC_split6.kwcoco.zip \
    --channels "red|green|blue,pan,sam.0:3,mae.0:3,landcover_hidden.0:3,invariants.0:3,mtm,materials.0:3,mat_feats.0:3,red_COLD_cv|green_COLD_cv|blue_COLD_cv" \
    --smart


#fixup="
#coco_images = dset.images().coco_images
#from watch.utils import util_gdal

#coco_img = dset.coco_image(408)

#for asset in coco_img.assets:
#    fpath = ub.Path(coco_img.bundle_dpath) / asset['file_name']
#    bak_fpath = fpath.augment(prefix='_backup_')
#    fpath.move(bak_fpath)
#    print(fpath)


#problematic_paths = []
#for img in coco_images:
#    for asset in img.assets:
#        if isinstance(asset['parent_file_name'], list) and len(asset['parent_file_name']) > 2:
#            print(len(asset['parent_file_name']))
#            problematic_paths.append(ub.Path(img.bundle_dpath) / asset['file_name'])

#for p in ub.ProgIter(problematic_paths):
#    p.delete()

#        #fpath = ub.Path(img.bundle_dpath) / asset['file_name']
#        #print(fpath)
#        #ptr = util_gdal.GdalOpen(fpath, mode='r')
#        #info = ptr.info()
#        #print(info['bands'])
#        #...
#"


#DATA_DVC_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
#geowatch visualize \
#    "$DATA_DVC_DPATH"/Drop7-MedianNoWinter10GSD/combo_imganns-KR_R002_EI2LMSC.kwcoco.zip \
#    --channels="(L8):(red_COLD_cv|green_COLD_cv|blue_COLD_cv)" \
#    --exclude_sensors=WV,PD \
#    --stack=True \
#    --animate=True \
#    --only_boxes=True \
#    --draw_labels=False



###########################
## BUILD SC CROPPED DATASET
###########################


#HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='hdd')
#python -m watch.cli.cluster_sites \
#    --src "$HDD_DATA_DPATH"/annotations/drop6_hard_v1/region_models/KR_R002.geojson \
#    --dst_dpath "$HDD_DATA_DPATH"/Drop7-Cropped2GSD/KR_R002/clusters \
#    --minimum_size="128x128@2GSD" \
#    --maximum_size="1024x1024@2GSD" \
#    --context_factor=1.3 \
#    --draw_clusters True


## Execute alignment / crop script
#python -m watch.cli.coco_align \
#    --src "$HDD_DATA_DPATH"/Aligned-Drop7/KR_R002/imgonly-KR_R002.kwcoco.zip \
#    --dst "$HDD_DATA_DPATH"/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip \
#    --regions "$HDD_DATA_DPATH/Drop7-Cropped2GSD/KR_R002/clusters/*.geojson" \
#    --rpc_align_method orthorectify \
#    --workers=10 \
#    --aux_workers=2 \
#    --force_nodata=-9999 \
#    --context_factor=1.0 \
#    --minimum_size="128x128@2GSD" \
#    --force_min_gsd=2.0 \
#    --convexify_regions=True \
#    --target_gsd=2.0 \
#    --geo_preprop=False \
#    --exclude_sensors=L8 \
#    --sensor_to_time_window "
#        S2: 1month
#    " \
#    --keep img

#HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='hdd')
#geowatch reproject_annotations \
#    --src "$HDD_DATA_DPATH"/Drop7-Cropped2GSD/KR_R002.kwcoco.zip \
#    --dst "$HDD_DATA_DPATH"/Drop7-Cropped2GSD/KR_R002.kwcoco.zip \
#    --io_workers=avail \
#    --region_models="$HDD_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
#    --site_models="$HDD_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"



#(cd "$HDD_DATA_DPATH"/Aligned-Drop7 && echo *)
# Create multiple items in a bash array, and loop over that array
#REGION_IDS=(
#    #KR_R001
#    #KR_R002
#    #AE_C001
#    #AE_C002
#    AE_C003 AE_R001 BH_R001 BR_R001 BR_R002 BR_R004 BR_R005 CH_R001 CN_C000
#    CN_C001 CO_C001 CO_C009 IN_C000  LT_R001 NG_C000 NZ_R001 PE_C001 PE_C003
#    PE_C004 PE_R001 QA_C001 SA_C001 SA_C005 SN_C000 US_C000 US_C001 US_C010
#    US_C011 US_C012 US_C016 US_R001 US_R004 US_R005 US_R006 US_R007 VN_C002
#)

# Create a new queue
HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='hdd')
SSD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')

SRC_DATA_DPATH=$HDD_DATA_DPATH
#DST_DATA_DPATH=$HDD_DATA_DPATH
#DST_DATA_DPATH=$SSD_DATA_DPATH
DST_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
ANN_DATA_DPATH=$SSD_DATA_DPATH

test -d "$ANN_DATA_DPATH/annotations/drop6_hard_v1/region_models" || echo "ERROR: ANNOTATIONS DONT EXIT"

REGION_IDS=(KR_R001 KR_R002 AE_R001 PE_R001 US_R007 BH_R001 BR_R001 BR_R002 BR_R004 BR_R005 CH_R001 LT_R001 NZ_R001 US_C010 US_C011 US_C012 US_C016 US_R001 US_R004 US_R005 US_R006)
#REGION_IDS=(KR_R001)

TRUE_REGION_DPATH="$DST_DATA_DPATH/annotations/drop6_hard_v1/region_models"
SRC_BUNDLE_DPATH=$SRC_DATA_DPATH/Aligned-Drop7
DST_BUNDLE_DPATH=$DST_DATA_DPATH/Drop7-Cropped2GSD


python -m cmd_queue new "crop_for_sc_queue"
for REGION_ID in "${REGION_IDS[@]}"; do
    echo "REGION_ID = $REGION_ID"
done
for REGION_ID in "${REGION_IDS[@]}"; do

    cmd_queue submit --jobname="cluster-$REGION_ID" --depends="None" -- crop_for_sc_queue \
        python -m watch.cli.cluster_sites \
            --src "$TRUE_REGION_DPATH/$REGION_ID.geojson" \
            --dst_dpath "$DST_BUNDLE_DPATH/$REGION_ID/clusters" \
            --draw_clusters True

    python -m cmd_queue submit --jobname="crop-$REGION_ID" --depends="cluster-$REGION_ID" -- crop_for_sc_queue \
        python -m watch.cli.coco_align \
            --src "$SRC_BUNDLE_DPATH/$REGION_ID/imgonly-$REGION_ID.kwcoco.zip" \
            --dst "$DST_BUNDLE_DPATH/$REGION_ID/$REGION_ID.kwcoco.zip" \
            --regions "$DST_BUNDLE_DPATH/$REGION_ID/clusters/*.geojson" \
            --rpc_align_method orthorectify \
            --workers=10 \
            --aux_workers=2 \
            --force_nodata=-9999 \
            --context_factor=1.0 \
            --minimum_size="128x128@2GSD" \
            --force_min_gsd=2.0 \
            --convexify_regions=True \
            --target_gsd=2.0 \
            --geo_preprop=False \
            --exclude_sensors=L8 \
            --sensor_to_time_window "
                S2: 1month
            " \
            --keep img

    python -m cmd_queue submit --jobname="reproject-$REGION_ID" --depends="crop-$REGION_ID" -- crop_for_sc_queue \
        geowatch reproject_annotations \
            --src "$DST_BUNDLE_DPATH/$REGION_ID/$REGION_ID.kwcoco.zip" \
            --dst "$DST_BUNDLE_DPATH/$REGION_ID/imgannots-$REGION_ID.kwcoco.zip" \
            --io_workers="avail/2" \
            --region_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/region_models/${REGION_ID}.geojson" \
            --site_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/site_models/${REGION_ID}_*.geojson"
done

# Show the generated script
python -m cmd_queue show "crop_for_sc_queue"

python -m cmd_queue run --workers=8 "crop_for_sc_queue"

DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='ssd')
ls "$DVC_DATA_DPATH/Drop7-Cropped2GSD"
python -m watch.cli.prepare_splits \
    --base_fpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/*/imgannots-*.kwcoco.zip \
    --dst_dpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD \
    --suffix=rawbands --run=1 --workers=2


#git clone git+ssh://namek.kitware.com/flash/smart_drop7/.git smart_drop7
#dvc remote add namek_ssd ssh://namek.kitware.com/flash/smart_drop7/.dvc/cache

geowatch_dvc add drop7_data_ssd --path=/flash/smart_drop7 --tags drop7_data --hardware ssd
geowatch_dvc add drop7_data_ssd --path=/media/joncrall/flash1/smart_drop7 --tags drop7_data --hardware ssd

#HDD_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='hdd')
#REGION_ID=KR_R001
#REGION_ID=KR_R002
#REGION_ID=AE_R001
#REGION_ID=PE_R001
#REGION_ID=CH_R001
#REGION_ID=US_R007
#REGION_IDS=(
#    #KR_R001 KR_R002 AE_R001 PE_R001
#    BH_R001 BR_R001 BR_R002 BR_R004 BR_R005
#    CH_R001 LT_R001 NZ_R001 US_C010 US_C011 US_C012 US_C016 US_R001 US_R004
#    US_R005 US_R006
#)
#for REGION_ID in "${REGION_IDS[@]}"; do
#    echo "REGION_ID = $REGION_ID"
#done
#for REGION_ID in "${REGION_IDS[@]}"; do
#    python -m watch.cli.cluster_sites \
#        --src "$HDD_DATA_DPATH"/annotations/drop6_hard_v1/region_models/"$REGION_ID".geojson \
#        --dst_dpath "$HDD_DATA_DPATH"/Drop7-Cropped2GSD/"$REGION_ID"/clusters \
#        --minimum_size="128x128@2GSD" \
#        --maximum_size="1024x1024@2GSD" \
#        --context_factor=1.3 \
#        --draw_clusters True

#    python -m watch.cli.coco_align \
#        --src "$HDD_DATA_DPATH/Aligned-Drop7/$REGION_ID/imgonly-$REGION_ID.kwcoco.zip" \
#        --dst "$HDD_DATA_DPATH/Drop7-Cropped2GSD/$REGION_ID/$REGION_ID.kwcoco.zip" \
#        --regions "$HDD_DATA_DPATH/Drop7-Cropped2GSD/$REGION_ID/clusters/*.geojson" \
#        --rpc_align_method orthorectify \
#        --workers=10 \
#        --aux_workers=2 \
#        --force_nodata=-9999 \
#        --context_factor=1.0 \
#        --minimum_size="128x128@2GSD" \
#        --force_min_gsd=2.0 \
#        --convexify_regions=True \
#        --target_gsd=2.0 \
#        --geo_preprop=False \
#        --exclude_sensors=L8 \
#        --sensor_to_time_window "
#            S2: 1month
#        " \
#        --keep img
#done


#cd Drop7-Cropped2GSD
#dvc add -- */*/S2 */*/WV* */*/PD
#dvc add -- */*.kwcoco.zip */clusters
dvc push -r namek_ssd -- */*.dvc


REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/KR_R002/KR_R002.kwcoco.zip \
    --dst "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/KR_R002/imgannots-KR_R002.kwcoco.zip \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"


REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip \
    --dst "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/KR_R001/imgannots-KR_R001.kwcoco.zip \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"

REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/PE_R001/PE_R001.kwcoco.zip \
    --dst "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/PE_R001/imgannots-PE_R001.kwcoco.zip \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/PE_R001*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/PE_R001*.geojson"



# On namek - test training - no teamfeat
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/KR_R001/imgannots-KR_R001.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/KR_R002/imgannots-KR_R002.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV1):(pan)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_kronly_small_V01
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '196,196'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.25y,-1.08y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.08y,1.25y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 2
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : True
    use_grid_negatives     : True
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 1.00
        global_saliency_weight : 0.20
        multimodal_reduce      : learned_linear
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"


# On namek - test training - no teamfeat (PE)
export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
echo "DVC_EXPT_DPATH = $DVC_EXPT_DPATH"
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER
DATASET_CODE=Drop7-Cropped2GSD
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_train_rawbands_split6.kwcoco.zip
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_vali_rawbands_split6.kwcoco.zip
CHANNELS="(L8,S2):(blue|green|red|nir),(WV):(blue|green|red),(WV1):(pan)"
EXPERIMENT_NAME=Drop7-Cropped2GSD_SC_bgrn_split6_V05
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
TARGET_LR=1e-4
WEIGHT_DECAY=$(python -c "print($TARGET_LR * 0.01)")
echo "WEIGHT_DECAY = $WEIGHT_DECAY"
MAX_STEPS=80000
WATCH_GRID_WORKERS=0 python -m watch.tasks.fusion fit --config "
data:
    select_videos          : $SELECT_VIDEOS
    num_workers            : 5
    train_dataset          : $TRAIN_FPATH
    vali_dataset           : $VALI_FPATH
    window_dims            : '256,256'
    time_steps             : 9
    time_sampling          : uniform-soft5-soft4-contiguous
    time_kernel            : '(-1.25y,-1.08y,-0.25y,-0.08y,0.0y,0.08y,0.25y,1.08y,1.25y)'
    window_resolution     : 2.0GSD
    input_resolution      : 2.0GSD
    output_resolution     : 2.0GSD
    neg_to_pos_ratio       : 1.0
    batch_size             : 4
    normalize_perframe     : false
    normalize_peritem      : 'blue|green|red|nir|pan'
    max_epoch_length       : 1000000
    channels               : '$CHANNELS'
    min_spacetime_weight   : 0.6
    temporal_dropout       : 0.5
    mask_low_quality       : False
    mask_samecolor_method  : None
    observable_threshold   : 0.0
    quality_threshold      : 0.0
    weight_dilate          : 10
    use_centered_positives : True
    use_grid_positives     : False
    use_grid_negatives     : False
    normalize_inputs       : 1024
    balance_areas          : True
model:
    class_path: MultimodalTransformer
    init_args:
        #saliency_weights      : '1:1'
        #class_weights         : auto
        tokenizer              : linconv
        arch_name              : smt_it_stm_p16
        decoder                : mlp
        positive_change_weight : 1
        negative_change_weight : 0.01
        stream_channels        : 16
        class_loss             : 'dicefocal'
        saliency_loss          : 'focal'
        saliency_head_hidden   : 6
        change_head_hidden     : 6
        class_head_hidden      : 6
        global_change_weight   : 0.00
        global_class_weight    : 1.00
        global_saliency_weight : 0.01
        multimodal_reduce      : learned_linear
optimizer:
    class_path: torch.optim.AdamW
    init_args:
        lr           : $TARGET_LR
        weight_decay : $WEIGHT_DECAY
lr_scheduler:
  class_path: torch.optim.lr_scheduler.OneCycleLR
  init_args:
    max_lr: $TARGET_LR
    total_steps: $MAX_STEPS
    anneal_strategy: cos
    pct_start: 0.05
trainer:
    accumulate_grad_batches: 24
    default_root_dir     : $DEFAULT_ROOT_DIR
    accelerator          : gpu
    devices              : 0,
    limit_val_batches    : 256
    limit_train_batches  : 2048
    num_sanity_val_steps : 0
    max_epochs           : 360
    callbacks:
        - class_path: pytorch_lightning.callbacks.ModelCheckpoint
          init_args:
              monitor: val_loss
              mode: min
              save_top_k: 5
              filename: '{epoch}-{step}-{val_loss:.3f}.ckpt'
              save_last: true

torch_globals:
    float32_matmul_precision: auto

initializer:
    init: $DVC_EXPT_DPATH/models/fusion/Drop6-MeanYear10GSD-V2/packages/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47/Drop6_TCombo1Year_BAS_10GSD_V2_landcover_split6_V47_epoch47_step3026.pt
"




echo "hello" > foo
aws s3 cp foo s3://kitware-smart-watch-data/dvc3/foo

aws s3 rm s3://kitware-smart-watch-data/dvc3/foo
