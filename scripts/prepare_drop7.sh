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


###################################################
# Reproject annotations onto time averaged BAS data
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DST_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD
REGION_IDS=(KR_R001 KR_R002 AE_R001 PE_R001 US_R007 BH_R001 BR_R001 BR_R002 BR_R004 BR_R005 CH_R001 LT_R001 NZ_R001 US_C010 US_C011 US_C012 US_C016 US_R001 US_R004 US_R005 US_R006)
#
ANN_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
python -m cmd_queue new "reproject_queue"
for REGION_ID in "${REGION_IDS[@]}"; do
    echo "REGION_ID = $REGION_ID"
done

for REGION_ID in "${REGION_IDS[@]}"; do
    SRC_GLOB_FPATH1="$DST_BUNDLE_DPATH/combo_imganns-${REGION_ID}_*.kwcoco.zip"
    SRC_GLOB_FPATH2="$DST_BUNDLE_DPATH/imganns-${REGION_ID}.kwcoco.zip"
    for SRC_FPATH in $SRC_GLOB_FPATH1 $SRC_GLOB_FPATH2; do
        if test -f "$SRC_FPATH"; then
            HASHID=$(echo "$SRC_FPATH" | sha256sum | cut -c1-8)
            python -m cmd_queue submit --jobname="reproject-$REGION_ID-$HASHID" -- reproject_queue \
                geowatch reproject_annotations \
                    --src="$SRC_FPATH"  \
                    --inplace \
                    --io_workers="avail/6" \
                    --region_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/region_models/${REGION_ID}.geojson" \
                    --site_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/site_models/${REGION_ID}_*.geojson"
       fi
    done
done
python -m cmd_queue show "reproject_queue"
python -m cmd_queue run --workers=8 "reproject_queue"

# Also update splits
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DST_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD
cd "$DST_BUNDLE_DPATH"
#
ANN_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
python -m cmd_queue new "reproject_queue"
for SRC_FPATH in data_*.kwcoco.zip; do
    echo "SRC_FPATH = $SRC_FPATH"
    HASHID=$(echo "$SRC_FPATH" | sha256sum | cut -c1-8)
    if test -f "$SRC_FPATH".dvc; then
        python -m cmd_queue submit --jobname="reproject-$HASHID" -- reproject_queue \
            geowatch reproject_annotations \
                --src="$SRC_FPATH"  \
                --inplace \
                --io_workers="avail/6" \
                --region_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/region_models" \
                --site_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/site_models"
    else
        echo "not updating file without a dvc sidecar"
    fi
done
python -m cmd_queue show "reproject_queue"
python -m cmd_queue run --workers=8 "reproject_queue"

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m watch.cli.prepare_splits \
    --base_fpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD/combo_imganns*-*_[RC]*_EI2LMSC.kwcoco.zip \
    --suffix=EI2LMSC \
    --backend=tmux --tmux_workers=6 \
    --run=1

dvc add combo*_EI2LMSC.kwcoco.zip data_*_EI2LMSC_*.kwcoco.zip data.kwcoco.zip

# Push updated annotations to DVC
python -c "if 1:
    import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce('.')
    bundle_dpath = dvc.dpath / 'Drop7-MedianNoWinter10GSD'
    to_update = [str(p)[:-4] for p in list(dvc.find_sidecar_paths_in_dpath(bundle_dpath)) if 'kwcoco' in str(p)]
    dvc.add(to_update, verbose=3)
    dvc.git_commitpush(message='update annotations')
    dvc.push(to_update, verbose=3)
"



###################################################



###
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


DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
python -m watch.utils.simple_dvc request \
    "$DVC_EXPT_DPATH"/models/depth/weights_v1.pt

# AC/SC Features on Clusters
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware="auto")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-Cropped2GSD
python -m watch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/*/[A-Z][A-Z]_R*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_invariants2=0 \
    --with_sam=0 \
    --with_materials=1 \
    --with_depth=0 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, \
    --cold_workermode=process \
    --cold_workers=8 \
    --tmux_workers=2 \
    --backend=tmux --run=0 --print-commands
    #--base_fpath "$BUNDLE_DPATH"/*/imganns-*[0-9].kwcoco.zip \


python -m watch.tasks.depth.predict \
    --dataset=/data2/dvc-repos/smart_drop7/Drop7-Cropped2GSD/NZ_R001/NZ_R001.kwcoco.zip \
    --deployed=/home/local/KHQ/jon.crall/remote/yardrat/data/dvc-repos/smart_expt_dvc/models/depth/weights_v1.pt \
    --output=/data2/dvc-repos/smart_drop7/Drop7-Cropped2GSD/NZ_R001/NZ_R001_dzyne_depth.kwcoco.zip \
    --data_workers=2 \
    --window_size=1440


python -m watch.tasks.rutgers_material_seg_v2.predict \
    --kwcoco_fpath=/media/joncrall/flash1/smart_drop7/Drop7-Cropped2GSD/KR_R001/KR_R001.kwcoco.zip \
    --model_fpath=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/rutgers/ru_model_05_25_2023.ckpt \
    --config_fpath=/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/models/rutgers/ru_config_05_25_2023.yaml \
    --output_kwcoco_fpath=/media/joncrall/flash1/smart_drop7/Drop7-Cropped2GSD/KR_R001/KR_R001_rutgers_material_seg_v4.kwcoco.zip \
    --workers=2

DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='ssd')
ls "$DVC_DATA_DPATH/Drop7-Cropped2GSD"
python -m watch.cli.prepare_splits \
    --base_fpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/*/combo_*_D.kwcoco.zip \
    --dst_dpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD \
    --suffix=depth --run=1 --workers=2

REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/data_vali_depth_split6.kwcoco.zip \
    --inplace \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"

REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/data_vali_depth_split6.kwcoco.zip \
    --inplace \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"

REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/data_train_depth_split6.kwcoco.zip \
    --inplace \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"




DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware="auto")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-Cropped2GSD
python -m watch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/*/[A-Z][A-Z]_R*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_invariants2=0 \
    --with_sam=0 \
    --with_materials=0 \
    --with_depth=0 \
    --with_mae=1 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0,1 \
    --cold_workermode=process \
    --cold_workers=8 \
    --tmux_workers=2 \
    --backend=tmux --run=1

DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='ssd')
ls "$DVC_DATA_DPATH/Drop7-Cropped2GSD"
python -m watch.cli.prepare_splits \
    --base_fpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/*/combo_*_E.kwcoco.zip \
    --dst_dpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD \
    --suffix=mae --run=1 --workers=2 \
    --splits split6

REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/data_train_mae_split6.kwcoco.zip \
    --inplace \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"


REGION_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
geowatch reproject_annotations \
    --src "$DVC_DATA_DPATH"/Drop7-Cropped2GSD/data_vali_mae_split6.kwcoco.zip \
    --inplace \
    --io_workers=avail \
    --region_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/region_models/*.geojson" \
    --site_models="$REGION_DATA_DPATH/annotations/drop6_hard_v1/site_models/*.geojson"


/media/joncrall/flash1/smart_drop7/Drop7-Cropped2GSD-Features/Drop7-Cropped2GSD/KR_R001

cd /media/joncrall/flash1/smart_drop7/Drop7-Cropped2GSD-Features/Drop7-Cropped2GSD/KR_R001

python -c "if 1:
    import kwcoco
    import ubelt as ub
    dst_dpaths = [p.parent / 'rawbands' / p.name for p in src_dpaths]

    ub.Path('.').glob('*.kwcoco.*')

    import kwcoco
    kwcoco.CocoDataset('KR_R001.data.kwcoco

    mv_man = self = CocoMoveAssetManager()
    for src, dst in zip(src_dpaths, dst_dpaths):
        mv_man.submit(src, dst)
"

python -c "if 1:
    import kwcoco
    import ubelt as ub
    from kwcoco.cli.coco_move_assets import CocoMoveAssetManager
    region_dpaths = [p for p in ub.Path('.').glob('*_R*') if p.is_dir()]

    for region_dpath in ub.ProgIter(region_dpaths):
        region_dpath = region_dpath.absolute()
        region_id = region_dpath.name
        coco_paths = list(region_dpath.glob('*.kwcoco.*'))
        dst_dpath = (region_dpath / 'rawbands')
        if dst_dpath.exists():
            continue
        coco_dsets = list(kwcoco.CocoDataset.coerce_multiple(coco_paths))
        mv_man = CocoMoveAssetManager(coco_dsets)
        src_dpaths = list(region_dpath.glob('*_CLUSTER_*'))
        dst_dpaths = [p.parent / 'rawbands' / p.name for p in src_dpaths]
        for src, dst in zip(src_dpaths, dst_dpaths):
            mv_man.submit(src, dst)
        mv_man.run()
"

# Fixes on yardrat
cd /data2/dvc-repos/smart_drop7/Drop7-Cropped2GSD
python -c "if 1:
    import kwcoco
    import ubelt as ub
    root = ub.Path('.')
    region_dpaths = [p for p in root.glob('*_R*') if p.is_dir()]

    import ubelt as ub
    from kwutil.copy_manager import CopyManager
    cman = CopyManager()

    # Move the data from the old compute location to the new one.
    for region_dpath in ub.ProgIter(region_dpaths):
        region_dpath = region_dpath.absolute()
        region_id = region_dpath.name
        new_region_dpath = region_dpath.parent.parent / 'Drop7-Cropped2GSD-Features' / region_dpath.name

        for fpath in region_dpath.glob('*.kwcoco.zip'):
            cman.submit(fpath, fpath.augment(dpath=new_region_dpath))
        cman.submit(region_dpath / '_assets', new_region_dpath / '_assets')
    cman.run()

    # Now change the name from _assets to teamfeats

    # from kwcoco.cli.coco_move_assets import CocoMoveAssetManager
    root = ub.Path('/data2/dvc-repos/smart_drop7/Drop7-Cropped2GSD-Features')
    region_dpaths = [p for p in root.glob('*_R*') if p.is_dir()]
    for region_dpath in ub.ProgIter(region_dpaths):
        region_dpath = region_dpath.absolute()
        region_id = region_dpath.name
        coco_paths = list(region_dpath.glob('*.kwcoco.zip'))

        src_path = (region_dpath / '_assets' / 'dzyne_depth')
        dst_path = (region_dpath / 'teamfeats' / 'dzyne_depth')
        if dst_dpath.exists():
            continue

        coco_dsets = list(kwcoco.CocoDataset.coerce_multiple(coco_paths, workers='avail'))
        self = mv_man = CocoMoveAssetManager(coco_dsets)

        mv_man.submit(src_path, dst_path)

        # Also mark old stuff that happened
        prev_dst_dpaths = list((region_dpath / 'rawbands').glob('*'))
        prev_src_dpaths = [p.parent.parent / p.name for p in prev_dst_dpaths]
        for src, dst in zip(prev_src_dpaths, prev_dst_dpaths):
            assert not src.exists()
            mv_man.submit(src, dst)

        mv_man.run()
"

dvc add -- */teamfeats/dzyne_depth */*_depth.kwcoco.zip */*_D.kwcoco.zip


DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware="auto")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-Cropped2GSD-Features
python -m watch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/*/[A-Z][A-Z]_R*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=0 \
    --with_invariants2=0 \
    --with_sam=0 \
    --with_materials=0 \
    --with_depth=1 \
    --with_mae=1 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0,1 \
    --cold_workermode=process \
    --cold_workers=8 \
    --tmux_workers=2 \
    --backend=tmux --run=1



DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='ssd')
ls "$DVC_DATA_DPATH/Drop7-Cropped2GSD-Features"

# utils.sh
#ls_array REGION_DPATHS "$DVC_DATA_DPATH/Drop7-Cropped2GSD-Features/*_R*"
#bash_array_repr "${REGION_DPATHS[@]}"
#cd $DVC_DATA_DPATH/Drop7-Cropped2GSD-Features/
#ls_array REGION_IDS "*_R*"
#bash_array_repr "${REGION_IDS[@]}"

DST_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='auto')
DST_BUNDLE_DPATH=$DST_DATA_DPATH/Drop7-Cropped2GSD-Features
REGION_IDS=(KR_R001 KR_R002 AE_R001 PE_R001 US_R007 BH_R001 BR_R001 BR_R002 BR_R004 BR_R005 CH_R001 LT_R001 NZ_R001 US_C010 US_C011 US_C012 US_C016 US_R001 US_R004 US_R005 US_R006)
#
ANN_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware='ssd')
python -m cmd_queue new "reproject_queue"
for REGION_ID in "${REGION_IDS[@]}"; do
    echo "REGION_ID = $REGION_ID"
done
for REGION_ID in "${REGION_IDS[@]}"; do
    SRC_FPATH="$DST_BUNDLE_DPATH/$REGION_ID/combo_${REGION_ID}_DE.kwcoco.zip"
    test -f "$SRC_FPATH"
    python -m cmd_queue submit --jobname="reproject-$REGION_ID" -- reproject_queue \
        geowatch reproject_annotations \
            --src="$SRC_FPATH"  \
            --inplace \
            --io_workers="avail/2" \
            --region_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/region_models/${REGION_ID}.geojson" \
            --site_models="$ANN_DATA_DPATH/annotations/drop6_hard_v1/site_models/${REGION_ID}_*.geojson"
done
python -m cmd_queue show "reproject_queue"
python -m cmd_queue run --workers=8 "reproject_queue"

DVC_DATA_DPATH=$(geowatch_dvc --tags='drop7_data' --hardware='ssd')
ls "$DVC_DATA_DPATH/Drop7-Cropped2GSD-Features"
python -m watch.cli.prepare_splits \
    --base_fpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD-Features/*/combo_*_DE.kwcoco.zip \
    --dst_dpath "$DVC_DATA_DPATH"/Drop7-Cropped2GSD-Features \
    --suffix=DE --run=1 --workers=10


####################3

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
# FIXME: dst outpaths
python -m watch.cli.prepare_splits \
    --base_fpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/*/imganns*-*_[RC]*.kwcoco.zip \
    --suffix=rawbands \
    --backend=tmux --tmux_workers=6 \
    --run=0

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
kwcoco move \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/KW_C001/data_train_rawbands_split6.kwcoco.zip \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/data_train_rawbands_split6.kwcoco.zip


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
kwcoco move \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/KW_C001/data_vali_rawbands_split6.kwcoco.zip \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/data_vali_rawbands_split6.kwcoco.zip

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m watch reproject \
    --src "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/data_train_rawbands_split6.kwcoco.zip \
    --inplace \
    --status_to_catname="positive_excluded: positive" \
    --regions="$DVC_DATA_DPATH"/drop7/region_models/*.geojson \
    --sites="$DVC_DATA_DPATH"/annotations/drop7/site_models/*.geojson --io_workers=8


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m watch reproject \
    --src "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-iMERIT/data_vali_rawbands_split6.kwcoco.zip \
    --inplace \
    --status_to_catname="positive_excluded: positive" \
    --regions="$DVC_DATA_DPATH"/drop7/region_models/*.geojson \
    --sites="$DVC_DATA_DPATH"/annotations/drop7/site_models/*.geojson --io_workers=8

kwcoco union \
    --absolute=True \
    --src \
        /data/joncrall/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD/data_vali_EI2LMSC_split6.kwcoco.zip \
        /data/joncrall/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-iMERIT/data_vali_rawbands_split6.kwcoco.zip \
    --dst /data/joncrall/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-Both/data_vali_mixed_split6.kwcoco.zip


kwcoco union \
    --absolute=True \
    --src \
        /data/joncrall/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD/data_train_EI2LMSC_split6.kwcoco.zip \
        /data/joncrall/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-iMERIT/data_train_rawbands_split6.kwcoco.zip \
    --dst /data/joncrall/dvc-repos/smart_data_dvc/Drop7-MedianNoWinter10GSD-Both/data_train_mixed_split6.kwcoco.zip


#python ~/code/watch-smartflow-dags/reproduce_mlops.py imgonly-US_R006.kwcoco.zip
# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/watch/cli/queue_cli/prepare_time_combined_dataset.py \
    --regions=all_tne \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-NoMask \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop7/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop7/region_models \
    --spatial_tile_size=512 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=1 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --mask_low_quality=False \
    --run=1


########## COMBINED NO MASK

# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
python ~/code/watch/watch/cli/queue_cli/prepare_time_combined_dataset.py \
    --regions="['US_C010', 'NG_C000', 'KW_C001', 'SA_C005', 'US_C001', 'CO_C009',
        'US_C014', 'CN_C000', 'PE_C004', 'IN_C000', 'PE_C001', 'SA_C001', 'US_C016',
        'AE_C001', 'US_C011', 'AE_C002', 'PE_C003', 'RU_C000', 'CO_C001', 'US_C000',
        'US_C012', 'AE_C003', 'CN_C001', 'QA_C001', 'SN_C000']
        # 'VN_C002',
    " \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-NoMask \
    --true_site_dpath="$DVC_DATA_DPATH"/annotations/drop7/site_models \
    --true_region_dpath="$DVC_DATA_DPATH"/annotations/drop7/region_models \
    --spatial_tile_size=512 \
    --merge_method=median \
    --remove_seasons=winter \
    --tmux_workers=2 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --skip_existing=0 \
    --cache=1 \
    --mask_low_quality=False \
    --run=1

__doc__="
Ideal DVC:

* DatasetName

    - RegionName

        - RegionKwcocoVariant1.dvc
        - RegionKwcocoVariant2.dvc
        - rawbands/WV.dvc
        - rawbands/S2.dvc
        - rawbands/L8.dvc
        - rawbands/PD.dvc
        - teamfeats/WV/landcover.dvc
        - teamfeats/S2/landcover.dvc
        - teamfeats/WV/mae.dvc
        - teamfeats/S2/mae.dvc
"
