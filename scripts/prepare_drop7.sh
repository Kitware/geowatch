#!/bin/bash

source "$HOME"/code/watch/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-3
DATASET_SUFFIX=Drop7
REGION_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/region_models/*_[R]0*.geojson"
SITE_GLOBSTR="$DVC_DATA_DPATH/annotations/drop6_hard_v1/site_models/*_[R]0*.geojson"

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
    all_paths = list(bundle_dpath.glob('*_[R]*'))
    region_dpaths = []
    for p in all_paths:
        if p.is_dir() and str(p.name)[2] == '_':
            region_dpaths.append(p)

    import cmd_queue
    queue = cmd_queue.Queue.create('tmux', size=16)

    for dpath in region_dpaths:
        region_id = dpath.name
        fname = f'imgonly-{region_id}.kwcoco.zip'
        fpath = bundle_dpath / fname
        if fpath.exists():
            queue.submit(f'kwcoco move {fpath} {dpath}')
        fname = f'imganns-{region_id}.kwcoco.zip'
        fpath = bundle_dpath / fname
        if fpath.exists():
            queue.submit(f'kwcoco move {fpath} {dpath}')

    queue.run()
    "
    #kwcoco move imgonly-AE_R001.kwcoco.zip ./AE_R001
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
    tne_dpaths = list(bundle_dpath.glob('[A-Z][A-Z]_R0*'))

    # Try using DVC at the image level instead?
    old_dvc_paths = []
    for dpath in tne_dpaths:
        if dpath.name != 'BR_R005':
            old_dvc_paths += list(dpath.glob('*/affine_warp/*.dvc'))

    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce(bundle_dpath)
    dvc.remove(old_dvc_paths)

    dvc.add(img_dpaths)
    dvc.git_commitpush(message='Add images for {bundle_dpath.name}')
    dvc.push(img_dpaths, remote='aws')
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
    tne_dpaths = list(bundle_dpath.glob('[A-Z][A-Z]_R0*'))

    # Try using DVC at the image level instead?
    img_dpaths = []
    for dpath in tne_dpaths:
        candidates = list(dpath.glob('*/affine_warp/*'))
        for cand in candidates:
            if cand.is_dir():
                img_dpaths.append(cand)

    from watch.utils import simple_dvc
    dvc = simple_dvc.SimpleDVC.coerce(bundle_dpath)
    dvc.add(img_dpaths)
    dvc.git_commitpush(message='Add images for {bundle_dpath.name}')
    dvc.push(img_dpaths, remote='aws')
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



#DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
#geowatch clean_geotiffs \
#    --src "$DVC_DATA_DPATH/Aligned-Drop7/data.kwcoco.json" \
#    --channels="*" \
#    --prefilter_channels="red" \
#    --min_region_size=256 \
#    --nodata_value=-9999 \
#    --workers="min(2,avail)" \
#    --probe_scale=None \
#    --use_fix_stamps=True \
#    --dry=True \
#    --export_bad_fpath=bad_files.txt


#--regions="[
#        # T&E Regions
#        AE_R001, BH_R001, BR_R001, BR_R002, BR_R004, BR_R005, CH_R001,
#        KR_R001,
#        KR_R002, LT_R001, NZ_R001, US_R001, US_R004, US_R005,
#        US_R006, US_R007,
#        # # iMerit Regions
#        AE_C001,
#        AE_C002,
#        AE_C003, PE_C001, QA_C001, SA_C005, US_C000, US_C010,
#        US_C011, US_C012,
#]" \


export CUDA_VISIBLE_DEVICES="0"
DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase2_expt' --hardware='auto')
BUNDLE_DPATH=$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD
python -m watch.cli.prepare_teamfeats \
    --base_fpath "$BUNDLE_DPATH"/imganns-*[0-9].kwcoco.zip \
    --expt_dvc_dpath="$DVC_EXPT_DPATH" \
    --with_landcover=1 \
    --with_invariants2=1 \
    --with_sam=1 \
    --with_materials=0 \
    --with_depth=0 \
    --with_cold=0 \
    --skip_existing=1 \
    --assets_dname=teamfeats \
    --gres=0, --tmux_workers=8 --backend=tmux --run=0


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m watch.cli.prepare_splits \
    --base_fpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD/combo_imganns*-*_[RC]*_I2L*.kwcoco.zip \
    --constructive_mode=True \
    --suffix=I2L \
    --backend=tmux --tmux_workers=6 \
    --run=1


HDD_DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SSD_DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="ssd")

rsync -avprPR "$HDD_DVC_DATA_DPATH"/./Drop7-MedianNoWinter10GSD "$SSD_DVC_DATA_DPATH"


geowatch visualize data.kwcoco.json --smart

fixup="
coco_images = dset.images().coco_images
from watch.utils import util_gdal

coco_img = dset.coco_image(408)

for asset in coco_img.assets:
    fpath = ub.Path(coco_img.bundle_dpath) / asset['file_name']
    bak_fpath = fpath.augment(prefix='_backup_')
    fpath.move(bak_fpath)
    print(fpath)


problematic_paths = []
for img in coco_images:
    for asset in img.assets:
        if isinstance(asset['parent_file_name'], list) and len(asset['parent_file_name']) > 2:
            print(len(asset['parent_file_name']))
            problematic_paths.append(ub.Path(img.bundle_dpath) / asset['file_name'])

for p in ub.ProgIter(problematic_paths):
    p.delete()

        #fpath = ub.Path(img.bundle_dpath) / asset['file_name']
        #print(fpath)
        #ptr = util_gdal.GdalOpen(fpath, mode='r')
        #info = ptr.info()
        #print(info['bands'])
        #...

"
