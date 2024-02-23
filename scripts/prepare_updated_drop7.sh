source "$HOME"/code/watch/secrets/secrets

DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
SENSORS=TA1-S2-L8-WV-PD-ACC-3
DATASET_SUFFIX=Drop7
test -e "$DVC_DATA_DPATH/annotations/drop7/region_models"
echo $?
REGIONS="$DVC_DATA_DPATH/annotations/drop7/region_models/*_[RC]*.geojson"
SITES="$DVC_DATA_DPATH/annotations/drop7/site_models/*_[RC]*.geojson"

#REGIONS="$DVC_DATA_DPATH/annotations/drop7/region_models/KR_R002.geojson"
#SITES="$DVC_DATA_DPATH/annotations/drop7/site_models/KR_R002_*.geojson"

export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR

#simple_dvc request "$DVC_DATA_DPATH/annotations/drop7" --verbose

# Current set of regions with ACC-3 data.
REGIONS="
- $DVC_DATA_DPATH/annotations/drop7/region_models/AE_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/AE_C002.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/AE_C003.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/AE_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/BH_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/BR_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/BR_R002.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/BR_R004.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/BR_R005.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/CH_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/CN_C000.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/CN_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/CO_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/CO_C009.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/CO_C011.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/IN_C000.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/KR_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/KR_R002.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/KW_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/LT_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/NG_C000.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/NZ_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/PE_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/PE_C003.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/PE_C004.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/PE_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/QA_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/RU_C000.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/SA_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/SA_C005.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/SA_C006.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/SN_C000.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_C000.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_C001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_C010.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_C011.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_C012.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_C014.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_C016.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_R001.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_R004.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_R005.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_R006.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/US_R007.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/VN_C002.geojson
- $DVC_DATA_DPATH/annotations/drop7/region_models/VN_C003.geojson
"

# Construct the TA2-ready dataset
python -m geowatch.cli.prepare_ta2_dataset \
    --dataset_suffix=$DATASET_SUFFIX \
    --stac_query_mode=auto \
    --cloud_cover=20 \
    --sensors="$SENSORS" \
    --api_key=env:SMART_STAC_API_KEY \
    --collated True \
    --dvc_dpath="$DVC_DATA_DPATH" \
    --regions="$REGIONS" \
    --sites="$SITES" \
    --aws_profile=iarpa \
    --requester_pays=False \
    --fields_workers=8 \
    --convert_workers=0 \
    --align_workers=8 \
    --align_aux_workers=0 \
    --ignore_duplicates=1 \
    --visualize=0 \
    --target_gsd=10 \
    --reproject_annotations=False \
    --cache=1 \
    --verbose=100 \
    --skip_existing=1 \
    --force_min_gsd=2.0 \
    --force_nodata=-9999 \
    --hack_lazy=False \
    --backend=tmux \
    --tmux_workers=8 \
    --final_union=False \
    --run=0

# Remove old kwcoco files
rm -f "$DVC_DATA_DPATH"/Aligned-Drop7/*/imgonly-*[0-9].kwcoco.zip
rm -f "$DVC_DATA_DPATH"/Aligned-Drop7/*/imgonly-*[0-9].kwcoco.zip.dvc
rm -f "$DVC_DATA_DPATH"/Aligned-Drop7/*/imganns-*[0-9].kwcoco.zip.dvc
rm -f "$DVC_DATA_DPATH"/Aligned-Drop7/*.kwcoco.zip
rm -f "$DVC_DATA_DPATH"/Aligned-Drop7/*.kwcoco.zip.dvc
rm -f "$DVC_DATA_DPATH"/Aligned-Drop7/*/subdata.kwcoco.*

git commit -am "Remove old kwcoco files"

rm -f "$DVC_DATA_DPATH"/Aligned-Drop7/*/subdata.kwcoco.*

# Add the new image only kwcoco files
sdvc add -- "$DVC_DATA_DPATH"/Aligned-Drop7/*/imgonly-*-rawbands.kwcoco.zip
git commit -am "Added updated imgonly kwcoco files"

# Add updated imagery
cd "$DVC_DATA_DPATH"/Aligned-Drop7
dvc add -- */L8 */S2 */WV */WV1
git commit -am "Update images"


####
# Prepare BAS Time Averaged Dataset
####

REGION_IDS="
- AE_C001
- AE_C002
- AE_C003
- AE_R001
- BH_R001
- BR_R001
- BR_R002
- BR_R004
- BR_R005
- CH_R001
- CN_C000
- CN_C001
- CO_C001
- CO_C009
- CO_C011
- IN_C000
- KR_R001
- KR_R002
- KW_C001
- LT_R001
- NG_C000
- NZ_R001
- PE_C001
- PE_C003
- PE_C004
- PE_R001
- QA_C001
- RU_C000
- SA_C001
- SA_C005
- SA_C006
- SN_C000
- US_C000
- US_C001
- US_C010
- US_C011
- US_C012
- US_C014
- US_C016
- US_R001
- US_R004
- US_R005
- US_R006
- US_R007
- VN_C002
- VN_C003
"


# ~/code/watch/dev/poc/prepare_time_combined_dataset.py
DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
TRUTH_DPATH=$DVC_DATA_DPATH/annotations/drop7-hard-v1

python -m geowatch.cli.queue_cli.prepare_time_combined_dataset \
    --regions="
    $REGION_IDS" \
    --reproject=False \
    --input_bundle_dpath="$DVC_DATA_DPATH"/Aligned-Drop7 \
    --output_bundle_dpath="$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-V2 \
    --spatial_tile_size=256 \
    --merge_method=median \
    --remove_seasons=winter \
    --mask_low_quality=True \
    --tmux_workers=4 \
    --time_window=1y \
    --combine_workers=4 \
    --resolution=10GSD \
    --backend=tmux \
    --skip_existing=1 \
    --cache=1 \
    --run=0 --print-commands

DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
cd "$DVC_DATA_DPATH"
sdvc add \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-V2/*/raw_bands \
    "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-V2/*/imgonly-*-rawbands.kwcoco.zip


REGION_ID_ARR=($(python -c "from kwutil.util_yaml import Yaml; import sys; print(' '.join(Yaml.coerce(sys.argv[1])))" "$REGION_IDS"))
for REGION_ID in "${REGION_ID_ARR[@]}"; do
    echo "REGION_ID = $REGION_ID"
done

python -m cmd_queue new "reproject_for_bas"
for REGION_ID in "${REGION_ID_ARR[@]}"; do
    python -m cmd_queue submit --jobname="reproject-$REGION_ID" -- reproject_for_bas \
        geowatch reproject_annotations \
            --src "$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/$REGION_ID/imgonly-$REGION_ID-rawbands.kwcoco.zip" \
            --dst "$DVC_DATA_DPATH/Drop7-MedianNoWinter10GSD-V2/$REGION_ID/imganns-$REGION_ID-rawbands.kwcoco.zip" \
            --io_workers="avail/2" \
            --region_models="$TRUTH_DPATH/region_models/${REGION_ID}.geojson" \
            --site_models="$TRUTH_DPATH/site_models/${REGION_ID}_*.geojson"
done
python -m cmd_queue show "reproject_for_bas"
python -m cmd_queue run --workers=8 "reproject_for_bas"


DVC_DATA_DPATH=$(geowatch_dvc --tags=phase2_data --hardware="hdd")
python -m geowatch.cli.prepare_splits \
    --src_kwcocos "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-V2/*/imganns*-rawbands.kwcoco.zip \
    --dst_dpath "$DVC_DATA_DPATH"/Drop7-MedianNoWinter10GSD-V2 \
    --suffix=rawbands \
    --backend=tmux --tmux_workers=6 \
    --splits split6 \
    --run=1


DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=hdd)
cd "$DVC_DATA_DPATH"
dvc add Drop7-MedianNoWinter10GSD-V2/*.kwcoco.zip Drop7-MedianNoWinter10GSD-V2/*/imganns-*-rawbands.kwcoco.zip -vvv

git commit -am "Add annotations and split6 to Drop7-MedianNoWinter10GSD-V2"
