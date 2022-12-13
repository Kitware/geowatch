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
    --align_workers=0 \
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

add_dvc_data(){
    # TODO: before doing this, remember to change the bundle name
    DATA_DVC_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=hdd)
    cd "$DATA_DVC_DPATH"
    git add Drop6
    cd "$DATA_DVC_DPATH/Drop6"
    python -m watch.cli.prepare_splits data.kwcoco.json --cache=0 --run=1
    7z a splits.zip data*.kwcoco.json imganns-*.kwcoco.json
    dvc add -- */L8 */S2 */WV *.zip && dvc push -r horologic -R . && git commit -am "Add Drop6 ACC-2" && git push 
}
