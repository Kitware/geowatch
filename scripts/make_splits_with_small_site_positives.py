import cmd_queue
import pathlib
import ubelt as ub

queue = cmd_queue.Queue.create(backend='tmux', size=1)

geojson_annot_dpath = pathlib.Path("/flash/smart_data_dvc/annotations/drop6/")
hdd_bundle_dpath = pathlib.Path("/flash/smart_data_dvc/Drop6/")

imgonly_fpaths = list(hdd_bundle_dpath.glob('imgonly*.kwcoco.*'))
for imgonly_fpath in imgonly_fpaths:
    region_id = imgonly_fpath.name.split('.')[0].split('-')[1]
    region_id = region_id
    imgonly_fpath = imgonly_fpath
    imganns_fpath = imgonly_fpath.parent / f"imganns-{region_id}_wsmall.kwcoco.zip"

    assert imgonly_fpath.exists()
    command = ub.codeblock(
        fr'''
        python -m watch reproject_annotations \
            --src "{str(imgonly_fpath)}" \
            --dst "{str(imganns_fpath)}" \
            --propogate_strategy="SMART" \
            --site_models="{geojson_annot_dpath}/site_models/{region_id}_*" \
            --region_models="{geojson_annot_dpath}/region_models/{region_id}*"
        ''')
    print(command)
    queue.submit(command)

# command = ub.codeblock(
# fr'''
# python ~/code/watch/watch/cli/prepare_splits.py \
#     --base_fpath="imganns*.kwcoco.*" \
#     --workers=5 \
#     --constructive_mode=True --run=1
# ''')
# print(command)
# queue.submit(command)

queue.rprint()
queue.run()