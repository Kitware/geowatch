#!/bin/bash


DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
BAS_HEATMAPS=$DVC_DPATH/models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json

smartwatch stats "$BAS_HEATMAPS"

jq .info "$BAS_HEATMAPS"

python -m watch.cli.kwcoco_to_geojson \
    "$BAS_HEATMAPS" \
    --default_track_fn saliency_heatmaps \
    --clear_annots \
    --out_dir "$RESULT_DIR" \
    --out_fpath "$TRACK_DIR/tracks.json" \
    --track_kwargs "{\"use_viterbi\": false, \"thresh\": 0.1, \"thresh\": 0.1, \"morph_kernel\": 3, \"time_filtering\": true, \"norm_ord\": 1}"


TEST_DATASET=$HOME/data/dvc-repos/smart_watch_dvc-hdd/Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json
PACKAGE_FPATH=$HOME/data/dvc-repos/smart_watch_dvc-hdd/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt

smartwatch visualize "$BAS_HEATMAPS" --channels="cloudmask" --animate=True --workers=8


pyblock "

fpath = '/data/projects/smart/smart_watch_dvc/models/fusion/eval3_candidates/pred/Drop3_SpotCheck_V323/pred_Drop3_SpotCheck_V323_epoch=18-step=12976/Aligned-Drop3-TA1-2022-03-10_combo_LM_nowv_vali.kwcoco/predcfg_abd043ec/pred.kwcoco.json'

import kwplot
plt = kwplot.autoplt()
sns = kwplot.autosns()
sns.set()

import kwcoco
import kwimage
coco_dset = kwcoco.CocoDataset(fpath)

coco_images = coco_dset.images().coco_images

# Loop over each image and record information about it
rows = []
for coco_img in ub.ProgIter(coco_images, desc='iter images'):


    delayed = coco_img.delay('cloudmask', space='auxiliary')
    cloud_im = delayed.finalize(
        nodata='ma',
        interpolation='nearest',
        antialias=False)
    cloud_bits = 1 << np.array([1, 2, 3])
    is_cloud_iffy = np.logical_or.reduce([cloud_im == b for b in cloud_bits])
    is_iffy = is_cloud_iffy | cloud_im.mask

    delayed = coco_img.delay('salient', space='auxiliary')
    salient_im = delayed.finalize(nodata='float')
    valid_saliency = 1 - np.isnan(salient_im)

    row = {
        'gid': coco_img.img['id'],
        'has_salient': 'salient' in coco_img.channels,
        'valid_saliency_percent': valid_saliency.sum() / valid_saliency.size,
        'iffy_percent': is_iffy.sum() / is_iffy.size,
        'cloud_percent': is_cloud_iffy.sum() / is_cloud_iffy.size,
    }
    rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)
df = df.sort_values('iffy_percent')
df['vidid'] = coco_dset.images(df.gid).get('video_id')

for vidid, group in df.groupby('vidid'):
    num_salient = group.has_salient.sum()
    num_no_salient = (~group.has_salient).sum()
    num_total = len(group)
    print(coco_dset.videos([vidid]).get('name'))
    print(f'{num_salient} / {num_total}')
    print(f'{num_no_salient} / {num_total}')

    print((~group.has_salient).sum())
    print(len(group))
    print((group.has_salient).sum())


df2 = df[~df.has_salient & (df.iffy_percent < 0.7)]
index = 5
row = df2.iloc[index].to_dict()
gid = row['gid']
has_salient = row['has_salient']
space = 'image'
coco_img = coco_dset.coco_image(gid)
imdata = coco_img.delay('red|green|blue', space=space).finalize(nodata='float')
cloud_im = coco_img.delay('cloudmask', space=space).finalize(nodata='ma', interpolation='nearest', antialias=False)
salient_img = coco_img.delay('salient', space=space).finalize(nodata='float')
is_cloud_iffy = np.logical_or.reduce([cloud_im == b for b in cloud_bits])

salient_overlay = kwimage.make_heatmask(salient_img[:, :, 0], with_alpha=0.5, cmap='Blues')
cloud_overlay = kwimage.make_heatmask(is_cloud_iffy[:, :, 0], with_alpha=0.5, cmap='Reds')
canvas = kwimage.normalize_intensity(imdata)
canvas = kwimage.fill_nans_with_checkers(canvas)
canvas = kwimage.overlay_alpha_images(cloud_overlay, canvas)
kwplot.imshow(canvas)
ax = plt.gca()
ax.set_title(ub.repr2(row, precision=3, nl=0))


ax = plt.gca()
ax.cla()
sns.scatterplot(data=df, x='iffy_percent', y='valid_saliency_percent', ax=ax, hue='has_salient')

dset_tag = '/'.join(ub.Path(coco_dset.fpath).parts[-4:])

ax = plt.gca()
ax.cla()
ax.plot(sorted_percents, )
ax.set_ylabel('percent cloudcover')
ax.set_xlabel('sorted image index')
ax.set_title(f'Images by Cloudcover in Validation Dataset{chr(10)}{dset_tag}')

ax.plot([len(coco_images) - 120] * 2, [0, 1])
ax.plot([0, len(coco_images)], [sorted_percents[-120], sorted_percents[-120]])


"


HDD_DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
SSD_DVC_DPATH=$(smartwatch_dvc --hardware="ssd")
echo "HDD_DVC_DPATH = $HDD_DVC_DPATH"
echo "SSD_DVC_DPATH = $SSD_DVC_DPATH"
kwcoco stats "$HDD_DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json" "$SSD_DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json"
kwcoco stats "$HDD_DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_vali.kwcoco.json" "$SSD_DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_vali.kwcoco.json"

TMP_DPATH=$HOME/tmp/debug_pred
DVC_DPATH=$(smartwatch_dvc --hardware="ssd")
TEST_DATASET=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/combo_LM_nowv_vali.kwcoco.json
SMALL_TEST_DATASET=$TMP_DPATH/small.kwcoco.json

PRED1_DATASET="$TMP_DPATH/test_pred_dset/pred.kwcoco.json"
PRED2_DATASET="$TMP_DPATH/test_pred_dset2/pred.kwcoco.json"
PRED3_DATASET="$TMP_DPATH/test_pred_dset3/pred.kwcoco.json"
PRED4_DATASET="$TMP_DPATH/test_pred_dset4/pred.kwcoco.json"

mkdir -p "$TMP_DPATH"
kwcoco validate "$TEST_DATASET"
kwcoco subset "$TEST_DATASET" "$SMALL_TEST_DATASET" --select_videos '.name == "KR_R002"'
kwcoco stats "$TEST_DATASET" "$SMALL_TEST_DATASET"
kwcoco validate "$SMALL_TEST_DATASET"


PACKAGE_FPATH=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
python -m watch.tasks.fusion.predict \
    --test_dataset="$SMALL_TEST_DATASET" \
    --package_fpath="$PACKAGE_FPATH" \
    --pred_dataset="$PRED3_DATASET" \
    --workers=4 \
    --write_preds=0 \
    --write_probs=1 \
    --chip_overlap=0 \
    --gpus=0 

kwcoco stats "$TEST_DATASET" "$SMALL_TEST_DATASET" "$PRED2_DATASET" "$PRED3_DATASET"
smartwatch stats "$TEST_DATASET" "$SMALL_TEST_DATASET" "$PRED1_DATASET" "$PRED2_DATASET" "$PRED3_DATASET"

echo "

import kwplot
kwplot.autompl()

from watch.tasks.fusion.predict import *  # NOQA
args = None
cmdline = False
torch.set_grad_enabled(False)
kwargs = {
    'test_dataset': '/home/joncrall/tmp/debug_pred/small.kwcoco.json',
    'package_fpath': '/home/joncrall/data/dvc-repos/smart_watch_dvc-ssd/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt',
    'pred_dataset': '/home/joncrall/tmp/debug_pred/test_pred_dset4/pred.kwcoco.json',
    'workers': 4,
    'write_preds': 0,
    'write_probs': 0,
    'chip_overlap': 0,
    'gpus': 0,
}


gids_seen_by_dataloader = set()
batch_iter = iter(test_dataloader)
prog = ub.ProgIter(batch_iter, desc='predicting', verbose=1)
_batch_iter = iter(prog)
for orig_batch in _batch_iter:
    for item in orig_batch:
        if item is None:
            continue
        batch_gids = [frame['gid'] for frame in item['frames']]
        gids_seen_by_dataloader.update(batch_gids)


dataset = test_dataloader.dataset
all_gids = list(dataset.sampler.dset.images())
missing_gids = list(set(all_gids) - set(gids_seen_by_dataloader))

missed_trs = []
for tr in dataset.new_sample_grid['targets']:
    if tr['main_gid'] in missing_gids:
        missed_trs.append(tr)

 
missed_items = []
for tr in ub.ProgIter(missed_trs, desc='check missing items'):
    item = dataset[tr]
    missed_items.append(item)


gid_to_reason = ub.ddict(list)
gid_to_grids = ub.ddict(list)
for tr, item in zip(missed_trs, missed_items):
    gid = tr['main_gid']
    reason = missed_items[0]['tr']['main_skip_reason']
    gid_to_grids[gid].append(tr['space_slice'])
    gid_to_reason[gid].append(reason)

import xdev
missed_gids = list(gid_to_reason.keys())
coco_dset = dataset.sampler.dset
for gid in xdev.InteractiveIter(missed_gids):
    space = 'video'

    grids = gid_to_grids[gid]
    grid_cells = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(s) for s in grids]).to_polygons()
    coco_img = coco_dset.coco_image(gid)
    imdata = coco_img.delay('red|green|blue', space=space).finalize(nodata='float')
    canvas = kwimage.normalize_intensity(imdata)
    canvas = kwimage.fill_nans_with_checkers(canvas)
    cloud_im = coco_img.delay('cloudmask', space=space).finalize(nodata='ma', interpolation='nearest', antialias=False)
    is_cloud_iffy = np.logical_or.reduce([cloud_im == b for b in cloud_bits])
    cloud_overlay = kwimage.make_heatmask(is_cloud_iffy[:, :, 0], with_alpha=0.4, cmap='Reds')
    canvas = grid_cells.draw_on(canvas, edgecolor='blue', facecolor='white', alpha=0.2, fastdraw=False)
    canvas = kwimage.overlay_alpha_images(cloud_overlay, canvas)
    kwplot.imshow(canvas)
    #canvas = grid_cells.draw(edgecolor='blue', facecolor='yellow', alpha=0.4)
    xdev.InteractiveIter.draw()



gid_to_reason_hist = ub.map_vals(ub.dict_hist, gid_to_reason)

tr = missed_trs[0]
item = dataset[tr]
tr_ = item['tr']

index = tr
self = dataset

missed_gids = set(tr['gids']) - set(tr_['gids'])
extra_gids = set(tr_['gids']) - set(tr['gids'])
"
