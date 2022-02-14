import kwimage
import kwcoco
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import shapely.ops
import pandas as pd
from collections import defaultdict
from watch.tasks.tracking.utils import pop_tracks, heatmaps
from watch.heuristics import CNAMES_DCT
from watch.tasks.tracking.from_heatmap import mean_normalized

def get_rgb(dset, gid):
    coco_img = dset.coco_image(gid)
    r = coco_img.delay('red', space='video').finalize()
    g = coco_img.delay('green', space='video').finalize()
    b = coco_img.delay('blue', space='video').finalize()
    rgb = np.concatenate((r, g, b), axis=2)
    rgb = kwimage.normalize_intensity(rgb)
    return rgb


def get_pred_seg(dset, gid, shape, render_track_id=False):
    img = dset.index.imgs[gid]
    pred_canvas = np.zeros(shape, dtype=np.uint8)

    pred_anns = dset.annots(gid=gid)
    pred_dets = dset.annots(gid=gid).detections

    vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
    pred_dets = pred_dets.warp(vid_from_img)

    aids = pred_anns.aids

    # filter annotations and only keep ones with 'track_id'
    track_ids = []
    valid_aids = []
    valid_segmentations = []
    for i, aid in enumerate(aids):
        if 'track_id' in dset.anns[aid]:
            # print('track id', dset.anns[aid]['track_id'])
            track_ids.append(dset.anns[aid]['track_id'])
            valid_aids.append(aid)
            valid_segmentations.append(pred_dets.data['segmentations'][i])
        else:
            print('skipping this ann')

    for i, pred_sseg in enumerate(valid_segmentations):
        track_now = track_ids[i]
        # print('track id', track_now)

        render_value = track_now if render_track_id else 1
        pred_canvas = pred_sseg.fill(pred_canvas, value=render_value)

    return pred_canvas


def get_gt_seg(dset, gid, shape, render_track_id=False):
    img = dset.index.imgs[gid]

    # Create a truth "panoptic segmentation" style mask
    true_canvas = np.zeros(shape, dtype=np.uint8)
    true_dets = dset.annots(gid=gid).detections

    vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
    true_dets = true_dets.warp(vid_from_img)

    true_anns = dset.annots(gid=gid)
    aids = true_anns.aids
    track_ids = [dset.anns[aid]['track_id'] for aid in aids]
    for i, true_sseg in enumerate(true_dets.data['segmentations']):
        track_now = track_ids[i]
        render_value = track_now if render_track_id else 1

        true_canvas = true_sseg.fill(true_canvas, value=render_value)

    return true_canvas


def render_pred_gt(pred_canvas, gt_canvas):
    # assumes both canvases to be binary
    # output color coding:
    # TN=white(R=1,G=1,B=1), TP=Green(R=0,G=1,B=0), FN=Yellow(R=1,G=1,B=0), FP=Red(R=1,G=0,B=0)
    out_canvas = np.zeros((pred_canvas.shape[0], pred_canvas.shape[1], 3))

    tn = (gt_canvas == 0) & (pred_canvas == 0)
    tp = (gt_canvas == 1) & (pred_canvas == 1)
    # fn = (gt_canvas == 1) & (pred_canvas == 0)
    fp = (gt_canvas == 0) & (pred_canvas == 1)

    # R
    out_canvas[:, :, 0] = np.clip(1 - tp, a_min=0, a_max=1)

    # G
    out_canvas[:, :, 1] = np.clip(1 - fp, a_min=0, a_max=1)

    # B
    out_canvas[:, :, 2] = np.clip(tn, a_min=0, a_max=1)

    return out_canvas


def get_heatmap(dset, gid, key):
    coco_img = dset.coco_image(gid)
    heatmap = coco_img.delay(key, space='video').finalize()
    if len(heatmap.shape) == 3:
        heatmap = heatmap[:, :, 0]
    return heatmap


def visualize_videos(pred_dset,
                     true_dset,
                     out_dir='./_assets',
                     hide_axis=False,
                     coco_dset_sc=None):
    os.makedirs(out_dir, exist_ok=True)
    for vidid, _ in pred_dset.index.videos.items():
        gids = pred_dset.index.vidid_to_gids[vidid]

        # save average heatmaps
        _heatmaps = heatmaps(pred_dset,
                             gids, {'fg': 'salient'},
                             skipped='interpolate')['fg']

        mean_heatmap = np.array(_heatmaps).mean(0)
        plt.figure()
        plt.imshow(mean_heatmap, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, 'average_heatmap_' + str(vidid) + '.jpg'))
        plt.close()

        # visualize pred and GT polygons
        shape = (pred_dset.index.videos[vidid]['height'], pred_dset.index.videos[vidid]['width'])

        n_images_to_visualize = 8
        sample_spacing = len(gids) // n_images_to_visualize
        gid_list = np.arange(start=0, stop=len(gids), step=sample_spacing)

        plt.figure(figsize=(20, 5))
        for j in range(n_images_to_visualize):
            plt.subplot(2, n_images_to_visualize, j + 1)
            pred_canvas = get_pred_seg(pred_dset, gids[gid_list[j]], shape)
            if true_dset is not None:
                gt_canvas = get_gt_seg(true_dset, gids[gid_list[j]], shape)
                coded_canvas = render_pred_gt(pred_canvas, gt_canvas)
            else:
                coded_canvas = pred_canvas
            plt.imshow(coded_canvas, interpolation='nearest')
            plt.title('image:' + str(gid_list[j]))
            if hide_axis:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

            # RGB
            plt.subplot(2, n_images_to_visualize,
                        j + 1 + n_images_to_visualize)
            rgb = get_rgb(pred_dset, gids[gid_list[j]])
            plt.imshow(rgb)
            if hide_axis:
                ax = plt.gca()
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)

        fname = out_dir + '/video_' + str(vidid) + '_tracks.jpg'
        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

        # visualize threhsolds
        thresholds = np.linspace(start=0.15, stop=0.4, num=5)
        plt.figure(figsize=(20, 15))
        for j in range(n_images_to_visualize):
            plt.subplot(6, n_images_to_visualize, j + 1)
            pred_canvas = get_heatmap(pred_dset, gids[gid_list[j]], key='salient') > thresholds[0]
            if true_dset is not None:
                gt_canvas = get_gt_seg(true_dset, gid=gids[gid_list[j]], shape=shape)
                coded_canvas = render_pred_gt(pred_canvas, gt_canvas)
            else:
                coded_canvas = pred_canvas

            plt.imshow(coded_canvas, interpolation='nearest')
            plt.title('image:' + str(gid_list[j]))

            if (n_images_to_visualize / (j + 1)) == 2:
                plt.title('image :' + str(gid_list[j]) + ', threshold' + '{0:.2f}'.format(thresholds[0]))

            plt.subplot(6, n_images_to_visualize, j + 1 + n_images_to_visualize)
            pred_canvas = get_heatmap(pred_dset, gids[gid_list[j]], key='salient') > thresholds[1]
            if true_dset is not None:
                gt_canvas = get_gt_seg(true_dset, gid=gids[gid_list[j]], shape=shape)
                coded_canvas = render_pred_gt(pred_canvas, gt_canvas)
            else:
                coded_canvas = pred_canvas
            plt.imshow(coded_canvas, interpolation='nearest')

            if (n_images_to_visualize / (j + 1)) == 2:
                plt.title('image:' + str(gid_list[j]) + ', threshold' + '{0:.2f}'.format(thresholds[1]))

            plt.subplot(6, n_images_to_visualize, j + 1 + 2 * n_images_to_visualize)
            pred_canvas = get_heatmap(pred_dset, gids[gid_list[j]], key='salient') > thresholds[2]
            if true_dset is not None:
                gt_canvas = get_gt_seg(true_dset, gid=gids[gid_list[j]], shape=shape)
                coded_canvas = render_pred_gt(pred_canvas, gt_canvas)
            else:
                coded_canvas = pred_canvas
            plt.imshow(coded_canvas, interpolation='nearest')

            if (n_images_to_visualize / (j + 1)) == 2:
                plt.title('image:' + str(gid_list[j]) + ', threshold' + '{0:.2f}'.format(thresholds[2]))

            plt.subplot(6, n_images_to_visualize, j + 1 + 3 * n_images_to_visualize)
            pred_canvas = get_heatmap(pred_dset, gids[gid_list[j]], key='salient') > thresholds[3]
            if true_dset is not None:
                gt_canvas = get_gt_seg(true_dset, gid=gids[gid_list[j]], shape=shape)
                coded_canvas = render_pred_gt(pred_canvas, gt_canvas)
            else:
                coded_canvas = pred_canvas
            plt.imshow(coded_canvas, interpolation='nearest')

            if (n_images_to_visualize / (j + 1)) == 2:
                plt.title('image:' + str(gid_list[j]) + ', threshold' + '{0:.2f}'.format(thresholds[3]))

            plt.subplot(6, n_images_to_visualize, j + 1 + 4 * n_images_to_visualize)
            pred_canvas = get_heatmap(pred_dset, gids[gid_list[j]], key='salient') > thresholds[4]
            if true_dset is not None:
                gt_canvas = get_gt_seg(true_dset, gid=gids[gid_list[j]], shape=shape)
                coded_canvas = render_pred_gt(pred_canvas, gt_canvas)
            else:
                coded_canvas = pred_canvas
            plt.imshow(coded_canvas, interpolation='nearest')

            if (n_images_to_visualize / (j + 1)) == 2:
                plt.title('image:' + str(gid_list[j]) + ', threshold' + '{0:.2f}'.format(thresholds[4]))

            # RGB
            plt.subplot(6, n_images_to_visualize, j + 1 + 5 * n_images_to_visualize)
            rgb = get_rgb(pred_dset, gids[gid_list[j]])
            plt.imshow(rgb)

        fname = out_dir + '/video_' + str(vidid) + 'thresholds.jpg'
        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight')

        # visualize SC heatmaps
        if coco_dset_sc is not None:
            plt.figure(figsize=(20, 15))
            for j in range(n_images_to_visualize):
                plt.subplot(6, n_images_to_visualize, j + 1)
                keys = [
                    'Site Preparation', 'Active Construction',
                    'Post Construction', 'No Activity'
                ]
                heatmap = get_heatmap(coco_dset_sc, gids[gid_list[j]], keys[0])
                plt.imshow(heatmap, vmin=0, vmax=1)
                if j == 3:
                    plt.title('image:' + str(gid_list[j]) + ' Prep')
                else:
                    plt.title('image:' + str(gid_list[j]))

                plt.subplot(6, n_images_to_visualize,
                            n_images_to_visualize + j + 1)
                heatmap = get_heatmap(coco_dset_sc, gids[gid_list[j]], keys[1])
                plt.imshow(heatmap, vmin=0, vmax=1)
                if j == 3:
                    plt.title('Active')

                plt.subplot(6, n_images_to_visualize,
                            2 * n_images_to_visualize + j + 1)
                heatmap = get_heatmap(coco_dset_sc, gids[gid_list[j]], keys[2])
                plt.imshow(heatmap, vmin=0, vmax=1)
                if j == 3:
                    plt.title('Post')

                plt.subplot(6, n_images_to_visualize,
                            3 * n_images_to_visualize + j + 1)
                heatmap = get_heatmap(coco_dset_sc, gids[gid_list[j]], keys[3])
                plt.imshow(heatmap, vmin=0, vmax=1)
                if j == 3:
                    plt.title('No actvty')

                plt.subplot(6, n_images_to_visualize, 4 * n_images_to_visualize + j + 1)
                pred_canvas = get_pred_seg(pred_dset, gids[gid_list[j]], shape)
                plt.imshow(pred_canvas, vmin=0, vmax=1)
                if j == 3:
                    plt.title('Pred polygons')

                if hide_axis:
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)

                # RGB
                plt.subplot(6, n_images_to_visualize,
                            j + 1 + 5 * n_images_to_visualize)
                rgb = get_rgb(pred_dset, gids[gid_list[j]])
                plt.imshow(rgb)
                if hide_axis:
                    ax = plt.gca()
                    ax.axes.xaxis.set_visible(False)
                    ax.axes.yaxis.set_visible(False)

            fname = out_dir + '/video_' + str(vidid) + '_sc_heatmaps.jpg'
            plt.tight_layout()
            plt.savefig(fname, bbox_inches='tight')
            plt.close()

        # Save class predictions of every track
        track_ids = list(pred_dset.index.trackid_to_aids.keys())
        track_labels = {}
        for tid in track_ids:
            track_labels[tid] = []
        for gid in gids:
            pred_anns = pred_dset.annots(gid=gid)
            aids = pred_anns.aids
            for i, aid in enumerate(aids):
                ann = pred_dset.anns[aid]
                tid = ann['track_id']
                track_labels[tid].append(ann['category_id'])

        # save track visualization
        out_dir_track = os.path.join(out_dir, 'track_viz')
        os.makedirs(out_dir_track, exist_ok=True)

        for tid in track_ids:
            plt.figure()
            plt.plot(track_labels[tid])
            plt.xlabel('images')
            plt.ylabel('class ID')
            fname = os.path.join(out_dir_track, str(tid) + '_track.jpg')
            plt.savefig(fname)
            plt.close()


keys_to_score_bas = kwcoco.ChannelSpec('salient')
keys_to_score_sc = kwcoco.ChannelSpec(
    '|'.join(CNAMES_DCT['positive']['scored']))


def viz_track_scores(coco_dset, track_cats, keys_to_score, out_pth):
    '''
    Example:
        >>> # xdoctest: +SKIP
        >>> import kwcoco as kc
        >>> from watch.cli.kwcoco_to_geojson import add_site_summary_to_kwcoco
        >>> from watch.heuristics import SITE_SUMMARY_CNAME
        >>> from watch.tasks.tracking.visualize import (
        >>>     viz_track_scores, keys_to_score_sc)
        >>> d = kc.CocoDataset(
        >>>     '/home/local/KHQ/matthew.bernstein/smart/data/smart_watch_dvc/'
        >>>     'Drop1-Aligned-TA1-2022-01/sc.kwcoco.json')
        >>> d = d.subset(d.index.vidid_to_gids[6])
        >>> kr_r002_path = (
        >>>     '/home/local/KHQ/matthew.bernstein/smart/data/smart_watch_dvc/'
        >>>     'annotations/region_models/KR_R002.geojson')
        >>> d = add_site_summary_to_kwcoco(kr_r002_path, d)
        >>> viz_track_scores(d, [SITE_SUMMARY_CNAME], keys_to_score_sc)
    '''
    def _score_track(track, gids=None):
        if gids is None:  # include only gids in track
            gids = [obs.gid for obs in track.observations]
        # dct = defaultdict(lambda: 0)
        dct = defaultdict(lambda: pd.NA)
        for obs in track.observations:
            dct[obs.gid] = obs.score
        return [dct[g] for g in gids]

    def _area_track(track):
        return shapely.ops.unary_union(
            [obs.poly.to_shapely() for obs in track.observations]).area

    def _img_to_poly(img):
        return kwimage.Boxes([[0, 0, *img.shape[::-1]]],
                             'xywh').to_polygons()[0]

    known_gids = set()  # hack to dedupe from melt()

    def _add_lines(gids, sensor_coarses, **kwargs):
        colors = dict(zip(['S2', 'L8', 'WV'], kwimage.Color.distinct(3)))
        nonlocal known_gids  # hack to dedupe from melt()
        for gid, sensor_coarse in zip(gids, sensor_coarses):
            if gid not in known_gids:
                known_gids.add(gid)
                plt.axvspan(gid - 0.45,
                            gid + 0.45,
                            color=colors[sensor_coarse],
                            alpha=0.1)

    dfs = []
    for i, (vidid, gids) in enumerate(coco_dset.index.vidid_to_gids.items()):
        vid_name = coco_dset.index.videos[vidid]['name']
        sub_dset = coco_dset.subset(gids, copy=True)
        tracks = pop_tracks(sub_dset,
                            cnames=track_cats,
                            score_chan=keys_to_score,
                            remove=False)
        df = pd.DataFrame({
            'gid':
            gids,
            'vid_name':
            vid_name,
            'sensor_coarse':
            coco_dset.images(gids).get('sensor_coarse'),
            **{
                f'track {i:3} area={_area_track(t):.2f}': _score_track(
                    t, gids)
                for i, t in enumerate(tracks)
            },
        })
        dfs.append(df)

    sns.set_context('paper')

    df = pd.concat(dfs)
    df = pd.melt(df, ['gid', 'vid_name', 'sensor_coarse']).dropna()
    g = sns.FacetGrid(df,
                      row='vid_name',
                      hue='variable',
                      aspect=3,
                      sharex=False)
    g = g.map(sns.lineplot, 'gid', 'value').add_legend()
    g = g.map(_add_lines, 'gid', 'sensor_coarse')

    plt.yscale('log')
    plt.ylabel('mean foreground response')
    plt.ylim(1e-10, 5)

    plt.savefig(out_pth)
