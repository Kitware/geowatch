import kwimage
import kwcoco
import numpy as np
import os
import ubelt as ub
import shapely.ops
import pandas as pd
import itertools
from collections import defaultdict
from watch.tasks.tracking.utils import pop_tracks
from watch.heuristics import CNAMES_DCT
from watch.tasks.tracking.from_heatmap import mean_normalized  # NOQA


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
                     out_dir,
                     hide_axis=False,
                     coco_dset_sc=None):
    import matplotlib.pyplot as plt
    for vidid, _ in pred_dset.index.videos.items():
        gids = pred_dset.index.vidid_to_gids[vidid]

        # save average heatmaps
        """
        _heatmaps = build_heatmaps(
            pred_dset, gids, {'fg': 'salient'}, skipped='interpolate')['fg']

        mean_heatmap = np.array(_heatmaps).mean(0)
        plt.figure()
        plt.imshow(mean_heatmap, vmin=0, vmax=1)
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, 'average_heatmap_' + str(vidid) + '.jpg'))
        plt.close()
        """

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

        fname = os.path.join(out_dir, 'video_' + str(vidid) + '_tracks.jpg')
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

        fname = os.path.join(out_dir, 'video_' + str(vidid) + 'thresholds.jpg')
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

            fname = os.path.join(out_dir, 'video_' + str(vidid) + '_sc_heatmaps.jpg')
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
            fname = os.path.join(out_dir_track, str(vidid) + '_' + str(tid) + '_track.jpg')
            plt.savefig(fname)
            plt.close()


def visualize_videos2(pred_dset,
                      true_dset,
                      out_dir):
    bas_mode = NotImplemented
    if bas_mode:
        keys = keys_to_score_bas
        # draw videos (regions) separately
        pass
    else:
        keys = keys_to_score_sc
        # draw videos (sites) together within a region
    from watch.cli import coco_visualize_videos

    def add_panoptic_img(pred_dset, true_dset):
        pass
    pan_key = add_panoptic_img(pred_dset, true_dset)
    coco_visualize_videos(
        src=pred_dset,
        space='video',
        viz_dpath=out_dir,
        channels=keys + pan_key,
        any3=True,
        draw_anns=False,
        animate=True,
        zoom_to_tracks=False,
        stack=True,
    )


keys_to_score_bas = kwcoco.FusedChannelSpec.coerce('salient')
keys_to_score_sc = kwcoco.FusedChannelSpec.coerce(
    '|'.join(CNAMES_DCT['positive']['scored']))

# def chans_intersect(c1: kwcoco.ChannelSpec, c2: kwcoco.ChannelSpec) -> kwcoco.FusedChannelSpec:

def are_bas_dct(dset):
    '''
    This isn't needed because BAS annots will get normalized to SC anyway.

    Assumes:
        - every image is in a video
        - every video has either only BAS tracks or only SC tracks
    Returns:
        Dict[video_id, True if BAS else SC]
    '''
    bas_cnames = keys_to_score_bas.code_list().to_set()
    sc_cnames = keys_to_score_sc.code_list().to_set()
    vids = dset.videos()
    # vid_names = vids.lookup('name')  # match on region/site
    # are_region = [p['type'] == 'region' for p in vids.lookup('properties')]
    # region = [p['region_id'] for p in vids.lookup('properties')]
    are_bas = []
    for images in vids.images:
        cnames = set(itertools.chain.from_iterable(a.cnames for a in images.annots))
        is_bas = cnames.issubset(bas_cnames)
        is_sc = cnames.issubset(sc_cnames)
        if is_bas == is_sc:
            print('WARNING: multiple or unknown track types in video!')
        are_bas.append(is_bas)
    return dict(zip(vids.lookup('id'), are_bas))


def viz_track_scores(dset, out_fpath, gt_dset=None):
    # import json
    import watch
    import kwplot
    import datetime
    from matplotlib.collections import LineCollection
    from matplotlib.dates import date2num
    from matplotlib.colors import to_rgba
    plt = kwplot.autoplt()
    sns = kwplot.autosns()

    # choose img channels to score
    are_bas_imgs = []
    are_sc_imgs = []
    for i in dset.images().coco_images:
        f = i.channels.fuse()
        are_bas_imgs.append(f.intersection(keys_to_score_bas).numel() == keys_to_score_bas.numel())
        are_sc_imgs.append(f.intersection(keys_to_score_sc).numel() == keys_to_score_sc.numel())
    assert (sum(are_bas_imgs) > 0 or sum(are_sc_imgs) > 0), 'no valid channels to score!'
    keys = (keys_to_score_bas if sum(are_bas_imgs) > sum(are_sc_imgs) else keys_to_score_sc)


    if gt_dset is not None:
        # have parallel keys to 'orig' for gt and post-vit
        NotImplemented
    # true_feats = json.load(open(f'gt_site_models/{track_id}.geojson'))['features'][1:]
    # true_labels = [f['properties']['current_phase'] for f in true_feats]
    # true_dates = [f['properties']['observation_date'] for f in true_feats]
    # true_dates = pd.to_datetime(true_dates).date

    # detailed viz
    annots = dset.annots()
    try:
        scores = annots.lookup('scores')
        tid = annots.lookup('track_id')
        dates = pd.to_datetime(annots.images.lookup('date_captured')).date
        sens = annots.images.lookup('sensor_coarse')
    except KeyError as e:
        print('cannot viz tracks ', e)
        return

    df = pd.DataFrame(dict(date=dates, sens=sens, tid=tid)).join(pd.DataFrame.from_records(scores))
    df['No Activity'] = 1 - df[keys.as_list()]
    ordered_phases = ['No Activity'] + keys.as_list()
    df['orig'] = df[ordered_phases].idxmax(axis=1)
    df['y'] = df['orig'].map(dict(zip(ordered_phases, np.linspace(0, 1, len(ordered_phases)))))

    palette = {c['name']: c['color'] for c in watch.heuristics.CATEGORIES}
    palette['salient'] = watch.heuristics.CATEGORIES_DCT['positive']['unscored'][0]['color']

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # ax2.stackplot(df['date'], df[ordered_phases].T, labels=ordered_phases, colors=[palette[p] for p in ordered_phases])
    # ax2.legend()
    # ax1.plot(true_dates, true_labels, label='true')
    # ax1.plot(df['date'], df['orig'], label='orig')

    def add_scores(x, *ordered_phases_cols, **kwargs):
        # if isinstance(x.iloc[0], datetime.date):
            # x = date2num(x)

        ax = plt.gca()
        ax.stackplot(x, pd.DataFrame(ordered_phases_cols).values, labels=ordered_phases, colors=[palette[p] for p in ordered_phases])

    def add_colored_linesegments(x, y, phase, **kwargs):
        # if isinstance(x.iloc[0], datetime.date):
            # x = date2num(x)

        _df = pd.DataFrame(dict(
            xy=zip(x, y),
            hue=pd.Series(phase).map(palette).astype('string').map(to_rgba),
            phase=phase,  # need to keep this due to float comparisons in searchsorted
        ))

        lines = []
        colors = []
        # drop consecutive dups
        ph = _df['phase']
        ixs = ph.loc[ph.shift() != ph].index.values
        for start, end, hue in zip(
                ixs,
                ixs[1:].tolist() + [None],
                _df['hue'].loc[ixs]
        ):
            line = _df['xy'].loc[start:end].values.tolist()
            if len(line) > 0:
                lines.append(line)
                colors.append(hue)

        lc = LineCollection(lines, alpha=1.0, colors=colors)
        ax = plt.gca()
        ax.add_collection(lc)

    def add_ticks(xs, sensor_coarses, **kwargs):
        # if isinstance(xs.iloc[0], datetime.date):
            # xs = date2num(xs)
        colors = dict(zip(['Sentinel-2', 'Landsat 8', 'WorldView'], kwimage.Color.distinct(3)))
        colors['S2'] = colors['Sentinel-2']
        colors['L8'] = colors['Landsat 8']
        colors['WW'] = colors['WorldView']
        ax = plt.gca()
        for x, sensor_coarse in zip(xs, sensor_coarses):
            plt.axvline(x,
                        ymin=0.8,
                        color=colors[sensor_coarse],
                        alpha=0.1)
        ax.legend()

    g = sns.FacetGrid(df,
                      col='tid',
                      aspect=3,
                      col_wrap=4,
                      sharex=False)
    g = g.map(add_scores, 'date', *ordered_phases)
    # TODO figure out why these aren't showing up
    # g = g.map(sns.scatterplot, 'date', 'y', 'orig', palette=palette, s=10)
    # g = g.map(add_colored_linesegments, 'date', 'y', 'orig')
    # g = g.map(add_ticks, 'date', 'sens')
    g = g.set(ylim=(0, 1), ylabel='score')
    g = g.add_legend()

    # summary viz from run_metrics_framework

    # from phase import viterbi

    g.savefig(out_fpath)
    plt.close()
