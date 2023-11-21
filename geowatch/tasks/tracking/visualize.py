import kwimage
import kwcoco
import numpy as np
import pandas as pd
import itertools
from geowatch.heuristics import CNAMES_DCT


def visualize_videos(pred_dset,
                     out_dir,
                     true_dset=None):
    bas_mode = NotImplemented
    if bas_mode:
        keys = keys_to_score_bas
        # draw videos (regions) separately
        pass
    else:
        keys = keys_to_score_sc
        # draw videos (sites) together within a region
    from geowatch.cli import coco_visualize_videos

    def add_panoptic_img(pred_dset, true_dset):
        '''
        handle cross product of these keys:
Active Construction                         4500
No Activity                                  713
Post Construction                          13179
Site Preparation                            1042
Unknown                                     1499
background                                     0
ignore                                     20755
negative                                    7455
positive                                    1287

        and decide when to merge sites into regions
        '''

        # TODO
        return []

    if true_dset is not None:
        pan_key = add_panoptic_img(pred_dset, true_dset)
        keys += pan_key

    coco_visualize_videos(
        src=pred_dset,
        space='video',
        viz_dpath=out_dir,
        channels=keys,
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
    import geowatch
    import kwplot
    from matplotlib.collections import LineCollection
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
        assert len(annots) > 0
        scores = annots.lookup('scores')
        tid = annots.lookup('track_id')
        dates = pd.to_datetime(annots.images.lookup('date_captured')).date
        sens = annots.images.lookup('sensor_coarse')
    except (KeyError, AssertionError) as e:
        print('cannot viz tracks ', e)
        return

    df = pd.DataFrame(dict(date=dates, sens=sens, tid=tid)).join(pd.DataFrame.from_records(scores))
    df['No Activity'] = 1 - df[keys.as_list()].sum(axis=1)
    ordered_phases = ['No Activity'] + keys.as_list()
    df['orig'] = df[ordered_phases].idxmax(axis=1)
    df['y'] = df['orig'].map(dict(zip(ordered_phases, np.linspace(0, 1, len(ordered_phases)))))

    palette = {c['name']: c['color'] for c in geowatch.heuristics.CATEGORIES}
    palette['salient'] = geowatch.heuristics.CATEGORIES_DCT['positive']['unscored'][0]['color']

    # fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    # ax2.stackplot(df['date'], df[ordered_phases].T, labels=ordered_phases, colors=[palette[p] for p in ordered_phases])
    # ax2.legend()
    # ax1.plot(true_dates, true_labels, label='true')
    # ax1.plot(df['date'], df['orig'], label='orig')

    def add_scores(x, *ordered_phases_cols, **kwargs):
        ax = plt.gca()
        ax.stackplot(x, pd.DataFrame(ordered_phases_cols).values, labels=ordered_phases, colors=[palette[p] for p in ordered_phases])

    def add_colored_linesegments(x, y, phase, **kwargs):
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
