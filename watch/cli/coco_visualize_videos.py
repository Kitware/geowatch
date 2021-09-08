import kwimage
import ubelt as ub
import kwcoco
import scriptconfig as scfg


class CocoVisualizeConfig(scfg.Config):
    """
    Visualizes annotations on kwcoco video frames on each band

    TODO:
        - [ ] Could parameterize which bands are displayed if that is useful
        - [ ] Could finalize by creating an animation if we need these for slides
        - [X] Could parallelize with ub.JobPool

    """
    default = {
        'src': scfg.Value('data.kwcoco.json', help='input dataset'),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            Where to save the visualizations. If unspecified,
            writes them adjacent to the input kwcoco file
            ''')),

        'num_workers': scfg.Value(0, help='number of parallel draw jobs'),

        'space': scfg.Value('image', help='can be image or video space'),
    }


def main(cmdline=True, **kwargs):
    """
    Ignore:
        src = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/data.kwcoco.json')
        cmdline = False
        kwargs = {
            'src': src,
        }
    """
    import kwcoco
    import pathlib
    config = CocoVisualizeConfig(default=kwargs, cmdline=cmdline)
    space = config['space']
    print('config = {}'.format(ub.repr2(dict(config), nl=2)))

    coco_dset = kwcoco.CocoDataset.coerce(config['src'])
    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    print('coco_dset = {!r}'.format(coco_dset))

    bundle_dpath = pathlib.Path(coco_dset.bundle_dpath)
    if config['viz_dpath'] is not None:
        viz_dpath = pathlib.Path(config['viz_dpath'])
    else:
        viz_dpath = bundle_dpath / '_viz'

    prog = ub.ProgIter(
        coco_dset.index.videos.items(), total=len(coco_dset.index.videos),
        desc='viz videos', newlines=True)

    pool = ub.JobPool(mode='thread', max_workers=config['num_workers'])

    for vidid, video in prog:
        sub_bundle_dpath = viz_dpath / video['name']
        sub_bundle_dpath.mkdir(parents=True, exist_ok=1)

        gids = coco_dset.index.vidid_to_gids[vidid]
        for gid in gids:
            img = coco_dset.index.imgs[gid]
            anns = coco_dset.annots(gid=gid).objs

            pool.submit(_write_ann_visualizations2,
                        coco_dset, img, anns, sub_bundle_dpath, space=space)

        for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='write vids'):
            job.result()

        pool.jobs.clear()


_CLI = CocoVisualizeConfig


def _write_ann_visualizations2(coco_dset : kwcoco.CocoDataset, img, anns,
                               sub_bundle_dpath, space):
    """
    TODO:
        refactor because similar code is also used in coco_align_geotiffs
    """
    # See if we can look at what we made
    from watch.utils.util_norm import normalize_intensity
    from watch.utils.kwcoco_extensions import CocoImage

    sensor_coarse = img.get('sensor_coarse', 'unknown')
    align_method = img.get('align_method', 'unknown')
    name = img.get('name', 'unknown')

    delayed = coco_dset.delayed_load(img['id'], space=space)

    coco_img = CocoImage(img)
    chan_groups = coco_img.channels.spec.split(',')

    img_view_dpath = sub_bundle_dpath / '_imgs'
    ann_view_dpath = sub_bundle_dpath / '_anns'
    dets = kwimage.Detections.from_coco_annots(anns, dset=coco_dset)

    if space == 'video':
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        dets = dets.warp(vid_from_img)

    for chan_group in chan_groups:
        chan = delayed.take_channels(chan_group)

        # spec = str(chan.channels.spec)
        canvas = chan.finalize()
        canvas = normalize_intensity(canvas)

        if len(canvas.shape) > 2 and canvas.shape[2] > 4:
            # hack for wv
            canvas = canvas[..., 0]

        img_chan_dpath = img_view_dpath / chan_group
        ann_chan_dpath = ann_view_dpath / chan_group

        ann_chan_dpath.mkdir(parents=True, exist_ok=1)
        img_chan_dpath.mkdir(parents=True, exist_ok=1)

        canvas = kwimage.ensure_float01(canvas)

        suffix = '_'.join([chan_group, sensor_coarse, align_method])

        view_img_fpath = ub.augpath(name, dpath=img_chan_dpath) + '_' + suffix + '.view_img.jpg'
        kwimage.imwrite(view_img_fpath, kwimage.ensure_uint255(canvas))

        view_ann_fpath = ub.augpath(name, dpath=ann_chan_dpath) + '_' + suffix + '.view_ann.jpg'
        ann_canvas = dets.draw_on(canvas)
        kwimage.imwrite(view_ann_fpath, kwimage.ensure_uint255(ann_canvas))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/coco_visualize_videos.py
    """
    main(cmdline=True)
