

def animate_visualizations(viz_dpath, channels=None, video_names=None,
                           frames_per_second=0.7, draw_anns=True,
                           draw_imgs=True, workers=0, zoom_to_tracks=False,
                           verbose=0):
    r"""
    Helper that roughly does the same thing as this bash script:

    Args:
        viz_dpath (str): the path where visualizations were dumped with the
            coco_visualize_videos script.

        zoom_to_tracks (bool):
            if specified uses "track" based-logic find paths to animate

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(--ffmpeg-test')
        >>> import ubelt as ub
        >>> dpath = ub.ensure_app_cache_dir('watch/test/ani_video')
        >>> ub.delete(dpath)
        >>> ub.ensuredir(dpath)
        >>> import kwcoco
        >>> from watch.utils import kwcoco_extensions
        >>> dset = kwcoco.CocoDataset.demo('vidshapes2-msi', num_frames=5)
        >>> img = dset.dataset['images'][0]
        >>> coco_img = dset.coco_image(img['id'])
        >>> channel_chunks = list(ub.chunks(coco_img.channels.fuse().parsed, chunksize=3))
        >>> channels = ','.join(['|'.join(p) for p in channel_chunks])
        >>> kwargs = {
        >>>     'src': dset.fpath,
        >>>     'viz_dpath': dpath,
        >>>     'space': 'video',
        >>>     'channels': channels,
        >>>     'zoom_to_tracks': False,
        >>> }
        >>> from watch.cli.coco_visualize_videos import main
        >>> cmdline = False
        >>> main(cmdline=cmdline, **kwargs)
        >>> viz_dpath = dpath
        >>> channels = None
        >>> video_names = None
        >>> frame_per_second = 0.7
        >>> from watch.cli.animate_visualizations import *  # NOQA
        >>> animate_visualizations(viz_dpath, verbose=1, workers=0)
    """
    import pathlib
    from watch.cli import gifify
    import ubelt as ub
    import kwcoco
    from watch.utils.lightning_ext import util_globals

    if channels is not None:
        channels = kwcoco.ChannelSpec.coerce(channels)

    workers = util_globals.coerce_num_workers(workers)
    viz_dpath = pathlib.Path(viz_dpath)

    if video_names is None:
        video_dpaths = [p for p in viz_dpath.glob('*') if p.is_dir()]
    else:
        video_dpaths = [viz_dpath / n for n in video_names]

    pool = ub.JobPool(mode='thread', max_workers=workers)

    types = []
    if draw_imgs:
        types.append('_imgs')
    if draw_anns:
        types.append('_anns')

    verbose_worker = verbose and workers <= 1

    # We make heavy reliance on a known directory structure here.
    # In general I don't like this, but this is not a system-critical part
    # so we can leave refactoring as a todo.

    prog = ub.ProgIter(desc='submit video jobs', verbose=3)
    prog.begin()

    for type_ in types:
        for video_dpath in video_dpaths:
            prog.set_extra(f'{type_=!r} {video_dpath=!r}')
            prog.step()
            video_name = video_dpath.name

            if zoom_to_tracks:
                track_subdpath = video_dpath / '_tracks'
                track_dpaths = list(track_subdpath.glob('*'))
                for track_dpath in track_dpaths:
                    track_name = track_dpath.name
                    type_dpath = track_dpath / type_

                    if channels is None:
                        channel_dpaths = [p for p in type_dpath.glob('*') if p.is_dir()]
                    else:
                        def sanatize_chan_pnams(cs):
                            return cs.replace('|', '_').replace(':', '-')
                        channel_dpaths = [type_dpath / sanatize_chan_pnams(c.spec) for c in channels.streams()]

                    for chan_dpath in channel_dpaths:
                        frame_fpaths = sorted(chan_dpath.glob('*'))
                        gif_fname = '{}{}_{}.gif'.format(track_name, type_, chan_dpath.name)
                        gif_fpath = track_subdpath / gif_fname
                        pool.submit(
                            gifify.ffmpeg_animate_frames, frame_fpaths,
                            gif_fpath, in_framerate=frames_per_second,
                            verbose=verbose_worker)

            else:
                type_dpath = video_dpath / type_
                if channels is None:
                    channel_dpaths = [p for p in type_dpath.glob('*') if p.is_dir()]
                else:
                    # channel_dpaths = [type_dpath / c.spec for c in channels.streams()]
                    def sanatize_chan_pnams(cs):
                        return cs.replace('|', '_').replace(':', '-')
                    channel_dpaths = [type_dpath / sanatize_chan_pnams(c.spec) for c in channels.streams()]
                for chan_dpath in channel_dpaths:
                    frame_fpaths = sorted(chan_dpath.glob('*'))
                    gif_fname = '{}{}_{}.gif'.format(video_name, type_, chan_dpath.name)
                    gif_fpath = video_dpath / gif_fname
                    pool.submit(
                        gifify.ffmpeg_animate_frames, frame_fpaths, gif_fpath,
                        in_framerate=frames_per_second, verbose=verbose_worker)

    for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='collect animate jobs'):
        job.result()

    print('Wrote animations to viz_dpath = {!r}'.format(viz_dpath))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/animate_visualizations.py
    """
    import fire
    fire.Fire(animate_visualizations)
