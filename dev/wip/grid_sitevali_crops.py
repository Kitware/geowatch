#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class GridSiteVizConfig(scfg.DataConfig):
    viz_dpath = scfg.Value(None, help='input', position=1)
    sub = '_imgs'


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict()
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = GridSiteVizConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = ' + ub.urepr(config, nl=1))
    import itertools as it
    from watch.cli import gifify
    import kwimage

    viz_dpath = ub.Path(config.viz_dpath)

    track_dpaths = [p for p in viz_dpath.ls() if not p.name.startswith('_')]

    track_to_frames = {}
    for track_dpath in track_dpaths:
        img_dpath = track_dpath / config.sub / 'stack'
        print(f'img_dpath={img_dpath}')
        track_to_frames[track_dpath.name] = sorted(img_dpath.ls())

    out_dpath = viz_dpath.augment(suffix='_gallary')
    out_dpath.ensuredir()

    chunk_rtsize = 5
    chunked_tracks = ub.chunks(track_to_frames, chunksize=chunk_rtsize ** 2)
    for chunk_idx, tracks in enumerate(ub.ProgIter(chunked_tracks, desc='chunks')):
        track_to_frame_ = ub.udict(track_to_frames).subdict(tracks)

        chunk_dpath = (out_dpath / f'chunk_{chunk_idx:03d}').ensuredir()

        track_to_now = {}
        grid_fpaths = []
        for fx, frames_at_time in enumerate(it.zip_longest(*track_to_frame_.values())):
            for track, fpath in zip(track_to_frame_, frames_at_time):
                if fpath is not None:
                    track_to_now[track] = kwimage.imread(fpath)

            frame = kwimage.stack_images_grid(track_to_now.values())
            grid_fpath = chunk_dpath / f'frame_{fx:03d}.jpg'
            grid_fpaths.append(grid_fpath)
            kwimage.imwrite(grid_fpath, frame)

        gif_fpath = out_dpath / f'chunk_{chunk_idx:03d}.gif'
        gifify.ffmpeg_animate_frames(grid_fpaths, gif_fpath)

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/wip/grid_sitevali_crops.py
        python -m grid_sitevali_crops
    """
    main()
