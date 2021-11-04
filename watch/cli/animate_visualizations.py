#!/usr/bin/env python
# -*- coding: utf-8 -*-

__notes__ = r"""

.. :code: bash

    # Make an animated gif for specified bands (use "," to separate)
    # Requires a CD
    CHANNELS="red|green|blue"
    mapfile -td \, _BANDS < <(printf "%s\0" "$CHANNELS")
    items=$(jq -r '.videos[] | .name' $OUTPUT_COCO_FPATH)
    for item in ${items[@]}; do
        echo "item = $item"
        for bandname in ${_BANDS[@]}; do
            echo "_BANDS = $_BANDS"
            BAND_DPATH="$VIZ_DPATH/${item}/_anns/${bandname}/"
            GIF_FPATH="$VIZ_DPATH/${item}_anns_${bandname}.gif"
            python -m watch.cli.gifify --frames_per_second .7 \
                --input "$BAND_DPATH" --output "$GIF_FPATH"
        done
    done

"""


def animate_visualizations(viz_dpath, channels=None, video_names=None,
                           frames_per_second=0.7, draw_anns=True,
                           draw_imgs=True, verbose=0):
    r"""
    Helper that roughly does the same thing as this bash script:

    Args:
        viz_dpath (str): the path where visualizations were dumped with the
            coco_visualize_videos script.

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
        >>> coco_img = kwcoco_extensions.CocoImage(img, dset)
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
    """
    import pathlib
    from watch.cli import gifify
    import kwcoco

    if channels is not None:
        channels = kwcoco.ChannelSpec.coerce(channels)

    viz_dpath = pathlib.Path(viz_dpath)

    if video_names is None:
        video_dpaths = [p for p in viz_dpath.glob('*') if p.is_dir()]
    else:
        video_dpaths = [viz_dpath / n for n in video_names]

    types = []
    if draw_imgs:
        types.append('_imgs')
    if draw_anns:
        types.append('_anns')
    for type_ in types:
        for video_dpath in video_dpaths:
            type_dpath = video_dpath / type_
            if channels is None:
                channel_dpaths = [p for p in type_dpath.glob('*') if p.is_dir()]
            else:
                channel_dpaths = [type_dpath / c.spec for c in channels.streams()]

            for chan_dpath in channel_dpaths:
                frame_fpaths = sorted(chan_dpath.glob('*'))
                gif_fpath = str(video_dpath) + type_ + '_' + chan_dpath.name + '.gif'

                gifify.ffmpeg_animate_frames(
                    frame_fpaths, gif_fpath, in_framerate=frames_per_second,
                    verbose=verbose)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/animate_visualizations.py
    """
    import fire
    fire.Fire(animate_visualizations)
