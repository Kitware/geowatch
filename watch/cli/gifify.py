#!/usr/bin/env python
"""
A gif-ify script

Wrapper around imgmagik convert or ffmpeg

TODO:
    - [ ] Moving this to kwplot
"""

import ubelt as ub
from os.path import isdir, join
import scriptconfig as scfg


class AnimateConfig(scfg.DataConfig):
    """
    Convert a sequence of images into a video or gif.
    """
    image_list = scfg.Value(None, required=True, help=ub.paragraph(
            '''
            list of images (or a text file containing a list of images)
            '''), position=1, nargs='*', alias=['input'])
    delay = scfg.Value(10, type=float, short_alias=['d'], help='delay between frames', nargs=1)
    output = scfg.Value('auto', short_alias=['o'], help=ub.paragraph(
        '''
        Path to the output file. If "auto", then the name will be chosen
        automatically.  If the input is a folder, it will be the folder name
        .mp4 otherwise it will be out.mp4.
        '''))
    max_width = scfg.Value(None, type=int, help='resize to max width')
    frames_per_second = scfg.Value(10, type=float, alias=['fps'], help=ub.paragraph(
        '''
        number of frames per second
        '''))

__config__ = AnimateConfig


def main(cmdline=True, **kwargs):
    import glob
    config = AnimateConfig.legacy(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    image_paths = config['image_list']

    print('Converting:')
    print('image_paths = ' + ub.repr2(image_paths))

    assert image_paths is not None

    auto_outname = 'out.mp4'

    frame_fpaths = []
    for p in image_paths:
        if isdir(p):
            if len(image_paths) == 1:
                auto_outname = ub.Path(p) + '.mp4'
            toadd = sorted(glob.glob(join(p, '*.png')))
            toadd += sorted(glob.glob(join(p, '*.jpg')))
            frame_fpaths.extend(toadd)
        else:
            if str(p).endswith('.txt'):
                with open(p, 'r') as f:
                    lines = list(f.read().split('\n'))
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if line and not line.startswith('#')]
                frame_fpaths.extend(lines)
            else:
                frame_fpaths.append(p)

    if config['output'] == 'auto':
        print(f'Resolved output to {auto_outname}')
        config['output'] = auto_outname

    # frame_fpaths = frame_fpaths[::2]

    print('frame_fpaths = {!r}'.format(frame_fpaths))

    backend = 'imagemagik'
    backend = 'ffmpeg'
    if backend == 'imagemagik':
        escaped_gif_fpath = config['output'].replace('%', '%%')
        command = ['convert', '-delay', str(config['delay']), '-loop', '0']
        command += frame_fpaths
        command += [escaped_gif_fpath]
        # print('command = {!r}'.format(command))
        print('Converting {} images to gif: {}'.format(len(frame_fpaths), escaped_gif_fpath))
        info = ub.cmd(command, verbose=3)
        print('finished')
        if info['ret'] != 0:
            print(info['out'])
            print(info['err'])
            raise RuntimeError(info['err'])
        return info['err']
    elif backend == 'ffmpeg':
        output_fpath = config['output']
        config['delay']
        # config['delay']
        in_framerate = config['frames_per_second']
        ffmpeg_animate_frames(frame_fpaths, output_fpath,
                              in_framerate=in_framerate,
                              max_width=config['max_width'])


def ffmpeg_animate_frames(frame_fpaths, output_fpath, in_framerate=1, verbose=3, max_width=None):
    """
    Use ffmpeg to transform a series of frames into a video.

    Args:
        frame_fpaths (List[PathLike]):
            ordered list of frames to be combined into an animation

        output_fpath (PathLike):
            output video name, as either a gif, avi, mp4, etc.

        in_framerate (int):
            number of input frames per second to use (lower is slower)

    References:
        https://superuser.com/questions/624563/how-to-resize-a-video-to-make-it-smaller-with-ffmpeg

    Example:
        >>> from watch.cli.gifify import *  # NOQA
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('shapes8')
        >>> ffmpeg_exe = ub.find_exe('ffmpeg')
        >>> if ffmpeg_exe is None:
        >>>     import pytest
        >>>     pytest.skip('test requires ffmpeg')
        >>> frame_fpaths = sorted(dset.images().gpath)
        >>> test_dpath = ub.Path.appdir('gifify', 'test').ensuredir()
        >>> # Test output to GIF
        >>> output_fpath = join(test_dpath, 'test.gif')
        >>> ffmpeg_animate_frames(frame_fpaths, output_fpath, in_framerate=0.5)
        >>> # Test number of frames is correct
        >>> from PIL import Image
        >>> pil_gif = Image.open(output_fpath)
        >>> try:
        >>>     while 1:
        >>>         pil_gif.seek(pil_gif.tell()+1)
        >>>         # do something to im
        >>> except EOFError:
        >>>     pass # end of sequence
        >>> assert pil_gif.tell() + 1 == dset.n_images
        >>> # Test output to video
        >>> output_fpath = join(test_dpath, 'test.mp4')
    """
    from os.path import join, abspath
    import uuid
    import os
    output_fpath = os.fspath(output_fpath)

    ffmpeg_exe = ub.find_exe('ffmpeg')
    if ffmpeg_exe is None:
        raise Exception('cannot find ffmpeg')

    try:
        temp_dpath = ub.Path.appdir('gifify', 'temp').ensuredir()
        temp_fpath = join(temp_dpath, 'temp_list_{}.txt'.format(str(uuid.uuid4())))
        if verbose:
            print('temp_fpath = {!r}'.format(temp_fpath))
        lines = ["file '{}'".format(abspath(fpath)) for fpath in frame_fpaths]
        text = '\n'.join(lines)
        with open(temp_fpath, 'w') as file:
            file.write(text + '\n')

        # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
        # evan_pad_option = '-filter:v pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2"'
        # vid_options = '-c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p'

        fmtkw = dict(
            IN=temp_fpath,
            OUT=output_fpath,
        )

        global_options = []
        input_options = [
            '-r {IN_FRAMERATE} ',
            '-f concat -safe 0',
            # '-framerate {IN_FRAMERATE} ',
        ]
        fmtkw.update(dict(
            IN_FRAMERATE=in_framerate,
        ))

        output_options = [
            # '-qscale 0',
            # '-crf 20',
            # '-r {OUT_FRAMERATE}',
            # '-filter:v scale=512:-1',

            # higher quality
            # https://stackoverflow.com/questions/42980663/ffmpeg-high-quality-animated-gif
            # '-filter_complex "fps=10;scale=500:-1:flags=lanczos,palettegen=stats_mode=full"'
            # '-filter_complex "fps=10;scale=500:-1:flags=lanczos,palettegen=stats_mode=full"'
            # '-filter_complex "fps=10;scale=500:-1:flags=lanczos,split[v1][v2]; [v1]palettegen=stats_mode=full [palette];[v2]palette]paletteuse=dither=sierra2_4a" -t 10'
        ]
        fmtkw.update(dict(
            # OUT_FRAMERATE=5,
        ))

        filtergraph_parts = []

        if max_width is not None:
            filtergraph_parts.append(f'scale={max_width}:-1')

        # scale_options.append(
        #     'force_original_aspect_ratio=decrease'
        # )
        # output_options += [
        #     '-vf scale="{}:-1"'.format(max_width)
        # ]

        if output_fpath.endswith('.mp4'):
            filtergraph_parts += [
                'pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2:x=0:y=0',
            ]
            # output_options += [
            #     # MP4 needs even width
            #     # https://stackoverflow.com/questions/20847674/ffmpeg-div2
            #     '-filter:v "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2"',
            # ]

        if filtergraph_parts:
            filtergraph = ','.join(filtergraph_parts)
            output_options += [
                f'-filter:v "{filtergraph}"'
            ]

        cmd_fmt = ' '.join(
            [ffmpeg_exe, '-y'] +
            global_options +
            input_options +
            ['-i {IN}'] +
            output_options +
            ["'{OUT}'"]
        )

        command = cmd_fmt.format(**fmtkw)

        if verbose > 0:
            print('Converting {} images to animation: {}'.format(len(frame_fpaths), output_fpath))

        info = ub.cmd(command, verbose=3 if verbose > 1 else 0, shell=True)

        if verbose > 0:
            print('finished')

    finally:
        pass
        # ub.delete(temp_dpath)

    import sys
    if sys.stdout.isatty():
        ub.cmd('stty sane')

    if info['ret'] != 0:
        # if not verbose:
        # print(info['out'])
        # print(info['err'])
        raise RuntimeError(info['err'])
    # -f concat -i mylist.txt
    #   ffmpeg \
    # -framerate 60 \
    # -pattern_type glob \
    # -i '*.png' \
    # -r 15 \
    # -vf scale=512:-1 \
    # out.gif \


__notes__ = """

Video to GiF

ffmpeg -ss 30 -t 3 -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif

ffmpeg -i "$HOME/2022-06-29 18-36-30.mkv" -vf "fps=3,scale=1000:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif



w, h = 800, 1164



/usr/bin/ffmpeg -y -r 2.0  -f concat -safe 0 \
        -i /home/joncrall/.cache/gifify/temp/temp_list_22a90c96-3547-4252-baa3-2a063690fdab.txt \
        -filter:v "pad=800:1164:-1:-1:color=black,scale=800:1164:force_original_aspect_ratio=decrease:eval=frame" \
        '/data/joncrall/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/eval/act/Drop4_SC_RGB_scratch_V002_epoch=155-step=78468.pt/crop_id_0db95966_crop.kwcoco/act_pxl_4758a8eb/act_poly_c8e01153/_pred_link/_viz_act_poly_c8e01153_activity_tracks.kwcoco_59eb7098/N37.649560E128.654510_N37.668000E128.689287/ab.mp4'


/usr/bin/ffmpeg -y -r 2.0  -f concat -safe 0 \
        -i /home/joncrall/.cache/gifify/temp/temp_list_22a90c96-3547-4252-baa3-2a063690fdab.txt \
        -filter:v "pad=w=iw:h='max(1164,ih)':x=0:y=0:color=white" \
        '/data/joncrall/dvc-repos/smart_expt_dvc/models/fusion/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/eval/act/Drop4_SC_RGB_scratch_V002_epoch=155-step=78468.pt/crop_id_0db95966_crop.kwcoco/act_pxl_4758a8eb/act_poly_c8e01153/_pred_link/_viz_act_poly_c8e01153_activity_tracks.kwcoco_59eb7098/N37.649560E128.654510_N37.668000E128.689287/ab.mp4'

"""

if __name__ == '__main__':
    """
    CommandLine:
        6 -i "$(ls -tr batch)"
    """
    main()
