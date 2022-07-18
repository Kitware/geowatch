#!/usr/bin/env python
"""
A gif-ify script

Wrapper around imgmagik convert or ffmpeg

TODO:
    - [ ] Moving this to kwplot
"""

import ubelt as ub
from os.path import isdir, join


def main():
    import argparse
    import glob
    description = ub.codeblock(
        '''
        Convert a sequence of images into a gif
        ''')
    parser = argparse.ArgumentParser(prog='gifify', description=description)

    parser.add_argument('image_list', nargs='*', help='list of images (or a text file containing a list of images)')
    parser.add_argument('-i', '--input', nargs='*', help='alternate way to specify list of images')
    parser.add_argument('-d', '--delay', nargs=1, type=float, default=10, help='delay between frames')
    parser.add_argument('-o', '--output', default='out.gif', help='output file')
    parser.add_argument('--max_width', default=None, type=int, help='resize to max width')
    parser.add_argument('--frames_per_second', default=10, type=float, help='number of frames per second')
    args, unknown = parser.parse_known_args()
    # print('unknown = {!r}'.format(unknown))
    # print('args = {!r}'.format(args))
    ns = args.__dict__.copy()

    image_paths1 = ns['image_list']
    image_paths2 = ns['input']

    print('Converting:')
    print('image_paths1 = ' + ub.repr2(image_paths1))
    print('image_paths2 = ' + ub.repr2(image_paths2))

    if image_paths1:
        image_paths = image_paths1
        assert not image_paths2, 'can only specify inputs one way'
    elif image_paths2:
        image_paths = image_paths2
        assert not image_paths1, 'can only specify inputs one way'

    assert image_paths is not None

    frame_fpaths = []
    for p in image_paths:
        if isdir(p):
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

    # frame_fpaths = frame_fpaths[::2]

    print('frame_fpaths = {!r}'.format(frame_fpaths))

    backend = 'imagemagik'
    backend = 'ffmpeg'
    if backend == 'imagemagik':
        escaped_gif_fpath = ns['output'].replace('%', '%%')
        command = ['convert', '-delay', str(ns['delay']), '-loop', '0']
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
        output_fpath = ns['output']
        ns['delay']
        # ns['delay']
        in_framerate = ns['frames_per_second']
        ffmpeg_animate_frames(frame_fpaths, output_fpath,
                              in_framerate=in_framerate,
                              max_width=ns['max_width'])


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
        >>> import kwcoco
        >>> dset = kwcoco.CocoDataset.demo('shapes8')
        >>> ffmpeg_exe = ub.find_exe('ffmpeg')
        >>> if ffmpeg_exe is None:
        >>>     import pytest
        >>>     pytest.skip('test requires ffmpeg')
        >>> frame_fpaths = sorted(dset.images().gpath)
        >>> test_dpath = ub.ensure_app_cache_dir('gifify', 'test')
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
        temp_dpath = ub.ensure_app_cache_dir('gifify', 'temp')
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

        if max_width is not None:
            output_options += [
                '-vf scale="{}:-1"'.format(max_width)
            ]

        if output_fpath.endswith('.mp4'):
            output_options += [
                # MP4 needs even width
                # https://stackoverflow.com/questions/20847674/ffmpeg-div2
                '-filter:v pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2"',
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

"""

if __name__ == '__main__':
    """
    CommandLine:
        6 -i "$(ls -tr batch)"
    """
    main()
