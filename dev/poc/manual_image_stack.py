"""
Requires that the align script wrote teh _debug_regions viz

Also need to run:

ALIGNED_KWCOCO_BUNDLE=$HOME/data/dvc-repos/smart_watch_dvc/Drop1-Aligned-L1-2022

smartwatch visualize --src \
    $ALIGNED_KWCOCO_BUNDLE/data.kwcoco.json \
    --any3=only \
    --viz_dpath=$ALIGNED_KWCOCO_BUNDLE/_viz \
    --draw_anns=False \
    --select_videos '.name == "BR_R002"' --animate=True

"""

import ubelt as ub
import kwimage


def stack_debug_regions():
    # HACK
    path1 = ub.Path('$HOME/remote/namek/smart_watch_dvc/Drop1-Aligned-L1-2022/_viz/BR_R002/_imgs/any3').expand()
    path2 = ub.Path('$HOME/remote/namek/smart_watch_dvc/Drop1-Aligned-L1-2022/_debug_regions').expand()

    paths1 = sorted(path1.glob('*.jpg'))
    paths2 = sorted(path2.glob('debug_BR_R002_*_crs84.jpg'))
    print(paths1[0])
    print(paths2[0])
    print(paths1[0].stem)
    print(paths2[0].stem)

    print(f'{len(paths1)=!r}')
    print(f'{len(paths2)=!r}')

    k_to_p1 = {}
    k_to_p2 = {}
    for p1 in paths1:
        k1 = '_'.join(p1.stem.split('_')[1:5])
        k_to_p1[k1] = p1

    for p2 in paths2:
        k2 = '_'.join(p2.stem.split('_')[3:7])
        k_to_p2[k2] = p2

    print('k1 = {!r}'.format(k1))
    print('k2 = {!r}'.format(k2))

    common = sorted(set(k_to_p1) & set(k_to_p2))
    print(f'{len(common)=!r}')

    manual_fpath = ub.Path('$HOME/remote/namek/smart_watch_dvc/Drop1-Aligned-L1-2022/_debug_regions_manual').expand().ensuredir()
    frame_fpath = (manual_fpath / 'frames_v3').ensuredir()

    fpaths = []
    for k in ub.ProgIter(common):
        p1 = k_to_p1[k]
        p2 = k_to_p2[k]

        fpath = frame_fpath / (k + '.jpg')
        fpaths.append(fpath)

        if not fpath.exists():
            d1 = kwimage.imread(p1)
            d2 = kwimage.imread(p2)
            d3 = kwimage.stack_images([d1, d2], axis=1)
            kwimage.imwrite(fpath, d3)

    output_fpath = manual_fpath / 'compare_ani.gif'
    from watch.cli import gifify
    gifify.ffmpeg_animate_frames(fpaths, output_fpath, in_framerate=0.7)


def stack_ann_img_animations():
    import watch
    import tempfile
    import kwimage
    tmp_dpath = ub.Path(tempfile.mkdtemp())
    dpath = watch.find_smart_dvc_dpath()
    viz_dpath = dpath / 'Drop1-Aligned-L1-2022-01/_viz3'
    for region in ub.ProgIter(list(viz_dpath.glob('*'))):
        vid_name = region.name
        fpaths1 = sorted(region.glob('_anns/depth/*.jpg'))
        fpaths2 = sorted(region.glob('_imgs/depth/*.jpg'))
        fpaths3 = []
        for fpath1, fpath2 in zip(fpaths1, fpaths2):
            img1 = kwimage.imread(fpath1)
            img2 = kwimage.imread(fpath2)
            img3 = kwimage.stack_images([img1, img2], axis=0)
            fpath3 = tmp_dpath / fpath2.name
            kwimage.imwrite(fpath3, img3)
            fpaths3.append(fpath3)
        output_fpath = viz_dpath / f'stack_{vid_name}.gif'
        if fpaths3:
            from watch.cli import gifify
            gifify.ffmpeg_animate_frames(fpaths3, output_fpath, in_framerate=0.7)
