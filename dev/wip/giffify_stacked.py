
def main():
    #
    # TODO: make this work
    import argparse
    import ubelt as ub
    import glob
    description = ub.codeblock(
        '''
        Stack and then giffify multiple seqeunces of images.
        ''')
    parser = argparse.ArgumentParser(prog='gifify_stacked', description=description)
    parser.add_argument(*('-i', '--input'), dest='input', nargs='+', action='append',
                        help='specify a list of images. Specifying this argument multiple times will stack results')
    args = parser.parse_args()
    ns = args.__dict__.copy()
    image_lists = ns['input']

    def resolve_imglist(imglist):
        from watch.utils import util_path
        if len(imglist) == 1:
            v = imglist[0]
            # if os.path.isdir(v):
            #     print(f'v={v}')
            #     imglist = util_path.coerce_patterned_paths(v + '/*.jpg', expected_extension='.jpg')
            # else:
            imglist = sorted(util_path.coerce_patterned_paths(v, expected_extension='.jpg'))
        else:
            imglist = util_path.coerce_patterned_paths(imglist, expected_extension='.jpg')
        return imglist

    resolved_list = []
    for imglist in image_lists:
        resolved_list.append(resolve_imglist(imglist))
    print('resolved_list = {}'.format(ub.repr2(resolved_list, nl=2)))
    import xdev
    xdev.embed()

    lens = list(map(len, resolved_list))
    assert ub.allsame(lens)


def hackit():
    import watch
    import ubelt as ub
    from watch.utils import util_path
    dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
    path1 = dvc_dpath / '_tmp/_viz__tmp_without_tta.kwcoco_4133893d/KR_R001/_anns/salient'
    path2 = dvc_dpath / '_tmp/_viz__tmp_with_tta.kwcoco_aab1bc51/KR_R001/_anns/salient'
    path3 = dvc_dpath / '_tmp/_viz__tmp_without_tta.kwcoco_4133893d/KR_R001/_anns/red_green_blue'
    path4 = dvc_dpath / '_tmp/_viz__tmp_with_tta.kwcoco_aab1bc51/KR_R001/_anns/red_green_blue'

    paths1 = sorted(util_path.coerce_patterned_paths(path1, '.jpg'))
    paths2 = sorted(util_path.coerce_patterned_paths(path2, '.jpg'))
    paths3 = sorted(util_path.coerce_patterned_paths(path3, '.jpg'))
    paths4 = sorted(util_path.coerce_patterned_paths(path4, '.jpg'))

    def infos(paths):
        rows = []
        for p in paths:
            row = {}
            row['id'] = p.stem.split('_0')[0]
            row['p'] = p
            rows.append(row)
        return rows
    tostack = [
        paths1, paths2, paths3, paths4
    ]
    infos = [infos(paths) for paths in tostack]
    ids = [[r['id'] for r in info] for info in infos]
    common = sorted(set.intersection(*map(set, ids)))

    id_to_paths = [{r['id']: r['p'] for r in info} for info in infos]
    valid_paths = [list(ub.take(lut, common)) for lut in id_to_paths]

    import ubelt as ub
    import kwimage
    tmp_dpath = ub.Path.appdir('watch/giffify/stacking2').ensuredir()
    tmp_dpath2 = ub.Path.appdir('watch/giffify/gifs').ensuredir()

    def stack_job(paths, stack_fpath):
        imdatas = [kwimage.imread(p) for p in paths]
        stacked = kwimage.stack_images_grid(imdatas, pad=10)
        kwimage.imwrite(stack_fpath, stacked)

    jobs = ub.JobPool(max_workers=8)
    stacked_gpaths = []
    for idx, paths in enumerate(zip(*valid_paths)):
        stack_fpath = tmp_dpath / f'tmp_{idx}.jpg'
        stacked_gpaths.append(stack_fpath)
        jobs.submit(stack_job, paths, stack_fpath)

    for job in jobs.as_completed(desc='stacking'):
        job.result()

    from watch.cli import gifify
    frame_fpaths = stacked_gpaths
    gif_fpath = dvc_dpath / '_tmp' / 'gif.gif'
    frames_per_second = 0.7
    gifify.ffmpeg_animate_frames(frame_fpaths, gif_fpath,
                                 in_framerate=frames_per_second,
                                 verbose=1)


if __name__ == '__main__':
    """
    CommandLine:

        DVC_DPATH=$(smartwatch_dvc)
        python ~/code/watch/dev/giffify_stacked.py \
            -i $DVC_DPATH/_tmp/_viz__tmp_without_tta.kwcoco_7c4ce4ca/KR_R001/_anns/salient \
            -i $DVC_DPATH/_tmp/_viz__tmp_without_tta.kwcoco_7c4ce4ca/KR_R001/_anns/red_green_blue \
            -i $DVC_DPATH/_tmp/_viz__tmp_with_tta.kwcoco_caf37b86/KR_R001/_anns/salient \
            -i $DVC_DPATH/_tmp/_viz__tmp_with_tta.kwcoco_caf37b86/KR_R001/_anns/red_green_blue
    """
    main()
