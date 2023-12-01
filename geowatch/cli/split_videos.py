r"""
Split a coco dataset into one per video.

Ignore:
    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase2_data' --hardware=auto)
    python -m geowatch.cli.split_videos \
        --src "$DVC_DATA_DPATH/Drop4-BAS/data_train.kwcoco.json" \
              "$DVC_DATA_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
        --dst_dpath "$DVC_DATA_DPATH/Drop4-BAS/"

    python -m geowatch.cli.split_videos \
        --dst_dpath "$DVC_DATA_DPATH/Drop4-BAS/"

"""
import scriptconfig as scfg


class SplitVideoConfig(scfg.DataConfig):
    """
    Breaks one or more kwcoco file containing multiple videos into single
    kwcoco files per video. The new kwcoco file names use the same name as the
    input dataset, but prefix it with the video name.
    """
    src = scfg.Value(None, nargs='+', help='one or more datasets to split', position=1)

    dst_dpath = scfg.Value(None, help=(
        'path to write to. If None, uses the src dataset path'))

    dst = scfg.Value(None, help=(
        'If specified, this is a format string template for the path that '
        'will be written to. An error will be thrown if they are not unique.'
        'Available keys are {src_name} and {video_name} in this version.'
        'This name will be relative to dst_dpath. Defaults to '
        '``{src_name}_{video_name}.kwcoco.json``'
    ))

    io_workers = scfg.Value(0, help='number of background IO workers')


def main(cmdline=1, **kwargs):
    """
    Ignore:
        from geowatch.cli.split_videos import *  # NOQA
        import geowatch
        data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        src_fpath = [
            str(data_dvc_dpath / 'Drop4-BAS/data_train.kwcoco.json'),
            str(data_dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'),
        ]
        from kwutil import util_pattern
        coco_fpaths = list(util_pattern.MultiPattern.coerce(list(map(str, src_fpath))) .paths())
        print(f'coco_fpaths={coco_fpaths}')
        cmdline = 0
        kwargs = dict(src=src_fpath)
    """
    import kwcoco
    import ubelt as ub
    from kwutil import util_pattern
    from kwutil import util_parallel
    config = SplitVideoConfig.cli(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.urepr(config, nl=1)))

    coco_fpaths = list(util_pattern.MultiPattern.coerce(config.src).paths())
    print(f'coco_fpaths={coco_fpaths}')

    io_workers = util_parallel.coerce_num_workers(config.io_workers)
    writer = util_parallel.BlockingJobQueue(max_workers=io_workers)

    writen_fpaths = set()

    if config['dst'] is None:
        dst_template = '{src_name}_{video_name}.kwcoco.json'
    else:
        dst_template = config['dst']
        assert '{' in dst_template
        assert '}' in dst_template

    for coco_fpath in coco_fpaths:
        print(f'splitting coco_fpath={coco_fpath}')
        dset = kwcoco.CocoDataset.coerce(coco_fpath)

        src_name = ub.Path(dset.fpath).name.split('.')[0]

        for video in ub.ProgIter(dset.videos().objs, desc='Splitting dataset'):
            video_name = video['name']

            fmtkw = {
                'video_name': video_name,
                'src_name': src_name,
            }
            if config.dst_dpath is None:
                dst_dpath = dset.bundle_dpath
            else:
                dst_dpath = ub.Path(config.dst_dpath)
            vid_fpath = ub.Path(dst_dpath) / dst_template.format(**fmtkw)
            if vid_fpath in writen_fpaths:
                raise Exception('Split name template did not generate unique names')
            writen_fpaths.add(vid_fpath)
            # print(f'vidname={vidname}')
            video_gids = list(dset.images(video_id=video['id']))
            # print(f'video_gids={video_gids}')
            vid_subset = dset.subset(video_gids)
            vid_subset.fpath = vid_fpath
            writer.submit(vid_subset.dump)
            # vid_subset.dump(vid_subset.fpath, newlines=False)

        writer.wait_until_finished(desc="Finish write jobs")
        print(f'finished splitting coco_fpath={coco_fpath}')
    print('Finished splits')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/split_videos.py
    """
    main()
