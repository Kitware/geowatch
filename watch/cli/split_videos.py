r"""
Split a coco dataset into one per video.

Ignore:
    DATA_DVC_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)
    python -m watch.cli.split_videos \
        --src "$DATA_DVC_DPATH/Drop4-BAS/data_train.kwcoco.json" \
              "$DATA_DVC_DPATH/Drop4-BAS/data_vali.kwcoco.json" \
        --dst_dpath "$DATA_DVC_DPATH/Drop4-BAS/"

    python -m watch.cli.split_videos \
        --dst_dpath "$DATA_DVC_DPATH/Drop4-BAS/"

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

    io_workers = scfg.Value(2, help='number of background IO workers')


def main(cmdline=1, **kwargs):
    """
    Ignore:
        from watch.cli.split_videos import *  # NOQA
        import watch
        data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        src_fpath = [
            str(data_dvc_dpath / 'Drop4-BAS/data_train.kwcoco.json'),
            str(data_dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'),
        ]
        from watch.utils import util_pattern
        coco_fpaths = list(util_pattern.MultiPattern.coerce(list(map(str, src_fpath))) .paths())
        print(f'coco_fpaths={coco_fpaths}')
        cmdline = 0
        kwargs = dict(src=src_fpath)
    """
    import kwcoco
    import ubelt as ub
    from watch.utils import util_pattern
    from watch.utils import util_parallel
    from watch.utils import util_globals
    config = SplitVideoConfig.legacy(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.repr2(config, nl=1)))

    coco_fpaths = list(util_pattern.MultiPattern.coerce(config.src).paths())
    print(f'coco_fpaths={coco_fpaths}')

    io_workers = util_globals.coerce_num_workers(config.io_workers)
    writer = util_parallel.BlockingJobQueue(max_workers=io_workers)

    for coco_fpath in coco_fpaths:
        print(f'splitting coco_fpath={coco_fpath}')
        dset = kwcoco.CocoDataset.coerce(coco_fpath)

        prefix = ub.Path(dset.fpath).name.split('.')[0]

        for video in ub.ProgIter(dset.videos().objs, desc='Splitting dataset'):
            vidname = video['name']

            if config.dst_dpath is None:
                dst_dpath = dset.bundle_dpath
            else:
                dst_dpath = ub.Path(config.dst_dpath)

            vid_fpath = ub.Path(dst_dpath) / (prefix + '_' + vidname + '.kwcoco.json')
            # print(f'vidname={vidname}')
            video_gids = list(dset.images(video_id=video['id']))
            # print(f'video_gids={video_gids}')
            vid_subset = dset.subset(video_gids)
            vid_subset.fpath = vid_fpath
            vid_subset.dump(vid_subset.fpath, newlines=False)

        writer.wait_until_finished(desc="Finish write jobs")
        print(f'finished splitting coco_fpath={coco_fpath}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/split_videos.py
    """
    main()
