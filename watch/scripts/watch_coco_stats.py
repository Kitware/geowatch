import scriptconfig as scfg
import ubelt as ub


class WatchCocoStats(scfg.Config):
    """
    Print watch-relevant information about a kwcoco dataset
    """
    default = {
        'src': scfg.Value(['special:shapes8'], nargs='+', help='path to dataset', position=1),
    }

    @classmethod
    def main(cls, cmdline=True, **kw):
        import kwcoco
        config = WatchCocoStats(kw, cmdline=cmdline)

        fpaths = config['src']
        assert len(fpaths) == 1, 'only 1 for now'

        fpath = fpaths[0]
        dset = kwcoco.CocoDataset.coerce(fpath)

        print('Per-video stats summary')
        for vidid, gids in dset.index.vidid_to_gids.items():
            avail_sensors = dset.images(gids).lookup('sensor_coarse', None)
            sensor_freq = ub.dict_hist(avail_sensors)
            video = dset.index.videos[vidid]
            print('video = {}'.format(ub.repr2(video, nl=1)))
            video_info = ub.dict_union({
                'name': video['name'],
                'vidid': vidid,
                'sensor_freq': sensor_freq,
            }, video)
            print('video_info = {}'.format(ub.repr2(video_info, nl=-1, sort=False)))


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.scripts.watch_coco_stats --src=special:vidshapes8-multispectral
    """
    WatchCocoStats.main()
