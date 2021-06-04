"""
Adds fields needed by ndsampler to correctly "watch" a region.

Some of this is done hueristically. We assume images come from certain sensors.
We assume input is orthorectified.  We assume some GSD "target" gsd for video
and image processing. Note a video GSD will typically be much higher (i.e.
lower resolution) than an image GSD.
"""
import kwcoco
import ubelt as ub
import scriptconfig as scfg
# from watch.tools.kwcoco_extensions import populate_watch_fields


class AddWatchFieldsConfig(scfg.Config):
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory for the output'),

        'channels': scfg.Value(['r|g|b'], help='expected channels'),

        # 'overwrite': scfg.Value(False, help='if True overwrites introspectable fields'),
    }


def main(**kwargs):
    r"""
    CommandLine:

        python ~/code/watch/scripts/filter_by_sensor.py \
            --src toydata.kwcoco.json \
            --dst toydata-gsd10.kwcoco.json \

    """
    config = AddWatchFieldsConfig(kwargs, cmdline=True)
    # print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    print('read dataset')
    dset = kwcoco.CocoDataset(config['src'])

    print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=2, precision=4)))
    gids_to_remove = []
    aids_to_remove = []
    for gid, img in dset.index.imgs.items():
        # print(gid)
        if img['channels'] not in config['channels']:
            gids_to_remove.append(gid)
    for aid, ann in dset.index.anns.items():
        if ann['image_id'] in gids_to_remove:
            aids_to_remove.append(aid)
            

    dset.remove_images(gids_to_remove)

    if config['dst'] is not None:
        print('write dataset')
        dset.fpath = config['dst']
        dset.dump(dset.fpath, newlines=True)
    else:
        print('not writing')

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/filter_by_sensor.py --src <existing kwcoco json> --dst <path to write new kwcoco json>
    """
    main()
