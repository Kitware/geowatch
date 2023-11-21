"""
Filter out images that don't correspond with the desired channels and
creates a new dataset json file.

Warning: make sure that the dst path is different than the source path,
or it will re-write the original dataset json file.
"""
import kwcoco
import ubelt as ub
import scriptconfig as scfg


class FilterBySensorConfig(scfg.Config):
    __default__ = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory for the output'),

        'channels': scfg.Value(['r|g|b'], help='expected channels'),

    }


def main(**kwargs):
    r"""
    CommandLine:
        python -m geowatch.tasks.rutgers_material_seg.scripts.filter_by_sensor \
            --src toydata.kwcoco.json \
            --dst toydata-gsd10.kwcoco.json \

    """
    config = FilterBySensorConfig(kwargs, cmdline=True)
    # print('config = {}'.format(ub.urepr(dict(config), nl=1)))

    print('read dataset')
    dset = kwcoco.CocoDataset(config['src'])

    print('dset.index.videos = {}'.format(ub.urepr(dset.index.videos, nl=2, precision=4)))
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
        python -m geowatch.tasks.rutgers_material_seg.scripts.filter_by_sensor \
            --src <existing kwcoco json> --dst <path to write new kwcoco json>
    """
    main()
