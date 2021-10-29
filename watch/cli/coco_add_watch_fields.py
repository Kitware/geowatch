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
import numpy as np
import kwimage
from watch.utils import kwcoco_extensions


class AddWatchFieldsConfig(scfg.Config):
    """
    Updates image transforms in a kwcoco json file to align all videos to a
    target GSD.
    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory for the output'),

        'target_gsd': scfg.Value(10.0, help='compute transforms for a target gsd'),

        'overwrite': scfg.Value(False, help='if True overwrites introspectable fields'),

        'edit_geotiff_metadata': scfg.Value(False, help='if True MODIFIES THE UNDERLYING IMAGES to ensure geodata is propogated'),

        'default_gsd': scfg.Value(None, help='if specified, assumed any images without geo-metadata have this GSD')
    }


def main(**kwargs):
    r"""
    CommandLine:

        kwcoco toydata --key vidshapes8-multispectral --dst toydata.kwcoco.json
        jq .images[0].auxiliary[0].file_name toydata.kwcoco.json

        kwcoco stats toydata.kwcoco.json
        kwcoco validate toydata.kwcoco.json

        jq .videos toydata.kwcoco.json
        jq .images[0] toydata.kwcoco.json

        python -m watch.cli.coco_add_watch_fields \
            --src toydata.kwcoco.json \
            --dst toydata-gsd10.kwcoco.json \
            --target_gsd=10

        jq .videos toydata-gsd10.kwcoco.json
        jq .images[0] toydata-gsd10.kwcoco.json

    Ignore:
        python -m watch.cli.coco_add_watch_fields \
            --src=$HOME/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi/data.kwcoco.json \
            --dst=$HOME/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi/data.kwcoco.new.json \
            --target_gsd=10

    jq .images[0].auxiliary[0] $HOME/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi/data.kwcoco.new.json
    jq .images[0].auxiliary[0] $HOME/data/dvc-repos/smart_watch_dvc/drop0_aligned_msi/data.kwcoco.json

    Example:
        >>> from watch.cli.coco_add_watch_fields import *  # NOQA
        >>> import kwcoco
        >>> # TODO: make a demo dataset with some sort of gsd metadata
        >>> dset = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> print('dset = {!r}'.format(dset))
        >>> target_gsd = 13.0
        >>> main(src=dset, target_gsd=target_gsd, default_gsd=1)
        >>> print('dset.index.imgs[1] = ' + ub.repr2(dset.index.imgs[1], nl=2))
        >>> print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=1)))

    Ignore:
        kwargs = {
            'src': ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0_aligned/data.kwcoco.json'),
            'target_gsd': 10.0,
            'dst': None,
        }
        kwargs['src'] = kwargs['dst']
        main(**kwargs)
    """
    config = AddWatchFieldsConfig(kwargs, cmdline=True)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    print('read dataset')
    dset = kwcoco.CocoDataset.coerce(config['src'])

    hard_coded_colors = {
        'No Activity': 'tomato',
        'Site Preparation': 'gold',
        'Active Construction': 'lime',
        'Post Construction': 'darkturquoise',
        'Unknown': 'blueviolet',
    }

    for cat in dset.cats.values():
        if cat['name'] in hard_coded_colors:
            cat['color'] = hard_coded_colors[cat['name']]

    print('start populate')
    target_gsd = config['target_gsd']
    overwrite = config['overwrite']
    default_gsd = config['default_gsd']
    kwcoco_extensions.populate_watch_fields(
        dset, target_gsd=target_gsd, overwrite=overwrite,
        default_gsd=default_gsd)
    print('dset.index.videos = {}'.format(ub.repr2(dset.index.videos, nl=2, precision=4)))

    if config['edit_geotiff_metadata']:
        kwcoco_extensions.ensure_transfered_geo_data(dset)

    for gid, img in dset.index.imgs.items():
        offset =  np.asarray(kwimage.Affine.coerce(img['warp_img_to_vid']))[:, 2]
        if np.any(np.abs(offset) > 100):
            print('img = {}'.format(ub.repr2(img, nl=-1)))
            print('warning there is a large offset')
            print('offset = {!r}'.format(offset))
            print('{}, {}'.format(gid, img['warp_img_to_vid']))

    if config['dst'] is not None:
        print('write dataset')
        dset.fpath = config['dst']
        dset.dump(dset.fpath, newlines=True)
    else:
        print('not writing')


_SubConfig = AddWatchFieldsConfig

if __name__ == '__main__':
    """
    CommandLine:
        python  -m watch.cli.coco_add_watch_fields
    """
    main()
