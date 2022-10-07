import sys
import kwcoco
import json
import argparse
import pystac
import ubelt as ub
import dateutil.parser


def hack_resolve_sensor_candidate(dset):
    """
    Make a sensor code that coarsely groups the different sensors

    set([tuple(sorted(g.get('sensor_candidates'))) for g in dset.imgs.values()])
    """
    known_mapping = {
        'GE01': 'WV',
        'WV01': 'WV',
        'WV02': 'WV',
        'WV03': 'WV',
        'WV03_VNIR': 'WV',
        'WV03_SWIR': 'WV',

        'LC08': 'LC',

        'LE07': 'LE',

        'S2A': 'S2',
        'S2B': 'S2',
        'S2-TrueColor': 'S2',
        'S2': 'S2'
    }

    for img in dset.imgs.values():
        coarsend = {known_mapping[k] for k in img['sensor_candidates']}
        assert len(coarsend) == 1
        img['sensor_coarse'] = ub.peek(coarsend)


def pystac_Catalog_coerce(data) -> pystac.Catalog:
    if isinstance(data, str):
        catalog = pystac.read_file(href=data)
    elif isinstance(data, dict):
        catalog = pystac.Catalog.from_dict(data)
    elif isinstance(data, pystac.Catalog):
        catalog = data
    else:
        raise TypeError(type(data))
    return catalog


def convert(out_file, cat, ignore_dem=True):
    from watch.gis import geotiff
    import watch.cli.geotiffs_to_kwcoco as gtk
    dset = kwcoco.CocoDataset()

    catalog = pystac_Catalog_coerce(cat)

    # index = 0
    for item in catalog.get_items():
        print('item = {!r}'.format(item))
        meta = item.to_dict()
        date = meta['properties']['datetime']
        images = []
        name = item.id
        for asset in item.get_assets():
            if asset != 'data' and 'data' not in item.assets[asset].roles:
                continue
            images.append(item.assets[asset].get_absolute_href())
        if len(images) > 1:
            img = gtk.make_coco_img_from_auxiliary_geotiffs(images, name)
        else:
            img = gtk.make_coco_img_from_geotiff(images[0], name)
            img['warp_pxl_to_wld'] = img['warp_pxl_to_wld'].__json__()
        info = geotiff.geotiff_metadata(images[0])
        date = dateutil.parser.parse(date).date()
        img['date_captured'] = date.isoformat()
        img['sensor_candidates'] = info['sensor_candidates']
        dset.add_image(**img)

    hack_resolve_sensor_candidate(dset)
    dset.fpath = out_file
    with open(out_file, 'w') as f:
        json.dump(dset.dataset, f, indent=2)

    return dset


def main(args):
    parser = argparse.ArgumentParser(description="Convert STAC catalog to KWCOCO")
    parser.add_argument("--out_file", help="Output KWCOCO")
    parser.add_argument("--catalog", help="Catalog to convert")
    parser.add_argument('--ignore_dem',
                        help="If set, don't use the digital elevation map",
                        default=False)
    args = parser.parse_args(args)
    convert(args.out_file, args.catalog, args.ignore_dem)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))