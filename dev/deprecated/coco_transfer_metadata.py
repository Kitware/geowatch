"""
Script to help transfer metadata from the original drop0 to drop0-msi

python -m watch.cli.geotiffs_to_kwcoco.py \
    --geotiff_dpath ~/data/dvc-repos/smart_watch_dvc/unannotated/AE-Dubai-0001 \
    --dst ~/data/dvc-repos/smart_watch_dvc/unannotated/dubai-msi.kwcoco.json \
    --relative True

python -m watch.cli.geotiffs_to_kwcoco.py \
    --geotiff_dpath ~/data/dvc-repos/smart_watch_dvc/unannotated/KR-Pyeongchang-S2 \
    --dst ~/data/dvc-repos/smart_watch_dvc/unannotated/korea-msi.kwcoco.json \
    --relative True

python -m watch.cli.geotiffs_to_kwcoco.py \
    --geotiff_dpath ~/data/dvc-repos/smart_watch_dvc/unannotated/US-Waynesboro-0001 \
    --dst ~/data/dvc-repos/smart_watch_dvc/unannotated/waynesboro-msi.kwcoco.json \
    --relative True

# FIXME: doesn't work without the leading ./
kwcoco union ./dubai-msi.kwcoco.json ./korea-msi.kwcoco.json ./waynesboro-msi.kwcoco.json --dst drop0-imgs-s2-lc-msi.json


python ~/code/watch/scripts/coco_transfer_metadata.py \
    --from_fpath $HOME/data/dvc-repos/smart_watch_dvc/unannotated/drop0-imgs-s2-lc-msi.json \
    --onto_fpath $HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
    --dst $HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json
"""
from os.path import join
import dateutil.parser
import watch
import kwcoco
import scriptconfig as scfg
import ubelt as ub
import itertools as it


class CocoTransferMetadataConfig(scfg.Config):
    default = {
        'from_fpath': scfg.Value(None, help='coco file with metadata to transfer'),
        'onto_fpath': scfg.Value(None, help='coco file to to be modified'),
        'dst': scfg.Value(None, help='destination coco path'),
        'relative': scfg.Value(False, help='if true make paths relative'),
    }


def main(**kwargs):
    """
    Example:
        kwargs = {
            'from_fpath': ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/unannotated/drop0-imgs-s2-lc-msi.json'),
            'onto_fpath': ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json'),
            'dst': ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0-msi.kwcoco.json'),
        }
    """
    config = CocoTransferMetadataConfig(default=kwargs, cmdline=True)
    dset1 = kwcoco.CocoDataset(config['from_fpath'])
    dset2 = kwcoco.CocoDataset(config['onto_fpath'])
    dset1.reroot(absolute=True)
    dset2.reroot(absolute=True)

    dst_fpath = config['dst']
    dset_out = transfer_data_onto_dset2(dset1, dset2)
    dset_out.fpath = dst_fpath
    if config['relative']:
        dset_out.reroot(dset_out.bundle_dpath, absolute=False)
    print('write to dset_out.fpath = {!r}'.format(dset_out.fpath))
    dset_out.dump(dset_out.fpath, newlines=True)


def transfer_data_onto_dset2(dset1, dset2):
    """
    Ignore:
        import kwcoco
        # dset1 = kwcoco.CocoDataset(ub.expandpath('$HOME/data/grab_tiles_out/fels/data.kwcoco.json'))
        dset1 = kwcoco.CocoDataset(ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/unannotated/drop0-imgs-s2-lc-msi.json'))
        dset2 = kwcoco.CocoDataset(ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json'))
    """
    dset_out = dset2.copy()

    association = find_association_based_on_sensor_metadata(dset1, dset2)

    print(f'len(association) = {len(association)}')
    # association = find_association_based_on_geotime_bounds(dset1, dset2)

    gids1, gids2 = zip(*association)
    # Make sure more matches are only 1 to 1
    gid2_to_gids1 = ub.group_items(gids1, gids2)
    gid1_to_gids2 = ub.group_items(gids2, gids1)

    if max(map(len, gid1_to_gids2.values())) > 1:
        raise AssertionError

    if max(map(len, gid2_to_gids1.values())) > 2:
        raise AssertionError

    for gid1, gid2 in ub.ProgIter(association, desc='transfer properties'):
        img1 = dset1.imgs[gid1]

        img_out = dset_out.imgs[gid2]

        img1['file_name']
        img1_base = ub.dict_diff(img1, {'auxiliary', 'filename_meta', 'utm_bounds'})
        assert img1_base['height'] == img_out['height']
        assert img1_base['width'] == img_out['width']

        img_out.pop('filename_meta', None)
        img_out.pop('num_bands', None)

        # How should the associated data be transfered.
        img_out['auxiliary'] = img1['auxiliary']
        img_out['file_name'] = None
        img_out['channels'] = None
    return dset_out


def find_association_based_on_sensor_metadata(dset1, dset2):

    for gid1, img1 in ub.ProgIter(dset1.imgs.items(), total=len(dset1.imgs)):
        fpath = coco_representative_fpath(dset1, img1)
        info = watch.gis.geotiff.geotiff_filepath_info(fpath)
        img1['filename_meta'] = info['filename_meta']

    for gid2, img2 in ub.ProgIter(dset2.imgs.items(), total=len(dset2.imgs)):
        fpath2 = coco_representative_fpath(dset2, img2)
        info = watch.gis.geotiff.geotiff_filepath_info(fpath2)
        img2['filename_meta'] = info['filename_meta']

    def build_sensor_key(img):
        meta = img['filename_meta']
        sensor =  meta.get('product_guess', '?')
        if sensor == 'sentinel2':
            key = (sensor, meta['tile_number'], meta['sense_start_time'])
        elif sensor == 'landsat':
            key = (sensor, meta['WRS_path'], meta['WRS_now'],
                   meta['acquisition_date'])
        else:
            key = None
        return key

    key_to_gids1 = ub.ddict(list)
    key_to_gids2 = ub.ddict(list)

    for gid1, img1 in ub.ProgIter(dset1.imgs.items(), total=len(dset1.imgs)):
        key1 = build_sensor_key(img1)
        if key1 is not None:
            key_to_gids1[key1].append(gid1)

    for gid2, img2 in ub.ProgIter(dset2.imgs.items(), total=len(dset2.imgs)):
        key2 = build_sensor_key(img2)
        if key2 is not None:
            key_to_gids2[key2].append(gid2)

    association = []
    for key, gids1 in key_to_gids1.items():
        gids2 = key_to_gids2.get(key, [])
        if len(gids2):
            if len(gids2) > 1 or len(gids1) > 1:
                print('warning ambiguous gids1, gids2 = {}, {}'.format(gids1, gids2))
                # hack
                if len(gids2) > 1 and len(gids1) == 1:
                    gids2 = gids2[0:1]
                elif len(gids2) == 1 and len(gids1) > 1:
                    gids1 = gids1[0:1]
                else:
                    raise AssertionError('complex case')
            for gid1, gid2 in it.product(gids1, gids2):
                association.append((gid1, gid2))
    return association


def find_association_based_on_geotime_bounds(dset1, dset2):
    for gid1, img1 in ub.ProgIter(dset1.imgs.items(), total=len(dset1.imgs)):
        fpath = coco_representative_fpath(dset1, img1)
        info = watch.gis.geotiff.geotiff_metadata(fpath)
        img1['utm_bounds'] = {
            'corners': info['utm_corners'],
            'crs_info': info['utm_crs_info'],
        }
        img1['filename_meta'] = info['filename_meta']

    for gid2, img2 in ub.ProgIter(dset2.imgs.items(), total=len(dset2.imgs)):
        fpath2 = dset2.get_image_fpath(img2)
        # fpath = coco_representative_fpath(dset1, img1)
        info = watch.gis.geotiff.geotiff_metadata(fpath2)
        img2['utm_bounds'] = {
            'corners': info['utm_corners'],
            'crs_info': info['utm_crs_info'],
        }
        img2['filename_meta'] = info['filename_meta']

    stamp_to_gids1 = ub.ddict(list)
    stamp_to_gids2 = ub.ddict(list)

    for gid1, img1 in ub.ProgIter(dset1.imgs.items(), total=len(dset1.imgs)):
        stamp1 = dateutil.parser.parse(img1['date_captured']).date()
        stamp_to_gids1[stamp1].append(gid1)

    for gid2, img2 in ub.ProgIter(dset2.imgs.items(), total=len(dset2.imgs)):
        stamp2 = dateutil.parser.parse(img2['date_captured']).date()
        stamp_to_gids2[stamp2].append(gid2)

    stamps1 = set(stamp_to_gids1)
    stamps2 = set(stamp_to_gids2)
    common_stamps = stamps1 & stamps2

    affinity = {}
    for stamp in common_stamps:
        gids1 = stamp_to_gids1[stamp]
        gids2 = stamp_to_gids2[stamp]

        imgs1 = dset1.images(gids1).objs
        imgs2 = dset2.images(gids2).objs

        import kwimage
        for img1 in imgs1:
            for img2 in imgs2:
                meta1 = img1['filename_meta']
                meta2 = img2['filename_meta']
                sensor1 =  meta1.get('product_guess', '?')
                sensor2 =  meta2.get('product_guess', '?')
                # This is not working correctly
                if sensor1 == sensor2:
                    b1 = img1['utm_bounds']
                    b2 = img2['utm_bounds']
                    if b1['crs_info'] == b2['crs_info']:
                        poly1 = kwimage.Polygon(exterior=b1['corners'])
                        poly2 = kwimage.Polygon(exterior=b2['corners'])
                        s1 = poly1.to_shapely()
                        s2 = poly2.to_shapely()
                        iou = s1.intersection(s2).area / s1.union(s2).area
                        if iou > 0.99:
                            affinity[(img1['id'], img2['id'])] = iou
    association = list(affinity.keys())
    return association


def coco_representative_fpath(dset, img):
    if img.get('file_name', None) is None:
        auxiliary = img.get('auxiliary', [])
        idx = ub.argmax(auxiliary, lambda x: (x['width'] * x['height']))
        aux = auxiliary[idx]
        fpath = join(dset.bundle_dpath, aux.get('file_name'))
    else:
        fpath = join(dset.bundle_dpath, img.get('file_name'))
    return fpath


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/coco_transfer_metadata.py
    """
    main()
