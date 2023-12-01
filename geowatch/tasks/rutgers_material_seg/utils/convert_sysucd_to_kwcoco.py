# flake8: noqa


from os.path import dirname
from os.path import exists
from os.path import join
import ubelt as ub
import glob

import os
import tarfile
import zipfile
from os.path import relpath


def convert_sysucd_to_kwcoco(extract_dpath, coco_fpath, type):
    """
    Converts the raw SpaceNet7 dataset to kwcoco
    Note:
        * The "train" directory contains 60 "videos" representing a region over time.
        * Each "video" directory contains :
            * images           - unmasked images
            * images_masked    - images with masks applied
            * labels           - geojson polys in wgs84?
            * labels_match     - geojson polys in wgs84 with track ids?
            * labels_match_pix - geojson polys in pixels with track ids?
            * UDM_masks - unusable data masks (binary data corresponding with an image, may not exist)
        File names appear like:
            "global_monthly_2018_01_mosaic_L15-1538E-1163N_6154_3539_13"
    Ignore:
        dpath = pathlib.Path("/home/joncrall/data/dvc-repos/smart_watch_dvc/extern/spacenet/")
        extract_dpath = dpath / 'extracted'
        coco_fpath = dpath / 'spacenet7.kwcoco.json'
    """
    import kwcoco
    import json
    import kwimage
    import parse
    import datetime
    print('Convert SYSUCD to kwcoco')

    coco_dset = kwcoco.CocoDataset()
    coco_dset.fpath = coco_fpath

    change_cid = coco_dset.ensure_category('change')
    ignore_cid = coco_dset.ensure_category('ignore')

    sysucd_fname_fmt = parse.Parser('global_monthly_{year:d}_{month:d}_mosaic_{}')

    # Add images
    tile_dpaths1 = list(glob.glob(f'{extract_dpath}{type}/time1/*'))
    tile_dpaths2 = list(glob.glob(f'{extract_dpath}{type}/time2/*'))
    label_dpaths = list(glob.glob(f'{extract_dpath}{type}/label/*'))
    # tile_dpaths2 = list(extract_dpath.glob('train/time2/*'))

    for tile_dpath1, tile_dpath2, label_dpath in ub.ProgIter(zip(tile_dpaths1, tile_dpaths2, label_dpaths), desc='add video'):
        tile_name = tile_dpath1.split('/')[-1].split('.')[0]
        vidid = coco_dset.add_video(name=tile_name)
        image_gpaths = [tile_dpath1, tile_dpath2]

        # sorted(tile_dpath.glob('labels/*'))
        # sorted(tile_dpath.glob('images_masked/*'))
        # sorted(tile_dpath.glob('labels_match/*'))
        # udm_fpaths = sorted(tile_dpath.glob('UDM_masks/*'))

        for frame_index, gpath in enumerate(image_gpaths):
            # gname = str(gpath.stem)
            # nameinfo = sysucd_fname_fmt.parse(tile_name)
            # print(coco_dset.bundle_dpath)
            # timestamp = datetime.datetime(
            #     year=nameinfo['year'], month=nameinfo['month'], day=1)
            gid = coco_dset.add_image(
                file_name=gpath,
                name=f"{tile_name}_{frame_index}",
                video_id=vidid,
                frame_index=frame_index,
                warp_to_wld={"type": "affine", "scale": 1.0},
                num_bands=3,
                # date_captured=timestamp.isoformat(),
                channels='r|g|b',
            )

        gid = coco_dset.index.name_to_img[f"{tile_name}_{frame_index}"]['id']
        c_mask = kwimage.imread(str(label_dpath))
        c_mask[c_mask == 255] = 1
        mask = kwimage.Mask(c_mask, 'c_mask')
        poly = mask.to_multi_polygon()
        xywh = ub.peek(poly.bounding_box().quantize().to_coco())
        ann = {
            'bbox': xywh,
            'image_id': gid,
            'category_id': ignore_cid,
            'segmentation': poly.to_coco(style='new')
        }
        coco_dset.add_annotation(**ann)

    coco_dset._ensure_imgsize()

    # Add annotations

    # def _from_geojson2(geometry):
    #     import numpy as np
    #     coords = geometry['coordinates']
    #     exterior = np.array(coords[0])[:, 0:2]
    #     interiors = [np.array(h)[:, 0:2] for h in coords[1:]]
    #     poly_data = dict(exterior=kwimage.Coords(exterior),
    #                      interiors=[kwimage.Coords(hole)
    #                                 for hole in interiors])
    #     self = kwimage.Polygon(data=poly_data)
    #     return self

    # all_udm_fpaths = sorted(extract_dpath.glob('train/*/UDM_masks/*'))
    # for udm_fpath in ub.ProgIter(all_udm_fpaths, desc='add ignore masks'):
    #     name_parts = udm_fpath.stem.split('_')
    #     assert name_parts[-1] == 'UDM'
    #     name = '_'.join(name_parts[:-1])

    #     gid = coco_dset.index.name_to_img[name]['id']
    #     c_mask = kwimage.imread(str(udm_fpath))
    #     c_mask[c_mask == 255] = 1
    #     mask = kwimage.Mask(c_mask, 'c_mask')
    #     poly = mask.to_multi_polygon()
    #     xywh = ub.peek(poly.bounding_box().quantize().to_coco())
    #     ann = {
    #         'bbox': xywh,
    #         'image_id': gid,
    #         'category_id': ignore_cid,
    #         'segmentation': poly.to_coco(style='new')
    #     }
    #     coco_dset.add_annotation(**ann)

    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    print('coco_dset = {!r}'.format(coco_dset))
    coco_dset.dump(str(coco_dset.fpath))

    # We will generally want an SQL cache when working with this dataset
    # if ub.argflag('--sql-hack'):
    #     from kwcoco.coco_sql_dataset import CocoSqlDatabase
    #     CocoSqlDatabase.coerce(coco_dset)

    return coco_dset


def main():
    # data_dpath = ub.ensure_app_cache_dir('kwcoco', 'data')
    data_dpath = "/media/native/data2/data/SYSU_CD/"
    type = "train"
    coco_fpath = f"{data_dpath}{type}_data.kwcoco.json"
    convert_sysucd_to_kwcoco(extract_dpath=data_dpath, coco_fpath=coco_fpath, type=type)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwcoco/kwcoco/data/grab_spacenet.py
    """
    main()
