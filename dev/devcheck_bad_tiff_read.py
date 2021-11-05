import kwimage
import kwplot
import watch
import ubelt as ub


def _recompute_bad_tiff():
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()

    bad_fpath = (
        dvc_dpath / 'drop1-S2-L8-WV-aligned/KR_R001/S2/affine_warp/crop_2015-10-21_N37.643680E128.649453_N37.683356E128.734073_S2_1/crop_2015-10-21_N37.643680E128.649453_N37.683356E128.734073_S2_1_B06.tif')

    parent_fpath = (
        dvc_dpath / 'drop1/_assets/smart-imagery/S2/T52SDG_20151021T022702/T52SDG_20151021T022702_B06.jp2')

    parent_data = kwimage.imread(parent_fpath)
    # bad_data = kwimage.imread(bad_fpath)

    kwplot.autompl()
    kwplot.imshow(kwimage.normalize_intensity(parent_data, nodata=0))

    print(ub.cmd('gdalinfo ' + str(bad_fpath))['out'])

    # TODO: parametarize
    compress = 'NONE'
    blocksize = 64
    crop_coordinate_srs = 'epsg:4326'
    target_srs = 'epsg:32652'

    lonmin = 128.649453
    lonmax = 128.73421371550342
    latmin = 37.64390140385888
    latmax = 37.683134282697765

    src_gpath = parent_fpath
    # dst_gpath = './tmp.tif'
    dst_gpath = bad_fpath

    prefix_template = (
        '''
        gdalwarp
        -multi
        --config GDAL_CACHEMAX 500 -wm 500
        --debug off
        -te {xmin} {ymin} {xmax} {ymax}
        -te_srs {crop_coordinate_srs}
        -t_srs {target_srs}
        -of COG
        -co OVERVIEWS=NONE
        -co BLOCKSIZE={blocksize}
        -co COMPRESS={compress}
        -co NUM_THREADS=2
        -overwrite
        ''')

    template_kw = {
        'crop_coordinate_srs': crop_coordinate_srs,
        'target_srs': target_srs,
        'ymin': latmin,
        'xmin': lonmin,
        'ymax': latmax,
        'xmax': lonmax,
        'blocksize': blocksize,
        'compress': compress,
        'SRC': src_gpath,
        'DST': dst_gpath,
    }

    template = ub.paragraph(
        prefix_template +
        '{SRC} {DST}')
    command = template.format(**template_kw)
    cmd_info = ub.cmd(command, verbose=3)  # NOQA

    dst_data = kwimage.imread(dst_gpath)

    kwplot.imshow(kwimage.normalize_intensity(dst_data))


def _check_questionable_cases():
    import kwcoco
    import pathlib
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    coco_fpath = dvc_dpath / 'drop1-S2-L8-WV-aligned/data.kwcoco.json'
    coco_dset = kwcoco.CocoDataset(coco_fpath)

    parent_dset = kwcoco.CocoDataset(dvc_dpath / 'drop1/data.kwcoco.json')

    # for img in coco_dset.index.imgs.values():
    #     parent_img = parent_dset.index.name_to_img[img['parent_name']]

    video = coco_dset.index.name_to_video['BH_R001']
    images = coco_dset.images(vidid=video['id'])
    flags = [d == '2020-04-14T07:06:21' for d in images.lookup('date_captured')]
    img = images.compress(flags).peek()

    coco_img = coco_dset.coco_image(img['id'])

    for obj in coco_img.iter_asset_objs():
        cropped_fpath = pathlib.Path(coco_dset.bundle_dpath) / obj['file_name']
        parent_fpath = pathlib.Path(coco_dset.bundle_dpath) / obj['parent_file_name']

        parent_geo_info = watch.gis.geotiff.geotiff_crs_info(str(parent_fpath))

        wgs84_region = kwimage.Polygon(exterior=obj['wgs84_corners'])
        wld_from_wgs84 = parent_geo_info['wgs84_to_wld']
        pxl_from_wld = parent_geo_info['wld_to_pxl']
        wld_region = wgs84_region.warp(wld_from_wgs84)
        pxl_region = wld_region.warp(pxl_from_wld)

        data = kwimage.imread(cropped_fpath)
        parent_data = kwimage.imread(parent_fpath)

        import kwplot
        plt = kwplot.autoplt()
        kwplot.imshow(parent_data)
        pxl_region.draw()

        ax = plt.gca()
        ax.set_title('parent_img: ' + obj['parent_file_name'])
        if data.sum() == 0:
            print('BAD DATA?')


def _create_annots_only_part():
    import json
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    site_dpath = dvc_dpath / 'drop1/site_models'
    for site_fpath in site_dpath.glob('*.geojson'):
        with open(str(site_fpath), 'r') as file:
            data = json.load(file)
