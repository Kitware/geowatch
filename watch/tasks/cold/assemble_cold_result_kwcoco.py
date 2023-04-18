"""
This is step 4/4 in predict.py

SeeAlso:

    predict.py

    prepare_kwcoco.py

    tile_processing_kwcoco.py

    export_cold_result_kwcoco.py

    assemble_cold_result_kwcoco.py *
"""
import os
import numpy as np
import pandas as pd
from osgeo import gdal
import pytz
import datetime as datetime_mod
from datetime import datetime as datetime_cls
import json
import kwcoco
import kwimage
import scriptconfig as scfg
import ubelt as ub
import logging
import shutil
import gc
try:
    from xdev import profile
except ImportError:
    from ubelt import identity as profile

logger = logging.getLogger(__name__)
# FIXME: note for Jon to be more general than assuming US/East locality in the future
tz = pytz.timezone('US/Eastern')


class AssembleColdKwcocoConfig(scfg.DataConfig):
    """
    TODO: write docs
    """
    stack_path = scfg.Value(None, help='folder directory of stacked data')
    reccg_path = scfg.Value(None, help='folder directory of cold processing result')
    coco_fpath = scfg.Value(None, help='file path of coco json')
    combined_coco_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to combined kwcoco file
        '''))
    mod_coco_fpath = scfg.Value(None, help='file path for modified coco json')
    meta_fpath = scfg.Value(None, help='file path of metadata json created by prepare_kwcoco script')
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, type=str, help="list of COLD coefficients for saving geotiff, e.g., a0,c1,a1,b1,a2,b2,a3,b3,cv,rmse")
    coefs_bands = scfg.Value(None, type=str, help='indicate the bands for output coefs_bands, e.g., 0,1,2,3,4,5')
    timestamp = scfg.Value(False, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')
    combine = scfg.Value(True, help='for temporal combined mode, Default is True')
    resolution = scfg.Value('30GSD', help=ub.paragraph(
        '''
        the resolution used when preparing the kwcoco data. Note: results will
        be wrong if this does not agree with what was used in
        PrepareKwcocoConfig
        '''))
    sensors = scfg.Value('L8', type=str, help='sensor type, default is "L8"')


@profile
def assemble_main(cmdline=1, **kwargs):
    """_summary_

    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m watch.tasks.cold.assemble_cold_result_kwcoco --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.assemble_cold_result_kwcoco assemble_main

    Example:
    >>> # xdoctest: +REQUIRES(env:TEST_COLD)
    >>> from watch.tasks.cold.assemble_cold_result_kwcoco import assemble_main
    >>> from watch.tasks.cold.assemble_cold_result_kwcoco import *
    >>> kwargs= dict(
    >>>    stack_path = "/home/jws18003/data/dvc-repos/smart_data_dvc/Drop6/_pycold_combine/stacked/KR_R001/",
    >>>    reccg_path = "/home/jws18003/data/dvc-repos/smart_data_dvc/Drop6/_pycold_combine/reccg/KR_R001/",
    >>>    coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Drop6/data_vali_split1_KR_R001.kwcoco.json'),
    >>>    mod_coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Drop6/_pycold_combine/test.json'),
    >>>    meta_fpath = '/home/jws18003/data/dvc-repos/smart_data_dvc/Drop6/_pycold_combine/stacked/KR_R001/block_x9_y9/crop_20210807T010000Z_N37.643680E128.649453_N37.683356E128.734073_L8_0.json',
    >>>    combined_coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Drop6_MeanYear/data_vali_split1_KR_R001_MeanYear.kwcoco.json'),
    >>>    coefs = 'cv,rmse,a0,a1,b1,c1',
    >>>    year_lowbound = None,
    >>>    year_highbound = None,
    >>>    coefs_bands = '0,1,2,3,4,5',
    >>>    timestamp = False,
    >>>    combine = True,
    >>>    sensors = 'L8',
    >>>    resolution = '10GSD',
    >>>    )
    >>> cmdline=0
    >>> assemble_main(cmdline, **kwargs)
    """
    # a hacky way to pass the process context and progress manager from the
    # caller when this is called as a subroutine
    pman = kwargs.pop('pman', None)
    proc_context = kwargs.pop('proc_context', None)

    config_in = AssembleColdKwcocoConfig.cli(cmdline=cmdline, data=kwargs)
    stack_path = ub.Path(config_in['stack_path'])
    reccg_path = ub.Path(config_in['reccg_path'])
    coco_fpath = ub.Path(config_in['coco_fpath'])
    combined_coco_fpath = ub.Path(config_in['combined_coco_fpath'])
    mod_coco_fpath = ub.Path(config_in['mod_coco_fpath'])
    out_path = reccg_path / 'cold_feature'
    tmp_path = out_path / 'tmp'
    meta_fpath = ub.Path(config_in['meta_fpath'])
    year_lowbound = config_in['year_lowbound']
    year_highbound = config_in['year_highbound']
    coefs = config_in['coefs']
    coefs_bands = config_in['coefs_bands']
    timestamp = config_in['timestamp']
    combine = config_in['combine']
    resolution = config_in['resolution']
    sensors = config_in['sensors']

    # define variables
    config = json.loads(meta_fpath.read_text())
    vid_w = config['video_w']
    vid_h = config['video_h']
    n_block_x = config['n_block_x']
    n_block_y = config['n_block_y']
    n_blocks = n_block_x * n_block_y  # total number of blocks

    cold_param = json.loads((reccg_path / 'log.json').read_text())
    method = cold_param['algorithm']

    coef_names = ['cv', 'rmse', 'a0', 'a1', 'b1', 'a2', 'b2', 'a3', 'b3', 'c1']
    band_names = [0, 1, 2, 3, 4, 5]

    BAND_INFO = {0: 'blue',
                 1: 'green',
                 2: 'red',
                 3: 'nir',
                 4: 'swir16',
                 5: 'swir22'}

    if coefs is not None:
        try:
            coefs = list(coefs.split(","))
        except Exception:
            print("Illegal coefs inputs: example, --coefs='a0, c1, a1, b1, a2, b2, a3, b3, cv, rmse'")

        try:
            coefs_bands = list(coefs_bands.split(","))
            coefs_bands = [int(coefs_band) for coefs_band in coefs_bands]
        except Exception:
            print("Illegal coefs_bands inputs: example, --coefs_bands='0, 1, 2, 3, 4, 5, 6'")

    # Get original transform from projection to image space
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    L8_new_gdal_transform, L8_proj = get_gdal_transform(coco_dset, 'L8', resolution=resolution)
    S2_new_gdal_transform, S2_proj = get_gdal_transform(coco_dset, 'S2', resolution=resolution)

    available_transforms = [L8_new_gdal_transform, S2_new_gdal_transform]
    if all(t is None for t in available_transforms):
        raise RuntimeError('There are no images of known sensors')

    # Get ordinal day list
    block_folder = stack_path / 'block_x1_y1'

    # if timestamp:
    meta_files = [m for m in os.listdir(block_folder) if m.endswith('.json')]

    # sort image files by ordinal dates
    img_dates = []
    img_names = []

    # read metadata and
    for meta in meta_files:
        meta_config = json.loads((block_folder / meta).read_text())
        ordinal_date = meta_config['ordinal_date']
        img_name = meta_config['image_name'] + '.npy'
        img_dates.append(ordinal_date)
        img_names.append(img_name)

    if year_lowbound is None:
        year_low_ordinal = min(img_dates)
        year_lowbound = pd.Timestamp.fromordinal(year_low_ordinal).year
    else:
        year_low_ordinal = pd.Timestamp.toordinal(datetime_mod.datetime(int(year_lowbound), 1, 1))

    img_dates, img_names = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                        zip(img_dates, img_names)))
    if year_highbound is None:
        year_high_ordinal = max(img_dates)
        year_highbound = pd.Timestamp.fromordinal(year_high_ordinal).year
    else:
        year_high_ordinal = pd.Timestamp.toordinal(datetime_mod.datetime(int(year_highbound + 1), 1, 1))

    img_dates, img_names = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                            zip(img_dates, img_names)))
    img_dates = sorted(img_dates)
    img_names = sorted(img_names)
    if timestamp:
        ordinal_day_list = img_dates
    if combine:
        combined_coco_dset = kwcoco.CocoDataset(combined_coco_fpath)

        # filter by sensors
        all_images = combined_coco_dset.images(list(ub.flatten(combined_coco_dset.videos().images)))
        flags = [s in sensors for s in all_images.lookup('sensor_coarse')]
        all_images = all_images.compress(flags)
        image_id_iter = iter(all_images)

        # Get ordinal date of combined coco image
        ordinal_dates = []
        img_names = []
        for image_id in image_id_iter:
            combined_coco_image: kwcoco.CocoImage = combined_coco_dset.coco_image(image_id)
            coco_image: kwcoco.CocoImage = coco_dset.coco_image(image_id)
            ts = combined_coco_image.img['timestamp']
            coco_img_name = coco_image.img['name']
            timestamp_local = datetime_cls.fromtimestamp(ts, tz=tz)
            timestamp_utc = timestamp_local.astimezone(pytz.utc)
            ordinal = timestamp_utc.toordinal()
            ordinal_dates.append(ordinal)
            img_names.append(coco_img_name)
        ordinal_day_list = ordinal_dates

    # else:
    #     # Get only the first ordinal date of each year
    #     first_ordinal_dates = []
    #     first_img_names = []
    #     last_year = None
    #     for ordinal_day, img_name in zip(img_dates, img_names):
    #         year = pd.Timestamp.fromordinal(ordinal_day).year
    #         if year != last_year:
    #             first_ordinal_dates.append(ordinal_day)
    #             first_img_names.append(img_name)
    #             last_year = year

    #     ordinal_day_list = first_ordinal_dates
    #     img_names = first_img_names

    # assemble
    logger.info('Generating COLD output geotiff')

    if coefs is not None:
        day_iter = range(len(ordinal_day_list))
        if pman is not None:
            day_iter = pman.progiter(day_iter, total=len(ordinal_day_list))
        for day in day_iter:
            tmp_map_blocks = [np.load(
                tmp_path / f'tmp_coefmap_block{x + 1}_{ordinal_day_list[day]}.npy')
                for x in range(n_blocks)]

            results = np.hstack(tmp_map_blocks)
            results = np.vstack(np.hsplit(results, n_block_x))
            ninput = 0
            for band_idx, band_name in enumerate(coefs_bands):
                for coef_index, coef in enumerate(coefs):
                    kwcoco_img_name = img_names[day]
                    band = BAND_INFO[band_name]
                    name_parts = list(map(str, (kwcoco_img_name, band, method, coef)))
                    outname = '_'.join(name_parts) + '.tif'
                    outfile = out_path / outname
                    if '_L8_' in outname:
                        outdriver_L8 = gdal.GetDriverByName('GTiff')
                        outdata_L8 = outdriver_L8.Create(os.fspath(outfile), vid_w, vid_h, 1, gdal.GDT_Float32)
                        outdata_L8.GetRasterBand(1).WriteArray(results[:vid_h, :vid_w, ninput])
                        outdata_L8.FlushCache()
                        outdata_L8.SetGeoTransform(L8_new_gdal_transform)
                        outdata_L8.FlushCache()
                        outdata_L8.SetProjection(L8_proj)
                        outdata_L8.FlushCache()
                        ninput = ninput + 1

                    if '_S2_' in outname:
                        outdriver_S2 = gdal.GetDriverByName('GTiff')
                        outdata_S2 = outdriver_S2.Create(os.fspath(outfile), vid_w, vid_h, 1, gdal.GDT_Float32)
                        outdata_S2.GetRasterBand(1).WriteArray(results[:vid_h, :vid_w, ninput])
                        outdata_S2.FlushCache()
                        outdata_S2.SetGeoTransform(S2_new_gdal_transform)
                        outdata_S2.FlushCache()
                        outdata_S2.SetProjection(S2_proj)
                        outdata_S2.FlushCache()
                        ninput = ninput + 1

            # for x in range(n_blocks):
                # TODO: would be nice to have a structure that controls these
                # name formats so we can use padded inter suffixes for nicer
                # sorting, or nest files to keep folder sizes small

    # Remove tmp files
    shutil.rmtree(tmp_path)

    logger.info('Starting adding new asset to kwcoco json')
    if combine:
        combined_coco_dset = kwcoco.CocoDataset(combined_coco_fpath)
        for image_id in combined_coco_dset.images():
            combined_coco_image: kwcoco.CocoImage = combined_coco_dset.coco_image(image_id)
            coco_image: kwcoco.CocoImage = coco_dset.coco_image(image_id)
            image_name = coco_image.img['name']

            asset_w = vid_w
            asset_h = vid_h

            for band_name in band_names:
                for coef in coef_names:
                    band = BAND_INFO[band_name]
                    new_fpath = out_path / f'{image_name}_{band}_{method}_{coef}.tif'
                    if new_fpath.exists():
                        channels = kwcoco.ChannelSpec.coerce(f'{band}_{method}_{coef}')

                        # COLD output was wrote based on transform information of
                        # coco_dset, so it aligned to a scaled video space.
                        warp_img_from_vid = combined_coco_image.warp_img_from_vid

                        if resolution is None:
                            scale_asset_from_vid = (1., 1.)
                        else:
                            scale_asset_from_vid = combined_coco_image._scalefactor_for_resolution(
                                space='video', resolution=resolution)
                        warp_asset_from_vid = kwimage.Affine.scale(scale_asset_from_vid)
                        warp_vid_from_asset = warp_asset_from_vid.inv()
                        warp_img_from_asset = warp_img_from_vid @ warp_vid_from_asset

                        # Use the CocoImage helper which will augment the coco dictionary with
                        # your information.
                        combined_coco_image.add_asset(os.fspath(new_fpath),
                                                      channels=channels, width=asset_w,
                                                      height=asset_h, warp_aux_to_img=warp_img_from_asset)
                        logger.info(f'Added to the asset {new_fpath}')
    else:
        # add new asset to each image
        for image_id in coco_dset.images():
            # Create a CocoImage object for each image.
            coco_image: kwcoco.CocoImage = coco_dset.coco_image(image_id)
            image_name = coco_image.img['name']

            asset_w = vid_w
            asset_h = vid_h

            for band_name in band_names:
                for coef in coef_names:
                    band = BAND_INFO[band_name]
                    new_fpath = out_path / f'{image_name}_{band}_{method}_{coef}.tif'
                    if new_fpath.exists():
                        channels = kwcoco.ChannelSpec.coerce(f'{band}_{method}_{coef}')

                        # COLD output was wrote based on transform information of
                        # coco_dset, so it aligned to a scaled video space.
                        warp_img_from_vid = coco_image.warp_img_from_vid

                        if resolution is None:
                            scale_asset_from_vid = (1., 1.)
                        else:
                            scale_asset_from_vid = coco_image._scalefactor_for_resolution(
                                space='video', resolution=resolution)
                        warp_asset_from_vid = kwimage.Affine.scale(scale_asset_from_vid)
                        warp_vid_from_asset = warp_asset_from_vid.inv()
                        warp_img_from_asset = warp_img_from_vid @ warp_vid_from_asset

                        # Use the CocoImage helper which will augment the coco dictionary with
                        # your information.
                        coco_image.add_asset(os.fspath(new_fpath),
                                                channels=channels, width=asset_w,
                                                height=asset_h,
                                                warp_aux_to_img=warp_img_from_asset)

                        logger.info(f'Added to the asset {new_fpath}')

    if proc_context is not None:
        context_info = proc_context.stop()
        if combine:
            combined_coco_dset.dataset['info'].append(context_info)
        else:
            coco_dset.dataset['info'].append(context_info)

    # Write a modified kwcoco.json file
    logger.info(f'Writing kwcoco file to: {mod_coco_fpath}')
    if combine:
        combined_coco_dset.fpath = mod_coco_fpath
        combined_coco_dset._ensure_json_serializable()
        combined_coco_dset.dump()
    else:
        coco_dset.fpath = mod_coco_fpath
        coco_dset._ensure_json_serializable()
        coco_dset.dump()
    logger.info(f'Finished writing kwcoco file to: {mod_coco_fpath}')

    gc.collect()


@profile
def get_gdal_transform(coco_dset, sensor_name, resolution=None):
    video_ids = list(coco_dset.videos())
    if len(video_ids) != 1:
        raise AssertionError('currently expecting one video per coco file; todo: be robust to this')
    video_id = video_ids[0]

    # Get all the images in the video.
    images = coco_dset.images(video_id=video_id)
    sensors = images.lookup('sensor_coarse', None)
    is_target_sensor = [s == sensor_name for s in sensors]

    # Filter to only the images from target sensor
    target_images = images.compress(is_target_sensor)
    if len(target_images) == 0:
        return None, None
        raise RuntimeError(f'Video {video_id} in {coco_dset} contains no {sensor_name} images')

    # Take the first target image
    target_coco_img = target_images.coco_images[0]

    # Get the transform for the original asset
    target_primary_asset = target_coco_img.primary_asset()
    target_primary_fpath = os.path.join(ub.Path(target_coco_img.bundle_dpath), target_primary_asset['file_name'])
    try:
        ref_image = gdal.Open(target_primary_fpath, gdal.GA_ReadOnly)
        proj = ref_image.GetProjection()     # This transforms from world space to CRS84
        trans = ref_image.GetGeoTransform()  # This transforms from the underlying asset to world space.
    except AttributeError:
        return None, None
    warp_wld_from_primary = kwimage.Affine.from_gdal(trans)
    warp_img_from_primary = kwimage.Affine.coerce(target_primary_asset['warp_aux_to_img'])

    # If we requested a specific processing resolution, our *new* asset on disk
    # will differ by a scale factor.
    if resolution is None:
        scale_asset_from_vid = (1.0, 1.0)
    else:
        scale_asset_from_vid = target_coco_img._scalefactor_for_resolution(
            space='video', resolution=resolution)
    warp_asset_from_vid = kwimage.Affine.scale(scale_asset_from_vid)
    warp_img_from_vid = target_coco_img.warp_img_from_vid

    warp_vid_from_asset = warp_asset_from_vid.inv()
    warp_primary_from_img = warp_img_from_primary.inv()

    #
    # Calculate the new GDAL transform mapping our new asset on disk to the
    # world space
    warp_wld_from_asset = (
        warp_wld_from_primary @ warp_primary_from_img @
        warp_img_from_vid @ warp_vid_from_asset
    )
    new_geotrans = tuple(warp_wld_from_asset.to_gdal())
    return new_geotrans, proj


if __name__ == '__main__':
    assemble_main()
