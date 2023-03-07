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
import datetime as datetime
import json
import kwcoco
import kwimage
import scriptconfig as scfg
import ubelt as ub
import logging
import fnmatch

try:
    from xdev import profile
except ImportError:
    from ubelt import identity as profile

logger = logging.getLogger(__name__)


class AssembleColdKwcocoConfig(scfg.DataConfig):
    """
    TODO: write docs
    """
    stack_path = scfg.Value(None, help='folder directory of stacked data')
    reccg_path = scfg.Value(None, help='folder directory of cold processing result')
    coco_fpath = scfg.Value(None, help='file path of coco json')
    mod_coco_fpath = scfg.Value(None, help='file path for modified coco json')
    meta_fpath = scfg.Value(None, help='file path of metadata json created by prepare_kwcoco script')
    year_lowbound = scfg.Value(None, help='min year for saving geotiff, e.g., 2017')
    year_highbound = scfg.Value(None, help='max year for saving geotiff, e.g., 2022')
    coefs = scfg.Value(None, type=str, help="list of COLD coefficients for saving geotiff, e.g., a0,c1,a1,b1,a2,b2,a3,b3,cv,rmse")
    coefs_bands = scfg.Value(None, type=str, help='indicate the bands for output coefs_bands, e.g., 0,1,2,3,4,5')
    timestamp = scfg.Value(True, help='True: exporting cold result by timestamp, False: exporting cold result by year, Default is False')


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
    >>>    stack_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/KR_R001",
    >>>    reccg_path = "/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/reccg/KR_R001",
    >>>    coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/imgonly-KR_R001.kwcoco.json'),
    >>>    mod_coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/KR_R001/imgonly-KR_R001.kwcoco.modified.json'),
    >>>    meta_fpath = '/gpfs/scratchfs1/zhz18039/jws18003/kwcoco/stacked/KR_R001/block_x10_y1/crop_20140115T020000Z_N37.643680E128.649453_N37.683356E128.734073_L8_0.json',
    >>>    coefs = ['cv'],
    >>>    year_lowbound = 2017,
    >>>    year_highbound = 2022,
    >>>    coefs_bands = [0, 1, 2, 3, 4, 5],
    >>>    timestamp = True,
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
    mod_coco_fpath = ub.Path(config_in['mod_coco_fpath'])
    out_path = reccg_path / 'cold_feature'
    meta_fpath = ub.Path(config_in['meta_fpath'])
    year_lowbound = config_in['year_lowbound']
    year_highbound = config_in['year_highbound']
    coefs = config_in['coefs']
    coefs_bands = config_in['coefs_bands']
    timestamp = config_in['timestamp']

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

    # dt = np.dtype([('t_start', np.int32),
    #                ('t_end', np.int32),
    #                ('t_break', np.int32),
    #                ('pos', np.int32),
    #                ('num_obs', np.int32),
    #                ('category', np.short),
    #                ('change_prob', np.short),
    #                ('coefs', np.float32, (7, 8)),   # note that the slope coefficient was scaled up by 10000
    #                ('rmse', np.float32, 7),
    #                ('magnitude', np.float32, 7)])

    # if coefs is not None:
    #     assert all(elem in coef_names for elem in coefs)
    #     assert all(elem in band_names for elem in coefs_bands)

    # Get original transform from projection to image space
    coco_dset = kwcoco.CocoDataset(coco_fpath)
    L8_new_gdal_transform, L8_proj = get_gdal_transform(coco_dset, 'L8')
    S2_new_gdal_transform, S2_proj = get_gdal_transform(coco_dset, 'S2')

    available_transforms = [L8_new_gdal_transform, S2_new_gdal_transform]
    if all(t is None for t in available_transforms):
        raise RuntimeError('There are no images of known sensors')

    # video_ids = list(coco_dset.videos())
    # if len(video_ids) != 1:
    #     raise AssertionError('currently expecting one video per coco file; todo: be robust to this')
    # video_id = video_ids[0]

    # # Get all the images in the first video.
    # images = coco_dset.images(video_id=video_id)
    # sensors = images.lookup('sensor_coarse', None)
    # is_landsat = [s == 'L8' for s in sensors]

    # # Filter to only the landsat images
    # landsat_images = images.compress(is_landsat)
    # if len(landsat_images) == 0:
    #     raise RuntimeError(f'Video {video_id} in {coco_dset} contains no landsat images')

    # # Take the first landsat image
    # coco_img = landsat_images.coco_images[0]
    # primary_asset = coco_img.primary_asset()
    # primary_fpath = ub.Path(coco_img.bundle_dpath) / primary_asset['file_name']
    # ref_image = gdal.Open(os.fspath(primary_fpath), gdal.GA_ReadOnly)
    # trans = ref_image.GetGeoTransform()
    # proj = ref_image.GetProjection()

    # original = kwimage.Affine.from_gdal(trans)
    # # c, a, b, f, d, e = trans
    # # original = kwimage.Affine(np.array([
    # #     [a, b, c],
    # #     [d, e, f],
    # #     [0, 0, 1],
    # # ]))

    # warp_vid_from_img = kwimage.Affine.coerce(coco_img.img['warp_img_to_vid']).inv()
    # new_geotrans =  original @ warp_vid_from_img
    # a, b, c, d, e, f, g, h, i = np.array(new_geotrans).ravel().tolist()
    # new_gdal_transform = (c, a, b, f, d, e)

    # Get ordinal day list
    block_folder = stack_path / 'block_x1_y1'

    if timestamp:
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
            year_low_ordinal = pd.Timestamp.toordinal(datetime.datetime(int(year_lowbound), 1, 1))

        img_dates, img_names = zip(*filter(lambda x: x[0] >= year_low_ordinal,
                                            zip(img_dates, img_names)))
        if year_highbound is None:
            year_high_ordinal = max(img_dates)
            year_highbound = pd.Timestamp.fromordinal(year_high_ordinal).year
        else:
            year_high_ordinal = pd.Timestamp.toordinal(datetime.datetime(int(year_highbound + 1), 1, 1))

        img_dates, img_names = zip(*filter(lambda x: x[0] < year_high_ordinal,
                                                zip(img_dates, img_names)))
        img_dates = sorted(img_dates)
        img_names = sorted(img_names)
        ordinal_day_list = img_dates

    # assemble
    logger.info('Generating COLD output geotiff')

    if coefs is not None:
        day_iter = range(len(ordinal_day_list))
        if pman is not None:
            day_iter = pman.progiter(day_iter, total=len(ordinal_day_list))
        for day in day_iter:
            tmp_map_blocks = [np.load(
                out_path / f'tmp_coefmap_block{x + 1}_{ordinal_day_list[day]}.npy')
                for x in range(n_blocks)]

            results = np.hstack(tmp_map_blocks)
            results = np.vstack(np.hsplit(results, n_block_x))
            # ninput = 0
            for band_idx, band_name in enumerate(coefs_bands):
                for coef_index, coef in enumerate(coefs):
                    kwcoco_img_name = img_names[day]
                    band = BAND_INFO[band_name]
                    name_parts = list(map(str, (kwcoco_img_name[:-4], band, method, coef)))
                    outname = '_'.join(name_parts) + '.tif'
                    outfile = out_path / outname
                    # outdriver1 = gdal.GetDriverByName("GTiff")
                    # outdata = outdriver1.Create(os.fspath(outfile), vid_w, vid_h, 1, gdal.GDT_Float32)
                    # outdata.GetRasterBand(1).WriteArray(results[:vid_h, :vid_w, ninput])
                    # outdata.FlushCache()
                    # outdata.SetGeoTransform(new_gdal_transform)
                    # outdata.FlushCache()
                    # outdata.SetProjection(proj)
                    # outdata.FlushCache()
                    # ninput = ninput + 1
                    if '_L8_' in outname:
                        outdriver_L8 = gdal.GetDriverByName('GTiff')
                        outdata_L8 = outdriver_L8.Create(os.fspath(outfile), vid_w, vid_h, 1, gdal.GDT_Float32)
                        outdata_L8.GetRasterBand(1).WriteArray(results[:vid_h, :vid_w, coef_index])
                        outdata_L8.FlushCache()
                        outdata_L8.SetGeoTransform(L8_new_gdal_transform)
                        outdata_L8.FlushCache()
                        outdata_L8.SetProjection(L8_proj)
                        outdata_L8.FlushCache()

                    if '_S2_' in outname:
                        outdriver_S2 = gdal.GetDriverByName('GTiff')
                        outdata_S2 = outdriver_S2.Create(os.fspath(outfile), vid_w, vid_h, 1, gdal.GDT_Float32)
                        outdata_S2.GetRasterBand(1).WriteArray(results[:vid_h, :vid_w, coef_index])
                        outdata_S2.FlushCache()
                        outdata_S2.SetGeoTransform(S2_new_gdal_transform)
                        outdata_S2.FlushCache()
                        outdata_S2.SetProjection(S2_proj)
                        outdata_S2.FlushCache()

            # for x in range(n_blocks):
                # TODO: would be nice to have a structure that controls these
                # name formats so we can use padded inter suffixes for nicer
                # sorting, or nest files to keep folder sizes small

    # Remove tmp files
    for file in os.listdir(out_path):
        if fnmatch.fnmatch(file, 'tmp_coefmap*'):
            os.remove(out_path / file)

    logger.info('Starting adding new asset to kwcoco json')

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

                    # COLD output was wrote based on transform information of coco_dset, so it aligned
                    warp_aux_to_img = coco_image.warp_img_from_vid

                    # Use the CocoImage helper which will augment the coco dictionary with
                    # your information.
                    coco_image.add_asset(new_fpath, channels=channels, width=asset_w,
                                            height=asset_h, warp_aux_to_img=warp_aux_to_img)
                    logger.info(f'Added to the asset {new_fpath}')

    if proc_context is not None:
        context_info = proc_context.stop()
        coco_dset.dataset['info'].append(context_info)

    # Write a modified kwcoco.json file
    logger.info(f'Writing kwcoco file to: {mod_coco_fpath}')
    coco_dset.fpath = mod_coco_fpath
    coco_dset._ensure_json_serializable()
    coco_dset.dump()
    logger.info(f'Finished writing kwcoco file to: {mod_coco_fpath}')


@profile
def get_gdal_transform(coco_dset, sensor_name):
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
    target_primary_asset = target_coco_img.primary_asset()
    target_primary_fpath = os.path.join(ub.Path(target_coco_img.bundle_dpath), target_primary_asset['file_name'])
    ref_image = gdal.Open(target_primary_fpath, gdal.GA_ReadOnly)
    trans = ref_image.GetGeoTransform()
    proj = ref_image.GetProjection()

    # Calculate the new GDAL transform
    original_affine = kwimage.Affine.from_gdal(trans)
    warp_affine = kwimage.Affine.coerce(target_coco_img.img['warp_img_to_vid']).inv()
    new_affine = original_affine @ warp_affine
    new_geotrans = tuple(new_affine.to_gdal())

    return new_geotrans, proj

if __name__ == '__main__':
    assemble_main()
