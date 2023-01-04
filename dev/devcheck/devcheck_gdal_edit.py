

def test_gdal_edit_data():
    import kwimage
    import numpy as np
    data = (np.random.rand(256, 256) * 100).astype(np.int16)
    data[20:30, 20:80] = -9999
    data[90:120, 30:50] = 0
    src_fpath = 'test.tif'
    dst_fpath = 'result.tif'
    kwimage.imwrite(src_fpath, data, backend='gdal', nodata_value=-9999, overviews=4)

    nodata_value = -9999
    src_fpath =  '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/./US_R007/L8/affine_warp/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0_nir.tif'
    src_fpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/CN_C001/L8/affine_warp/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0_swir22.tif'
    data = kwimage.imread(src_fpath)
    from watch.utils import util_gdal
    dset = util_gdal.GdalDataset.open(src_fpath)
    self = dset

    proj = self.GetProjection()
    transform = self.GetGeoTransform()
    crs = self.GetSpatialRef()  # NOQA

    overviews = self.get_overview_info()[0]

    new_data = data.copy()
    new_data[new_data == 0] = nodata_value

    r = util_gdal.GdalDataset.open(dst_fpath)  # NOQA

    kwimage.imwrite(dst_fpath, data, transform=transform, crs=proj, nodata_value=nodata_value, overviews=overviews)

    import rasterio as rio
    src = rio.open(src_fpath, 'r')
    dst = rio.open(dst_fpath, 'w', **src.profile)

    with rio.open(src_fpath, 'r') as src:
        src.profile
        new = src.read()
        mask = (new == 0)
        nodata_value = -9999
        new[mask] = nodata_value
        with rio.open(dst_fpath, 'w', **src.profile) as dst:
            dst.write(new)

    # foo = kwimage.imread('result.tif')
    """
    gdal_calc.py -A test.tif --outfile=result.tif --calc="(-9999 * (A == 0)) + ((A != 0) * A)"
    echo "----"
    gdalinfo test.tif
    echo "----"
    gdalinfo result.tif

    gdalinfo /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/./US_R007/L8/affine_warp/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0/crop_20151126T160000Z_N34.190052W083.941277_N34.327136W083.776956_L8_0_nir.tif


    gdalinfo /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-SC/BR_R005_0036_box/WV/affine_warp/crop_20200220T130000Z_S23.434718W046.499921_S23.424402W046.492905_WV_0/crop_20200220T130000Z_S23.434718W046.499921_S23.424402W046.492905_WV_0_blue.tif


    gdalinfo /home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/Drop4-BAS/CN_C001/L8/affine_warp/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0/crop_20200823T020000Z_N30.114986E119.908343_N30.593740E120.466058_L8_0_swir22.tif

    """
    # foo = kwimage.imread('result.tif')
    # band = z.GetRasterBand(1)
    # from osgeo import gdal
    # info = gdal.Info(z, format='json')


def mwe():
    fpath = 'test.tif'
    new_fpath = 'result.tif'

    correct_nodata_value = -9999

    from osgeo import gdal
    src_dset = gdal.Open(fpath, gdal.GA_ReadOnly)
    assert src_dset.RasterCount == 1
    src_band = src_dset.GetRasterBand(1)
    num_overviews = src_band.GetOverviewCount()

    driver1 = gdal.GetDriverByName(str('MEM'))
    copy1 = driver1.CreateCopy(str(''), src_dset)
    src_dset.FlushCache()
    src_dset = None

    # Modify the pixel contents
    band = copy1.GetRasterBand(1)
    band_data = band.ReadAsArray()
    band_data[band_data == 0] = correct_nodata_value
    curr_nodat_value = band.GetNoDataValue()
    band.WriteArray(band_data)
    if curr_nodat_value != correct_nodata_value:
        band.SetNoDataValue(correct_nodata_value)

    overviewlist = (2 ** np.arange(1, num_overviews + 1)).tolist()
    copy1.BuildOverviews('AVERAGE', overviewlist)

    _options = [
        'BIGTIFF=YES',
        'TILED=YES',
        'BLOCKXSIZE={}'.format(256),
        'BLOCKYSIZE={}'.format(256),
    ]
    _options += ['COMPRESS={}'.format('DEFLATE')]
    _options.append('COPY_SRC_OVERVIEWS=YES')

    # Flush the in-memory dataset to an on-disk GeoTiff
    driver1 = None
    driver2 = gdal.GetDriverByName(str('GTiff'))
    copy2 = driver2.CreateCopy(new_fpath, copy1, options=_options)
    copy2.FlushCache()
    copy1 = None
    copy2 = None  # NOQA
    driver2 = None
