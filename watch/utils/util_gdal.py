import kwimage
import os
import ubelt as ub


'''
References:
    https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-multi
    https://gis.stackexchange.com/a/241810
    https://trac.osgeo.org/gdal/wiki/UserDocs/GdalWarp#WillincreasingRAMincreasethespeedofgdalwarp
    https://github.com/OpenDroneMap/ODM/issues/778

TODO test this and see if it's safe to add:
    --config GDAL_PAM_ENABLED NO
Removes .aux.xml sidecar files and puts them in the geotiff metadata
ex. histogram from fmask
https://stackoverflow.com/a/51075774
https://trac.osgeo.org/gdal/wiki/ConfigOptions#GDAL_PAM_ENABLED
https://gdal.org/drivers/raster/gtiff.html#georeferencing
'''
gdalwarp_performance_opts = ub.paragraph('''
        -multi
        --config GDAL_CACHEMAX 15%
        -wm 15%
        -co NUM_THREADS=ALL_CPUS
        -wo NUM_THREADS=1
        ''')


def gdal_multi_warp(in_fpaths, out_fpath, space_box, local_epsg, nodata=None, rpcs=None,
                    blocksize=256, compress='DEFLATE', use_perf_opts=False):
    """
    Ignore:
        # Uses data from the data cube with extra=1
        from watch.cli.coco_align_geotiffs import *  # NOQA
        cube, region_df = SimpleDataCube.demo(with_region=True, extra=True)
        local_epsg = 32635
        space_box = kwimage.Polygon.from_shapely(region_df.geometry.iloc[1]).bounding_box().to_ltrb()
        dpath = ub.ensure_app_cache_dir('smart_watch/test/gdal_multi_warp')
        out_fpath = join(dpath, 'test_multi_warp.tif')
        in_fpath1 = cube.coco_dset.get_image_fpath(2)
        in_fpath2 = cube.coco_dset.get_image_fpath(3)
        in_fpaths = [in_fpath1, in_fpath2]
        rpcs = None
        gdal_multi_warp(in_fpaths, out_fpath, space_box, local_epsg, rpcs)
    """
    # Warp then merge
    import tempfile

    # Write to a temporary file and then rename the file to the final
    # Destination so ctrl+c doesn't break everything
    tmp_out_fpath = ub.augpath(out_fpath, prefix='.tmp.')

    tempfiles = []  # hold references
    warped_gpaths = []
    for in_fpath in in_fpaths:
        tmpfile = tempfile.NamedTemporaryFile(suffix='.tif')
        tempfiles.append(tmpfile)
        tmp_out = tmpfile.name
        gdal_single_warp(in_fpath, tmp_out, space_box, local_epsg, rpcs=rpcs,
                         nodata=nodata, blocksize=blocksize, compress=compress,
                         use_perf_opts=use_perf_opts)
        warped_gpaths.append(tmp_out)

    if nodata is not None:
        from watch.utils import util_raster
        valid_polygons = []
        for tmp_out in warped_gpaths:
            sh_poly = util_raster.mask(tmp_out, tolerance=10,
                                       default_nodata=nodata)
            valid_polygons.append(sh_poly)
        valid_areas = [p.area for p in valid_polygons]
        # Determine order by valid data
        warped_gpaths = list(ub.sorted_vals(ub.dzip(warped_gpaths, valid_areas)).keys())
        warped_gpaths = warped_gpaths[::-1]
    else:
        # Last image is copied over earlier ones, but we expect first image to
        # be the primary one, so reverse order
        warped_gpaths = warped_gpaths[::-1]

    merge_cmd_parts = ['gdal_merge.py']
    if nodata is not None:
        merge_cmd_parts.extend(['-n', str(nodata)])
    merge_cmd_parts.extend(['-o', tmp_out_fpath])
    merge_cmd_parts.extend(warped_gpaths)
    merge_cmd = ' '.join(merge_cmd_parts)
    cmd_info = ub.cmd(merge_cmd_parts, check=True)
    if cmd_info['ret'] != 0:
        print('\n\nCOMMAND FAILED: {!r}'.format(merge_cmd))
        print(cmd_info['out'])
        print(cmd_info['err'])
        raise Exception(cmd_info['err'])
    os.rename(tmp_out_fpath, out_fpath)

    if 0:
        datas = []
        for p in warped_gpaths:
            d = kwimage.imread(p)
            d = kwimage.normalize_intensity(d, nodata=0)
            datas.append(d)

        import kwplot
        kwplot.autompl()
        combo = kwimage.imread(out_fpath)
        combo = kwimage.normalize_intensity(combo, nodata=0)
        datas.append(combo)
        kwplot.imshow(kwimage.stack_images(datas, axis=1))

        datas2 = []
        for p in in_fpaths:
            d = kwimage.imread(p)
            d = kwimage.normalize_intensity(d, nodata=0)
            datas2.append(d)
        kwplot.imshow(kwimage.stack_images(datas2, axis=1), fnum=2)


def gdal_single_warp(in_fpath, out_fpath, space_box, local_epsg, nodata=None, rpcs=None,
                     blocksize=256, compress='DEFLATE', use_perf_opts=False):
    r"""
    TODO:
        - [ ] This should be a kwgeo function?

    Ignore:
        in_fpath =
        s3://landsat-pds/L8/001/002/LC80010022016230LGN00/LC80010022016230LGN00_B1.TIF?useAnon=true&awsRegion=US_WEST_2

        gdalwarp 's3://landsat-pds/L8/001/002/LC80010022016230LGN00/LC80010022016230LGN00_B1.TIF?useAnon=true&awsRegion=US_WEST_2' foo.tif

    aws s3 --profile iarpa cp s3://kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif foo.tif

    gdalwarp 's3://kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif' bar.tif

    Note:
        Proof of concept for warp from S3:

        aws s3 --profile iarpa ls s3://kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/

        gdalinfo \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/LC08_L2SP_017039_20190404_20211102_02_T1_T17RMP_B7_BRDFed.tif"

        gdalwarp \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            -te_srs epsg:4326 \
            -te -81.51 29.99 -81.49 30.01 \
            -t_srs epsg:32617 \
            -overwrite \
            -of COG \
            -co OVERVIEWS=AUTO \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/0bff43f318c14a97b19b682a36f28d26/LC08_L2SP_017039_20190404_20211102_02_T1_T17RMP_B7_BRDFed.tif" \
            partial_crop2.tif
        gdalinfo partial_crop2.tif
        kwplot partial_crop2.tif

        gdalinfo \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            --config AWS_CONFIG_FILE "$HOME/.aws/config" \
            --config CPL_AWS_CREDENTIALS_FILE "$HOME/.aws/credentials" \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif"

        gdalwarp \
            --config AWS_DEFAULT_PROFILE "iarpa" \
            --config AWS_CONFIG_FILE "$HOME/.aws/config" \
            --config CPL_AWS_CREDENTIALS_FILE "$HOME/.aws/credentials" \
            -te_srs epsg:4326 \
            -te -43.51 -23.01 -43.49 -22.99 \
            -t_srs epsg:32723 \
            -overwrite \
            -of COG \
            "/vsis3/kitware-smart-watch-data/processed/ta1/drop1/mtra/f9e7e52029bb4dfaadfe306e92641481/S2A_MSI_L2A_T23KPQ_20190509_20211103_SR_B05.tif" \
            partial_crop.tif
        kwplot partial_crop.tif
    """
    # Data is from geo-pandas so this should be traditional order
    lonmin, latmin, lonmax, latmax = space_box.data[0]

    # Coordinate Reference System of the "te" crop coordinates
    # te_srs = spatial reference of query points
    crop_coordinate_srs = 'epsg:4326'

    # NUM_THREADS=2

    # Coordinate Reference System of the "target" destination image
    # t_srs = target spatial reference for output image
    if local_epsg is None:
        target_srs = 'epsg:4326'
    else:
        target_srs = 'epsg:{}'.format(local_epsg)

    # Use the new COG output driver
    template_parts = [
        '''
        gdalwarp
        --debug off
        -te {xmin} {ymin} {xmax} {ymax}
        -te_srs {crop_coordinate_srs}
        -t_srs {target_srs}
        -of COG
        -co OVERVIEWS=AUTO
        -co BLOCKSIZE={blocksize}
        -co COMPRESS={compress}
        -overwrite
        '''
    ]

    template_kw = {
        'crop_coordinate_srs': crop_coordinate_srs,
        'target_srs': target_srs,
        'ymin': latmin,
        'xmin': lonmin,
        'ymax': latmax,
        'xmax': lonmax,
        'blocksize': blocksize,
        'compress': compress,
        'SRC': in_fpath,
        'DST': out_fpath,
    }
    if nodata is not None:
        # TODO: Use cloudmask?
        template_parts.append(
            '''
            -srcnodata {NODATA_VALUE}
            ''')
        template_kw['NODATA_VALUE'] = nodata

    # HACK TO FIND an appropirate DEM file
    if rpcs is not None:
        dems = rpcs.elevation
        if hasattr(dems, 'find_reference_fpath'):
            # TODO: get a better DEM path for this image if possible
            dem_fpath, dem_info = dems.find_reference_fpath(latmin, lonmin)
            template_parts.append(ub.paragraph(
                '''
                -rpc -et 0
                -to RPC_DEM={dem_fpath}
                '''))
            template_kw['dem_fpath'] = dem_fpath
        else:
            dem_fpath = None
            template_parts.append('-rpc -et 0')

    if compress == 'RAW':
        compress = 'NONE'

    if use_perf_opts:
        template_parts.append(gdalwarp_performance_opts)
    else:
        # use existing options
        template_parts.append(ub.paragraph(
            '''
            -multi
            --config GDAL_CACHEMAX 500
            -wm 500
            -co NUM_THREADS=2
            '''))

    template_parts.append('{SRC} {DST}')
    template = ' '.join(template_parts)

    command = template.format(**template_kw)
    cmd_info = ub.cmd(command, verbose=0)  # NOQA
    if cmd_info['ret'] != 0:
        print('\n\nCOMMAND FAILED: {!r}'.format(command))
        print(cmd_info['out'])
        print(cmd_info['err'])
        raise Exception(cmd_info['err'])
