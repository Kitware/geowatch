"""
Created on Wed Mar 24 14:32:42 2021

@author: skakun
"""

from osgeo import gdal, osr
import os
import glob
import time
import sys
import numpy as np
import pandas as pd
# from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
import itertools
from collections import defaultdict

S2_BANDS = ['B%02d' % (x) for x in np.arange(1, 13)]
S2_BANDS.append('B8A')
S2_ANGLE_BANDS = ['SEA4', 'SEZ4', 'SOA4', 'SOZ4']
S2_ANCILLARY_RASTERS = ['cloudmask']


def print_usage():
    print("Usage: python s2_coreg_l1c.py input_folder_safe output_folder")
    return


def get_utm_zone(ds, form='epsg'):
    prj = ds.GetProjection()
    srs = osr.SpatialReference(wkt=prj)
    if (form == 'epsg'):
        #        return srs.GetUTMZone()
        return srs.GetAttrValue("AUTHORITY", 1)
    elif (form == 'proj4'):
        return srs.ExportToProj4()
    else:
        return ''


def get_gcp_for_registration(master_ds,
                             slave_ds,
                             master_array,
                             slave_array,
                             w_size=64,
                             w_step=64,
                             pfname_out=''):
    '''
   this function runs phase correlation and returns the GCPs and error for each tile w_size
   '''
    # for slave image
    # pixel,line,X,Y,error
    res = []
    # check if raster of the same size
    if (master_array.shape != slave_array.shape):
        print("Slave and master image have different size")
        return res
    geo_slave = slave_ds.GetGeoTransform()
    # print(geo_slave)
    total_start_time = time.time()
    if (pfname_out != ''):
        f_out = open(pfname_out, 'w')
        f_out.write(
            'j,i,X_geo_slave_adj,Y_geo_slave_adj,error,offset_pixels_x,offset_pixels_y\n'
        )

    w_size_2 = int(w_size / 2.)
    for i in np.arange(w_size_2, master_array.shape[0], w_step):  # y-axis
        for j in np.arange(w_size_2, master_array.shape[1], w_step):  # x-axis
            if (i + w_size / 2.) > master_array.shape[0]:
                continue
            if (j + w_size / 2.) > master_array.shape[1]:
                continue
            master_array_window = master_array[(i - w_size_2):(i + w_size_2),
                                               (j - w_size_2):(j + w_size_2)]
            slave_array_window = slave_array[(i - w_size_2):(i + w_size_2),
                                             (j - w_size_2):(j + w_size_2)]
            if np.sum((master_array_window == 0)
                      | (slave_array_window == 0)) > 0.5 * w_size * w_size:
                #print("Window (%s,%s) too many nodata (>50%%). Unable to correlate" % (i,j))
                s = "%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (j, i, 0.0, 0.0, np.nan,
                                                        0., 0.)
                res.append(s)
                continue
            # start_time = time.time()
            # offset_pixels, error, diffphase = register_translation(master_array_window, slave_array_window, 100)
            offset_pixels, error, diffphase = phase_cross_correlation(
                master_array_window, slave_array_window, upsample_factor=100)
            # end_time = time.time()
            #print( "Window (%s,%s) processed in %.5f sec" % (i,j,end_time-start_time))
            #          #print("\tDetected pixel offset (y, x) and error: (%.3f, %.3f) %.5f" %(offset_pixels[0], offset_pixels[1], error))
            #          #print "\tDetected pixel offset (y, x) and (error, CCmax_norm): (%.3f, %.3f) (%.5f, %.5f)" %(offset_pixels[0], offset_pixels[1], error, 1-error)
            # this is the center of window in slave coordinates
            X_geo_slave = geo_slave[0] + geo_slave[1] * j
            Y_geo_slave = geo_slave[3] + geo_slave[5] * i
            # adjusting due to offset
            X_geo_slave_adj = X_geo_slave + geo_slave[1] * offset_pixels[1]
            Y_geo_slave_adj = Y_geo_slave + geo_slave[5] * offset_pixels[0]
            s = "%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (
                j, i, X_geo_slave_adj, Y_geo_slave_adj, error,
                offset_pixels[1], offset_pixels[0])
            res.append(s)
            if (pfname_out != ''):
                f_out.write(s + '\n')
    total_end_time = time.time()
    print("--- Total processing time %s seconds ---" %
          (total_end_time - total_start_time))
    if (pfname_out != ''):
        f_out.close()
    return res


def ensure_baseline_scene(baseline_scene):
    '''
    Return baseline_scene from a dir, csv, or df as a valid dir or None.
    '''
    def get_granuledir(df):
        if len(df) == 0:
            print('[ERROR]: file %s is empty' % (baseline_scene))
            return
        df = df.squeeze()
        if 'granuledir' in df:
            return df['granuledir']

        raise NotImplementedError
        input_folder = os.path.commonpath(xmls)  # NOQA  # FIXME?
        candidates = glob.glob(
            os.path.join(input_folder, '*.SAFE', 'GRANULE', df['granule_id']))
        return candidates[0]

    if isinstance(baseline_scene, str):

        if os.path.isdir(baseline_scene):
            return baseline_scene

        elif os.path.isfile(baseline_scene):
            df = pd.read_csv(baseline_scene)
            return get_granuledir(df)

    elif isinstance(baseline_scene, pd.DataFrame):

        df = baseline_scene
        return get_granuledir(df)


def s2_coregister(granuledirs, output_folder, baseline_scene, tile):
    '''
    Coregister Sentinel-2 scenes and save them to disk.

    This assumes that all scenes belong to the same MGRS tile.

    Creates output scenes, GCPs, and log files in the following structure:
    {output_folder}/
      T{mgrs_tile_id}/
        {granule_id}/
          GCP/
            {band_file_used_for_coreg}_gcp_10.txt
            {band_file_used_for_coreg}_gcp_20.txt
            {band_file_used_for_coreg}_gcp_60.txt
          Log/
            {band_file_used_for_coreg}_coreg.log
          {band_file}.tif
          {band_file}.vrt

    Args:
        granuledirs: list of granuledirs for scenes to coregister.
            Should include baseline_scene.
        output_folder: to save coregistered scenes in the same format as input_folder
        baseline_scene: one of
            path to granuledir of baseline scene
            dataframe containing the granule_id of the baseline scene in safedirs' folder
            path to .csv containing the above dataframe
    '''
    error_threshold = 0.5  # threshold for peak magnitude of phase correlation is used for initial rejection of bad matches
    max_shift_threshold = 3
    elev = 0.
    base_band = 'B04'

    def get_band(granuledir):
        bands = glob.glob(
            os.path.join(granuledir, f'*{base_band}.tif'))
        assert len(bands) == 1
        return bands[0]

    # getting a primary baseline scene
    baseline_scene = ensure_baseline_scene(baseline_scene)
    if baseline_scene is None:
        print('[ERROR]: file %s with baseline scene not found' %
              (baseline_scene))
        return
    pfname_master = get_band(baseline_scene)

    # Harvesting all scenes
    pfnames = list(map(get_band, granuledirs))

    # run coregistration
    for x in pfnames:
        fname_base = os.path.basename(x)
        path_data = os.path.dirname(x)
        x = os.path.normpath(x)
        scene_id = os.path.basename(path_data)
        path_out_data = os.path.join(output_folder, "T{}".format(tile), scene_id)
        os.makedirs(path_out_data, exist_ok=True)

        if fname_base == os.path.basename(pfname_master):
            print('This is a master scene - just copy/translate to GTiff %s' %
                  (fname_base))
            for b in itertools.chain(S2_BANDS,
                                     S2_ANGLE_BANDS,
                                     S2_ANCILLARY_RASTERS):
                fname_band = os.path.basename(x.replace(base_band, b))
                if b in S2_BANDS:
                    pfname_band = os.path.join(path_data, fname_band)
                elif b in S2_ANGLE_BANDS:
                    pfname_band_base, _ = os.path.splitext(
                        os.path.join(path_data, fname_band))
                    pfname_band = "{}.tif".format(pfname_band_base)
                    if not os.path.isfile(pfname_band):
                        pfname_band = os.path.join(
                            path_data, "{}.tif".format(b))
                elif b in S2_ANCILLARY_RASTERS:
                    pfname_band = os.path.join(
                        path_data, "{}.tif".format(b))
                else:
                    raise NotImplementedError(
                        "Unsure how to build path to band "
                        "file '{}'".format(b))

                if((b in S2_ANGLE_BANDS or b in S2_ANCILLARY_RASTERS)
                   and not os.path.isfile(pfname_band)):
                    print("* Warning * Missing optional S2 band file '{}'"
                          ", skipping!".format(pfname_band))
                    continue

                fname_out = fname_band[:-4] + '.tif'
                com = 'gdal_translate -of GTiff -co "COMPRESS=DEFLATE" %s %s' % (
                    pfname_band, os.path.join(path_out_data, fname_out))

                os.system(com)

        else:
            print('Coregistration is performed!')
            # this is where GCP will be stored
            path_to_gcp = os.path.join(path_out_data, 'GCP')
            os.makedirs(path_to_gcp, exist_ok=True)
            # some info for logging
            path_to_log = os.path.join(path_out_data, 'Log')
            os.makedirs(path_to_log, exist_ok=True)

            # getting info on the master
            # Open master file and reading the array
            s2_master_ds = gdal.Open(pfname_master)
            s2_master_array = np.array(
                s2_master_ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            # Reading geotransform
            s2_master_gt = s2_master_ds.GetGeoTransform()
            xsize = s2_master_ds.RasterXSize
            ysize = s2_master_ds.RasterYSize
            x_res = s2_master_gt[1]
            y_res = s2_master_gt[5]
            ul_x = s2_master_gt[0]
            ul_y = s2_master_gt[3]
            lr_x = s2_master_gt[0] + xsize * x_res
            lr_y = s2_master_gt[3] + ysize * y_res
            x_min = min(ul_x, lr_x)
            y_min = min(ul_y, lr_y)
            x_max = max(ul_x, lr_x)
            y_max = max(ul_y, lr_y)

            # Getting UTM zone
            utm_epsg = get_utm_zone(s2_master_ds, form='epsg')
            utm_epsg = int(utm_epsg)

            # Working with the secondary image
            fname_slave = fname_base

            # Reading the secondary scene
            s2_slave_ds = gdal.Open(x)
            s2_slave_array = np.array(
                s2_slave_ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)

            # Running GCP search
            res_coregistration = get_gcp_for_registration(
                s2_master_ds,
                s2_slave_ds,
                s2_master_array,
                s2_slave_array,
                w_size=64,
                w_step=64)  # production mode
            # res_coregistration = get_gcp_for_registration(s2_master_ds, s2_slave_ds, s2_master_array, s2_slave_array, w_size=64, w_step=512) # debug mode

            if (len(res_coregistration) > 0):
                fname_log = fname_slave[:-4] + '_coreg.log'
                f_log = open(os.path.join(path_to_log, fname_log), 'w')
                f_log.write("==========Filtering==============\n")

                # we create 3 files targetting various resolutions: 10m, 20, 60m
                fname_gcp_10 = fname_slave[:-4] + '_gcp_10.txt'
                fname_gcp_20 = fname_slave[:-4] + '_gcp_20.txt'
                fname_gcp_60 = fname_slave[:-4] + '_gcp_60.txt'

                # These are files where we put GCP for subsequent warping
                f_10 = open(os.path.join(path_to_gcp, fname_gcp_10), 'w')
                f_20 = open(os.path.join(path_to_gcp, fname_gcp_20), 'w')
                f_60 = open(os.path.join(path_to_gcp, fname_gcp_60), 'w')

                num_gcp = 0
                for g in res_coregistration:
                    array = [float(x) for x in g.split(',')
                             ]  # pixel,line,X,Y,error,shift_x,shift_y
                    if ((array[4] < error_threshold) & (not np.isnan(array[4]))
                            & (abs(array[5]) < max_shift_threshold) &
                              (abs(array[6]) < max_shift_threshold)):
                        # For 10 m we select origianl values since B04 is 10 m
                        output_str_10 = "-gcp %s %s %.5f %.5f %.f " % (
                            array[0], array[1], array[2], array[3], elev)
                        f_10.write(output_str_10)
                        # For 20m, we divide by 2 pixel/line
                        output_str_20 = "-gcp %.5f %.5f %.5f %.5f %.f " % (
                            array[0] / 2., array[1] / 2., array[2], array[3],
                            elev)
                        f_20.write(output_str_20)
                        # For 60m, we divide by 6 pixel/line
                        output_str_60 = "-gcp %.5f %.5fs %.5f %.5f %.f " % (
                            array[0] / 6., array[1] / 6., array[2], array[3],
                            elev)
                        f_60.write(output_str_60)

                        num_gcp = num_gcp + 1
                        f_log.write(
                            "Point (%s,%s) with error %.5f and shift (%.5f, %.5f) PASSED threshold %.5f\n"
                            % (array[0], array[1], array[4], array[5],
                               array[6], error_threshold))
                    else:
                        # print "Point (%s,%s) with error %.5f did not pass threshold %.5f" % (array[0],array[1],array[4], error_threshold)
                        f_log.write(
                            "Point (%s,%s) with error %.5f and shift (%.5f, %.5f) NOT PASSED threshold %.5f\n"
                            % (array[0], array[1], array[4], array[5],
                               array[6], error_threshold))
                f_log.write("Total of good GCPs is %s" % (num_gcp))
                f_10.close()
                f_20.close()
                f_60.close()

                # now transforming files
                # TODO: this is easy to parallize
                for b in itertools.chain(S2_BANDS,  # ['B01', 'B04', 'B11']:
                                         S2_ANGLE_BANDS,  # ['SEA4, 'SEZ4', 'SOZ4', 'SOA4']
                                         S2_ANCILLARY_RASTERS):  # ['cloudmask']
                    if b in S2_BANDS:
                        fname_band = os.path.basename(x.replace(base_band, b))
                        pfname_band = os.path.join(path_data, fname_band)
                    elif b in S2_ANGLE_BANDS:
                        fname_band_base, _ = os.path.splitext(
                            os.path.basename(x.replace(base_band, b)))
                        pfname_band = os.path.join(
                            path_data, "{}.tif".format(fname_band_base))
                        if not os.path.isfile(pfname_band):
                            pfname_band = os.path.join(
                                path_data, "{}.tif".format(b))
                    elif b in S2_ANCILLARY_RASTERS:
                        pfname_band = os.path.join(
                            path_data, "{}.tif".format(b))
                    else:
                        raise NotImplementedError(
                            "Unsure how to build path to band "
                            "file '{}'".format(b))

                    if((b in S2_ANGLE_BANDS or b in S2_ANCILLARY_RASTERS)
                       and not os.path.isfile(pfname_band)):
                        print("* Warning * Missing optional S2 band file '{}'"
                              ", skipping!".format(pfname_band))
                        continue

                    # First, checking spatial resolution based on the band
                    fname_gcp = fname_gcp_10  # deafault value
                    x_res = 10
                    y_res = 10
                    resampling_method = 'cubic'
                    if ((b == 'B05') | (b == 'B06') | (b == 'B07') |
                        (b == 'B8A') | (b == 'B11') | (b == 'B12')):  # NOQA
                        fname_gcp = fname_gcp_20
                        x_res = 20
                        y_res = 20
                    elif ((b == 'B01') | (b == 'B09') | (b == 'B10')):
                        fname_gcp = fname_gcp_60
                        x_res = 60
                        y_res = 60
                    elif (b == 'cloudmask'):
                        fname_gcp = fname_gcp_20
                        x_res = 20
                        y_res = 20
                        resampling_method = 'near'

                    com_gdal_translate_prefix = 'gdal_translate -of VRT --optfile %s -r %s -a_srs "epsg:%s"' % (
                        os.path.join(path_to_gcp, fname_gcp), resampling_method, utm_epsg)
                    com_gdalwarp_prefix = 'gdalwarp -overwrite -of GTiff -order 3 -et 0.05 -r %s -co "COMPRESS=DEFLATE" -tr %s %s -te %s %s %s %s -t_srs "epsg:%s"' %\
                                          (resampling_method, abs(x_res), abs(x_res), x_min, y_min, x_max, y_max, utm_epsg)

                    fname_out_prefix = fname_base[:-8]
                    if fname_out_prefix == '':
                        fname_vrt = '%s.vrt' % (b)
                        fname_out = '%s.tif' % (b)
                    else:
                        fname_vrt = '%s_%s.vrt' % (fname_base[:-8], b)
                        fname_out = '%s_%s.tif' % (fname_base[:-8], b)

                    os.system('%s %s %s' %
                              (com_gdal_translate_prefix, pfname_band,
                               os.path.join(path_out_data, fname_vrt)))
                    os.system('%s %s %s' %
                              (com_gdalwarp_prefix,
                               os.path.join(path_out_data, fname_vrt),
                               os.path.join(path_out_data, fname_out)))
                f_log.close()
            else:
                print('[ERROR]: cannot do co-registration: no GCP found')
            s2_master_ds = None
            s2_slave_ds = None


def s2_coregister_all_tiles(input_folder_or_safedirs,
                            output_folder,
                            baseline_scenes={},
                            dry_run=False,
                            granuledirs_input=False):
    '''
    Driver script to group S2 scenes into MGRS tiles, find baseline scenes for them,
    coregister them within each tile, and save them to disk.

    Args:
        input_folder_or_safedirs: list of S2 L1C .SAFE scenes to coregister
        output_folder: to save coregistered scenes in the same format as input_folder
        baseline_scenes: dict where keys are MGRS tiles and values are one of
            path to granuledir of baseline scene
            dataframe containing the granule_id of the baseline scene in safedirs' folder
            path to .csv containing the above dataframe

            If None, looks in f'{input_folder}/{mgrs_tile}.baseline.scene.csv'

            Finally, if that is not found, creates the baseline scenes.
        dry_run: if True, just populate the dict of scenes to coregister.

    Returns:
        scenes, baseline_scenes:
            Dict[tile: str, scenes: list(path)],
            Dict[tile: str, scenes: path]

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # xdoctest: +REQUIRES(--ta1)
        >>> from watch.datacube.registration.s2_coreg_l1c import *
        >>> from watch.demo.sentinel2_demodata import grab_sentinel2_product
        >>> #
        >>> safedirs = [str(grab_sentinel2_product(i).path) for i in range(3)]
        >>> output_folder = './coregistered'
        >>> scenes, baseline_scenes = s2_coregister_all_tiles(safedirs, output_folder)
        >>> assert len(scenes['52SDG']) == 3
        >>> #
        >>> # clean up
        >>> for d in safedirs:
        >>>     os.system(f'rm -r {d}')
        >>> os.system(f'rm -r {output_folder}')
    '''
    # Get list of all granuledirs
    if isinstance(input_folder_or_safedirs, list):
        if granuledirs_input:
            granuledirs = input_folder_or_safedirs
            input_folder = os.path.commonpath(granuledirs)
        else:
            safedirs = input_folder_or_safedirs
            input_folder = os.path.commonpath(safedirs)
            granuledirs = list(
                itertools.chain.from_iterable(
                    glob.glob(os.path.join(s, 'GRANULE', f'*{os.path.sep}'))
                    for s in safedirs))
    else:
        input_folder = input_folder_or_safedirs
        granuledirs = glob.glob(
            os.path.join(input_folder, '*.SAFE', 'GRANULE', f'*{os.path.sep}'))

    granuledirs = list(map(os.path.normpath, granuledirs))

    # extract MGRS tiles from them
    scenes = defaultdict(list)
    if granuledirs_input:
        for g in granuledirs:
            tile = os.path.basename(os.path.normpath(g)).split('_')[3][1:]
            scenes[tile].append(g)
    else:
        for g in granuledirs:
            tile = os.path.basename(os.path.normpath(g)).split('_')[1][1:]
            scenes[tile].append(g)

    def ensure_baseline_scenes(baseline_scenes):
        '''
        Do ensure_baseline_scene for each tile.

        Add additional defaults of:
        1. f'{tile}.baseline.scene.csv'
        2. output of s2_baseline_scene.find_baseline_scene()
        '''
        for tile in scenes.keys():
            scene = ensure_baseline_scene(baseline_scenes.get(tile))
            if scene is None:
                csv = os.path.join(input_folder, f'{tile}.baseline.scene.csv')
                if os.path.isfile(csv):
                    scene = csv
                else:
                    from watch.datacube.registration.s2_baseline_scene import find_baseline_scene
                    baseline_scenes.update(**find_baseline_scene(
                        [os.path.join(g, 'MTD_TL.xml') for g in granuledirs],
                        return_paths=True))
                    scene = baseline_scenes[tile]
                scene = ensure_baseline_scene(scene)
                assert scene is not None
                assert scene in scenes[tile]
                baseline_scenes[tile] = scene

    ensure_baseline_scenes(baseline_scenes)

    # run coregistration
    if not dry_run:
        for tile in scenes:
            s2_coregister(scenes[tile], output_folder, baseline_scenes[tile], tile)

    return scenes, baseline_scenes


def main():
    if len(sys.argv) < 4:
        print_usage()
        return

    mgrs_tile = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]

    s2_coregister(mgrs_tile, input_folder, output_folder)


if __name__ == '__main__':
    main()
