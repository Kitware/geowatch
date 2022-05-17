# -*- coding: utf-8 -*-
"""
@author: skakun

This scripts takes ortho-rectified Planet and baseline
Sentinel-2 scene (red B04) and coregister Planet to Sentinel-2.

The output is both VRT with new GCP and warped geotif (with compression).
Output filename will be {input_filename}_coreg.tif
If GCPs cannot be found or not enough, the output would be VRT pointing on the original input.

Examples:
single MS
python planet_to_s2.py 20210130_130826_96_2307_3B_AnalyticMS.tif T23KPQ_20220314T130251_B04.jp2 output_folder/

"""

try:
    import gdal
except ImportError:
    from osgeo import gdal

try:
    import osr
except ImportError:
    from osgeo import osr

import os
import time
import sys
import numpy as np
from skimage.registration import phase_cross_correlation


def print_usage():
    print("Usage: planet_to_s2.py path_to_input_planet path_to_baseline_s2 output_folder")
    return


def get_gcp_for_registration(master_ds, slave_ds, master_array, slave_array, w_size=64, w_step=64, pfname_out=''):
    # this function runs phase correlation and returns the GCPs and error for each tile w_size
    # for slave image
    # pixel,line,X,Y,error
    res = []
    # check if raster of the same size
    if (master_array.shape != slave_array.shape):
        print("[ERROR]: Primary and secondary images have different size")
        return res
    geo_slave = slave_ds.GetGeoTransform()
#    print(geo_slave)
    total_start_time = time.time()
    if (pfname_out != ''):
        f_out = open(pfname_out, 'w')
        f_out.write('j,i,X_geo_slave_adj,Y_geo_slave_adj,error,offset_pixels_x,offset_pixels_y\n')

    w_size_2 = int(w_size / 2.)
    for i in np.arange(w_size_2, master_array.shape[0], w_step): # y-axis
        for j in np.arange(w_size_2, master_array.shape[1], w_step): # x-axis
            if (i + w_size / 2.) > master_array.shape[0]:
                continue
            if (j + w_size / 2.) > master_array.shape[1]:
                continue
            master_array_window = master_array[(i - w_size_2):(i + w_size_2), (j - w_size_2):(j + w_size_2)]
            slave_array_window = slave_array[(i - w_size_2):(i + w_size_2), (j - w_size_2):(j + w_size_2)]
            if np.sum((master_array_window == 0) | (slave_array_window == 0)) > 0.5 * w_size * w_size:
                # print("Window (%s,%s) too many nodata (>50%%). Unable to correlate" % (i,j))
                s = "%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (j, i, 0.0, 0.0, np.nan, 0., 0.)
                res.append(s)
                continue
            # start_time = time.time()
            # offset_pixels, error, diffphase = register_translation(master_array_window, slave_array_window, 100)
            offset_pixels, error, diffphase = phase_cross_correlation(master_array_window, slave_array_window, upsample_factor=100)
            # end_time = time.time()
#            print( "Window (%s,%s) processed in %.5f sec" % (i,j,end_time-start_time))
#            print("\tDetected pixel offset (y, x) and error: (%.3f, %.3f) %.5f" %(offset_pixels[0], offset_pixels[1], error))
#            print "\tDetected pixel offset (y, x) and (error, CCmax_norm): (%.3f, %.3f) (%.5f, %.5f)" %(offset_pixels[0], offset_pixels[1], error, 1-error)
            # this is the center of window in slave coordinates
            X_geo_slave = geo_slave[0] + geo_slave[1] * j
            Y_geo_slave = geo_slave[3] + geo_slave[5] * i
            # adjusting due to offset
            X_geo_slave_adj = X_geo_slave + geo_slave[1] * offset_pixels[1]
            Y_geo_slave_adj = Y_geo_slave + geo_slave[5] * offset_pixels[0]
            s = "%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f" % (j, i, X_geo_slave_adj, Y_geo_slave_adj, error, offset_pixels[1], offset_pixels[0])
            res.append(s)
            if (pfname_out != ''):
                f_out.write(s + '\n')
    total_end_time = time.time()
    print("--- Total processing time %s seconds ---" % (total_end_time - total_start_time))
    if (pfname_out != ''):
        f_out.close()
    return res


def planet_to_s2_coregister(path_to_input_planet, path_to_baseline_s2, output_folder):
    error_threshold = 0.5 # treshold for peak magnitude of phase correlation is used for initial rejection of bad matches
    max_shift_threshold = 5 # in pixels, so in 3 m effectively ~15 m
    elev = 0.

    fname_planet = os.path.basename(path_to_input_planet)
    planet_ds = gdal.Open(path_to_input_planet)
    planet_gt = planet_ds.GetGeoTransform()
    planet_num_bands = planet_ds.RasterCount

    # Reference band
    red_target_band = 1
    if (planet_num_bands == 1):
        red_target_band = 1
    elif((planet_num_bands == 3) | (planet_num_bands == 4) | (planet_num_bands == 5)):
        red_target_band = 3
    elif(planet_num_bands == 8):
        red_target_band = 6

    # Spatial reference
    planet_proj = planet_ds.GetProjectionRef()
    proj_ref = osr.SpatialReference()
    proj_ref.ImportFromWkt(planet_proj) # Well known format
    proj4 = proj_ref.ExportToProj4()

    xsize = planet_ds.RasterXSize
    ysize = planet_ds.RasterYSize

    planet_xres = planet_gt[1]
    planet_yres = abs(planet_gt[5])

    ul_x = planet_gt[0]
    ul_y = planet_gt[3]
    lr_x = planet_gt[0] + planet_gt[1] * xsize
    lr_y = planet_gt[3] + planet_gt[5] * ysize

    x_min = min(ul_x, lr_x)
    y_min = min(lr_y, ul_y)
    x_max = max(ul_x, lr_x)
    y_max = max(lr_y, ul_y)

    if not (os.path.isdir(output_folder)):
        os.makedirs(output_folder)

    # Here we are checking projection for targ
    # print(f'{proj4}')

    # Creating S2 in the same projection/resolution as Planet
    string_options = f'-overwrite -r lanczos -te {x_min} {y_min} {x_max} {y_max} '\
                     f'-tr  {planet_xres} {planet_yres}'\
                     f'-t_srs "{proj4}" -co COMPRESS=DEFLATE'

    warp_options = gdal.WarpOptions(gdal.ParseCommandLine(string_options))
    s2_resampled_ds = gdal.Warp(os.path.join(output_folder, f'{fname_planet[:-4]}_s2_tmp.tif'),
                                path_to_baseline_s2,
                                options=warp_options)

    s2_primary_arr = s2_resampled_ds.GetRasterBand(1).ReadAsArray()
    planet_secondary_arr = planet_ds.GetRasterBand(red_target_band).ReadAsArray()

    print(s2_primary_arr.shape)
    print(planet_secondary_arr.shape)

    # Running GCP search
    res_coregistration = get_gcp_for_registration(s2_resampled_ds, planet_ds,
                                                  s2_primary_arr, planet_secondary_arr,
                                                  w_size=64, w_step=64) # production mode

    # Multi spectral
    fname_gcp = fname_planet[:-4] + '_gcp.txt'
    f_gcp = open(os.path.join(output_folder, fname_gcp), 'w')

    num_gcp = 0
    for g in res_coregistration:
        array = [float(x) for x in g.split(',')] # pixel,line,X,Y,error,shift_x,shift_y
        if ((array[4] < error_threshold) & (not np.isnan(array[4])) & (abs(array[5]) < max_shift_threshold) & (abs(array[6]) < max_shift_threshold)):
            # We are using factor to convert to WV geometry
            pixel_ms = array[0]
            line_ms = array[1]
            output_str = "-gcp %.5f %.5fs %.5f %.5f %.f " % (pixel_ms, line_ms, array[2], array[3], elev)
            f_gcp.write(output_str)

            num_gcp = num_gcp + 1

    f_gcp.close()

    # if (pan_to_coreg):
    #    f_gcp_pan.close()

    if num_gcp > 5:
        com_gdal_translate_prefix = f'gdal_translate -of VRT --optfile {os.path.join(output_folder, fname_gcp)} ' \
                                    f'-r cubic -a_srs "{proj4}" -a_nodata 0 '
        com_gdalwarp_prefix = f'gdalwarp -overwrite -of GTiff -order 3 -et 0.05 -r cubic -co "COMPRESS=DEFLATE" ' \
                              f'-tr {planet_xres} {planet_yres} -te {x_min} {y_min} {x_max} {y_max} ' \
                              f'-t_srs "{proj4}" -srcnodata 0 -dstnodata 0 '

        fname_vrt = f'{fname_planet[:-4]}_coreg.vrt'
        fname_out = f'{fname_planet[:-4]}_coreg.tif'
        os.system(f'{com_gdal_translate_prefix} {path_to_input_planet} {os.path.join(output_folder, fname_vrt)}')
        os.system(f'{com_gdalwarp_prefix} {os.path.join(output_folder, fname_vrt)} {os.path.join(output_folder,  fname_out)}')

    else:
        print('Not enough GCPs - just VRT input files')
        # If we don't find enough GCPs - just vrt data
        fname_vrt = f'{fname_planet[:-4]}.vrt'
        fname_out = None
        os.system(f'gdal_translate -of VRT {path_to_input_planet} {os.path.join(output_folder, fname_vrt)}')

    planet_ds = None
    s2_resampled_ds = None

    # Deleting temporary files
    if os.path.isfile(os.path.join(output_folder, f'{fname_planet[:-4]}_s2_tmp.tif')):
        os.remove(os.path.join(output_folder, f'{fname_planet[:-4]}_s2_tmp.tif'))

    return fname_out


if __name__ == '__main__':
    num_arg = len(sys.argv)

    if (num_arg != 4):
        print_usage()
        sys.exit(1)

    if (num_arg == 4):
        path_to_input_planet = sys.argv[1]
        path_to_baseline_s2 = sys.argv[2]
        output_folder = sys.argv[3]

    planet_to_s2_coregister(path_to_input_planet, path_to_baseline_s2, output_folder)
