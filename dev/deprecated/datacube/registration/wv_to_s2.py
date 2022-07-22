"""
Created on Mon Oct  4 14:43:31 2021

@author: skakun

This scripts takes ortho-rectified VHR data (e.g. WV-2, WV-3) and baseline
Sentinel-2 scene (red B04) and coregister VHR to Sentinel-2. The script can be run:
(i) for single multi-spectral (MS) VHR
(ii) both MS and PAN to make sure that MS and PAN are aligned
(iii) single PAN

The output is both VRT with new GCP and warped geotif (with compression).
Output filename will be {input_filename}_coreg.tif
If GCPs cannot be found or not enough, the output would VRT pointing on the original input.

Examples:
single MS
python wv_to_s2.py vhr_ms.tif T52SDG_20180602T021639_B04.tif output_folder/

MS + PAN
python wv_to_s2.py vhr_ms.tif vhr_pan.tif T52SDG_20180602T021639_B04.tif output_folder/

PAN (in case MS is not available)
python wv_to_s2.py vhr_pan.tif T52SDG_20180602T021639_B04.tif output_folder/

NOTE: running MS and PAN separately might result in misalignment between MS and PAN VHR
"""

from osgeo import gdal, osr
import os
import time
import sys
import numpy as np
from skimage.registration import phase_cross_correlation

from watch.utils.util_gdal import gdalwarp_performance_opts
import ubelt as ub


def gdal_translate_prefix(optfile, proj4):
    return ub.paragraph(f'''
        gdal_translate -of VRT --optfile {optfile}
        -r cubic -a_srs "{proj4}" -a_nodata 0
        ''')


def gdalwarp_prefix(xres, yres, x_min, y_min, x_max, y_max, proj4):
    return ub.paragraph(f'''
        gdalwarp -overwrite -of GTiff -order 3 -et 0.05 -r cubic
        -co "COMPRESS=DEFLATE" -co "BIGTIFF=YES"
        -tr {xres} {yres} -te {x_min} {y_min} {x_max} {y_max}
        -t_srs "{proj4}" -srcnodata 0 -dstnodata 0
        {gdalwarp_performance_opts}
        ''')


def print_usage():
    print("Usage: wv_to_s2.py path_to_input_wv {path_to_input_pan} path_to_baseline_s2 output_folder")
    return


def get_gcp_for_registration(master_ds, slave_ds, master_array, slave_array, w_size=64, w_step=64, pfname_out=''):
    # this function runs phase correlation and returns the GCPs and error for each tile w_size
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
        f_out.write('j,i,X_geo_slave_adj,Y_geo_slave_adj,error,offset_pixels_x,offset_pixels_y\n')

    w_size_2 = int(w_size / 2.)
    for i in np.arange(w_size_2, master_array.shape[0], w_step):  # y-axis
        for j in np.arange(w_size_2, master_array.shape[1], w_step):  # x-axis
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
            # print( "Window (%s,%s) processed in %.5f sec" % (i,j,end_time-start_time))
            # print("\tDetected pixel offset (y, x) and error: (%.3f, %.3f) %.5f" %(offset_pixels[0], offset_pixels[1], error))
            # print "\tDetected pixel offset (y, x) and (error, CCmax_norm): (%.3f, %.3f) (%.5f, %.5f)" %(offset_pixels[0], offset_pixels[1], error, 1-error)
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


def wv_to_s2_coregister(path_to_input_wv, path_to_baseline_s2, output_folder, path_to_input_wv_pan=''):
    error_threshold = 0.5  # treshold for peak magnitude of phase correlation is used for initial rejection of bad matches
    max_shift_threshold = 5  # in pixels, so in 10 m effectively ~30 m
    elev = 0.

# Checking existence of PAN band
    pan_to_coreg = False
    if (os.path.isfile(path_to_input_wv_pan) & (path_to_input_wv_pan != '')):
        pan_to_coreg = True

    fname_wv = os.path.basename(path_to_input_wv)
    wv_ds = gdal.Open(path_to_input_wv)
    wv_gt = wv_ds.GetGeoTransform()
    wv_num_bands = wv_ds.RasterCount

# Reference band
    red_target_band = 1
    if (wv_num_bands == 1):
        red_target_band = 1
    elif(wv_num_bands == 4):
        red_target_band = 3
    elif(wv_num_bands == 8):
        red_target_band = 5

# MS: Spatial reference
    wv_proj = wv_ds.GetProjectionRef()
    proj_ref = osr.SpatialReference()
    proj_ref.ImportFromWkt(wv_proj)  # Well known format
    proj4 = proj_ref.ExportToProj4()

    xsize = wv_ds.RasterXSize
    ysize = wv_ds.RasterYSize

    wv_xres = wv_gt[1]
    wv_yres = abs(wv_gt[5])

    ul_x = wv_gt[0]
    ul_y = wv_gt[3]
    lr_x = wv_gt[0] + wv_gt[1] * xsize
    lr_y = wv_gt[3] + wv_gt[5] * ysize

    x_min = ul_x
    y_min = lr_y
    x_max = lr_x
    y_max = ul_y

# print(f'{x_min} {y_min} {x_max} {y_max}')
# print(f'{wv_xres} {wv_yres}')

    if (pan_to_coreg):
        pan_ds = gdal.Open(path_to_input_wv_pan)
        pan_gt = pan_ds.GetGeoTransform()

        # PAN: Spatial reference
        # to make sure that they are the same
        pan_proj = pan_ds.GetProjectionRef()
        pan_proj_ref = osr.SpatialReference()
        pan_proj_ref.ImportFromWkt(pan_proj)  # Well known format

        if not (pan_proj_ref.IsSame(proj_ref)):
            print('[ERROR]: spatial system for MS and PAN not the same')
            print(f'MS: {wv_proj}')
            print(f'PAN: {pan_proj}')
            pan_to_coreg = False
            pan_ds = None

        if not (os.path.isdir(output_folder)):
            os.makedirs(output_folder)

# Here we are checking projection for targ
# print(f'{proj4}')

# Creating S2 in the same projection as WV
    string_options = f'-r cubic -te {x_min} {y_min} {x_max} {y_max} '\
                     f'-t_srs "{proj4}" -co COMPRESS=DEFLATE'

    warp_options = gdal.WarpOptions(gdal.ParseCommandLine(string_options))
    s2_resampled_ds = gdal.Warp(os.path.join(output_folder, f'{fname_wv[:-4]}_s2_tmp.tif'),
                                path_to_baseline_s2,
                                options=warp_options)

    s2_gt = s2_resampled_ds.GetGeoTransform()
    s2_xres = s2_gt[1]
    s2_yres = abs(s2_gt[5])

# Factors
    factor_x = s2_xres / wv_xres
    factor_y = s2_yres / wv_yres

# Resampling WV data to the S2 resolution
    string_options = f'-r average -tr {s2_xres} {s2_yres} -te {x_min} {y_min} {x_max} {y_max} '\
                     f'-t_srs "{proj4}" -co COMPRESS=DEFLATE'

    warp_options = gdal.WarpOptions(gdal.ParseCommandLine(string_options))
    wv_resampled_ds = gdal.Warp(os.path.join(output_folder, f'{fname_wv[:-4]}_vhr_tmp.tif'),
                                path_to_input_wv,
                                options=warp_options)

    s2_primary_arr = s2_resampled_ds.GetRasterBand(1).ReadAsArray()
    wv_secondary_arr = wv_resampled_ds.GetRasterBand(red_target_band).ReadAsArray()

# Running GCP search
    res_coregistration = get_gcp_for_registration(s2_resampled_ds, wv_resampled_ds,
                                                  s2_primary_arr, wv_secondary_arr,
                                                  w_size=64, w_step=64)  # production mode

# Multi spectral
    fname_gcp_ms = fname_wv[:-4] + '_gcp_ms.txt'
    f_gcp_ms = open(os.path.join(output_folder, fname_gcp_ms), 'w')

    if (pan_to_coreg):
        fname_gcp_pan = fname_wv[:-4] + '_gcp_pan.txt'
        f_gcp_pan = open(os.path.join(output_folder, fname_gcp_pan), 'w')

    num_gcp = 0
    for g in res_coregistration:
        array = [float(x) for x in g.split(',')]  # pixel,line,X,Y,error,shift_x,shift_y
        if ((array[4] < error_threshold) & (not np.isnan(array[4])) & (abs(array[5]) < max_shift_threshold) & (abs(array[6]) < max_shift_threshold)):
            # We are using factor to convert to WV geometry
            pixel_ms = array[0] * factor_x
            line_ms = array[1] * factor_y
            output_str = "-gcp %.5f %.5fs %.5f %.5f %.f " % (pixel_ms, line_ms, array[2], array[3], elev)
            f_gcp_ms.write(output_str)

            # for PAN: we will convert pixel/line from MS to pixel/line PAN
            # they are not perfectly coreg in terms of pixels/lines
            if (pan_to_coreg):
                # first convert to geo
                x_geo = wv_gt[0] + wv_gt[1] * pixel_ms
                y_geo = wv_gt[3] + wv_gt[5] * line_ms

                # convert geo to pixel/line in PAN
                # this will ensure that we apply the same transformation for PAN as with MS
                # to keep them co-registered
                pixel_pan = (x_geo - pan_gt[0]) / pan_gt[1]
                line_pan =  (y_geo - pan_gt[3]) / pan_gt[5]

                output_str = "-gcp %.5f %.5fs %.5f %.5f %.f " % (pixel_pan, line_pan, array[2], array[3], elev)
                f_gcp_pan.write(output_str)

            num_gcp = num_gcp + 1

    f_gcp_ms.close()

    if (pan_to_coreg):
        f_gcp_pan.close()

    if num_gcp > 5:
        com_gdal_translate_prefix = gdal_translate_prefix(optfile=os.path.join(output_folder, fname_gcp_ms), proj4=proj4)
        com_gdalwarp_prefix = gdalwarp_prefix(wv_xres, wv_yres, x_min, y_min, x_max, y_max, proj4)

        fname_vrt = f'{fname_wv[:-4]}_coreg.vrt'
        fname_out = f'{fname_wv[:-4]}_coreg.tif'
        os.system(f'{com_gdal_translate_prefix} {path_to_input_wv} {os.path.join(output_folder, fname_vrt)}')
        os.system(f'{com_gdalwarp_prefix} {os.path.join(output_folder, fname_vrt)} {os.path.join(output_folder, fname_out)}')

        if (pan_to_coreg):
            fname_pan = os.path.basename(path_to_input_wv_pan)

            # Parameters for PAN - we keep the same extent
            pan_xsize = pan_ds.RasterXSize
            pan_ysize = pan_ds.RasterYSize

            pan_xres = pan_gt[1]
            pan_yres = abs(pan_gt[5])

            pan_ul_x = pan_gt[0]
            pan_ul_y = pan_gt[3]
            pan_lr_x = pan_gt[0] + pan_gt[1] * pan_xsize
            pan_lr_y = pan_gt[3] + pan_gt[5] * pan_ysize

            pan_x_min = pan_ul_x
            pan_y_min = pan_lr_y
            pan_x_max = pan_lr_x
            pan_y_max = pan_ul_y

            com_gdal_translate_prefix = gdal_translate_prefix(optfile=os.path.join(output_folder, fname_gcp_pan), proj4=proj4)
            com_gdalwarp_prefix = gdalwarp_prefix(pan_xres, pan_yres, pan_x_min, pan_y_min, pan_x_max, pan_y_max, proj4)

            fname_pan_vrt = f'{fname_pan[:-4]}_coreg.vrt'
            fname_pan_out = f'{fname_pan[:-4]}_coreg.tif'
            os.system(f'{com_gdal_translate_prefix} {path_to_input_wv_pan} {os.path.join(output_folder, fname_pan_vrt)}')
            os.system(f'{com_gdalwarp_prefix} {os.path.join(output_folder, fname_pan_vrt)} {os.path.join(output_folder, fname_pan_out)}')
    else:
        print('Not enough GCPs - just VRT input files')
        # If we don't find enough GCPs - just vrt data
        fname_vrt = f'{fname_wv[:-4]}.vrt'
        os.system(f'gdal_translate -of VRT {path_to_input_wv} {os.path.join(output_folder, fname_vrt)}')
        if (pan_to_coreg):
            fname_pan = os.path.basename(path_to_input_wv_pan)
            fname_pan_vrt = f'{fname_pan[:-4]}.vrt'
            os.system(f'gdal_translate -of VRT {path_to_input_wv_pan} {os.path.join(output_folder, fname_pan_vrt)}')

    wv_ds = None
    s2_resampled_ds = None
    wv_resampled_ds = None
    if (pan_to_coreg):
        pan_ds = None

# Deleting temporary files
    if os.path.isfile(os.path.join(output_folder, f'{fname_wv[:-4]}_vhr_tmp.tif')):
        os.remove(os.path.join(output_folder, f'{fname_wv[:-4]}_vhr_tmp.tif'))
    if os.path.isfile(os.path.join(output_folder, f'{fname_wv[:-4]}_s2_tmp.tif')):
        os.remove(os.path.join(output_folder, f'{fname_wv[:-4]}_s2_tmp.tif'))

    return


if __name__ == '__main__':
    num_arg = len(sys.argv)

    if (num_arg < 4) | (num_arg > 5):
        print_usage()
        sys.exit(1)

    if (num_arg == 4):
        path_to_input_wv = sys.argv[1]
        path_to_baseline_s2 = sys.argv[2]
        output_folder = sys.argv[3]
        path_to_input_wv_pan = ''
    elif (num_arg == 5):
        path_to_input_wv = sys.argv[1]
        path_to_input_wv_pan = sys.argv[2]
        path_to_baseline_s2 = sys.argv[3]
        output_folder = sys.argv[4]

    wv_to_s2_coregister(path_to_input_wv, path_to_baseline_s2, output_folder, path_to_input_wv_pan)
