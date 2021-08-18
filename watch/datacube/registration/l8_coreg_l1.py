# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:32:42 2021

@author: skakun
"""

from osgeo import gdal, osr
import os, shutil
import glob
import time
import sys
import numpy as np
import pandas as pd
# from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation
import scipy.ndimage
from shutil import copyfile

S2_BANDS = ['B%02d' % (x) for x in np.arange(1, 13)]
S2_BANDS.append('B8A')
L8_BANDS = ['B%01d' % (x) for x in np.arange(1, 12)]
L8_BANDS_EXTRA_COL1 = ['BQA']
L8_BANDS_EXTRA_COL2 = ['QA_PIXEL', 'QA_RADSAT', 'SAA', 'SZA', 'VAA', 'VZA']


def print_usage():
    print(
        "Usage: python l8_coreg_l1.py mgrs_tile input_folder_safe output_folder s2_folder"
    )
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
    # this function runs phase correlation and returns the GCPs and error for each tile w_size
    # for slave image
    # pixel,line,X,Y,error
    res = []
    # check if raster of the same size
    if (master_array.shape != slave_array.shape):
        print("Slave and master image have different size")
        return res
    geo_slave = slave_ds.GetGeoTransform()
    #print(geo_slave)
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
            start_time = time.time()
            # offset_pixels, error, diffphase = register_translation(master_array_window, slave_array_window, 100)
            offset_pixels, error, diffphase = phase_cross_correlation(
                master_array_window, slave_array_window, upsample_factor=100)
            end_time = time.time()
            #print( "Window (%s,%s) processed in %.5f sec" % (i,j,end_time-start_time))
            #          #print("\tDetected pixel offset (y, x) and error: (%.3f, %.3f) %.5f" %(offset_pixels[0], offset_pixels[1], error))
            #          #print "\tDetected pixel offset (y, x) and (error, CCmax_norm): (%.3f, %.3f) (%.5f, %.5f)" %(offset_pixels[0], offset_pixels[1], error, 1-error)
            #this is the center of window in slave coordinates
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


def l8_coregister(mgrs_tile, input_folder, output_folder, baseline_scene):
    error_threshold = 0.5  # treshold for peak magnitude of phase correlation is used for initial rejection of bad matches
    max_shift_threshold = 3
    elev = 0.
    s2_base_band = 'B04'
    l8_base_band = 'B4'
    padding_px = 5

    if not (os.path.isdir(output_folder)):
        os.makedirs(output_folder)

    def get_s2_band(granuledir):
        bands = glob.glob(
            os.path.join(granuledir, 'IMG_DATA', f'*_{s2_base_band}.jp2'))
        assert len(bands) == 1
        return bands[0]

    pfname_master = get_s2_band(baseline_scene)

    print(f'Primary scene found {pfname_master}')

    # Harveting all Landsat scenes
    pfname_list = glob.glob(os.path.join(input_folder, f'LC*_{l8_base_band}.TIF'))

    # Creating a dictionary of scenes
    # key is scene id
    db = dict()
    for pfname in pfname_list:
        fname = os.path.basename(pfname)
        scene_id = '_'.join(fname.split('_')[:-1])
        if not (scene_id in db.keys()):
            db[scene_id] = []
        db[scene_id] = pfname
    print(db)

    # Dict for primary scene
    db_master = dict()
    # adding T in front to be consistent with S2 code
    db_master[f'T{mgrs_tile}'] = pfname_master
    print(db_master)

    for tile in sorted(db_master.keys()):
        print(tile, os.path.basename(db_master[tile]))

        for x in db.keys():  # x is scene id here
            print(x, db[x])
            fname_base = os.path.basename(db[x])
            path_data = os.path.dirname(db[x])
            scene_id = x
            lc8_collection = scene_id.split('_')[5]
            scene_id_mgrs = f'{scene_id}_T{mgrs_tile}'
            path_out_data = os.path.join(output_folder, tile, scene_id_mgrs)

            if not (os.path.isdir(path_out_data)):
                os.makedirs(path_out_data)

            print('Coregistration is performed!')
            # this is where GCP will be stored
            path_to_gcp = os.path.join(path_out_data, 'GCP')
            # some info for logging
            path_to_log = os.path.join(path_out_data, 'Log')
            if not (os.path.isdir(path_to_gcp)):
                os.makedirs(path_to_gcp)
            if not (os.path.isdir(path_to_log)):
                os.makedirs(path_to_log)

            # getting info on the S2 baseline scene
            pfname_master = db_master[tile]
            # Open master file and reading the array
            s2_master_ds = gdal.Open(pfname_master)
            s2_master_array = np.array(
                s2_master_ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            # we resample s2 baseline to 30m from 10m
            s2_master_array = scipy.ndimage.zoom(s2_master_array,
                                                 1. / 3,
                                                 order=3)

            # Reading geotransform
            s2_master_gt = s2_master_ds.GetGeoTransform()
            xsize = s2_master_ds.RasterXSize
            ysize = s2_master_ds.RasterYSize
            x_res = s2_master_gt[1]  # 10 m
            y_res = s2_master_gt[5]  # 10 m
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

            # conerting UTM to lat/lon for using in the metadata update
            utm = osr.SpatialReference()
            utm.ImportFromEPSG(utm_epsg)
            wgs84 = osr.SpatialReference()
            wgs84.ImportFromEPSG(4326)
            tx = osr.CoordinateTransformation(utm, wgs84)
            (ul_x_lon, ul_y_lat, z) = tx.TransformPoint(ul_x, ul_y)
            (lr_x_lon, lr_y_lat, z) = tx.TransformPoint(lr_x, lr_y)

            # Working with the secondary LC8 image
            # First we convert original L8 scene into MGRS system
            # for this we create a temproray vrt file
            # we add padding of padding_px pixels
            com_prefix = (
                f'gdalwarp -overwrite -of VRT -r cubic -tr 30 30  '
                f'-te {x_min-30*padding_px} {y_min-30*padding_px} {x_max+30*padding_px} {y_max+30*padding_px} -t_srs "epsg:{utm_epsg}" -srcnodata 0 -dstnodata 0'
            )
            l8_fname_tmp_base_band = f'{scene_id_mgrs}_{l8_base_band}_tmp.vrt'
            com = f'{com_prefix} {db[x]} {os.path.join(path_out_data, l8_fname_tmp_base_band)}'
            os.system(com)

            # Reading the L8 secondary scene from temporary B4 file
            l8_slave_ds = gdal.Open(
                os.path.join(path_out_data, l8_fname_tmp_base_band))
            l8_slave_array = np.array(
                l8_slave_ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint16)
            # since we are adding padding we take a subset
            l8_slave_array = l8_slave_array[padding_px:(-padding_px),
                                            padding_px:(-padding_px)]

            # checking if the created tmp file has any data
            maxx = np.max(l8_slave_array)  # finding the max value
            if maxx == 0:  # if max is 0 then it means the subset over MGRS tile
                continue

            # Running GCP search
            res_coregistration = get_gcp_for_registration(
                s2_master_ds,
                l8_slave_ds,
                s2_master_array,
                l8_slave_array,
                w_size=64,
                w_step=64)  # production mode
            #res_coregistration = get_gcp_for_registration(s2_master_ds, l8_slave_ds, s2_master_array, l8_slave_array, w_size=64, w_step=512) # debug mode

            if (len(res_coregistration) > 0):
                fname_log = f'{scene_id_mgrs}_coreg.log'
                f_log = open(os.path.join(path_to_log, fname_log), 'w')
                f_log.write("==========Filtering==============\n")

                # we create two files targetting various resolutions: 15m (pan), 30m (everything else)
                fname_gcp_15 = f'{scene_id_mgrs}_gcp_15.txt'  # for pancrhomatic 15 m band
                fname_gcp_30 = f'{scene_id_mgrs}_gcp_30.txt'  # all other bands

                # These are files where we put GCP for subsequent warping
                f_15 = open(os.path.join(path_to_gcp, fname_gcp_15), 'w')
                f_30 = open(os.path.join(path_to_gcp, fname_gcp_30), 'w')

                num_gcp = 0
                for g in res_coregistration:
                    array = [float(x) for x in g.split(',')
                             ]  # pixel,line,X,Y,error,shift_x,shift_y
                    if ((array[4] < error_threshold) & (not np.isnan(array[4]))
                            & (abs(array[5]) < max_shift_threshold) &
                        (abs(array[6]) < max_shift_threshold)):
                        # For 30 m we select origianl values since B4 is 30 m
                        output_str_30 = "-gcp %s %s %.5f %.5f %.f " % (
                            array[0] + padding_px, array[1] + padding_px,
                            array[2] + 30 * padding_px,
                            array[3] - 30 * padding_px, elev)
                        f_30.write(output_str_30)
                        # For 15 m, we multiply by 2 pixel/line
                        output_str_15 = "-gcp %.5f %.5f %.5f %.5f %.f " % (
                            array[0] * 2. + padding_px, array[1] * 2. +
                            padding_px, array[2] + 30 * padding_px,
                            array[3] - 30 * padding_px, elev)
                        f_15.write(output_str_15)

                        num_gcp = num_gcp + 1
                        f_log.write(
                            "Point (%s,%s) with error %.5f and shift (%.5f, %.5f) PASSED threshold %.5f\n"
                            % (array[0], array[1], array[4], array[5],
                               array[6], error_threshold))
                    else:
                        #print "Point (%s,%s) with error %.5f did not pass threshold %.5f" % (array[0],array[1],array[4], error_threshold)
                        f_log.write(
                            "Point (%s,%s) with error %.5f and shift (%.5f, %.5f) NOT PASSED threshold %.5f\n"
                            % (array[0], array[1], array[4], array[5],
                               array[6], error_threshold))
                f_log.write("Total of good GCPs is %s" % (num_gcp))
                f_15.close()
                f_30.close()

                # Update metadata
                fname_meta = f'{scene_id}_MTL.txt'
                fname_meta_updated = f'{scene_id_mgrs}_MTL.txt'
                f_meta_out = open(
                    os.path.join(path_out_data, fname_meta_updated), 'w')
                with open(os.path.join(path_data, fname_meta)) as f_meta_in:
                    for line in f_meta_in:
                        line_update = line
                        if (('FILE_NAME_BAND' in line) | ('FILE_NAME_QUALITY' in line) | ('METADATA_FILE_NAME' in line) |\
                            ('FILE_NAME_METADATA_ODL' in line) | ('FILE_NAME_ANGLE_SENSOR' in line) | ('FILE_NAME_ANGLE_SOLAR' in line)):
                            line_update = line.replace(f'{scene_id}',
                                                       f'{scene_id_mgrs}')
                        if ('CORNER_UL_PROJECTION_X_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_x}\n'
                        if ('CORNER_UL_PROJECTION_Y_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_y}\n'
                        if ('CORNER_LR_PROJECTION_X_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_x}\n'
                        if ('CORNER_LR_PROJECTION_Y_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_y}\n'
                        if ('CORNER_UR_PROJECTION_X_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_x}\n'
                        if ('CORNER_UR_PROJECTION_Y_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_y}\n'
                        if ('CORNER_LL_PROJECTION_X_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_x}\n'
                        if ('CORNER_LL_PROJECTION_Y_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_y}\n'
                        if (('PANCHROMATIC_LINES' in line) |
                            ('PANCHROMATIC_SAMPLES' in line)):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {int(2*xsize/3.)}\n'
                        if (('REFLECTIVE_LINES' in line) |
                            ('REFLECTIVE_SAMPLES' in line) |
                            ('THERMAL_LINES' in line) |
                            ('THERMAL_SAMPLES' in line)):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {int(xsize/3.)}\n'
                        if ('CORNER_UL_LON_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_x_lon}\n'
                        if ('CORNER_UL_LAT_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_y_lat}\n'
                        if ('CORNER_LR_LON_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_x_lon}\n'
                        if ('CORNER_LR_LAT_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_y_lat}\n'
                        if ('CORNER_UR_LON_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_x_lon}\n'
                        if ('CORNER_UR_LAT_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_y_lat}\n'
                        if ('CORNER_LL_LON_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {ul_x_lon}\n'
                        if ('CORNER_LL_LAT_PRODUCT' in line):
                            tmp = line.split(' = ')
                            line_update = f'{tmp[0]} = {lr_y_lat}\n'
                        f_meta_out.write(line_update)
                f_meta_out.close()
                # Also copying original MTL file
                copyfile(os.path.join(path_data, fname_meta),
                         os.path.join(path_out_data, fname_meta))

                # now transforming files
                for b in L8_BANDS:
                    fname_band = f'{scene_id}_{b}.TIF'
                    pfname_band = os.path.join(path_data, fname_band)

                    # convert band to tmp file into MGRS gridding scheme
                    x_res = 30
                    y_res = 30
                    fname_gcp = fname_gcp_30  # default value
                    if (b == 'B8'):  # pan band
                        fname_gcp = fname_gcp_15
                        x_res = 15
                        y_res = 15

                    # convert band to MGRS tmp file
                    com_prefix = (
                        f'gdalwarp -overwrite -of VRT -r cubic -tr {abs(x_res)} {abs(y_res)} '
                        f'-te {x_min-abs(x_res)*padding_px} {y_min-abs(x_res)*padding_px} {x_max+abs(x_res)*padding_px} {y_max+abs(x_res)*padding_px} '
                        f'-t_srs "epsg:{utm_epsg}" -srcnodata 0 -dstnodata 0')
                    l8_fname_tmp_band = f'{scene_id_mgrs}_{b}_tmp.vrt'
                    com = f'{com_prefix} {pfname_band} {os.path.join(path_out_data, l8_fname_tmp_band)}'
                    os.system(com)

                    # running re-sampling using GCP
                    com_gdal_translate_prefix = f'gdal_translate -of VRT --optfile {os.path.join(path_to_gcp, fname_gcp)} -r cubic -a_srs "epsg:{utm_epsg}" -a_nodata 0'
                    com_gdalwarp_prefix = f'gdalwarp -overwrite -of GTiff -order 3 -et 0.05 -r cubic -co "COMPRESS=DEFLATE" -tr {abs(x_res)} {abs(y_res)} -te {x_min} {y_min} {x_max} {y_max} -t_srs "epsg:{utm_epsg}" -srcnodata 0 -dstnodata 0'
                    fname_vrt = f'{scene_id_mgrs}_{b}.vrt'
                    fname_out = f'{scene_id_mgrs}_{b}.tif'
                    os.system(
                        f'{com_gdal_translate_prefix} {os.path.join(path_out_data, l8_fname_tmp_band)} {os.path.join(path_out_data, fname_vrt)}'
                    )
                    os.system(
                        f'{com_gdalwarp_prefix} {os.path.join(path_out_data, fname_vrt)} {os.path.join(path_out_data, fname_out)}'
                    )

                if lc8_collection == '01':  # collection 1
                    extra_bands = L8_BANDS_EXTRA_COL1
                elif (lc8_collection == '02'):  # collection 2
                    extra_bands = L8_BANDS_EXTRA_COL2
                else:
                    extra_bands = []

                for b in extra_bands:
                    fname_band = f'{scene_id}_{b}.TIF'
                    pfname_band = os.path.join(path_data, fname_band)

                    # convert band to tmp file into MGRS gridding scheme
                    x_res = 30
                    y_res = 30
                    fname_gcp = fname_gcp_30
                    resampling_method = 'cubic'
                    nodata = 0

                    if ('QA' in b):
                        resampling_method = 'near'
                        nodata = 'None'

                    # convert band to MGRS tmp file
                    com_prefix = (
                        f'gdalwarp -overwrite -of VRT -r {resampling_method} -tr {abs(x_res)} {abs(y_res)} '
                        f'-te {x_min-abs(x_res)*padding_px} {y_min-abs(x_res)*padding_px} {x_max+abs(x_res)*padding_px} {y_max+abs(x_res)*padding_px} '
                        f'-t_srs "epsg:{utm_epsg}" -srcnodata {nodata} -dstnodata {nodata}'
                    )
                    l8_fname_tmp_band = f'{scene_id_mgrs}_{b}_tmp.vrt'
                    com = f'{com_prefix} {pfname_band} {os.path.join(path_out_data, l8_fname_tmp_band)}'
                    os.system(com)

                    # running re-sampling using GCP
                    com_gdal_translate_prefix = f'gdal_translate -of VRT --optfile {os.path.join(path_to_gcp, fname_gcp)} -r {resampling_method} -a_srs "epsg:{utm_epsg}" -a_nodata {nodata}'
                    com_gdalwarp_prefix = f'gdalwarp -overwrite -of GTiff -order 3 -et 0.05 -r {resampling_method} -co "COMPRESS=DEFLATE" -tr {abs(x_res)} {abs(y_res)} -te {x_min} {y_min} {x_max} {y_max} -t_srs "epsg:{utm_epsg}" -srcnodata {nodata} -dstnodata {nodata}'
                    fname_vrt = f'{scene_id_mgrs}_{b}.vrt'
                    fname_out = f'{scene_id_mgrs}_{b}.tif'
                    os.system(
                        f'{com_gdal_translate_prefix} {os.path.join(path_out_data, l8_fname_tmp_band)} {os.path.join(path_out_data, fname_vrt)}'
                    )
                    os.system(
                        f'{com_gdalwarp_prefix} {os.path.join(path_out_data, fname_vrt)} {os.path.join(path_out_data, fname_out)}'
                    )

                f_log.close()
            else:
                print('[ERROR]: cannot do co-registration: no GCP found')
            s2_master_ds = None
            s2_slave_ds = None

    return


def main():

    if len(sys.argv) < 5:
        print_usage()
        sys.exit(1)

    mgrs_tile = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]
    s2_folder = sys.argv[4]

    l8_coregister(mgrs_tile, input_folder, output_folder, s2_folder)


if __name__ == '__main__':

    main()
