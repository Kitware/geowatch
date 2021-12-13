import os
import numpy as np
from skimage import io
from scipy.ndimage import zoom


def adjust_shape(Im, s):
    "Adjust shape of grayscale image Im to s."

    # crop if necesary
    Im = Im[:s[0], :s[1]]
    si = Im.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0 , s[1] - si[1])

    return np.pad(Im, ((0, p0), (0, p1)), 'edge')


def read_sentinel_img(path, normalize=False):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")

    Im = np.stack((r, g, b), axis=2).astype('float32')

    return Im


def read_sentinel_img_4(path, normalize=False):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    Im = np.stack((b, g, r, nir), axis=2).astype('float32')

    return Im


def read_sentinel_img_leq20(path, normalize=False):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")
    s = r.shape

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    Im = np.stack((b, g, r, ir1, ir2, ir3, nir, nir2, swir2, swir3), axis=2).astype('float32')

    return Im


def read_sentinel_img_leq60(path, normalize=False):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif"), 6), s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif"), 6), s)
    swirc = adjust_shape(zoom(io.imread(path + im_name + "B10.tif"), 6), s)

    Im = np.stack((uv, b, g, r, ir1, ir2, ir3, nir, nir2, wv, swirc, swir2, swir3), axis=2).astype('float32')

    return Im


def read_sentinel_img_trio(img_path, mask_path, num_channels=13, normalize=False):
    """Read cropped Sentinel-2 image pair and change map."""
    #read images
    if num_channels == 3:
        I1 = read_sentinel_img(img_path + '/imgs_1/', normalize)
        I2 = read_sentinel_img(img_path + '/imgs_2/', normalize)
    elif num_channels == 4:
        I1 = read_sentinel_img_4(img_path + '/imgs_1/', normalize)
        I2 = read_sentinel_img_4(img_path + '/imgs_2/', normalize)
    elif num_channels == 10:
        I1 = read_sentinel_img_leq20(img_path + '/imgs_1/', normalize)
        I2 = read_sentinel_img_leq20(img_path + '/imgs_2/', normalize)
    elif num_channels == 13:
        I1 = read_sentinel_img_leq60(img_path + '/imgs_1/', normalize)
        I2 = read_sentinel_img_leq60(img_path + '/imgs_2/', normalize)
    cm = io.imread(mask_path + '/cm/cm.png', as_gray=True) != 0

    # crop if necessary
    s1 = I1.shape
    s2 = I2.shape
    I2 = np.pad(I2, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), 'edge')

    return I1, I2, cm
