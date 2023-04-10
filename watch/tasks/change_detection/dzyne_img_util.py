#original script: https://github.com/fangchangma/sparse-to-dense/blob/master/utils.lua

import cv2
import rasterio
import torch

import numpy as np
import tifffile as tiff

from PIL import Image
from rasterio.windows import Window

LOAD_ORIGINAL = True
RGB_ONLY = True
BAND_SWAP = False


def pad(img, pad_size=32):
    """
    Pad image on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array,
        tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad
    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def pad_enlarge(img, pad_size=1024):
    """
    Pad image on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array,
        tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height >= pad_size:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width >= pad_size:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_CONSTANT)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def unpad(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]


def minmax(img):

    img_dim = len(img.shape)
    if (img_dim < 3):
        img = np.expand_dims(img, axis=2)

    out = np.zeros_like(img).astype(np.float32)

    if img.sum() == 0:
        if img_dim < 3:
            return out[:, :, 0].astype(np.float32)
        else:
            return out.astype(np.float32)

    # c = img.min()
    # d = img.max()

    for i in range(img.shape[2]):

        c = img[:, :, i].min()
        d = img[:, :, i].max()

        t = (img[:, :, i] - c) / (float(d - c) + 0.001)
        out[:, :, i] = t

    if img_dim < 3:
        return out[:, :, 0].astype(np.float32)
    else:
        return out.astype(np.float32)


def equalizeRGB(img):

    img = minmax(img) * 255

    img_y_cr_cb = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    return img_rgb_eq

# This is from Antonio


def normalizeRGB(img, percent_range=(2, 98)):

    min_val, max_val = percent_range

    img = img.astype(np.float32)

    img_dim = len(img.shape)
    if (img_dim < 3):
        img = np.expand_dims(img, axis=2)

    for b in range(img.shape[2]):

        # min = np.percentile(img[:, :, b].flatten(), min_val)
        # max = np.percentile(img[:, :, b].flatten(), max_val)
        # pdb.set_trace()

        img_b = img[:, :, b]
        valid_values = img_b[img_b != 0]

        # perc_valid = float(valid_values.size) / float(img_b.shape[0] * img_b.shape[1])
        # print('valid percetn is:', perc_valid)
        # if perc_valid > 0.5:
        #    valid_values = img[:, :, b].flatten()
        # pdb.set_trace()

        if valid_values.size == 0:
            min = max = 0
        else:
            min, max = np.percentile(valid_values, (min_val, max_val))

        if min == max:
            img[:, :, b] = 255 * (img[:, :, b] - min)
        else:
            img[:, :, b] = 255 * (img[:, :, b] - min) / (max - min)

    img /= 255.0
    img[img < 0] = 0
    img[img > 1] = 1

    if img_dim < 3:
        return img[:, :, 0]
    else:
        return img


def load_image(file_name_rgb, file_name_tif):

    # pdb.set_trace()

    if RGB_ONLY:
        if file_name_rgb.lower().endswith(('.png', '.jpg', '.jpeg', 'jp2')):
            rgb = np.asarray(Image.open(str(file_name_rgb)))
        elif file_name_rgb.lower().endswith(('.tif', '.tiff')):
            rgb = tiff.imread(str(file_name_rgb))
        elif file_name_rgb.lower().endswith(('.ntf')):
            rgb = readRasterImage(file_name_rgb)[0]
        else:
            print("Not valid RGB image format")

        if rgb.max() > 255:
            rgb = (255 * minmax(rgb)).astype(np.uint8)

        if len(rgb.shape) < 3:
            rgb =  np.dstack((rgb, rgb, rgb))

        if rgb.shape[2] == 4 or RGB_ONLY or not BAND_SWAP:
            rgb = rgb[:, :, :3]
            if BAND_SWAP:
                rgb = rgb[:, :, [2, 1, 0]]
        else:
            rgb = rgb[:, :, [4, 2, 1]]

        if LOAD_ORIGINAL:
            return rgb

        # normalize it to [0, 1]
        rgb = minmax(rgb)

        tf = np.concatenate([rgb, rgb, rgb[:, :, 0:2]], axis=2)

    else:

        if file_name_rgb.lower().endswith(('.tif', '.tiff')):
            rgb = tiff.imread(str(file_name_tif))
        elif file_name_rgb.lower().endswith(('.ntf')):
            rgb = readRasterImage(file_name_rgb)[0]

        rgb = minmax(rgb)

        if len(rgb.shape) < 3:
            rgb =  np.dstack((rgb, rgb, rgb))

        #tf = tiff.imread(str(file_name_tif)).astype(np.float32) / (2 ** 11 - 1)

        if file_name_tif.lower().endswith(('.tif', '.tiff')):
            tf = tiff.imread(str(file_name_tif)).astype(np.float32)
        elif file_name_tif.lower().endswith(('.ntf')):
            tf = readRasterImage(file_name_tif)[0]

        tf = minmax(tf)

        if len(tf.shape) < 3:
            tf =  np.dstack((tf, tf, tf))

        if tf.shape[2] == 4:
            tf = np.concatenate([tf, tf], axis=2)

    if tf.shape[2] == 4 or rgb.shape[2] == 4 or RGB_ONLY or not BAND_SWAP:
        return np.concatenate([rgb[:, :, :3], tf], axis=2) * (2 ** 8 - 1)
    else:
        if tf.shape[2] <= 4:
            if not BAND_SWAP:
                return np.concatenate([rgb[:, :, :3], tf], axis=2) * (2 ** 8 - 1)
            else:
                return np.concatenate([rgb[:, :, [2, 1, 0]], tf], axis=2) * (2 ** 8 - 1)
        else:
            return np.concatenate([rgb[:, :, [4, 2, 1]], tf], axis=2) * (2 ** 8 - 1)

#-----------------------------------------------------------------------------


def readRasterImage(filename: str, geoProps={}, convert=False):
    """Reads ntf files

    [description]

    Decorators:
        task

    Arguments:
        filename {str} -- filename of the file to read in

    Keyword Arguments:
        geoProps {dict} -- additional properties, like bounding box (default: {{}})
        convert {bool} -- should raster be converted to float and normalized (default: {False})

    Returns:
        Tuple of (img, transform, bands, datatype)
    """

    print('processing %s' % filename)
    with rasterio.open(filename, "r") as src_ds:

        bands = src_ds.count  # num bands
        datatype = np.dtype(src_ds.dtypes[0])
        transform = src_ds.transform

        # --- Initilizing img ---
        # Running under the assumption all bands are the same data type
        # if bool(geoProps):
        #     vrt = WarpedVRT(src_ds)
        #     dst_window = vrt.window( *geoProps["bounds"])
        #     src_ds = vrt.read(1, window=dst_window)
        #     img = np.ndarray(
        #         (*src_ds.shape, bands),
        #         dtype=datatype)
        # else:
        img = np.ndarray(
            (src_ds.height, src_ds.width, bands),
            dtype=datatype)
        # --- End: Initilizing img ---

        # --- Copying imagery by ROIs ---

        # Imagery needs to be cut (cropped?) into ROIs that satisfy
        # a 32-pixel size based geometry, due to the network
        print(img.shape)

        # Creating ROIs
        width = 128
        shape = img.shape
        rois = []
        for y in range(0, shape[0], width):
            for x in range(0, shape[1], width):
                locx = x
                locy = y
                if x + width >= shape[1]:
                    locx = shape[1] - width
                if y + width >= shape[0]:
                    locy = shape[0] - width

                rois.append((locx, locx + width, locy, locy + width))
        # End: Creating ROIs

        # Copying by ROI, transposed s.t. the number of bands is the last
        # dimension (AD: is this openCV shennanigans?)
        for i in range(len(rois)):
            win = Window(int(rois[i][0]), int(rois[i][2]), width, width)
            for j in range(bands):
                # if bool(geoProps):
                #     data = src_ds[ int(rois[i][0]):int(rois[i][1]), int(rois[i][2]):int(rois[i][3]) ]
                # else:
                data = src_ds.read(j + 1, window=win)
                np.copyto(img[rois[i][2]:rois[i][3],
                              rois[i][0]:rois[i][1], j], data)
        # End: Copying by ROI

        if convert:
            img = img.astype(np.float32)
            if datatype == np.uint8:
                img /= 255.0

            if bands == 3:
                normalizeRGB(img)
            else:
                # normalize(img, bands, datatype)
                raise RuntimeError("Unsupported band")

        # --- End: Copying imagery by ROIs ---

        return (img, transform, bands, datatype)


#-----------------------------------------------------------------------------
# Convert tensor to numpy array
#-----------------------------------------------------------------------------

def tensor2np(input_image, if_normalize=True, percent_range=(3, 97)):
    """
    :param input_image: C*H*W / H*W
    :return: ndarray, H*W*C / H*W
    """
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        #  如果输入为灰度图C==1，则输出array，ndim==2；
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize and C == 3:
            '''
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
            #  add to prevent extreme noises in visual images
            image_numpy[image_numpy<0]=0
            image_numpy[image_numpy>255]=255
            image_numpy = image_numpy.astype(np.uint8)
            '''
            image_numpy = (255 * normalizeRGB(image_numpy, percent_range)).astype(np.uint8)

    return image_numpy

#-----------------------------------------------------------------------------
# Convert numpy array to tensor
#-----------------------------------------------------------------------------


def np_to_tensor(image):
    """
    input: nd.array: H*W*C/H*W
    """
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = np.transpose(image, [2, 0, 1])
        elif image.ndim == 2:
            image = np.newaxis(image, 0)
        image = torch.from_numpy(image)
        return image.unsqueeze(0)
