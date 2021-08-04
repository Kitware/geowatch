"""
containing some widely-used pre-processing method for change detection
"""
import numpy as np


def norm_img(img, channel_first=True):
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        max_value = np.max(img, axis=1, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=1, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (channel, height * width)
        max_value = np.max(img, axis=0, keepdims=True)  # (channel, 1)
        min_value = np.min(img, axis=0, keepdims=True)  # (channel, 1)
        diff_value = max_value - min_value
        nm_img = (img - min_value) / diff_value
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def stad_img(img, channel_first=True, get_para=False):
    """
    normalization image
    :param channel_first:
    :param img: (C, H, W)
    :return:
        norm_img: (C, H, W)
    """
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        mean = np.mean(img, axis=1, keepdims=True)  # (channel, 1)
        center = img - mean  # (channel, height * width)
        var = np.sum(np.power(center, 2), axis=1, keepdims=True) / (img_height * img_width)  # (channel, 1)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (height * width, channel)
        mean = np.mean(img, axis=0, keepdims=True)  # (1, channel)
        center = img - mean  # (height * width, channel)
        var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (img_height * img_width)  # (1, channel)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    print('mean is ', mean)
    print('std is ', std)
    if get_para:
        return nm_img, mean, std
    else:
        return nm_img


def CVA(img_X, img_Y, stad=False):
    # CVA has not affinity transformation consistency, so it is necessary to normalize multi-temporal images to
    # eliminate the radiometric inconsistency between them
    if stad:
        img_X = stad_img(img_X)
        img_Y = stad_img(img_Y)
    img_diff = img_X - img_Y
    L2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))
    return L2_norm


def main():
    from osgeo import gdal
    import time
    import imageio

    data_set_X = gdal.Open('../../../Dataset/Landsat/Taizhou/2000TM')  # data set X
    data_set_Y = gdal.Open('../../../Dataset/Landsat/Taizhou/2003TM')  # data set Y

    img_width = data_set_X.RasterXSize  # image width
    img_height = data_set_X.RasterYSize  # image height

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    channel, img_height, img_width = img_X.shape
    tic = time.time()
    L2_norm = CVA(img_X, img_Y)

    bcm = np.ones((img_height, img_width))

    raise NotImplementedError
    otsu = None
    thre = otsu(L2_norm.reshape(1, -1))
    bcm[L2_norm > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))
    imageio.imwrite('CVA_Taizhou.png', bcm)
    toc = time.time()
    print(toc - tic)


if __name__ == '__main__':
    main()
