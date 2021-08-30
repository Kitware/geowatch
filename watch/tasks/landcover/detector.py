import logging
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import torch
from shapely.geometry import shape
from torchvision.transforms.functional import to_tensor

from .nets import LinkNet34

log = logging.getLogger(__name__)

# the output of the detector (value) maps to these entries
# feature_mapping[value + 1]
feature_mapping = [
    'NONE',
    'BH135',  # Agriculture, Paddy
    'EA010',  # Agriculture, General
    'BA040',  # Ocean < 10m depth
    'BH082',  # Lakes & Water
    'BH140',  # Rivers
    'BH160',  # Sabkha
    'BJ100',  # Permanent or Nearly Permanent Ice and/or Snow
    'DA010',  # Barren/Minimal Vegetation, Beaches, Non-herbaceous Alluvial Deposits/Fans
    'DB170',  # Sand dunes
    'AL020',  # Urban/Built-Up
    'EB010',  # Grassland
    'EB070',  # Shrub/Scrub
    'EC015',  # Forest, Deciduous, Evergreen
    'ED010',  # Wetland, Permanent/Herbaceous
    'AP030'  # Road
]

# outputs from the model
channels = [
    'rice_field',  # 0
    'cropland',  # 1
    'water',  # 2
    'inland_water',  # 3
    'river_or_stream',  # 4
    'sebkha',  # 5
    'snow_or_ice_field',  # 6
    'bare_ground',  # 7
    'sand_dune',  # 8
    'built_up',  # 9
    'grassland',  # 10
    'brush',  # 11
    'forest',  # 12
    'wetland',  # 13
    'road',  # 14
]

cmap8 = np.array([[0, 0, 0],  # 0  noInformation
                  [79, 235, 52],  # 1  Agriculture, Paddy
                  [235, 211, 52],  # 2  Agriculture, General
                  [114, 151, 255],  # 3  Ocean
                  [0, 152, 203],  # 4  Lakes & Water
                  [77, 182, 255],  # 5  Rivers
                  [168, 198, 227],  # 6  Sabkha
                  [30, 254, 254],  # 7  Permanent or Nearly Permanent Ice and/or Snow
                  [145, 141, 118],  # 8  Barren/Minimal Vegetation, Beaches, Non-herbaceous Alluvial Deposits/Fans
                  [255, 194, 168],  # 9  Sand dunes
                  [255, 141, 0],  # 10 Urban/Built-Up
                  [21, 255, 0],  # 11 Grassland
                  [72, 190, 54],  # 12 Shrub/Scrub
                  [200, 0, 255],  # 13 Forest, Deciduous, Evergreen
                  [100, 21, 255],  # 14 Wetland, Permanent/Herbaceous
                  [255, 0, 0],  # 15 Roads
                  ],
                 np.uint8)
cmap = cmap8 / 255.

# The nodata value in the output from the model
PRED_NODATA = -1


def run(model, img, metadata):
    if np.all(img == 0):
        log.warning('skipping all black image: gid:{}'.format(metadata['id']))
        return None

    img = preprocess(img)
    pred = predict_image(img, model)
    return pred


def preprocess(img):
    img = img.astype(np.float32)
    img = normalize(img)
    return img


def pad(fn):
    def wrapped(img, *args, **kwargs):
        pads = [(c - s % c) % c for s, c in zip(img.shape, (512, 512, 8))]
        # log.debug('{} pad with {}'.format(img.shape, pads))

        # pad right and bottom only
        img = np.pad(img, [(0, p) for p in pads], mode='edge')

        out = fn(img, *args, **kwargs)

        # remove padding
        out = out[:out.shape[0] - pads[0], :out.shape[1] - pads[1]]

        return out

    return wrapped


@pad
def predict_image(img, model):
    dtype = np.int8

    mask = get_nodata_mask(img)

    if not np.any(mask):
        pred = np.full(img.shape[:2], PRED_NODATA, dtype=dtype)
        return pred

    device = get_device()

    t_image = to_tensor(img).float().unsqueeze(0).to(device)

    output = model(t_image)

    pred = torch.softmax(output, dim=1)

    # convert tensor to numpy array
    pred = pred.squeeze().detach().cpu().numpy()

    pred = np.where(mask == True, pred, PRED_NODATA)  # NOQA

    # reorder axes to (height, width, num_channels)
    pred = np.moveaxis(pred, 0, -1)

    return pred


def normalize(image, low=2, high=98) -> np.array:
    normalized_bands = []
    for band in range(image.shape[2]):
        local_band = image[:, :, band]
        local_band = local_band[local_band != 0]
        a = np.percentile(local_band, low)
        b = np.percentile(local_band, high)
        local_band = (image[:, :, band] - a) / (b - a)
        local_band = np.clip(local_band, 0, 1)
        normalized_bands.append(local_band)
    return np.stack(normalized_bands, 2)


def get_nodata_mask(img, nodata=0):
    """
    Get a mask of a continous no data region.
    Return an numpy array with values True for data, False for no data
    """
    mask = np.ones(img.shape[:2], bool)
    img_nodata = img == nodata
    for i in range(img.shape[0]):
        if np.all(img_nodata[i, :]):
            mask[i, :] = False

    for i in range(img.shape[1]):
        if np.all(img_nodata[:, i]):
            mask[:, i] = False
    return mask


def get_features(img):
    """
    Generate features from image

    """

    img = img.astype(np.int16)

    mask = img != PRED_NODATA

    # geometries are in pixel coords
    geometries = rasterio.features.shapes(img, mask=mask)
    geometries = [(shape(geom), v) for geom, v in geometries]

    return geometries


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(filename, num_outputs, num_channels):
    if isinstance(filename, str):
        filename = Path(filename)
    torch.hub.set_dir('/tmp')
    model = LinkNet34(num_outputs=num_outputs, num_channels=num_channels)
    model.load_state_dict(torch.load(filename))
    device = get_device()
    log.debug('  device {}'.format(device))
    model.to(device)
    model.eval()
    return model
