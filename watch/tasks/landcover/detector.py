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

# The nodata value in the output from the model
PRED_NODATA = -1


def run(model, img, metadata):
    if np.all(img == 0):
        log.warning('skipping all black image: gid:{}'.format(metadata['id']))
        return None

    img = preprocess(img)
    try:
        pred = predict_image(img, model)
    except Exception:
        log.error('error processing image with shape {}'.format(img.shape))
        raise
    return pred


def preprocess(img):
    img = img.astype(np.float32)
    img = normalize(img)
    return img


def pad(fn):
    def wrapped(img, *args, **kwargs):
        pads = [(c - s % c) % c for s, c in zip(img.shape, (512, 512, 1))]
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

    device = get_model_device(model)

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


def get_model_device(model):
    """
    Return the device associated with the model
    """
    device = next(model.parameters()).device
    return device


def load_model(filename, num_outputs, num_channels, device='auto'):
    if isinstance(filename, str):
        filename = Path(filename)
    torch.hub.set_dir('/tmp')
    model = LinkNet34(num_outputs=num_outputs, num_channels=num_channels)
    if device == 'auto':
        device = get_device()
    log.debug('  device {}'.format(device))
    model.to(device)
    device = get_model_device(model)  # ensure a proper torch.device
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model
