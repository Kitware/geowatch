import logging
from pathlib import Path

import numpy as np
# import rasterio
# import rasterio.features
import torch
# from shapely.geometry import shape
# from torchvision.transforms.functional import to_tensor

from .nets import UNetR

log = logging.getLogger(__name__)

# The nodata value in the output from the model
PRED_NODATA = np.nan


def run(model, img, metadata):
    if np.all(img == 0):
        log.warning('skipping all black image: gid:{}'.format(metadata['id']))
        return None

    img = normalize(img)
    try:
        pred = predict_image(img, model)
    except Exception:
        log.error('error processing image with shape {}'.format(img.shape))
        raise
    return pred


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
def predict_image(img, model, device=None):
    """
    Example:
        >>> from watch.tasks.landcover.detector import *  # NOQA
        >>> import kwimage
        >>> def fake_model(t_image):
        ...     np_img = t_image.cpu().numpy()[0].transpose(1, 2, 0)
        ...     np_out = kwimage.gaussian_blur(np_img)
        ...     output = torch.from_numpy(np_out).permute(2, 0, 1)[None, ...]
        ...     return output
        >>> model = fake_model
        >>> device = 'cpu'
        >>> # orig_image = np.random.rand(32, 32, 3)
        >>> orig_image = kwimage.ensure_float01(kwimage.grab_test_image())
        >>> nan_poly = kwimage.Polygon.random(rng=421).scale(orig_image.shape[0] // 2)
        >>> # Note: the zero polygon will not be contiguous, so we wont see it in the output
        >>> zero_poly = kwimage.Polygon.random(rng=49120).scale(orig_image.shape[0])
        >>> img = orig_image.copy()
        >>> img = zero_poly.fill(img, 0)
        >>> img = nan_poly.fill(img, np.nan)
        >>> # Set bands of regions to be -0
        >>> img[10:100, :] = 0
        >>> img[0:100, -250:] = 0
        >>> img[:, -50:-10] = 0
        >>> pred = predict_image(img, model, device)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(img, pnum=(1, 2, 1), doclf=True)
        >>> kwplot.imshow(pred, pnum=(1, 2, 2))
    """
    dtype = np.float32

    mask = get_nodata_mask(img)

    if not np.any(mask):
        num_classes = model.out_channels
        h, w = img.shape[:2]
        pred = np.full((h, w, num_classes), PRED_NODATA, dtype=dtype)
    else:
        if device is None:
            device = get_model_device(model)

        masked_img = img.copy()
        masked_img[~mask] = 0
        assert np.isnan(masked_img).sum() == 0

        t_image = torch.from_numpy(masked_img).permute(2, 0, 1)[None, ...].to(device)
        # t_image = to_tensor(img).float().unsqueeze(0).to(device)

        output = model(t_image)

        pred = torch.softmax(output, dim=1)

        # convert tensor to numpy array
        pred = pred.squeeze().detach().cpu().numpy()

        pred = np.where(mask, pred, PRED_NODATA)  # NOQA

        # reorder axes to (height, width, num_channels)
        pred = np.moveaxis(pred, 0, -1)
    return pred


def normalize(image, low=2, high=98) -> np.array:
    """

    Example:
        >>> from watch.tasks.landcover.detector import *  # NOQA
        >>> import kwimage
        >>> # orig_image = np.random.rand(32, 32, 3)
        >>> orig_image = kwimage.ensure_float01(kwimage.grab_test_image())
        >>> image = kwimage.Polygon.random().scale(orig_image.shape[0]).fill(orig_image.copy(), np.nan)
        >>> output = normalize(image)
        >>> assert np.isnan(image).sum() == np.isnan(output).sum()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(image, pnum=(1, 2, 1), doclf=True)
        >>> kwplot.imshow(output, pnum=(1, 2, 2))

    Example:
        >>> from watch.tasks.landcover.detector import *  # NOQA
        >>> # Test 100% nan case
        >>> image = np.full((32, 32, 3), fill_value=np.nan)
        >>> output = normalize(image)
        >>> assert image is not output
        >>> assert np.all(np.isnan(output))
        >>> # Test 100% nan in a single band case
        >>> image = np.random.rand(32, 32, 3)
        >>> image[..., 1] = np.nan
        >>> output = normalize(image)
        >>> assert np.isnan(image).sum() == np.isnan(output).sum()
    """
    normalized_bands = []
    for band in range(image.shape[2]):
        local_band = image[:, :, band]
        #
        valid_mask = ~np.isnan(local_band)
        if not valid_mask.any():
            pass
        else:
            valid_values = local_band[valid_mask]
            # TODO: do we need to handle this case anymore?
            # Should we assume nans are correctly inputed?
            # local_band = local_band[local_band != 0]
            # local_band = local_band[local_band > 0]
            if len(valid_values) > 0:
                a = np.percentile(valid_values, low)
                b = np.percentile(valid_values, high)
            else:
                a = b = 0
            denom = (b - a)
            numer = (image[:, :, band] - a)
            if denom > 0:
                local_band = numer / denom
            else:
                local_band = numer
        local_band = np.clip(local_band, 0, 1)
        normalized_bands.append(local_band)
    return np.stack(normalized_bands, 2)


def get_nodata_mask(img):
    """
    Return an numpy array with values True for data, False for no data
    """
    valid_mask = ~(np.isnan(img).any(axis=2))
    return valid_mask


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
    model = UNetR(num_outputs=num_outputs, num_channels=num_channels)
    if device == 'auto':
        device = get_device()
    log.debug('  device {}'.format(device))
    model.to(device)
    device = get_model_device(model)  # ensure a proper torch.device
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model
