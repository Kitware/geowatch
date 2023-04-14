import logging
from pathlib import Path

import numpy as np
import torch

from .nets import UNetR
from .utils import setup_logging

log = logging.getLogger(__name__)


def run(model, image, metadata):
    invalid_mask_chan = np.isnan(image)
    invalid_mask = invalid_mask_chan.any(axis=2)
    if np.all(invalid_mask):
        return None

    image = normalize(image, invalid_mask_chan)
    try:
        image[invalid_mask] = 0
        pred = predict_image(image, model)
        pred[invalid_mask] = np.nan
    except Exception:
        log.error('error processing image: gid:{}'.format(metadata['id']))
        raise
    return pred


def pad(fn):
    def wrapped(image, *args, **kwargs):
        pads = [(c - s % c) % c for s, c in zip(image.shape, (512, 512, 1))]
        log.debug('{} pad with {}'.format(image.shape, pads))

        # pad right and bottom only
        image = np.pad(image, [(0, p) for p in pads], mode='edge')

        out = fn(image, *args, **kwargs)

        # remove padding
        out = out[:out.shape[0] - pads[0], :out.shape[1] - pads[1]]
        return out
    return wrapped


@pad
def predict_image(image, model):
    device = get_model_device(model)
    t_image = torch.from_numpy(image).permute(2, 0, 1)[None, ...].to(device)

    output = model(t_image)

    pred = torch.softmax(output, dim=1)
    pred = pred.squeeze().detach().cpu().numpy().transpose(1,2,0)
    return pred


def normalize(image, invalid_mask, low=2, high=98):
    normalized_bands = []
    for band_idx in range(image.shape[2]):
        band = image[:, :, band_idx]
        mask = invalid_mask[:,:, band_idx]
        if np.all(mask):
            raise ValueError

        valid_values = band[~mask]
        if len(valid_values) > 0:
            a = np.percentile(valid_values, low)
            b = np.percentile(valid_values, high)
        else:
            raise ValueError

        denom = (b - a)
        numer = (band - a)
        if denom > 0:
            band = numer / denom
        else:
            band = numer

        band = np.clip(band, 0, 1)
        normalized_bands.append(band)
    image_norm = np.stack(normalized_bands, 2)
    return image_norm


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_device(model):
    """
    Return the device associated with the model
    """
    device = next(model.parameters()).device
    return device


def load_model(filename, num_outputs, num_channels, device='auto'):
    torch.hub.set_dir('/tmp')
    if isinstance(filename, str):
        filename = Path(filename)
    if device == 'auto':
        device = get_device()
    log.debug('  device {}'.format(device))
    model = UNetR(num_outputs=num_outputs, num_channels=num_channels).to(device)
    model.load_state_dict(torch.load(filename, map_location=device))
    model.eval()
    return model
