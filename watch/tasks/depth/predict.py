# import gdal here first otherwise the import fails due to conflict with rasterio
# TODO fix
from osgeo import gdal  # NOQA
import json
import logging
import warnings
from functools import partial
from pathlib import Path

import click
import kwcoco
import kwimage
import numpy as np
import torch
import torchvision.transforms
from medpy.filter.smoothing import anisotropic_diffusion
from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import modules_monkeypatch  # NOQA
from .datasets import WVRgbDataset
from .pl_highres_verify import MultiTaskModel, modify_bn, dfactor, local_utils
from .utils import process_image_chunked
from ..landcover.detector import get_device
from ..landcover.predict import get_output_file
from ..landcover.utils import setup_logging

log = logging.getLogger(__name__)


@click.command()
@click.option('--dataset', required=True, type=click.Path(exists=True), help='input kwcoco dataset')
@click.option('--deployed', required=True, type=click.Path(exists=True), help='pytorch weights file')
@click.option('--output', required=False, type=click.Path(), help='output kwcoco dataset')
def predict(dataset, deployed, output):
    coco_dset_filename = dataset
    weights_filename = Path(deployed)
    output_dset_filename = get_output_file(output)
    output_data_dir = output_dset_filename.parent.joinpath(
        output_dset_filename.name.split('.')[0])

    log.info('Input:          {}'.format(coco_dset_filename))
    log.info('Weights:        {}'.format(weights_filename))
    log.info('Output:         {}'.format(output_dset_filename))
    log.info('Output Images:  {}'.format(output_data_dir))

    output_dset = kwcoco.CocoDataset(coco_dset_filename).copy()

    # input data
    dataset = WVRgbDataset(coco_dset_filename)

    # model
    log.debug('loading model')
    model = MultiTaskModel(config=_load_config())
    state_dict = torch.load(weights_filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    model = modify_bn(model, track_running_stats=False, bn_momentum=0.01)
    model.to(get_device())

    log.debug('processing images')
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, collate_fn=lambda x: x)
    for batch in tqdm(dataloader, miniters=1, unit='image', disable=True):
        assert len(batch) == 1
        img_info = batch[0]
        gid = img_info['id']
        try:
            image = img_info['imgdata']
            pred = process_image_chunked(image,
                                         partial(run_inference, model=model),
                                         chip_size=(2048, 2048, 3)
                                         )

            # get clean img_info
            img_info = dataset.dset.imgs[gid]
            info = _write_output(img_info, pred, output_dir=output_data_dir)
            aux = output_dset.imgs[gid].get('auxiliary', [])
            aux.append(info)
            output_dset.imgs[gid]['auxiliary'] = aux

        except KeyboardInterrupt:
            log.info('interrupted')
            break
        except Exception:
            log.exception('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

    output_dset.dump(str(output_dset_filename), indent=2)
    log.info('output written to {}'.format(output_dset_filename))


def run_inference(image, model):
    with torch.no_grad():
        image_float = image / 255.0
        mean = np.mean(image_float.reshape(-1, image_float.shape[-1]), axis=0)
        std = np.std(image_float.reshape(-1, image_float.shape[-1]), axis=0)

        batch2 = {
            "image": torchvision.transforms.functional.to_tensor(image)[None, ...],
            "image_mean": torch.from_numpy(mean)[None, ...],
            "image_std": torch.from_numpy(std)[None, ...],
        }

        batch2 = local_utils.batch_to_cuda(batch2)

        pred2, batch2 = model(batch2, tta=True)

        output_depth = pred2['depth'][0, 0, :, :].cpu().data.numpy()
        output_label = pred2['seg'][0, 0, :, :].cpu().data.numpy()

        weighted_depth = dfactor * output_depth

        alpha = 0.9
        weighted_seg = alpha * output_label + (1.0 - alpha) * np.minimum(0.99, weighted_depth / 70.0)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tmp2 = 255 * anisotropic_diffusion(weighted_seg, niter=1, kappa=100, gamma=0.8)
        weighted_final = ndimage.median_filter(tmp2.astype(np.uint8), size=7)

    return weighted_final


def _write_output(img_info, image, output_dir):
    if img_info.get('file_name'):
        dir = Path(img_info.get('file_name')).parent
    else:
        dir = Path(img_info['auxiliary'][0]['file_name']).parent

    dirs = list(dir.parts)
    if dirs[0] == '_assets':
        dirs = dirs[1:]

    pred_filename = output_dir.joinpath('_assets', *dirs, img_info['name'] + '_depth.tif')

    info = {
        'file_name': str(pred_filename.relative_to(output_dir)),
        'channels': 'depth',
        'height': image.shape[0],
        'width': image.shape[1],
        'num_bands': 1,
        'warp_aux_to_img': {'scale': [img_info['width'] / image.shape[1],
                                      img_info['height'] / image.shape[0]],
                            'type': 'affine'}
    }
    pred_filename.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        kwimage.imwrite(str(pred_filename),
                        image,
                        backend='gdal',
                        compress='deflate')

    return info


def _load_config():
    from importlib import resources as importlib_resources
    fp = importlib_resources.open_text('watch.tasks.depth', 'config.json')
    return json.load(fp)


if __name__ == '__main__':
    setup_logging()
    torch.hub.set_dir('/tmp/weights')
    predict()
