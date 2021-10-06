# import gdal here first otherwise the import fails due to conflict with rasterio
# TODO fix
from osgeo import gdal
import json
import logging
import warnings
from pathlib import Path

import click
import kwcoco
import kwimage
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import WVRgbDataset
from .pl_highres_verify import MultiTaskModel, modify_bn
from ..landcover.detector import get_device
from ..landcover.predict import get_output_file
from ..landcover.utils import setup_logging

log = logging.getLogger(__name__)

CONFIG_FILE = Path(__file__).parent.joinpath('config.json')


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
    # model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), '/output/state_dict.pt')

    model = modify_bn(model, track_running_stats=False, bn_momentum=0.01)
    model.to(get_device())

    log.debug('processing images')
    dataloader = DataLoader(dataset, num_workers=0, batch_size=1, collate_fn=lambda x: x)
    for batch in tqdm(dataloader, miniters=1, unit='image'):
        assert len(batch) == 1
        try:
            img_info = dataset.dset.imgs[batch[0]['id']]
            results = model.test_step(batch, 0)
            gid, pred = results[0]
            info = _write_output(img_info, pred, output_dir=output_data_dir)
            output_dset.imgs[gid]['auxiliary'].append(info)

        except KeyboardInterrupt:
            log.info('interrupted')
            break
        except Exception:
            log.exception('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

    output_dset.dump(str(output_dset_filename), indent=2)
    log.info('output written to {}'.format(output_dset_filename))


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
    with open(CONFIG_FILE, 'r') as fp:
        return json.load(fp)


if __name__ == '__main__':
    setup_logging()
    torch.hub.set_dir('/tmp/weights')
    predict()
