import datetime
import logging
import warnings
from pathlib import Path

import click
import kwcoco
import kwimage
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from . import detector
from .datasets import L8asWV3Dataset, S2asWV3Dataset, S2Dataset
from .utils import setup_logging

from watch.utils.lightning_ext import util_globals
from watch.utils import util_parallel

log = logging.getLogger(__name__)


@click.command()
@click.option('--dataset', required=True, type=click.Path(exists=True), help='input kwcoco dataset')
@click.option('--deployed', required=True, type=click.Path(exists=True), help='pytorch weights file')
@click.option('--output', required=False, type=click.Path(), help='output kwcoco dataset')
@click.option('--num_workers', default=0, required=False, type=str, help='number of dataloading workers. Can be "auto"')
@click.option('--device', default='auto', required=False, type=str, help='auto, cpu, or integer of the device to use')
def predict(dataset, deployed, output, num_workers=0, device='auto'):
    coco_dset_filename = dataset
    weights_filename = Path(deployed)
    output_dset_filename = get_output_file(output)
    output_data_dir = output_dset_filename.parent.joinpath(
        output_dset_filename.name.split('.')[0])

    log.info('Input:          {}'.format(coco_dset_filename))
    log.info('Weights:        {}'.format(weights_filename))
    log.info('Output:         {}'.format(output_dset_filename))
    log.info('Output Images:  {}'.format(output_data_dir))

    try:
        device = int(device)
    except Exception:
        pass

    num_workers = util_globals.coerce_num_workers(num_workers)

    if weights_filename.stem == 'visnav_osm':
        #
        # This model was trained on 8-band WV3 data with 15 segmentation classes
        #
        ptdataset = ConcatDataset([
            L8asWV3Dataset(coco_dset_filename),
            S2asWV3Dataset(coco_dset_filename)
        ])
        model_outputs = [
            'rice_field', 'cropland', 'water', 'inland_water', 'river_or_stream',
            'sebkha', 'snow_or_ice_field', 'bare_ground', 'sand_dune', 'built_up',
            'grassland', 'brush', 'forest', 'wetland', 'road'
        ]
        assert len(model_outputs) == 15
        model = detector.load_model(weights_filename, num_outputs=15,
                                    num_channels=8, device=device)

    elif weights_filename.stem == 'visnav_sentinel2':
        #
        # This model was trained on 13-band Sentinel 2 data with 22 segmentation classes
        #
        ptdataset = S2Dataset(coco_dset_filename)
        model_outputs = [
            'forest_deciduous', 'forest_evergreen', 'brush', 'grassland', 'bare_ground',
            'built_up', 'cropland', 'rice_field', 'marsh', 'swamp',
            'inland_water', 'snow_or_ice_field', 'reef', 'sand_dune', 'sebkha',
            'ocean<10m', 'ocean>10m', 'lake', 'river', 'beach',
            'alluvial_deposits', 'med_low_density_built_up'
        ]
        assert len(model_outputs) == 22
        model = detector.load_model(weights_filename, num_outputs=22,
                                    num_channels=13, device=device)
    else:
        raise Exception('unknown weights file')

    log.info('Using {}'.format(type(ptdataset)))

    output_dset = kwcoco.CocoDataset(coco_dset_filename).copy()

    dataloader = DataLoader(ptdataset, num_workers=num_workers,
                            batch_size=None, collate_fn=lambda x: x)

    # Start the worker processes before we do threading
    dataloader_iter = iter(dataloader)

    # Create a queue that writes data to disk in the background
    writer = util_parallel.BlockingJobQueue(max_workers=num_workers)

    for img_info in tqdm(dataloader_iter, miniters=1):
        try:
            pred_filename, pred = _predict_single(
                img_info, model=model, model_outputs=model_outputs,
                output_dset=output_dset,
                output_dir=output_dset_filename.parent)

            if pred is not None:
                writer.submit(_write_worker, pred_filename, pred)

        except KeyboardInterrupt:
            log.info('interrupted')
            break
        except Exception:
            log.exception('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

    writer.wait_until_finished()

    # self.dset.dump(str(self.output_dset_filename)+'_orig.json', indent=2)
    output_dset.dump(str(output_dset_filename), indent=2)
    log.info('output written to {}'.format(output_dset_filename))


def _predict_single(img_info, model, model_outputs,
                    output_dset: kwcoco.CocoDataset,
                    output_dir: Path):
    """
    Modifies the coco dataset inplace, returns the data that needs to be
    written to disk.
    """
    gid = img_info['id']
    name = img_info['name']
    img = img_info['imgdata']

    pred = detector.run(model, img, img_info)

    if pred is None:
        return None, None

    if img_info.get('file_name'):
        dir = Path(img_info.get('file_name')).parent
    else:
        dir = Path(img_info['auxiliary'][0]['file_name']).parent

    pred_filename = output_dir.joinpath('_assets', dir, name + '_landcover.tif')

    info = {
        'file_name': str(pred_filename.relative_to(output_dir)),
        'channels': "|".join(model_outputs),
        'height': pred.shape[0],
        'width': pred.shape[1],
        'num_bands': pred.shape[2],
        'warp_aux_to_img': {'scale': [img_info['width'] / pred.shape[1],
                                      img_info['height'] / pred.shape[0]],
                            'type': 'affine'}
    }

    output_dset.imgs[gid]['auxiliary'].append(info)
    return (pred_filename, pred)


def _write_worker(pred_filename, pred):
    pred_filename.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        kwimage.imwrite(str(pred_filename), pred, backend='gdal',
                        compress='RAW', blocksize=64)


def get_output_file(output):
    default_output_filename = 'out_{}.kwcoco.json'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    if output is None:
        output_dir = Path('/tmp')
        return output_dir.joinpath(default_output_filename)
    else:
        output = Path(output)
        if output.is_dir():
            return output.joinpath(default_output_filename)
        else:
            return output


if __name__ == '__main__':
    setup_logging()
    predict()
