import datetime
import logging
import warnings
from pathlib import Path

import click
import kwcoco
import kwimage
from torch.utils.data import DataLoader
from tqdm import tqdm

from watch.utils import util_parallel
from watch.utils.lightning_ext import util_globals
from . import detector
from .model_info import lookup_model_info
from .utils import setup_logging

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

    # try:
    #     device = int(device)
    # except Exception:
    from watch.utils.lightning_ext import util_device
    device = util_device.coerce_devices(device)[0]

    log.info('device = {}'.format(device))

    num_workers = util_globals.coerce_num_workers(num_workers)

    model_info = lookup_model_info(weights_filename)
    ptdataset = model_info.create_dataset(coco_dset_filename)
    model = model_info.load_model(weights_filename, device)

    log.info('Using {}'.format(type(model_info).__name__))

    output_dset = kwcoco.CocoDataset(coco_dset_filename)

    dataloader = DataLoader(ptdataset, num_workers=num_workers,
                            batch_size=None, collate_fn=lambda x: x)

    # Start the worker processes before we do threading
    dataloader_iter = iter(dataloader)

    # Create a queue that writes data to disk in the background
    writer = util_parallel.BlockingJobQueue(max_workers=num_workers)

    for img_info in tqdm(dataloader_iter, miniters=1):
        try:
            pred_filename, pred = _predict_single(
                img_info, model=model, model_outputs=model_info.model_outputs,
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
    """
    CommandLine:
        export CUDA_VISIBLE_DEVICES="1"
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15
        DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH/models/landcover/visnav_remap_s2_subset.pt"
        python -m watch.tasks.landcover.predict \
            --dataset=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json \
            --deployed=$DZYNE_LANDCOVER_MODEL_FPATH  \
            --device=0 \
            --num_workers="avail" \
            --output=$KWCOCO_BUNDLE_DPATH/data_dzyne_landcover.kwcoco.json

        python -m watch stats $KWCOCO_BUNDLE_DPATH/data_dzyne_landcover.kwcoco.json

        python -m watch visualize $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json \
            --animate=True --channels="built_up|forest|water" --skip_missing=True \
            --workers=4 --draw_anns=False
    """
    setup_logging()
    predict()
