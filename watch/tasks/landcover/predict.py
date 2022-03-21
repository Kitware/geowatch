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
@click.option('--select_images', required=False, default=None, help='if specified, a jq operation to filter images')
@click.option('--select_videos', required=False, default=None, help='if specified, a jq operation to filter videos')
def predict(dataset, deployed, output, num_workers=0, device='auto', select_images=None, select_videos=None):
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

    input_dset = kwcoco.CocoDataset.coerce(coco_dset_filename)
    from watch.utils import kwcoco_extensions
    filtered_gids = kwcoco_extensions.filter_image_ids(
        input_dset, include_sensors=None, exclude_sensors=None,
        select_images=select_images, select_videos=select_videos)
    input_dset = input_dset.subset(filtered_gids)
    log.info('Selected input_dset = {!r}'.format(input_dset))

    model_info = lookup_model_info(weights_filename)
    ptdataset = model_info.create_dataset(input_dset)
    model = model_info.load_model(weights_filename, device)

    log.info('Using {}'.format(type(model_info).__name__))

    output_dset = input_dset.copy()

    dataloader = DataLoader(ptdataset, num_workers=num_workers,
                            batch_size=None, collate_fn=lambda x: x)

    # Start the worker processes before we do threading
    dataloader_iter = iter(dataloader)

    # Create a queue that writes data to disk in the background
    writer = util_parallel.BlockingJobQueue(max_workers=num_workers)

    for img_info in tqdm(dataloader_iter, miniters=1):
        try:
            pred_filename, pred, nodata = _predict_single(
                img_info, model=model, model_outputs=model_info.model_outputs,
                output_dset=output_dset,
                output_dir=output_dset_filename.parent)

            if pred is not None:
                writer.submit(_write_worker, pred_filename, pred, nodata)

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

    pred_filename = output_dir.joinpath('_assets/landcover', dir, name + '_landcover.tif')

    # Use the WATCH standards to transform float values with nans into
    # uint16 with nodata.
    from watch.tasks.fusion.predict import quantize_float01
    quant_pred, quantization = quantize_float01(pred)
    nodata = quantization['nodata']

    info = {
        'file_name': str(pred_filename.relative_to(output_dir)),
        'channels': "|".join(model_outputs),
        'height': pred.shape[0],
        'width': pred.shape[1],
        'num_bands': pred.shape[2],
        'quantization': quantization,
        'warp_aux_to_img': {'scale': [img_info['width'] / pred.shape[1],
                                      img_info['height'] / pred.shape[0]],
                            'type': 'affine'}
    }

    output_dset.imgs[gid]['auxiliary'].append(info)
    return (pred_filename, quant_pred, nodata)


def _write_worker(pred_filename, pred, nodata=None):
    pred_filename.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        kwimage.imwrite(str(pred_filename), pred, backend='gdal',
                        compress='DEFLATE', blocksize=128, nodata=nodata)


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


        # Drop3 Test
        # ==========
        export CUDA_VISIBLE_DEVICES="1"
        DVC_DPATH_SSH=$(python -m watch.cli.find_dvc --hardware=ssd)
        DVC_DPATH_HDD=$(python -m watch.cli.find_dvc --hardware=hdd)
        KWCOCO_BUNDLE_DPATH=$DVC_DPATH_SSH/Aligned-Drop3-TA1-2022-03-10
        DZYNE_LANDCOVER_MODEL_FPATH="$DVC_DPATH_HDD/models/landcover/visnav_remap_s2_subset.pt"
        python -m watch.tasks.landcover.predict \
            --dataset=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json \
            --deployed=$DZYNE_LANDCOVER_MODEL_FPATH  \
            --device=0 \
            --num_workers="avail" \
            --select_images '.sensor_coarse == "S2" and .frame_index < 100' \
            --select_videos '.name == "KR_R001"' \
            --output=$KWCOCO_BUNDLE_DPATH/test_data_dzyne_landcover.kwcoco.json

        python -m watch visualize $KWCOCO_BUNDLE_DPATH/test_data_dzyne_landcover.kwcoco.json \
            --animate=True --channels="built_up|forest|water" --skip_missing=True \
            --workers=4 --draw_anns=False
    """
    setup_logging()
    predict()
