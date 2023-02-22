import datetime
import warnings
import ubelt as ub
from pathlib import Path

import kwcoco
import kwimage
from torch.utils.data import DataLoader

from watch.utils import util_parallel
from watch.utils import util_progress
from . import detector
from .model_info import lookup_model_info
import scriptconfig as scfg


class LandcoverPredictConfig(scfg.DataConfig):
    dataset = scfg.Value(None, required=True, help='input kwcoco dataset')
    deployed = scfg.Value(None, required=True, help='pytorch weights file')
    output = scfg.Value(None, required=True, help='output kwcoco dataset')
    num_workers = scfg.Value(0, type=str, help='number of dataloading workers. Can be "auto"')
    device = scfg.Value('auto', type=str, help='auto, cpu, or integer of the device to use')
    select_images = scfg.Value(None, type=str, help='if specified, a jq operation to filter images')
    select_videos = scfg.Value(None, type=str, help='if specified, a jq operation to filter videos')


def predict(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.landcover.predict import *  # NOQA
        >>> import kwcoco
        >>> import watch
        >>> dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> dset = kwcoco.CocoDataset(dvc_data_dpath / 'Drop6/imganns-KR_R001.kwcoco.zip')
        >>> deployed = dvc_expt_dpath / 'models/landcover/sentinel2.pt'
        >>> kwargs = {
        >>>     'dataset': dset.fpath,
        >>>     'deployed': deployed,
        >>>     'output': ub.Path(dset.fpath).augment(stemsuffix='_landcover'),
        >>>     'select_images': '.sensor_coarse == "S2"',
        >>> }
        >>> cmdline = 0
        >>> predict(cmdline, **kwargs)
    """
    config = LandcoverPredictConfig.cli(cmdline=cmdline, data=kwargs)

    print('config = {}'.format(ub.urepr(dict(config), align=':', nl=1)))

    coco_dset_filename = config.dataset
    weights_filename = Path(config.deployed)
    output_dset_filename = get_output_file(config.output)

    deployed = ub.Path(config.deployed)
    if not deployed.is_file():
        raise ValueError('Landcover model does not exist')

    from watch.utils.lightning_ext import util_device
    device = util_device.coerce_devices(config.device)[0]
    print(f'device={device}')

    num_workers = util_parallel.coerce_num_workers(config.num_workers)

    input_dset = kwcoco.CocoDataset.coerce(coco_dset_filename)
    from watch.utils import kwcoco_extensions
    filtered_gids = kwcoco_extensions.filter_image_ids(
        input_dset, include_sensors=None, exclude_sensors=None,
        select_images=config.select_images, select_videos=config.select_videos)
    input_dset = input_dset.subset(filtered_gids)
    print('Selected input_dset = {!r}'.format(input_dset))

    model_info = lookup_model_info(weights_filename)
    ptdataset = model_info.create_dataset(input_dset)
    model = model_info.load_model(weights_filename, device)

    print('Using {}'.format(type(model_info).__name__))

    output_dset = input_dset.copy()

    dataloader = DataLoader(ptdataset, num_workers=num_workers,
                            batch_size=None, collate_fn=lambda x: x)

    # Start the worker processes before we do threading
    dataloader_iter = iter(dataloader)

    # Create a queue that writes data to disk in the background
    writer = util_parallel.BlockingJobQueue(max_workers=num_workers)

    pman = util_progress.ProgressManager()
    with pman:
        for img_info in pman.progiter(dataloader_iter, total=len(dataloader)):
            try:
                pred_filename, pred, nodata = _predict_single(
                    img_info, model=model, model_outputs=model_info.model_outputs,
                    output_dset=output_dset,
                    output_dir=output_dset_filename.parent)

                if pred is not None:
                    writer.submit(_write_worker, pred_filename, pred, nodata)

            except KeyboardInterrupt:
                print('interrupted')
                break
            except Exception as ex:
                print('warning: ex = {}'.format(ub.urepr(ex, nl=1)))
                print('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

    writer.wait_until_finished()

    output_dset.dump(str(output_dset_filename), indent=2)
    print('output written to {}'.format(output_dset_filename))


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

        DVC_EXPT_DPATH=$(smartwatch_dvc --tags=phase2_expt --hardware=auto)
        DVC_DATA_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware=auto)
        echo "
        DVC_DATA_DPATH = $DVC_DATA_DPATH
        DVC_EXPT_DPATH = $DVC_EXPT_DPATH
        "

        KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
        DZYNE_LANDCOVER_MODEL_FPATH="$DVC_EXPT_DPATH/models/landcover/sentinel2.pt"

        DATASET_FPATH=$KWCOCO_BUNDLE_DPATH/imganns-KR_R001.kwcoco.zip

        python -m watch.tasks.landcover.predict \
            --dataset=$DATASET_FPATH \
            --deployed=$DZYNE_LANDCOVER_MODEL_FPATH  \
            --device=0 \
            --num_workers=4 \
            --output=$KWCOCO_BUNDLE_DPATH/imganns-KR_R001_landcover.kwcoco.zip

        smartwatch stats $KWCOCO_BUNDLE_DPATH/data_dzyne_landcover.kwcoco.json

        smartwatch visualize $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json \
            --animate=True --channels="built_up|forest|water" --skip_missing=True \
            --workers=4 --draw_anns=False

    """
    predict()
