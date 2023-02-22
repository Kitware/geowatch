r"""
Prediction script for landcover features.

Given a checkout of the model and drop6 data, the following demos computing and
visualizing a subset of the features.

CommandLine:

    DVC_EXPT_DPATH=$(smartwatch_dvc --tags=phase2_expt --hardware=auto)
    DVC_DATA_DPATH=$(smartwatch_dvc --tags=phase2_data --hardware=auto)

    KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/Drop6
    DZYNE_LANDCOVER_MODEL_FPATH="$DVC_EXPT_DPATH/models/landcover/sentinel2.pt"

    INPUT_DATASET_FPATH=$KWCOCO_BUNDLE_DPATH/imganns-KR_R001.kwcoco.zip
    OUTPUT_DATASET_FPATH=$KWCOCO_BUNDLE_DPATH/imganns-KR_R001_landcover_small.kwcoco.zip

    echo "
    DVC_DATA_DPATH="$DVC_DATA_DPATH"
    DVC_EXPT_DPATH="$DVC_EXPT_DPATH"

    DZYNE_LANDCOVER_MODEL_FPATH="$DZYNE_LANDCOVER_MODEL_FPATH"

    INPUT_DATASET_FPATH="$INPUT_DATASET_FPATH"
    OUTPUT_DATASET_FPATH="$OUTPUT_DATASET_FPATH"
    "

    export CUDA_VISIBLE_DEVICES="1"
    python -m watch.tasks.landcover.predict \
        --dataset="$INPUT_DATASET_FPATH" \
        --deployed="$DZYNE_LANDCOVER_MODEL_FPATH"  \
        --device=0 \
        --num_workers=4 \
        --select_images='(.frame_index < 100) and (.sensor_coarse == "S2")' \
        --with_hidden=6 \
        --output="$OUTPUT_DATASET_FPATH"

    smartwatch stats $OUTPUT_DATASET_FPATH

    smartwatch visualize $OUTPUT_DATASET_FPATH \
        --animate=True --channels="red|green|blue,barren|forest|water,landcover_hidden.0:3,landcover_hidden.3:6" \
        --skip_missing=True --workers=4 --draw_anns=False --smart=True
"""
import datetime
# import warnings
import ubelt as ub
from pathlib import Path

import kwcoco
# import kwimage
from torch.utils.data import DataLoader

from watch.utils import util_parallel
from watch.utils import util_progress
from . import detector
from .model_info import lookup_model_info
from .utils import setup_logging

import scriptconfig as scfg


class LandcoverPredictConfig(scfg.DataConfig):
    dataset = scfg.Value(None, required=True, help='input kwcoco dataset')
    deployed = scfg.Value(None, required=True, help='pytorch weights file')
    output = scfg.Value(None, required=True, help='output kwcoco dataset')
    num_workers = scfg.Value(0, type=str, help='number of dataloading workers. Can be "auto"')
    device = scfg.Value('auto', type=str, help='auto, cpu, or integer of the device to use')
    select_images = scfg.Value(None, type=str, help='if specified, a jq operation to filter images')
    select_videos = scfg.Value(None, type=str, help='if specified, a jq operation to filter videos')
    with_hidden = scfg.Value(None, type=int, help='if true, also write out this many of the hidden activations')
    track_emissions = scfg.Value(True, help='Set to False to disable codecarbon')


def predict(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.landcover.predict import *  # NOQA
        >>> from watch.tasks.landcover.predict import _predict_single
        >>> import kwcoco
        >>> import watch
        >>> dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> dvc_expt_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
        >>> dset = kwcoco.CocoDataset(dvc_data_dpath / 'Drop6/imganns-KR_R001.kwcoco.zip')
        >>> deployed = dvc_expt_dpath / 'models/landcover/sentinel2.pt'
        >>> kwargs = {
        >>>     'dataset': dset.fpath,
        >>>     'deployed': deployed,
        >>>     'output': ub.Path(dset.fpath).augment(stemsuffix='_landcover', multidot=True),
        >>>     'select_images': '.sensor_coarse == "S2"',
        >>> }
        >>> cmdline = 0
        >>> predict(cmdline, **kwargs)
    """
    from watch.utils.lightning_ext import util_device
    from watch.tasks.fusion.predict import CocoStitchingManager
    from watch.utils import process_context
    from watch.utils import kwcoco_extensions

    config = LandcoverPredictConfig.cli(cmdline=cmdline, data=kwargs)

    print('config = {}'.format(ub.urepr(dict(config), align=':', nl=1)))

    coco_dset_filename = config.dataset
    weights_filename = Path(config.deployed)
    output_dset_filename = get_output_file(config.output)

    deployed = ub.Path(config.deployed)
    if not deployed.is_file():
        raise ValueError('Landcover model does not exist')

    device = util_device.coerce_devices(config.device)[0]
    print(f'device={device}')

    num_workers = util_parallel.coerce_num_workers(config.num_workers)

    input_dset = kwcoco.CocoDataset.coerce(coco_dset_filename)
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

    # Create a queue that writes data to disk in the background
    writer_queue = util_parallel.BlockingJobQueue(max_workers=num_workers)

    landcover_stitcher = CocoStitchingManager(
        output_dset,
        'landcover',
        chan_code='|'.join(model_info.model_outputs),
        stiching_space='image',
        writer_queue=writer_queue,
    )

    num_hidden = config.with_hidden
    if config.with_hidden:
        _register_hidden_layer_hook(model)

        hidden_stitcher = CocoStitchingManager(
            output_dset,
            'landcover_hidden',
            chan_code=f'landcover_hidden.0:{num_hidden}',
            stiching_space='image',
            writer_queue=writer_queue,
        )
        hidden_stitcher.num_hidden = num_hidden
    else:
        model._activation_cache = None
        hidden_stitcher = None

    proc_context = process_context.ProcessContext(
        type='process',
        name='watch.tasks.invariants.predict',
        config=config.to_dict(),
        track_emissions=config.track_emissions,
    )
    proc_context.start()

    dataloader = DataLoader(ptdataset, num_workers=num_workers,
                            batch_size=None, collate_fn=lambda x: x)

    # Start the worker processes before we do threading
    dataloader_iter = iter(dataloader)

    pman = util_progress.ProgressManager()
    with pman:
        _prog = pman.progiter(dataloader_iter, total=len(dataloader),
                              desc='predict landcover')
        for img_info in _prog:
            try:
                _predict_single(
                    img_info, model, model_info.model_outputs,
                    landcover_stitcher, hidden_stitcher,
                    output_dset=output_dset)

            except KeyboardInterrupt:
                print('interrupted')
                break
            except Exception as ex:
                print('warning: ex = {}'.format(ub.urepr(ex, nl=1)))
                print('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))
                raise

    writer_queue.wait_until_finished()

    print('Finish process context')
    proc_context.add_disk_info(ub.Path(input_dset.fpath).parent)
    proc_context.add_device_info(device)
    proc_context.stop()
    output_dset.dataset['info'].append(proc_context.obj)

    output_dset.dump(str(output_dset_filename), indent=2)
    print('output written to {}'.format(output_dset_filename))


def _register_hidden_layer_hook(model):
    # TODO: generalize to other models
    # Specific to UNetR model
    # These are at half of the output image resolution.

    model._activation_cache = {}
    def record_hidden_activation(layer, input, output):
        activation = output.detach()
        model._activation_cache['hidden'] = activation

    layer_of_interest = model.decoder1[3]
    layer_of_interest._forward_hooks.clear()
    layer_of_interest.register_forward_hook(record_hidden_activation)


def _predict_single(img_info,
                    model,
                    model_outputs,
                    landcover_stitcher,
                    hidden_stitcher,
                    output_dset: kwcoco.CocoDataset):
    """
    Modifies the coco dataset inplace, returns the data that needs to be
    written to disk.
    """
    gid = img_info['id']
    img = img_info['imgdata']

    pred = detector.run(model, img, img_info)

    if pred is None:
        return None, None

    landcover_stitcher.accumulate_image(gid, None, pred)
    landcover_stitcher.submit_finalize_image(gid)

    if model._activation_cache is not None:
        # Lots of hardcoded things here.
        # Hack to unpad here.
        hidden_raw = model._activation_cache['hidden'].cpu().numpy()
        hidden = hidden_raw[0].transpose(1, 2, 0)
        h, w = pred.shape[0:2]
        hidden_dsize = (w // 2, h // 2)
        hidden = hidden[0:h // 2, 0:w // 2, 0:hidden_stitcher.num_hidden]
        hidden_stitcher.accumulate_image(
            gid, None, hidden, asset_dsize=hidden_dsize,
            scale_asset_from_stitchspace=0.5)
        hidden_stitcher.submit_finalize_image(gid)


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
