# import gdal here first otherwise the import fails due to conflict with rasterio
# TODO fix
# NOTE: from Jon C, wrt the above fix, the underlying issue is C libraries, and
# there is some logic to work around this in the watch.__init__ module.
from osgeo import gdal
import json
import logging
import warnings
from functools import partial
import ubelt as ub

import os
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
@click.option('--window_size', required=False, type=int, default=1024, help='sliding window size')
@click.option('--dump_shards', required=False, default=False, help='if True, output partial kwcoco files as they are completed')
@click.option('--data_workers', required=False, default=0, help='background data loaders')
def predict(dataset, deployed, output, window_size=2048, dump_shards=False, data_workers=0, ):
    weights_filename = ub.Path(deployed)

    output_dset_filename = ub.Path(get_output_file(output))

    output_bundle_dpath = output_dset_filename.parent
    output_data_dir = output_bundle_dpath / 'dzyne_depth'

    log.info('Input:          {}'.format(dataset))
    log.info('Weights:        {}'.format(weights_filename))
    log.info('Output:         {}'.format(output_dset_filename))
    log.info('Output Images:  {}'.format(output_data_dir))

    input_dset = kwcoco.CocoDataset.coerce(dataset)
    input_bundle_dpath = ub.Path(input_dset.bundle_dpath)

    output_dset = input_dset.copy()
    if input_bundle_dpath != output_bundle_dpath:
        # Need to change the root of the output directory
        # The kwcoco reroot logic is flakey for complex cases, so be careful
        # In the normal case where the output and input kwcoco share the same
        # bundle, then this logic is avoided
        output_dset.reroot(absolute=True)
        output_dset.fpath = str(output_dset_filename)
        new_prefix = os.path.relpath(input_bundle_dpath, output_bundle_dpath)
        output_dset.reroot(old_prefix=str(input_bundle_dpath),
                           new_prefix=str(new_prefix), absolute=False,
                           check=True)

    # input data
    torch_dataset = WVRgbDataset(input_dset)
    cache = 1

    if cache:
        log.debug('checking for cached files')

        # Remove any image ids that are already computed
        gid_to_pred_filename = {}
        miss_gids = []
        hit_gids = []
        for gid in torch_dataset.gids:
            img_info = torch_dataset.dset.imgs[gid]
            pred_filename = _image_pred_filename(torch_dataset,
                                                 output_data_dir, img_info)
            gid_to_pred_filename[gid] = pred_filename
            if pred_filename.exists():
                hit_gids.append(gid)
            else:
                miss_gids.append(gid)

        log.info(f'Found {len(hit_gids)} / {len(gid_to_pred_filename)} cached depth maps')
        # Might be a better way to indicate a subset, but this works
        torch_dataset.gids = miss_gids

    # model
    log.debug('loading model')
    config = _load_config()
    config['backbone_params']['pretrained'] = False  # dont download on predict
    model = MultiTaskModel(config=config)
    state_dict = torch.load(weights_filename, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    model = modify_bn(model, track_running_stats=False, bn_momentum=0.01)
    model = model.eval()
    model.to(get_device())

    log.debug('processing images')
    dataloader = DataLoader(torch_dataset, num_workers=data_workers, batch_size=1, collate_fn=lambda x: x)
    with torch.no_grad():
        for batch in tqdm(dataloader, miniters=1, unit='image', disable=False):
            assert len(batch) == 1
            batch_item = batch[0]
            gid = batch_item['id']

            # get clean img_info
            img_info = torch_dataset.dset.imgs[gid]

            pred_filename = _image_pred_filename(torch_dataset,
                                                 output_data_dir, img_info)
            if cache and pred_filename.exists():
                continue

            try:
                S = window_size
                image = batch_item['imgdata']
                pred = process_image_chunked(
                    image, partial(run_inference, model=model),
                    chip_size=(S, S, 3),
                )

                info = _write_output(img_info, pred, pred_filename, output_bundle_dpath)
                aux = output_dset.imgs[gid].get('auxiliary', [])
                aux.append(info)
                output_dset.imgs[gid]['auxiliary'] = aux

                if dump_shards:
                    # Dump debugging shard (TODO: could cache process for quick
                    # reruns)
                    shard_dset = output_dset.subset([gid])
                    shard_dset.reroot(absolute=True)
                    shard_dset.fpath = pred_filename.augment(ext='.kwcoco.json')
                    # output_dpath / (imgname + '_depth.kwcoco.json')
                    shard_dset.dump(shard_dset.fpath, indent=2)

            except KeyboardInterrupt:
                log.info('interrupted')
                break
            except Exception:
                log.exception('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

    if cache and hit_gids:
        # add metadata for cache items
        for gid in hit_gids:
            img_info = torch_dataset.dset.imgs[gid]
            pred_filename = _image_pred_filename(torch_dataset,
                                                 output_data_dir, img_info)

            gdal_img = gdal.Open(str(pred_filename), gdal.GA_ReadOnly)
            if gdal_img is None:
                raise Exception(gdal.GetLastErrorMsg())
            pred_shape = (gdal_img.RasterYSize, gdal_img.RasterXSize,
                          gdal_img.RasterCount)
            gdal_img = None

            # pred_shape = kwimage.load_image_shape(pred_filename)
            info = _build_aux_info(img_info, pred_shape, pred_filename, output_bundle_dpath)
            aux = output_dset.imgs[gid].get('auxiliary', [])
            aux.append(info)
            output_dset.imgs[gid]['auxiliary'] = aux

    output_dset.dump(str(output_dset_filename), indent=2)
    output_dset.validate()
    log.info('output written to {}'.format(output_dset_filename))


def _image_pred_filename(torch_dataset, output_data_dir, img_info):
    # Construct an output file name based on the video and image name
    imgname = img_info['name']
    vidid = img_info.get('video_id', None)
    if vidid is not None:
        vidname = torch_dataset.dset.index.videos[vidid]['name']
        output_dpath = output_data_dir / vidname
    else:
        output_dpath = output_data_dir
    pred_filename = output_dpath / (imgname + '_depth.tif')
    return pred_filename


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


def _build_aux_info(img_info, pred_shape, pred_filename, output_bundle_dpath):
    info = {
        'file_name': str(pred_filename.relative_to(output_bundle_dpath)),
        'channels': 'depth',
        'height': pred_shape[0],
        'width': pred_shape[1],
        'num_bands': 1,
        'warp_aux_to_img': {'scale': [img_info['width'] / pred_shape[1],
                                      img_info['height'] / pred_shape[0]],
                            'type': 'affine'}
    }
    return info


def _write_output(img_info, pred, pred_filename, output_bundle_dpath):
    pred_shape = pred.shape
    info = _build_aux_info(img_info, pred_shape, pred_filename, output_bundle_dpath)
    pred_filename.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        kwimage.imwrite(str(pred_filename),
                        pred,
                        backend='gdal',
                        compress='DEFLATE')
    return info


def _load_config():
    from importlib import resources as importlib_resources
    fp = importlib_resources.open_text('watch.tasks.depth', 'config.json')
    return json.load(fp)


if __name__ == '__main__':
    r"""
    # Notes:

    # VRAM usage with weights_v2_gray
    # window_size=512:   4.951 GB
    # window_size=640:   7.406 GB
    # window_size=704:   8.912 GB
    # window_size=736:   9.310 GB
    # window_size=768:  10.099 GB
    # window_size=1024: 17.111 GB
    # window_size=1152: 21.007 GB

    DVC_DPATH=$(python -m watch.cli.find_dvc)
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15
    python -m watch.tasks.depth.predict \
        --dataset="$KWCOCO_BUNDLE_DPATH/data.kwcoco.json" \
        --output="$KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json" \
        --deployed="$DVC_DPATH/models/depth/weights_v2_gray.pt" \
        --dump_shards=True \
        --data_workers=4 \
        --window_size=736

    python -m watch visualize $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json \
        --viz_dpath $DVC_DPATH/Drop1-Aligned-L1-2022-01/_viz_depth \
        --animate=True --channels=depth --skip_missing=True

    python -m watch stats $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json

    python -m kwcoco stats $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json

    python -m watch visualize $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json \
        --viz_dpath $KWCOCO_BUNDLE_DPATH/_viz_depth \
        --animate=True --channels="red|green|blue" --skip_missing=True \
        --select_images '.sensor_coarse == "WV"' --workers=4

    """
    setup_logging()
    torch.hub.set_dir('/tmp/weights')
    predict()
