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
# import torchvision.transforms
# from medpy.filter.smoothing import anisotropic_diffusion
from torch.utils.data import DataLoader
from tqdm import tqdm

from scipy import ndimage

import geowatch_tpl  # NOQA

from .datasets import WVRgbDataset, WVSuperRgbDataset
# from .pl_highres_verify import MultiTaskModel, modify_bn, dfactor, local_utils
from .pl_highres_verify import MultiTaskModel, modify_bn, dfactor
from .utils import process_image_chunked
from ..landcover.detector import get_device
from ..landcover.predict import get_output_file
from ..landcover.utils import setup_logging

log = logging.getLogger(__name__)


# Debug variables
ENABLE_WRITES = 1  # set to zero to disable writing to disk
ENABLE_MODEL = 1  # set to zero to disable running the model forward
ENABLE_PROCESS = 1  # set to zero to disable chunked process
ENABLE_CHUNKS = 1  # set to zero to disable chunking (increases VRAM)


@click.command()
@click.option('--dataset', required=True, type=click.Path(exists=True), help='input kwcoco dataset')
@click.option('--deployed', required=True, type=click.Path(exists=True), help='pytorch weights file')
@click.option('--output', required=False, type=click.Path(), help='output kwcoco dataset')
@click.option('--window_size', required=False, type=int, default=1024, help='sliding window size')
@click.option('--dump_shards', required=False, default=False, help='if True, output partial kwcoco files as they are completed')
@click.option('--data_workers', required=False, default=0, help='background data loaders')
@click.option('--select_images', required=False, default=None, help='if specified, a jq operation to filter images')
@click.option('--select_videos', required=False, default=None, help='if specified, a jq operation to filter videos')
@click.option('--asset_suffix', required=False, default='_assets/dzyne_depth', help='folder relative to output to save features in')
@click.option('--cache', required=False, default=0, help='if True, enable caching of results')
@click.option('--use_super_res_bands', required=False, default=False, help='if True, uses super-res (upsampled) bands to form source RGB image')
@click.option('--scale', required=False, default=4, help='Specify scaling factor between image size and super-res bands, only applicable when use_super_res_bands is true')
def predict(dataset, deployed, output, window_size=2048, dump_shards=False,
            data_workers=0, select_images=None, select_videos=None,
            asset_suffix='_assets/dzyne_depth', cache=False, use_super_res_bands=False, scale=4):
    """
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from geowatch.tasks.depth.predict import *  # NOQA
        >>> import geowatch
        >>> dvc_dpath = geowatch.find_dvc_dpath()
        >>> dataset = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data_vali.kwcoco.json'
        >>> output = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/dzyne_depth_test.kwcoco.json'
        >>> deployed = dvc_dpath / "models/depth/weights_v1.pt"
        >>> data_workers = 0
        >>> cache = 0
        >>> select_images = '.name == "crop_20150123T020752Z_N37.734145E128.855484_N37.811709E128.946746_WV_0"'
        >>> select_videos = None
        >>> dump_shards = False
        >>> window_size = 1536
        >>> asset_suffix = '_assets/test_dzyne_depth'
        >>> predict.callback(dataset=dataset, deployed=deployed, output=output,
        >>>         window_size=window_size, dump_shards=dump_shards,
        >>>         data_workers=data_workers, select_images=select_images,
        >>>         select_videos=select_videos, cache=cache)
    """
    weights_filename = ub.Path(deployed)

    output_dset_filename = ub.Path(get_output_file(output))

    output_bundle_dpath = output_dset_filename.parent
    output_data_dir = output_bundle_dpath / asset_suffix

    log.info('Input:          {}'.format(dataset))
    log.info('Weights:        {}'.format(weights_filename))
    log.info('Output:         {}'.format(output_dset_filename))
    log.info('Output Images:  {}'.format(output_data_dir))

    input_dset = kwcoco.CocoDataset.coerce(dataset)
    input_bundle_dpath = ub.Path(input_dset.bundle_dpath)

    from geowatch.utils import kwcoco_extensions
    from geowatch.tasks.fusion.predict import quantize_float01
    filtered_gids = kwcoco_extensions.filter_image_ids(
        input_dset,
        include_sensors=['WV'],
        select_images=select_images,
        select_videos=select_videos
    )
    log.info('Valid Images:          {}'.format(len(filtered_gids)))
    input_dset = input_dset.subset(filtered_gids)

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
    if use_super_res_bands is True:
        torch_dataset = WVSuperRgbDataset(dset=input_dset, scale=scale)  # TODO - auto-detect scaling
    else:
        torch_dataset = WVRgbDataset(input_dset)

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

        log.info(
            f'Found {len(hit_gids)} / {len(gid_to_pred_filename)} cached depth maps')
        # Might be a better way to indicate a subset, but this works
        torch_dataset.gids = miss_gids

    # model
    log.debug('loading model')
    config = _load_config()
    config['backbone_params']['pretrained'] = False  # dont download on predict
    model = MultiTaskModel(config=config)
    state_dict = torch.load(
        weights_filename,
        map_location=lambda storage,
        loc: storage)
    model.load_state_dict(state_dict)

    model = modify_bn(model, track_running_stats=False, bn_momentum=0.01)
    model = model.eval()
    model.to(get_device())

    S = window_size
    chip_size = (S, S, 3)
    overlap = (128, 128, 0)
    output_dtype = np.float32  # Will be quantized as a final step

    process_func = partial(run_inference, model=model)

    log.debug('processing images')
    dataloader = DataLoader(torch_dataset, num_workers=data_workers,
                            batch_size=1, pin_memory=1, collate_fn=lambda x: x)
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
                # Dereference items after we are done with them
                batch_item = None
                image = None
                continue

            try:
                image = batch_item['imgdata']
                log.info('processing image {}'.format(image.shape))

                if ENABLE_PROCESS:
                    use_chunks = ENABLE_CHUNKS
                    if image.shape[0] < S and image.shape[1] < S:
                        use_chunks = 0

                    if use_chunks:
                        pred = process_image_chunked(image, process_func,
                                                     chip_size=chip_size,
                                                     overlap=overlap,
                                                     output_dtype=output_dtype)
                    else:
                        pred = process_func(image)
                else:
                    pred = np.zeros(image.shape[0:2], dtype=output_dtype)
                # Dereference items after we are done with them
                batch_item = None  # dereference for memory
                image = None  # dereference for memory

                quant_pred, quantization = quantize_float01(
                    pred, old_min=0, old_max=1, quantize_dtype=np.uint8)
                pred = None  # dereference for memory

                info = _write_output(img_info, quant_pred, pred_filename,
                                     output_bundle_dpath, quantization)
                quant_pred = None  # dereference for memory

                aux = output_dset.imgs[gid].get('auxiliary', [])
                aux.append(info)
                output_dset.imgs[gid]['auxiliary'] = aux

                if dump_shards:
                    # Dump debugging shard
                    shard_dset = output_dset.subset([gid])
                    shard_dset.reroot(absolute=True)
                    shard_dset.fpath = pred_filename.augment(
                        ext='.kwcoco.json')
                    # output_dpath / (imgname + '_depth.kwcoco.json')
                    shard_dset.dump(shard_dset.fpath, indent=2)

            except KeyboardInterrupt:
                log.info('interrupted')
                break
            except Exception:
                log.exception(
                    'Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

    if cache and hit_gids:
        from geowatch.utils import util_gdal
        # add metadata for cache items
        for gid in hit_gids:
            img_info = torch_dataset.dset.imgs[gid]
            pred_filename = _image_pred_filename(torch_dataset,
                                                 output_data_dir, img_info)
            with util_gdal.GdalDataset.open(pred_filename, 'r') as gdal_img:
                pred_shape = (gdal_img.RasterYSize, gdal_img.RasterXSize,
                              gdal_img.RasterCount)
            # pred_shape = kwimage.load_image_shape(pred_filename)

            # Hack to get the quantization dict that would have been computed
            # at predict time.
            _, quantization = quantize_float01(
                None, old_min=0, old_max=1, quantize_dtype=np.uint8)

            info = _build_aux_info(img_info, pred_shape, pred_filename,
                                   output_bundle_dpath, quantization)
            aux = output_dset.imgs[gid].get('auxiliary', [])
            aux.append(info)
            output_dset.imgs[gid]['auxiliary'] = aux

    if ENABLE_WRITES:
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


def fake_model(batch2, tta=True):
    # For testing
    np_data = batch2['image'][0].permute(1, 2, 0).numpy()
    x = kwimage.gaussian_blur(np_data, sigma=7)
    depth = torch.from_numpy(x.mean(axis=2))[None, None]
    pred2 = dict(depth=depth, seg=depth)
    return pred2, batch2


def _test():
    """
    Small test to check that stitching logic works when nan regions are
    involved.
    """
    import kwimage
    src = kwimage.ensure_float01(kwimage.grab_test_image(dsize=(2048, 2048)))
    nan_poly = kwimage.Polygon.random(rng=32021).scale(src.shape[0] * 3)
    image = nan_poly.fill(src.copy() * 255, np.nan)
    output_dtype = np.uint8
    overlap = (0, 0, 0)
    chip_size = (512, 512, 3)
    model = fake_model
    process_func = partial(run_inference, model=model, device='cpu')
    pred = process_image_chunked(image, process_func,
                                 chip_size=chip_size,
                                 overlap=overlap,
                                 output_dtype=output_dtype)

    from geowatch.tasks.fusion.predict import quantize_float01
    quant_pred, quantization = quantize_float01(pred, old_min=0, old_max=1,
                                                quantize_dtype=np.uint8)
    print('quantization = {}'.format(ub.urepr(quantization, nl=1)))

    import kwplot
    kwplot.autompl()
    kwplot.imshow(kwimage.normalize_intensity(
        image), pnum=(1, 3, 1), doclf=True)
    kwplot.imshow(pred, pnum=(1, 3, 2))
    kwplot.imshow(quant_pred, pnum=(1, 3, 3))


def run_inference(image, model, device=0):
    """
    Example:
        >>> from geowatch.tasks.depth.predict import *  # NOQA
        >>> import kwimage
        >>> import kwarray
        >>> src = kwimage.ensure_float01(kwimage.grab_test_image(dsize=(512, 512)))
        >>> src = kwimage.Polygon.random(rng=None).scale(src.shape[0]).fill(src.copy(), np.nan)
        >>> model = fake_model
        >>> image = src * 255
        >>> device = 'cpu'
        >>> result = run_inference(image, model, device=device)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(src, pnum=(1, 2, 1), doclf=True)
        >>> kwplot.imshow(result, pnum=(1, 2, 2))
    """
    if not ENABLE_MODEL:
        pred_shape = image.shape[0:2]
        return np.full(pred_shape, fill_value=np.nan, dtype=np.float32)

    img_h, img_w = image.shape[0:2]
    pad_h = pad_w = 0
    min_w, min_h = 64, 64
    if img_h < min_h or img_w < min_w:
        pad_h = min_h - img_h
        pad_w = min_w - img_w
        pad_dims = ((0, pad_h), (0, pad_w), (0, 0))
        image = np.pad(image, pad_dims, mode='constant',
                       constant_values=np.nan)

    with torch.no_grad():
        nodata_mask = np.isnan(image)
        if not np.all(nodata_mask):

            # Replace nans with zeros before going into the network
            image_float = image / 255.0  # not sure why we want to do this...

            # image_float = image.copy()
            image_float[nodata_mask] = 0
            image_tensor = torch.from_numpy(
                image_float.transpose(
                    (2, 0, 1))).contiguous()

            mean = np.nanmean(image.reshape(-1, image.shape[-1]), axis=0)
            std = np.nanstd(image.reshape(-1, image.shape[-1]), axis=0)

            batch2 = {
                "image": image_tensor[None, ...].to(device),
                "image_mean": torch.from_numpy(mean)[None, ...].to(device),
                "image_std": torch.from_numpy(std)[None, ...].to(device),
            }

            pred2, batch2 = model(batch2, tta=True)

            output_depth = pred2['depth'][0, 0, :, :].cpu().data.numpy()
            output_label = pred2['seg'][0, 0, :, :].cpu().data.numpy()
            # output_depth[nodata_mask.all(axis=2)] = np.nan

            weighted_depth = dfactor * output_depth

            alpha = 0.9
            weighted_seg = alpha * output_label + \
                (1.0 - alpha) * np.minimum(0.99, weighted_depth / 70.0)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # tmp2 = 255 * anisotropic_diffusion(weighted_seg, niter=1, kappa=100, gamma=0.8)
                tmp2 = anisotropic_diffusion(
                    weighted_seg, niter=1, kappa=100, gamma=0.8)

            # weighted_final = ndimage.median_filter(tmp2.astype(np.uint8), size=7)
            weighted_final = ndimage.median_filter(tmp2, size=7)
            weighted_final[nodata_mask.all(axis=2)] = np.nan
            # weighted_final = ndimage.median_filter(tmp2.astype(np.uint8), size=7)

            if pad_h or pad_w:
                # Undo pad
                weighted_final = weighted_final[:-pad_h, :-pad_w]

        else:
            pred_shape = image.shape[0:2]
            return np.full(pred_shape, fill_value=np.nan, dtype=np.float32)

    return weighted_final


# Vendored in to deal with 3.10 issue
def anisotropic_diffusion(img, niter=1, kappa=50,
                          gamma=0.1, voxelspacing=None, option=1):
    r"""
    Edge-preserving, XD Anisotropic diffusion.


    Parameters
    ----------
    img : array_like
        Input image (will be cast to numpy.float).
    niter : integer
        Number of iterations.
    kappa : integer
        Conduction coefficient, e.g. 20-100. ``kappa`` controls conduction
        as a function of the gradient. If ``kappa`` is low small intensity
        gradients are able to block conduction and hence diffusion across
        steep edges. A large value reduces the influence of intensity gradients
        on conduction.
    gamma : float
        Controls the speed of diffusion. Pick a value :math:`<= .25` for stability.
    voxelspacing : tuple of floats or array_like
        The distance between adjacent pixels in all img.ndim directions
    option : {1, 2, 3}
        Whether to use the Perona Malik diffusion equation No. 1 or No. 2,
        or Tukey's biweight function.
        Equation 1 favours high contrast edges over low contrast ones, while
        equation 2 favours wide regions over smaller ones. See [1]_ for details.
        Equation 3 preserves sharper boundaries than previous formulations and
        improves the automatic stopping of the diffusion. See [2]_ for details.

    Returns
    -------
    anisotropic_diffusion : ndarray
        Diffused image.

    Notes
    -----
    Original MATLAB code by Peter Kovesi,
    School of Computer Science & Software Engineering,
    The University of Western Australia,
    pk @ csse uwa edu au,
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal,
    Department of Pharmacology,
    University of Oxford,
    <alistair.muldal@pharm.ox.ac.uk>

    Adapted to arbitrary dimensionality and added to the MedPy library Oskar Maier,
    Institute for Medical Informatics,
    Universitaet Luebeck,
    <oskar.maier@googlemail.com>

    June 2000  original version. -
    March 2002 corrected diffusion eqn No 2. -
    July 2012 translated to Python -
    August 2013 incorporated into MedPy, arbitrary dimensionality -

    References
    ----------
    .. [1] P. Perona and J. Malik.
       Scale-space and edge detection using ansotropic diffusion.
       IEEE Transactions on Pattern Analysis and Machine Intelligence,
       12(7):629-639, July 1990.
    .. [2] M.J. Black, G. Sapiro, D. Marimont, D. Heeger
       Robust anisotropic diffusion.
       IEEE Transactions on Image Processing,
       7(3):421-432, March 1998.
    """
    # define conduction gradients functions
    import numpy
    if option == 1:
        def condgradient(delta, spacing):
            return numpy.exp(-(delta / kappa)**2.) / float(spacing)
    elif option == 2:
        def condgradient(delta, spacing):
            return 1. / (1. + (delta / kappa)**2.) / float(spacing)
    elif option == 3:
        kappa_s = kappa * (2**0.5)

        def condgradient(delta, spacing):
            top = 0.5 * ((1. - (delta / kappa_s)**2.)**2.) / float(spacing)
            return numpy.where(numpy.abs(delta) <= kappa_s, top, 0)

    # initialize output array
    out = numpy.array(img, dtype=numpy.float32, copy=True)

    # set default voxel spacing if not supplied
    if voxelspacing is None:
        voxelspacing = tuple([1.] * img.ndim)

    # initialize some internal variables
    deltas = [numpy.zeros_like(out) for _ in range(out.ndim)]

    for _ in range(niter):

        # calculate the diffs
        for i in range(out.ndim):
            slicer = tuple([slice(None, -1) if j == i else slice(None)
                           for j in range(out.ndim)])
            deltas[i][slicer] = numpy.diff(out, axis=i)

        # update matrices
        matrices = [
            condgradient(
                delta,
                spacing) *
            delta for delta,
            spacing in zip(
                deltas,
                voxelspacing)]

        # subtract a copy that has been shifted ('Up/North/West' in 3D case) by one
        # pixel. Don't as questions. just do it. trust me.
        for i in range(out.ndim):
            slicer = tuple([slice(1, None) if j == i else slice(None)
                           for j in range(out.ndim)])
            matrices[i][slicer] = numpy.diff(matrices[i], axis=i)

        # update the image
        out += gamma * (numpy.sum(matrices, axis=0))

    return out


def _build_aux_info(img_info, pred_shape, pred_filename, output_bundle_dpath,
                    quantization):
    info = {
        'file_name': str(pred_filename.relative_to(output_bundle_dpath)),
        'channels': 'depth',
        'height': pred_shape[0],
        'width': pred_shape[1],
        'num_bands': 1,
        'quantization': quantization,
        'warp_aux_to_img': {'scale': [img_info['width'] / pred_shape[1],
                                      img_info['height'] / pred_shape[0]],
                            'type': 'affine'}
    }
    return info


def _write_output(img_info, pred, pred_filename,
                  output_bundle_dpath, quantization):
    pred_shape = pred.shape
    info = _build_aux_info(img_info, pred_shape, pred_filename,
                           output_bundle_dpath, quantization)

    # with warnings.catch_warnings():
    #     warnings.simplefilter('ignore', UserWarning)
    if ENABLE_WRITES:
        pred_filename.parent.mkdir(parents=True, exist_ok=True)

        kwimage.imwrite(str(pred_filename),
                        pred, backend='gdal', blocksize=256,
                        nodata=quantization['nodata'],
                        compress='DEFLATE', overviews=3)
    return info


def _load_config():
    from importlib import resources as importlib_resources
    fp = importlib_resources.open_text('geowatch.tasks.depth', 'config.json')
    return json.load(fp)


if __name__ == '__main__':
    r"""
    # Notes:

        weights_v1 - for RGB
        weights_v2_gray - for PAN

        TODO: Predict with both models, one for RGB and one with PAN

    # VRAM usage with weights_v2_gray
    # window_size=512:   4.951 GB
    # window_size=640:   7.406 GB
    # window_size=704:   8.912 GB
    # window_size=736:   9.310 GB
    # window_size=768:  10.099 GB
    # window_size=1024: 17.111 GB
    # window_size=1152: 21.007 GB

    ## Drop 4 ##
    python3 -m geowatch.tasks.depth.predict \
    --deployed=/smart/backup/models/depth/weights_v1.pt \
    --dataset=/output/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_wv_superRes.kwcoco.json \
    --output=/output/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/data_wv_superRes_depth.kwcoco.json \
    --data_workers=8 \
    --window_size=2048 \
    --cache=1 \
    --use_super_res_bands=True \
    --scale=4

    python -m geowatch visualize $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json \
        --animate=True --channels="depth,red|green|blue" --skip_missing=True \
        --select_images '.sensor_coarse == "WV"' --workers=4 --draw_anns=False

    python -m geowatch stats $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json

    python -m kwcoco stats $KWCOCO_BUNDLE_DPATH/dzyne_depth.kwcoco.json


    Notes:
        Does export MALLOC_MMAP_THRESHOLD_=0 help with memory?

        export MALLOC_MMAP_THRESHOLD_=8192
        export MALLOC_ARENA_MAX=4

    """
    setup_logging()
    torch.hub.set_dir('/tmp/weights')
    predict()
