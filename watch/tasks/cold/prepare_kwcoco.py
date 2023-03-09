"""
This is a proof-of-concept for converting kwcoco files into the
expected data structures for pycold.

Relevant functions:
    * grab_demo_kwcoco_dataset - downloads a small kwcoco dataset for testing
    * stack_kwcoco - runs the stacking process on an entire kwcoco file
    * process_one_coco_image - runs the stacking for a single coco image.

Limitations:
    * Currently only handles Landsat-8

    * The quality bands are not exactly what I was expecting them to be,
      some of the quality filtering is stubbed out or disabled.

    * Not setup for an HPC environment yet, but that extension shouldn't be too
      hard.

    * Nodata values are currently not masked or handled

    * Configurations are hard-coded

TODO:
    - [ ] Incorporate watch/tasks/fusion/datamodules/qa_bands.py
"""
import kwcoco
import os
import json
import numpy as np
import einops
import functools
import operator
import ubelt as ub
import itertools as it
import logging
from datetime import datetime as datetime_cls
import numpy as geek
import scriptconfig as scfg

logger = logging.getLogger(__name__)


try:
    from xdev import profile
except ImportError:
    profile = ub.identity


class PrepareKwcocoConfig(scfg.DataConfig):
    """
    The docstring will be the description in the CLI help
    """

    coco_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        a path to a file to input kwcoco file
        '''))
    out_dpath = scfg.Value(None, help='output directory for the output')
    adj_cloud = scfg.Value(False, help='How to treat QA band, default is False: ignoring adj. cloud class')
    method = scfg.Value(None, help='stacking mode for original COLD or Hybrid, default is None, if HybridCOLD then stacked data include g, r, nir, swir16, swir22, ASI, tir, QA')


# TODO:
# For each sensor, register the specific bands we are interested in.
# This demo currently assumes landsat8
SENSOR_TO_INFO = {}
SENSOR_TO_INFO['L8'] = {
    'sensor_name': 'Landsat-8',
    'intensity_channels': 'blue|green|red|nir|swir16|swir22|lwir11',
    'quality_channels': 'quality',
    'quality_interpretation': 'FMASK'
}  # The name of quality_channels for Drop 4 is 'cloudmask'.

# Register different quality bit standards.
QA_INTERPRETATIONS = {}

# These are specs for TA1 processed data
QA_BIT = {
    'clear': 1 << 0,
    'cloud': 1 << 1,
    'cloud_adj': 1 << 2,
    'shadow': 1 << 3,
    'snow': 1 << 4,
    'water': 1 << 5,
}

QA_INTERPRETATIONS['FMASK'] = {
    'clear': 0,
    'water': 1,
    'cloud_shadow': 2,
    'snow': 3,
    'cloud': 4,
    'no_obs': 255,
}
QUALITY_BIT_INTERPRETATIONS = {}


@profile
def prepare_kwcoco_main(cmdline=1, **kwargs):
    """_summary_

    Args:
        cmdline (int, optional): _description_. Defaults to 1.

    Ignore:
        python -m watch.tasks.cold.prepare_kwcoco --help
        TEST_COLD=1 xdoctest -m watch.tasks.cold.prepare_kwcoco prepare_kwcoco_main

    Example:
        >>> # xdoctest: +REQUIRES(env:TEST_COLD)
        >>> from watch.tasks.cold.prepare_kwcoco import prepare_kwcoco_main
        >>> from watch.tasks.cold.prepare_kwcoco import *
        >>> kwargs= dict(
        >>>   coco_fpath = ub.Path('/home/jws18003/data/dvc-repos/smart_data_dvc/Aligned-Drop6-2022-12-01-c30-TA1-S2-L8-WV-PD-ACC-2/KR_R001/data_KR_R001.kwcoco.json'),
        >>>   out_dpath = ub.Path.appdir('/gpfs/scratchfs1/zhz18039/jws18003/kwcoco'),
        >>>   adj_cloud = False,
        >>>   method = None,
        >>> )
        >>> cmdline=0
        >>> prepare_kwcoco_main(cmdline, **kwargs)
    """
    config = PrepareKwcocoConfig.cli(cmdline=cmdline, data=kwargs)
    coco_fpath = config['coco_fpath']
    dpath = ub.Path(config['out_dpath']).ensuredir()
    adj_cloud = config['adj_cloud']
    method = config['method']
    out_dir = dpath / 'stacked'
    meta_fpath = stack_kwcoco(coco_fpath, out_dir, adj_cloud, method)
    return meta_fpath


# function for decoding HLS qa band
def qa_decoding(qa_array):
    """
    This function is modified from qabitval_array_HLS function
    (https://github.com/GERSL/pycold/blob/c5b380eccc2916e5c3aec0bbd2b1982e114b75b1/src/python/pycold/imagetool/prepare_ard.py#L74)
    """
    unpacked = np.full(qa_array.shape, QA_INTERPRETATIONS['FMASK']['clear'])

    QA_CLOUD_unpacked = geek.bitwise_and(qa_array, QA_BIT['cloud'])
    QA_CLOUD_ADJ = geek.bitwise_and(qa_array, QA_BIT['cloud_adj'])
    QA_SHADOW_unpacked = geek.bitwise_and(qa_array, QA_BIT['shadow'])
    QA_SNOW_unpacked = geek.bitwise_and(qa_array, QA_BIT['snow'])
    QA_WATER_unpacked = geek.bitwise_and(qa_array, QA_BIT['water'])

    unpacked[QA_WATER_unpacked > 0] = QA_INTERPRETATIONS['FMASK']['water']
    unpacked[QA_SNOW_unpacked > 0] = QA_INTERPRETATIONS['FMASK']['snow']
    unpacked[QA_SHADOW_unpacked >
             0] = QA_INTERPRETATIONS['FMASK']['cloud_shadow']
    unpacked[QA_CLOUD_ADJ > 0] = QA_INTERPRETATIONS['FMASK']['cloud']
    unpacked[QA_CLOUD_unpacked > 0] = QA_INTERPRETATIONS['FMASK']['cloud']
    unpacked[qa_array == QA_INTERPRETATIONS['FMASK']
             ['no_obs']] = QA_INTERPRETATIONS['FMASK']['no_obs']

    return unpacked


def qa_decoding_no_boundary(qa_array):
    """
    This function is modified from qabitval_array_HLS function
    (https://github.com/GERSL/pycold/blob/c5b380eccc2916e5c3aec0bbd2b1982e114b75b1/src/python/pycold/imagetool/prepare_ard.py#L74)
    """
    unpacked = np.full(qa_array.shape, QA_INTERPRETATIONS['FMASK']['clear'])

    QA_CLOUD_unpacked = geek.bitwise_and(qa_array, QA_BIT['cloud'])
    QA_CLOUD_ADJ = geek.bitwise_and(qa_array, QA_BIT['cloud_adj'])
    QA_SHADOW_unpacked = geek.bitwise_and(qa_array, QA_BIT['shadow'])
    QA_SNOW_unpacked = geek.bitwise_and(qa_array, QA_BIT['snow'])
    QA_WATER_unpacked = geek.bitwise_and(qa_array, QA_BIT['water'])

    unpacked[QA_WATER_unpacked > 0] = QA_INTERPRETATIONS['FMASK']['water']
    unpacked[QA_SNOW_unpacked > 0] = QA_INTERPRETATIONS['FMASK']['snow']
    unpacked[QA_SHADOW_unpacked >
             0] = QA_INTERPRETATIONS['FMASK']['cloud_shadow']
    unpacked[QA_CLOUD_unpacked > 0] = QA_INTERPRETATIONS['FMASK']['cloud']
    unpacked[QA_CLOUD_ADJ > 0] = QA_INTERPRETATIONS['FMASK']['clear']
    unpacked[qa_array == QA_INTERPRETATIONS['FMASK']
             ['no_obs']] = QA_INTERPRETATIONS['FMASK']['no_obs']

    return unpacked


def setup_logging():
    # TODO: handle HPC things here in addition to stdout for doctests
    logging.basicConfig(level='INFO')


##########################################################################
#                   Functions for Artificial Surface Index (ASI)                   #
#  See original code: https://github.com/GERSL/ASI_py/blob/main/ASI_standalone.py  #
##########################################################################

def hist_cut(band, mask, fill_value=-9999, k=3, minmax='std'):
    if minmax == 'std':
        mean = band[mask].mean()
        std = band[mask].std()
        low_val = (mean - k * std)
        high_val = (mean + k * std)
    else:
        low_val, high_val = minmax  # use specified value range.
    is_low = band < low_val
    is_high = band > high_val
    mask_invalid_index = is_low | is_high
    band[mask_invalid_index] = fill_value
    return band, ~mask_invalid_index


def minmax_norm(band, mask, fill_value=-9999):
    max_val = band[mask].max()
    min_val = band[mask].min()
    extent = max_val - min_val
    if extent != 0:
        shifted = band - min_val
        scaled = shifted / extent
        band[mask] = scaled[mask]
    band[~mask] = fill_value
    return band


# Artificial Surface Index (ASI) is designed based the surface reflectance
# imagery of Landsat 8.
def artificial_surface_index(
        Blue, Green, Red, NIR, SWIR1, SWIR2, Scale, MaskValid_Obs, fillV):
    # The calculation chain.

    # Artificial surface Factor (AF).
    AF = (NIR - Blue) / (NIR + Blue) + 0.000001
    AF, MaskValid_AF = hist_cut(AF, MaskValid_Obs, fillV, 6, [-1, 1])
    MaskValid_AF_U = MaskValid_AF & MaskValid_Obs
    AF_Norm = minmax_norm(AF, MaskValid_AF_U, fillV)

    # Vegetation Suppressing Factor (VSF).
    # Modified Soil Adjusted Vegetation Index (MSAVI).
    MSAVI = ((2 * NIR + 1 * Scale) -
             np.sqrt((2 * NIR + 1 * Scale)**2 - 8 * (NIR - Red))) / 2
    MSAVI, MaskValid_MSAVI = hist_cut(MSAVI, MaskValid_Obs, fillV, 6, [-1, 1])
    NDVI = (NIR - Red) / (NIR + Red) + 0.000001
    NDVI, MaskValid_NDVI = hist_cut(NDVI, MaskValid_Obs, fillV, 6, [-1, 1])
    VSF = 1 - MSAVI * NDVI
    MaskValid_VSF = MaskValid_MSAVI & MaskValid_NDVI & MaskValid_Obs
    VSF_Norm = minmax_norm(VSF, MaskValid_VSF, fillV)

    # Soil Suppressing Factor (SSF).
    # Derive the Modified Bare soil Index (MBI).
    MBI = (SWIR1 - SWIR2 - NIR) / (SWIR1 + SWIR2 + NIR) + 0.5
    MBI, MaskValid_MBI = hist_cut(MBI, MaskValid_Obs, fillV, 6, [-0.5, 1.5])
    # Deriving Enhanced-MBI based on MBI and MNDWI.
    MNDWI = (Green - SWIR1) / (Green + SWIR1) + 0.000001
    MNDWI, MaskValid_MNDWI = hist_cut(MNDWI, MaskValid_Obs, fillV, 6, [-1, 1])
    EMBI = ((MBI + 0.5) - (MNDWI + 1)) / ((MBI + 0.5) + (MNDWI + 1))
    EMBI, MaskValid_EMBI = hist_cut(EMBI, MaskValid_Obs, fillV, 6, [-1, 1])
    # Derive SSF.
    SSF = (1 - EMBI)
    MaskValid_SSF = MaskValid_MBI & MaskValid_MNDWI & MaskValid_EMBI & MaskValid_Obs
    SSF_Norm = minmax_norm(SSF, MaskValid_SSF, fillV)

    # Modulation Factor (MF).
    MF = (Blue + Green - NIR - SWIR1) / (Blue + Green + NIR + SWIR1) + 0.000001
    MF, MaskValid_MF = hist_cut(MF, MaskValid_Obs, fillV, 6, [-1, 1])
    MaskValid_MF_U = MaskValid_MF & MaskValid_Obs
    MF_Norm = minmax_norm(MF, MaskValid_MF_U, fillV)

    # Derive Artificial Surface Index (ASI).
    ASI = AF_Norm * SSF_Norm * VSF_Norm * MF_Norm
    MaskValid_ASI = MaskValid_AF_U & MaskValid_VSF & MaskValid_SSF & MaskValid_MF_U & MaskValid_Obs
    ASI[~MaskValid_ASI] = fillV

    return ASI


def stack_kwcoco(coco_fpath, out_dir, adj_cloud, method):
    """
    Args:
        coco_fpath (str | PathLike | CocoDataset):
            the kwcoco dataset to convert

        out_dir (str | PathLike): path to write the data

    Returns:
        List[Dict]: a list of dictionary result objects

    Example:
        >>> # xdoctest: +SKIP
        >>> # TODO: readd this doctest
        >>> from pycold.imagetool.prepare_kwcoco import *  # NOQA
        >>> setup_logging()
        >>> coco_fpath = grab_demo_kwcoco_dataset()
        >>> dpath = ub.Path.appdir('pycold/tests/stack_kwcoco').ensuredir()
        >>> out_dir = dpath / 'stacked'
        >>> results = stack_kwcoco(coco_fpath, out_dir)
    """
    # TODO: configure
    out_dir = ub.Path(out_dir)

    # Load the kwcoco dataset
    dset = kwcoco.CocoDataset.coerce(coco_fpath)
    videos = dset.videos()

    for video_id in videos:
        # Get the image ids of each image in this video seqeunce
        for image_id in dset.images():
            coco_image: kwcoco.CocoImage = dset.coco_image(image_id)
            coco_image = coco_image.detach()

            # For now, it supports only L8
            if coco_image.img['sensor_coarse'] == 'L8':
                # Transform the image data into the desired block structure.
                result = process_one_coco_image(
                    coco_image, out_dir, adj_cloud, method)

    return result


def process_one_coco_image(coco_image, out_dir, adj_cloud, method):
    """
    Args:
        coco_image (kwcoco.CocoImage): the image to process
        out_dir (Path): path to write the image data

    Returns:
        Dict: result dictionary with keys:
            status (str) : either a string passed or failed
            fpaths (List[str]): a list of files that were written
    """
    n_block_x = 20
    n_block_y = 20
    is_partition = True  # hard coded

    # Use the COCO name as a unique filename id.
    image_name = coco_image.img.get('name', None)
    video_name = coco_image.video.get('name', None)
    if image_name is None:
        image_name = 'img_{:06d}'.format(coco_image.img['id'])
    if video_name is None:
        video_name = 'vid_{:06d}'.format(coco_image.video['id'])

    video_dpath = (out_dir / video_name).ensuredir()

    # Other relevant coco metadata
    date_captured = coco_image.img['date_captured']
    ordinal_date = datetime_cls.strptime(
        date_captured[:10], '%Y-%m-%d').toordinal()
    # frame_index = coco_image.img['frame_index']
    n_cols = coco_image.img['width']
    n_rows = coco_image.img['height']
    # Determine what sensor the image is from.
    # Note: if kwcoco needs to register more fine-grained sensor
    # information we can do that.
    sensor = coco_image.img['sensor_coarse']
    assert sensor == 'L8', 'MWE only supports landsat-8 for now'

    # Given the sensor, determine what the intensity and quality band
    # we should request are.
    sensor_info = SENSOR_TO_INFO[sensor]
    intensity_channels = sensor_info['intensity_channels']
    quality_channels = sensor_info['quality_channels']
    quality_interpretation = sensor_info['quality_interpretation']
    quality_bits = QA_INTERPRETATIONS[quality_interpretation]
    # Specify how we are going to handle spatial resampling and nodata
    delay_kwargs = {
        'nodata_method': None,
        'space': 'video',
    }

    # Construct delayed images. These represent a tree of image
    # operations that will resample the image at the desired resolution
    # as well as align it with other images in the sequence.
    delayed_im = coco_image.delay(channels=intensity_channels, **delay_kwargs)
    delayed_qa = coco_image.delay(channels=quality_channels, **delay_kwargs)
    # Check what shape the data would be loaded with if we finalized right now.
    h, w = delayed_im.shape[0:2]
    # Determine if padding is necessary to properly break the data into blocks.
    padded_w = int(np.ceil(w / n_block_x) * n_block_x)
    padded_h = int(np.ceil(h / n_block_y) * n_block_y)

    if padded_w != h or padded_h != h:
        # cropping using an oversized slice with clip=False and wrap=False is
        # equivalent to padding. In the future a more efficient pad operation
        # where the padding value can be specified will be added, but this will
        # work well enough for now.
        slice_ = (slice(0, padded_h), slice(0, padded_w))
        delayed_im = delayed_im.crop(slice_, clip=False, wrap=False)
        delayed_qa = delayed_qa.crop(slice_, clip=False, wrap=False)

    # It is important that the categorical QA band is not interpolated or
    # antialiased, whereas the intensity bands should be.
    qa_data = delayed_qa.finalize(interpolation='nearest', antialias=False)
    # Decoding QA band
    if adj_cloud:
        qa_unpacked = qa_decoding(qa_data)
    else:
        qa_unpacked = qa_decoding_no_boundary(qa_data)

    # First check the quality bands before loading all of the image data.
    # FIXME: the quality bits in this example are wrong.
    # Setting the threshold to zero to bypass for now.
    clear_threshold = 0
    if clear_threshold > 0:
        clear_bits = functools.reduce(
            operator.or_, ub.take(quality_bits, ['clear_land', 'clear_water']))
        noobs_bits = functools.reduce(
            operator.or_, ub.take(quality_bits, ['no_observation']))
        is_clear = (qa_data & clear_bits) > 0
        is_noobs = (qa_data & noobs_bits) > 0
        is_obs = ~is_noobs
        is_obs_clear = is_clear & is_obs
        clear_ratio = is_obs_clear.sum() / is_obs.sum()
    else:
        clear_ratio = 1

    result = {
        'status': None,
        'fpaths': None,
    }

    if clear_ratio <= clear_threshold:
        logger.warn('Not enough clear observations for {}/{}'.format(
            video_name, image_name))
        result['status'] = 'failed'
        return result

    im_data = delayed_im.finalize(interpolation='cubic', antialias=True)

    if method == 'ASI':
        Scale = 10000
        fill_value = 0
        B1 = im_data[:, :, 0]
        B2 = im_data[:, :, 1]
        B3 = im_data[:, :, 2]
        B4 = im_data[:, :, 3]
        B5 = im_data[:, :, 4]
        B6 = im_data[:, :, 5]
        MaskValid_Obs = ((B1 > 0) & (B1 < 1 * Scale) &
                         (B2 > 0) & (B2 < 1 * Scale) &
                         (B3 > 0) & (B3 < 1 * Scale) &
                         (B4 > 0) & (B4 < 1 * Scale) &
                         (B5 > 0) & (B5 < 1 * Scale) &
                         (B6 > 0) & (B6 < 1 * Scale)
                         )

        # Calculating ASI
        ASI = artificial_surface_index(
            B1.astype(
                np.float32), B2.astype(
                np.float32), B3.astype(
                np.float32), B4.astype(
                    np.float32), B5.astype(
                        np.float32), B6.astype(
                            np.float32), Scale, MaskValid_Obs, fill_value)
        # Get land mask.
        MNDWI = (B2 - B5) / (B2 + B5)
        MNDWI, MaskValid_MNDWI = hist_cut(
            MNDWI, MaskValid_Obs, fill_value, 6, [-1, 1])
        # Water threshold for MNDWI (may need to be adjusted for different
        # study areas).
        Water_Th = 0
        MaskLand = (MNDWI < Water_Th)

        # Convert dtype from float32 to int16
        ASI = ASI * Scale
        ASI = ASI.astype('int16')
        ASI[ASI == 0] = fill_value

        # Exclude water pixels.
        ASI[~MaskLand] = fill_value
        ASI = ASI.reshape(ASI.shape[0], ASI.shape[1], 1)
        false_band = np.full((ASI.shape[0], ASI.shape[1], 1), 0)
        # input for Hybrid-COLD (with ASI) = B2, B3, B4, B5, B6, ASI
        data = np.concatenate(
            [im_data[:, :, 1:6], ASI, false_band, qa_unpacked], axis=2)

    else:
        data = np.concatenate([im_data, qa_unpacked], axis=2)

    result_fpaths = []

    metadata = {
        'image_name': image_name,
        'date_captured': date_captured,
        'ordinal_date': ordinal_date,
        'region_id': video_name,
        'n_cols': n_cols,
        'n_rows': n_rows,
        'video_w': w,
        'video_h': h,
        'padded_n_cols': padded_w,
        'padded_n_rows': padded_h,
        'n_block_x': n_block_x,
        'n_block_y': n_block_y,
        'adj_cloud': adj_cloud,
        'method': method
    }

    if is_partition:
        bw = int(padded_w / n_block_x)  # width of a block
        bh = int(padded_h / n_block_y)  # height of a block

        # Use einops to rearrange the data into blocks
        # Question: would using numpy strided tricks be faster?
        blocks = einops.rearrange(
            data, '(nby bh) (nbx bw) c -> nbx nby bh bw c', bw=bw, bh=bh)

        for i, j in it.product(range(n_block_y), range(n_block_x)):
            block = blocks[i, j]

            # FIXME: Disable skipping until QA bands are handled correctly
            SKIP_BLOCKS_WITH_QA = False
            if SKIP_BLOCKS_WITH_QA:
                # check if no valid pixels in the chip, then eliminate
                qa_unique = np.unique(block[..., -1])
                qa_unique
                # skip blocks are all cloud, shadow or filled values in DHTC,
                # we also don't need to save pixel that has qa value of
                # 'QA_CLOUD', 'QA_SHADOW', or FILLED value (255)
                if ... and False:
                    continue

            block_dname = 'block_x{}_y{}'.format(i + 1, j + 1)
            block_dpath = (video_dpath / block_dname).ensuredir()
            block_fpath = block_dpath / (image_name + '.npy')

            metadata.update({
                'x': i + 1,
                'y': j + 1,
                'total_pixels': int(np.prod(block.shape[0:2])),
                'total_bands': int(block.shape[-1]),
            })
            meta_fpath = block_dpath / (image_name + '.json')
            if not os.path.exists(block_fpath):
                meta_fpath.write_text(json.dumps(metadata))
                np.save(block_fpath, block)
                result_fpaths.append(block_fpath)
                result_fpaths.append(meta_fpath)
        logger.info(
            'Stacked blocked image {}/{}'.format(video_name, image_name))
    else:

        metadata.update({
            'total_pixels': int(np.prod(data.shape[0:2])),
            'total_bands': int(data.shape[-1]),
        })

        full_fpath = video_dpath / (image_name + '.npy')
        meta_fpath = video_dpath / (image_name + '.json')
        if not os.path.exists(block_fpath):
            meta_fpath.write_text(json.dumps(metadata))
            np.save(full_fpath, data)
            result_fpaths.append(full_fpath)
            result_fpaths.append(meta_fpath)
            logger.info(
                'Stacked full image {}/{}'.format(video_name, image_name))

    result['status'] = 'passed'
    result['fpaths'] = result_fpaths
    return meta_fpath


if __name__ == '__main__':
    prepare_kwcoco_main()