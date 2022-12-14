# import watch
# import kwcoco
# dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
# coco_fpath = dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'

# dset = kwcoco.CocoDataset(coco_fpath)

# coco_img = dset.images().coco_images[0]

# img_w = coco_img.img['width']
# img_h = coco_img.img['height']


# video = dset.index.videos[coco_img.img['video_id']]
# vid_w = video['width']
# vid_h = video['height']

# print(f'{img_w=} {img_h=}')
# print(f'{vid_w=} {vid_h=}')

# delayed_imgspace = coco_img.delay('red|green|blue', space='image')
# delayed_vidspace = coco_img.delay('red|green|blue', space='video')

# print(f'{delayed_imgspace.shape=}')
# print(f'{delayed_vidspace.shape=}')

# imgspace_im = delayed_imgspace.finalize()
# vidspace_im = delayed_vidspace.finalize()
# print(f'{imgspace_im.shape=}')
# print(f'{vidspace_im.shape=}')

# imgspace_im = delayed_imgspace.finalize()
# vidspace_im = delayed_vidspace.finalize()
# print(f'{imgspace_im.shape=}')
# print(f'{vidspace_im.shape=}')
import kwcoco
# import json
import numpy as np
# import einops
import functools
import operator
import ubelt as ub
# import itertools as it
import logging
from datetime import datetime
import numpy as geek
logger = logging.getLogger(__name__)

# TODO:
# For each sensor, register the specific bands we are interested in.
# This demo currently assumes landsat8
SENSOR_TO_INFO = {}
SENSOR_TO_INFO['L8'] = {
    'sensor_name': 'Landsat-8',
    'intensity_channels': 'blue|green|red|nir|swir16|swir22|lwir11',
    'quality_channels': 'cloudmask',
    'quality_interpretation': 'FMASK'  # I dont think this is right.
}

QUALITY_BIT_INTERPRETATIONS = {}

QUALITY_BIT_INTERPRETATIONS['TA1'] = {
    'TnE'           : 1 << 0,  # T&E binary mask
    'dilated_cloud' : 1 << 1,
    'cirrus'        : 1 << 2,
    'cloud'         : 1 << 3,
    'cloud_shadow'  : 1 << 4,
    'snow'          : 1 << 5,
    'clear'         : 1 << 6,
    'water'         : 1 << 7,
}

# This will be used for value of decoding.
# After decoding QA band, it will include only value of 0, 1, 2, 3, 4, and 255.
QUALITY_BIT_INTERPRETATIONS['FMASK'] = {
    'clear_land'     : 0,
    'clear_water'    : 1,
    'cloud_shadow'   : 2,
    'snow'           : 3,
    'cloud'          : 4,
    'no_observation' : 255,
}


def setup_logging():
    # TODO: handle HPC things here in addition to stdout for doctests
    logging.basicConfig(level='INFO')


def get_file_name(parent_name):
    sensor = parent_name.split('_')[0]
    path_hv = parent_name.split('_')[2]
    year = parent_name.split('_')[3][:4]
    doy = datetime(int(year), int(parent_name[21:23]), int(parent_name[23:25])).strftime('%j')
    collection = parent_name.split('_')[5]
    if sensor == 'LC08':
        sensor = 'LC8'
    if collection == '02':
        collection = 'C_2'
    file_name = sensor + path_hv + year + doy + collection
    return file_name

QA_CLEAR = 0
QA_WATER = 1
QA_SHADOW = 2
QA_SNOW = 3
QA_CLOUD = 4
QA_FILL = 255

def qabitval_array_HLS(packedint_array):
    """
    Institute a hierarchy of qa values that may be flagged in the bitpacked
    value.
    fill > cloud > shadow > snow > water > clear
    Args:
        packedint: int value to bit check
    Returns:
        offset value to use
    """
    unpacked = np.full(packedint_array.shape, 0)
    QA_CLOUD_unpacked = geek.bitwise_and(packedint_array, 1 << 1)
    QA_CLOUD_ADJ = geek.bitwise_and(packedint_array, 1 << 2)
    QA_SHADOW_unpacked = geek.bitwise_and(packedint_array, 1 << 3)
    QA_SNOW_unpacked = geek.bitwise_and(packedint_array, 1 << 4)
    QA_WATER_unpacked = geek.bitwise_and(packedint_array, 1 << 5)

    unpacked[QA_WATER_unpacked > 0] = QA_WATER
    unpacked[QA_SNOW_unpacked > 0] = QA_SNOW
    unpacked[QA_SHADOW_unpacked > 0] = QA_SHADOW
    unpacked[QA_CLOUD_ADJ > 0] = QA_CLOUD
    unpacked[QA_CLOUD_unpacked > 0] = QA_CLOUD
    unpacked[packedint_array == QA_FILL] = QA_FILL

    return unpacked

def process_one_coco_image(coco_image, config, out_dir):
    """
    Args:
        coco_image (kwcoco.CocoImage): the image to process
        out_dir (Path): path to write the image data
    Returns:
        Dict: result dictionary with keys:
            status (str) : either a string passed or failed
            fpaths (List[str]): a list of files that were written
    """
    n_block_x = config['n_block_x']
    n_block_y = config['n_block_y']
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
    coco_image.img['date_captured']
    coco_image.img['frame_index']
    print(coco_image.img['name'])
    print(coco_image.img['parent_name'])
    print(coco_image.img['height'], coco_image.img['width'])

    sensor = coco_image.img['sensor_coarse']
    assert sensor == 'L8', 'MWE only supports landsat-8 for now'

    # Given the sensor, determine what the intensity and quality band
    # we should request are.
    sensor_info = SENSOR_TO_INFO[sensor]
    intensity_channels = sensor_info['intensity_channels']
    quality_channels = sensor_info['quality_channels']
    quality_interpretation = sensor_info['quality_interpretation']
    quality_bits = QUALITY_BIT_INTERPRETATIONS[quality_interpretation]

    # Specify how we are going to handle spatial resampling and nodata
    delay_kwargs = {'nodata_method': None, 'space': 'image'}

    delayed_im = coco_image.delay(channels=intensity_channels, **delay_kwargs)
    delayed_qa = coco_image.delay(channels=quality_channels, **delay_kwargs)
    print(delayed_im)

    h, w = delayed_im.shape[0:2]
    print(h, w)

    # Determine if padding is necessary to properly break the data into blocks.
    padded_w = int(np.ceil(w / n_block_x) * n_block_x)
    padded_h = int(np.ceil(h / n_block_y) * n_block_y)

    if padded_w != h or padded_h != h:
        slice_ = (slice(0, padded_h), slice(0, padded_w))
        delayed_im = delayed_im.crop(slice_, clip=False, wrap=False)
        delayed_qa = delayed_qa.crop(slice_, clip=False, wrap=False)

    qa_data = delayed_qa.finalize(interpolation='nearest', antialias=False)

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

    # Decoding QA band
    qa_unpacked = qabitval_array_HLS(qa_data)

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
        'status': None
    }

    if clear_ratio <= clear_threshold:
        logger.warn('Not enough clear observations for {}/{}'.format(
            video_name, image_name))
        result['status'] = 'failed'
        return result

    im_data = delayed_im.finalize(interpolation='cubic', antialias=True)

    data = np.concatenate([im_data, qa_unpacked], axis=2)

    image_name = get_file_name(coco_image.img['parent_name'])

    result_fpaths = []


    return



import watch
import kwcoco
dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
coco_fpath = dvc_dpath / 'Drop4-BAS/data.kwcoco.json'

dpath = ub.Path.appdir('tmp/gpfs/scratchfs1/zhz18039/jws18003/kwcoco').ensuredir()
out_dir = dpath / 'stacked_US_C001_drop_fixed'

# TODO: determine the block settings from the config
config = {
    'n_block_x': 20,
    'n_block_y': 20,
}

# TODO: configure
out_dir = ub.Path(out_dir)

# Load the kwcoco dataset
dset = kwcoco.CocoDataset.coerce(coco_fpath)
videos = dset.videos()

results = []

for video_id in videos:
    if video_id == 17: # US_C000
        # Get the image ids of each image in this video seqeunce
        images = dset.images(video_id=video_id)

        for image_id in images:
            coco_image : kwcoco.CocoImage = dset.coco_image(image_id)
            coco_image = coco_image.detach()
            if coco_image.img['sensor_coarse'] == 'L8':
                result = process_one_coco_image(coco_image, config, out_dir)
                break
