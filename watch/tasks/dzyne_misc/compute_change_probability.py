import kwcoco
import json

import numpy as np
import pandas as pd

from shapely.geometry import shape
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser


def _get_args():
    parser = ArgumentParser(description='Gives a probability score ')
    parser.add_argument('-i', '--input', type=Path, required=True, help='input kwcoco json file')
    parser.add_argument('-o', '--output', type=Path, required=True, help='output kwcoco json file')
    args = parser.parse_args()

    return args


def _get_required_channels() -> list[str]:
    return ['red', 'green', 'blue', 'depth', 'change']


def _has_required_channels(auxiliary: list) -> bool:
    available_channels = [aux['channels'] for aux in auxiliary]
    required_channels = _get_required_channels()

    return set(required_channels).issubset(available_channels)


def _get_change_image(coco_image: kwcoco.CocoImage):

    delay_image = coco_image.delay(channels=['change'])
    image = delay_image.finalize(nodata='float')
    image = image.astype(np.float32)

    return image, ~np.isnan(image)


def _save_kwcoco_file(dataset: dict, dst_file: Path) -> Path:
    with open(dst_file, 'w') as f:
        json.dump(dataset, f)

    return dst_file


def _preprocess_image_dataframe(data: pd.DataFrame):

    # convert geos_corners from dict -> wkt; for grouping regions
    data['geos_corners'] = data.geos_corners.apply(lambda x: shape(x).wkt)
    data['image_index'] = data.index

    # apply some filtering to scope dataset
    data = data[data['sensor_coarse'] == 'WV']
    data['has_required_channels'] = data['auxiliary'].apply(lambda x: _has_required_channels(x))
    data = data[data['has_required_channels'] is True]
    data = data.drop('has_required_channels', axis=1)

    return data


if __name__ == '__main__':

    args = _get_args()
    data = kwcoco.CocoDataset(args.input)
    image_df = pd.DataFrame(data.dataset['images'])
    image_df = _preprocess_image_dataframe(image_df)

    # main
    region_uuid = 0
    for region_name, region_df in tqdm(image_df.groupby('geos_corners')):

        # preprocess region_df
        region_df = region_df.sort_values(by=['timestamp'])

        # iterate through all images for region
        gids = []
        probability = 0.0
        running_image = None
        running_mask = None

        for row in region_df.itertuples():

            gids.append(row.id)
            change_image, valid_data_mask = _get_change_image(data.coco_image(row.id))

            if running_image is None:
                running_image = change_image
            else:
                running_image += change_image

            if running_mask is None:
                running_mask = valid_data_mask
            else:
                running_mask &= valid_data_mask

        # TODO - refine probability scoring; placeholder
        probability = running_image[running_mask].mean()
        probability = np.float64(probability)  # casted to np.float64 because json-lib can't write float32 objs

        # update output kwcoco
        if 'region' not in data.dataset:
            # add region obj to dataset
            data.dataset['region'] = []

        entry = {
            'name': region_uuid,
            'gids' : gids,
            'probability' : probability
        }

        data.dataset['region'].append(entry)

        print('{} | {:.3f}'.format(region_uuid, probability))
        region_uuid += 1

    output_file = _save_kwcoco_file(data.dataset, args.output)
    print(f'Results saved to... {output_file}')
