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

log = logging.getLogger(__name__)


@click.command()
@click.option('--dataset', required=True, type=click.Path(exists=True), help='input kwcoco dataset')
@click.option('--deployed', required=True, type=click.Path(exists=True), help='pytorch weights file')
@click.option('--output', required=False, type=click.Path(), help='output kwcoco dataset')
def predict(dataset, deployed, output):
    coco_dset_filename = dataset
    weights_filename = Path(deployed)
    output_dset_filename = get_output_file(output)
    output_data_dir = output_dset_filename.parent.joinpath(
        output_dset_filename.name.split('.')[0])

    log.info('Input:          {}'.format(coco_dset_filename))
    log.info('Weights:        {}'.format(weights_filename))
    log.info('Output:         {}'.format(output_dset_filename))
    log.info('Output Images:  {}'.format(output_data_dir))

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
        model = detector.load_model(weights_filename, num_outputs=15, num_channels=8)

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
        model = detector.load_model(weights_filename, num_outputs=22, num_channels=13)
    else:
        raise Exception('unknown weights file')

    log.info('Using {}'.format(type(ptdataset)))

    output_dset = kwcoco.CocoDataset(coco_dset_filename).copy()
    for img_info in tqdm(DataLoader(ptdataset, num_workers=0,
                                    batch_size=None, collate_fn=lambda x: x),
                         miniters=1):
        try:
            _predict_single(img_info, model=model,
                            model_outputs=model_outputs,
                            output_dset=output_dset,
                            output_dir=output_dset_filename.parent)

        except KeyboardInterrupt:
            log.info('interrupted')
            break
        except Exception:
            log.exception('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

    # self.dset.dump(str(self.output_dset_filename)+'_orig.json', indent=2)
    output_dset.dump(str(output_dset_filename), indent=2)
    log.info('output written to {}'.format(output_dset_filename))


def _predict_single(img_info, model, model_outputs,
                    output_dset: kwcoco.CocoDataset,
                    output_dir: Path):
    gid = img_info['id']
    name = img_info['name']
    img = img_info['imgdata']

    pred = detector.run(model, img, img_info)

    if pred is None:
        return

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

    pred_filename.parent.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        kwimage.imwrite(str(pred_filename),
                        pred,
                        backend='gdal',
                        compress='deflate')

        # kwimage.imwrite(str(pred_filename)[:-4] + '_orig.tif',
        #                 img[:, :, [3, 2, 1]],
        #                 backend='gdal',
        #                 compress='deflate')
        # # single band images
        # for band in range(pred.shape[2]):
        #     kwimage.imwrite(str(pred_filename)[:-4] + '_color_{}.tif'.format(band),
        #                     pred[:, :, band],
        #                     backend='gdal',
        #                     compress='deflate')


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
