import datetime
import logging
import warnings
from pathlib import Path

import click
import kwcoco
import kwimage
import numpy as np
from tqdm import tqdm

from . import detector
from .utils import setup_logging

log = logging.getLogger(__name__)


class DatasetPredict:
    def __init__(self, dataset_filename, weights_filename, output=None):
        self.dset = kwcoco.CocoDataset(dataset_filename)
        self.dataset_filename = Path(dataset_filename)
        self.output_dset = self.dset.copy()
        self._set_output(output)

        log.debug('dset {}'.format(self.dset))
        log.debug('weights: {}'.format(weights_filename))
        log.debug('output: {}'.format(output))

        self.model = detector.load_model(weights_filename, num_outputs=15, num_channels=8)

    def _set_output(self, output):
        default_output_filename = 'out_{}.kwcoco.json'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        if output is None:
            self.output_dir = Path('/tmp')
            self.output_dset_filename = self.output_dir.joinpath(default_output_filename)
        else:
            output = Path(output)
            if output.is_dir():
                self.output_dir = output
                self.output_dset_filename = self.output_dir.joinpath(default_output_filename)
            else:
                self.output_dir = output.parent
                self.output_dset_filename = output

    def predict(self):
        for img_info in tqdm(self.dset.imgs.values(), miniters=1):
            try:
                self._predict_single(img_info)
            except KeyboardInterrupt:
                log.info('interrupted')
                break
            except Exception:
                log.exception('Unable to load id:{} - {}'.format(img_info['id'], img_info['name']))

        # self.dset.dump(str(self.output_dset_filename)+'_orig.json', indent=2)
        self.output_dset.dump(str(self.output_dset_filename), indent=2)
        log.info('output written to {}'.format(self.output_dset_filename))

    def _predict_single(self, img_info):
        gid = img_info['id']

        # pprint(img_info)
        name = img_info['name']
        sensor_coarse = img_info['sensor_coarse']

        if sensor_coarse == 'L8':
            img = self.load_L8(gid)
        elif sensor_coarse == 'S2':
            img = self.load_S2(gid)
        else:
            log.info('skipping {}'.format(sensor_coarse))
            return

        pred = detector.run(self.model, img, img_info)

        if pred is None:
            return

        if img_info.get('file_name'):
            dir = Path(img_info.get('file_name')).parent
        else:
            dir = Path(img_info['auxiliary'][0]['file_name']).parent

        pred_filename = self.output_dir.joinpath('landcover', dir, name + '_landcover.tif')

        info = {
            'file_name': str(pred_filename.relative_to(self.output_dir)),
            'channels': "|".join(detector.channels),
            'height': pred.shape[0],
            'width': pred.shape[1],
            'num_bands': pred.shape[2],
            'warp_aux_to_img': {'scale': [img_info['width'] / pred.shape[1],
                                          img_info['height'] / pred.shape[0]],
                                'type': 'affine'}
        }

        self.output_dset.imgs[gid]['auxiliary'].append(info)

        pred_filename.parent.mkdir(parents=True, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            kwimage.imwrite(str(pred_filename),
                            pred,
                            backend='gdal',
                            compress='deflate')

        # single band images
        # for band in range(pred.shape[2]):
        #     kwimage.imwrite(str(pred_filename)[:-4] + '_color_{}.tif'.format(band),
        #                     pred[:, :, band],
        #                     backend='gdal',
        #                     compress='deflate')

    def load_L8(self, gid):
        """
        Load an L8 image and stack it to look like a WV3 image.

        This will be removed after we train on L8 data.
        """
        channels_list = [
            'coastal',
            'blue',
            'green',
            'green',  # or pan
            'red',
            'red',
            'nir',
            'nir'
        ]
        # all of these channels are 30m GSD so no resizing is necessary
        img = np.dstack([self.dset.load_image(gid, channels) for channels in channels_list])
        return img

    def load_S2(self, gid):
        """
        Load an S2 image and stack it to look like a WV3 image.

        This will be removed after we train on S2 data.
        """

        channels_list = [
            'coastal',
            'blue',
            'green',
            'green',  # wrong, S2 doesn't cover this wavelength
            'red',
            'B06',
            ('B08', 'B8A', 'B07'),
            'B09'
        ]
        channel_images = [self._try_load_channel(gid, channels) for channels in channels_list]

        dshape = channel_images[1].shape
        # dsize is width, height
        dsize = (dshape[1], dshape[0])
        channel_images = [kwimage.imresize(img, dsize=dsize, interpolation='linear')
                          for img in channel_images]

        img = np.dstack(channel_images)
        return img

    def _try_load_channel(self, gid, channels):
        if isinstance(channels, (list, tuple)):
            ex = None
            for chan in channels:
                try:
                    return self._try_load_channel(gid, chan)
                except Exception as e:
                    ex = e
            raise Exception('Unable to load any channels {}: {}'.format(channels, str(ex))) from ex
        else:
            return self.dset.load_image(gid, channels)


@click.command()
@click.option('--dataset', required=True, type=click.Path(exists=True), help='input kwcoco dataset')
@click.option('--deployed', required=True, type=click.Path(exists=True), help='pytorch weights file')
@click.option('--output', required=False, type=click.Path(), help='output kwcoco dataset')
def predict(dataset, deployed, output):
    dp = DatasetPredict(dataset, deployed, output)
    dp.predict()


if __name__ == '__main__':
    setup_logging()
    predict()
