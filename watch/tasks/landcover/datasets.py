import logging
from copy import deepcopy

import kwcoco
import kwimage
import numpy as np
import torch.utils.data

log = logging.getLogger(__name__)


class CocoTorchDataset(torch.utils.data.Dataset):

    def __init__(self, dset):
        if isinstance(dset, kwcoco.CocoDataset):
            self.dset = dset
        else:
            self.dset = kwcoco.CocoDataset(dset)

        self.gids = sorted(list(filter(self._include, self.dset.imgs.keys())))

    def __len__(self):
        return len(self.gids)

    def __getitem__(self, idx):
        gid = self.gids[idx]
        img_info = deepcopy(self.dset.imgs[gid])
        img_info['imgdata'] = self._load(gid)
        return img_info

    def _include(self, gid):
        """
        Args:
            gid:
        Returns: True to include the given image in this dataset.  False to exclude.
        """
        return True

    def _load(self, img_info):
        """
        Load an image and return a numpy array.
        """
        raise NotImplementedError('subclass must override _load')


class L8asWV3Dataset(CocoTorchDataset):
    """
    Load L8 images and stack them to look like WV3 images.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] == 'L8'

    def _load(self, gid):
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


class S2asWV3Dataset(CocoTorchDataset):
    """
    Load S2 images and stack them to look like WV3 images.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] == 'S2'

    def _load(self, gid):

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


class S2Dataset(CocoTorchDataset):
    """
    Load S2 images an stack.
    """

    def _include(self, gid):
        return self.dset.imgs[gid]['sensor_coarse'] == 'S2'

    def _load(self, gid):
        all_channels = [
            'coastal', 'blue', 'green', 'red', 'B05',
            'B06', 'B07', 'nir', 'B8A', 'B09',
            'cirrus', 'swir16', 'swir22',
        ]

        dsize = (self.dset.imgs[gid]['width'], self.dset.imgs[gid]['height'])

        img_stack = []
        for channels in all_channels:
            img = self.dset.load_image(gid, channels)
            # log.debug('resizing {:<10} from {} to {}'.format(
            #     channels, (img.shape[1], img.shape[0]), dsize))
            img = kwimage.imresize(img, dsize=dsize, interpolation='linear')
            img_stack.append(img)

        img = np.dstack(img_stack)

        return img
