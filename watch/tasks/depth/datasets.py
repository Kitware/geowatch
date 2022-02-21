import numpy as np
from torchvision import transforms

from .demo_transform import Normalize, ToTensor, ToNumpy
from .dzyne_img_util import normalizeRGB
from ..landcover.datasets import _CocoTorchDataset

WV_CHANNELS = "coastal|blue|green|yellow|red|red-edge|near-ir1|near-ir2"


class WVRgbDataset(_CocoTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}

        self.transform = transforms.Compose([
            ToTensor(),
            Normalize(__imagenet_stats['mean'],
                      __imagenet_stats['std']),
            ToNumpy()
        ])

    def _include(self, gid):
        img = self.dset.imgs[gid]
        if img['sensor_coarse'] == 'WV':
            if img.get('channels') == WV_CHANNELS:
                return True

            for aux in img.get('auxiliary', []):
                if aux.get('channels') == WV_CHANNELS:
                    return True

        return False

    def _load(self, gid):
        delayed = self.dset.delayed_load(gid, channels='red|green|blue')
        img = delayed.finalize()

        # if self.dset.imgs[gid].get('channels') == WV_CHANNELS:
        #     img = self.dset.load_image(gid)
        # else:
        #     img = self.dset.load_image(gid, WV_CHANNELS)
        # img = img[:, :, [4, 2, 1]]

        img = normalizeRGB(np.asarray(img), (3, 97))
        img *= 255
        img = img.astype(np.uint8)

        # TODO original images are too big to fit in 12GB CUDA memory.
        # Will they be chipped for us when doing broad area search or do I need to chip them?
        # crop for testing
        # img = img[:1024, :1024, :]

        img = self.transform(img)
        return img
