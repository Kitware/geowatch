import kwcoco
import ndsampler
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils import data
import pathlib
from watch.tasks.fusion.datasets import common
from watch.tasks.fusion import utils


class OneraCD_2018(pl.LightningDataModule):
#     un-filtered
#     mean =[
#             1618.045472603135, #B01
#             1422.4117861742477, #B02
#             1359.4422181552754, #B03
#             1414.6326650140888, #B04
#             1558.066015014933, #B05
#             1986.5720639874007, #B06
#             2211.05960321667, #B07
#             2119.168043369016, #B08
#             711.6691026661, #B09
#             15.798994821955343, #B10
#             2134.1261592656733, #B11
#             1584.3473492925966, #B12
#             2345.39250148559, #B8A
#         ]
#     std = [
#             318.0413600546062, #B01
#             456.1716680330628, #B02
#             590.073089436455, #B03
#             849.3395398520843, #B04
#             808.8434944245414, #B05
#             810.2980239328889, #B06
#             888.134386002103, #B07
#             901.4549041572369, #B08
#             369.83128311537274, #B09
#             9.292564246350967, #B10
#             1114.8360249854718, #B11
#             983.4251876271745, #B12
#             950.995883516169, #B8A
#         ]
    mean = [
            1562.0766579032488, #B01
            1338.2290704889197, #B02
            1244.4365473161317, #B03
            1254.8445257885762, #B04
            1406.2908957584507, #B05
            1929.3345394166415, #B06
            2185.7971215083016, #B07
            2089.7967767112846, #B08
            664.2430765239151, #B09
            13.850620521359653, #B10
            1960.5317996244119, #B11
            1412.116801289823, #B12
            2343.9090645496567, #B8A
        ]
    std = [
            239.70035979139226, #B01
            325.0655318620384, #B02
            415.1683138256359, #B03
            625.9869373244433, #B04
            592.5562234734191, #B05
            631.5796533148324, #B06
            711.8276877371072, #B07
            747.0317373493228, #B08
            312.45130719530385, #B09
            4.51324437779879, #B10
            896.0873314714964, #B11
            752.2534022942613, #B12
            777.9910284369854, #B8A
        ]
    bce_weight = 30
    
    def __init__(
        self,
        train_kwcoco_path=None,
        test_kwcoco_path=None,
        time_steps=2,
        chip_size=128,
        time_overlap=0,
        chip_overlap=0.1,
        channels='B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A',
        valid_pct=0.1,
        batch_size=4,
        num_workers=4,
        preprocessing_step=None,
        tfms_channel_subset=None,
        tfms_train_channel_size=1000,
    ):
        super().__init__()
        self.train_kwcoco_path = train_kwcoco_path
        self.test_kwcoco_path = test_kwcoco_path
        self.time_steps = time_steps
        self.chip_size = chip_size
        self.time_overlap = time_overlap
        self.chip_overlap = chip_overlap
        self.channels = channels
        self.valid_pct = valid_pct
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.preprocessing_step = preprocessing_step
        self.tfms_train_channel_size = tfms_train_channel_size

        tfms_channel_subset = channels if (tfms_channel_subset is None) else tfms_channel_subset

        channel_split = channels.split("|")
        tfms_channel_subset = [
            idx
            for idx, channel in enumerate(tfms_channel_subset.split("|"))
            if channel in channel_split
        ]

        self.train_tfms = transforms.Compose([
            self.preprocessing_step,
            utils.DimensionDropout(1, self.tfms_train_channel_size),
        ])
        self.test_tfms = transforms.Compose([
            self.preprocessing_step,
            utils.Lambda(lambda x: x[:, tfms_channel_subset]),
        ])

    def setup(self, stage):

        if stage == "fit" or stage is None:
            if isinstance(self.train_kwcoco_path, pathlib.Path):
                coco_fpath = str(self.train_kwcoco_path.expanduser())
            else:
                coco_fpath = str(self.train_kwcoco_path)
            print('coco_fpath = {!r}'.format(coco_fpath))
            kwcoco_ds = kwcoco.CocoDataset(coco_fpath)
            kwcoco_sampler = ndsampler.CocoSampler(kwcoco_ds)
            train_val_ds = common.VideoDataset(
                kwcoco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                transform=self.train_tfms,
            )

            num_examples = len(train_val_ds)
            num_valid = int(self.valid_pct * num_examples)
            num_train = num_examples - num_valid

            self.train_dataset, self.valid_dataset = data.random_split(
                train_val_ds,
                [num_train, num_valid],
            )

        if stage == "test" or stage is None:
            if isinstance(self.test_kwcoco_path, pathlib.Path):
                coco_fpath = str(self.test_kwcoco_path.expanduser())
            else:
                coco_fpath = str(self.test_kwcoco_path)
            kwcoco_ds = kwcoco.CocoDataset(coco_fpath)
            kwcoco_sampler = ndsampler.CocoSampler(kwcoco_ds)
            self.test_dataset = common.VideoDataset(
                kwcoco_sampler,
                sample_shape=(self.time_steps, self.chip_size, self.chip_size),
                window_overlap=(self.time_overlap, self.chip_overlap, self.chip_overlap),
                channels=self.channels,
                transform=self.test_tfms,
            )

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("OneraCD_2018")
        parser.add_argument("--train_kwcoco_path", default=None, type=pathlib.Path)
        parser.add_argument("--test_kwcoco_path", default=None, type=pathlib.Path)
        parser.add_argument("--time_steps", default=2, type=int)
        parser.add_argument("--chip_size", default=128, type=int)
        parser.add_argument("--time_overlap", default=0, type=int)
        parser.add_argument("--chip_overlap", default=0.1, type=float)
        parser.add_argument("--channels", default='B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A', type=str)
        parser.add_argument("--valid_pct", default=0.1, type=float)
        parser.add_argument("--batch_size", default=4, type=int)
        parser.add_argument("--num_workers", default=4, type=int)
        parser.add_argument("--tfms_train_channel_size", default=1000, type=int)
        return parent_parser
