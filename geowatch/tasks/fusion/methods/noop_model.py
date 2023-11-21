import pytorch_lightning as pl
import torch
from torch import nn

import kwcoco
import netharn as nh
import ubelt as ub

from geowatch import heuristics
from geowatch.tasks.fusion.methods.network_modules import RobustModuleDict
from geowatch.tasks.fusion.methods.watch_module_mixins import WatchModuleMixins


class NoopModel(pl.LightningModule, WatchModuleMixins):
    """
    No-op example model. Contains a dummy parameter to satisfy the optimizer
    and trainer.

    TODO:
        - [ ] Minimize even further.
        - [ ] Identify mandatory steps in __init__ and move to a parent class.
    """

    _HANDLES_NANS = True

    def get_cfgstr(self):
        cfgstr = f'{self.hparams.name}_NOOP'
        return cfgstr

    def __init__(
        self,
        classes=10,
        dataset_stats=None,
        input_sensorchan=None,
        name: str = "unnamed_model",
    ):
        """
        Args:
            name: Specify a name for the experiment. (Unsure if the Model is the place for this)
        """

        super().__init__()
        self.save_hyperparameters()

        self.dummy_param = nn.Parameter(torch.randn(1), requires_grad=True)

        input_stats = self.set_dataset_specific_attributes(input_sensorchan, dataset_stats)

        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.num_classes = len(self.classes)

        # TODO: this data should be introspectable via the kwcoco file
        hueristic_background_keys = heuristics.BACKGROUND_CLASSES

        # FIXME: case sensitivity
        hueristic_ignore_keys = heuristics.IGNORE_CLASSNAMES
        if self.class_freq is not None:
            all_keys = set(self.class_freq.keys())
        else:
            all_keys = set(self.classes)

        self.background_classes = all_keys & hueristic_background_keys
        self.ignore_classes = all_keys & hueristic_ignore_keys
        self.foreground_classes = (all_keys - self.background_classes) - self.ignore_classes
        # hueristic_ignore_keys.update(hueristic_occluded_keys)

        self.saliency_num_classes = 2
        self.class_weights = self._coerce_class_weights('auto')
        self.saliency_weights = self._coerce_saliency_weights('auto')

        self.sensor_channel_tokenizers = RobustModuleDict()

        # Unique sensor modes obviously isn't very correct here.
        # We should fix that, but let's hack it so it at least
        # includes all sensor modes we probably will need.
        if input_stats is not None:
            sensor_modes = set(self.unique_sensor_modes) | set(input_stats.keys())
        else:
            sensor_modes = set(self.unique_sensor_modes)

        # important to sort so layers are always created in the same order
        sensor_modes = sorted(sensor_modes)
        for k in sensor_modes:
            if isinstance(k, str):
                if k == '*':
                    s = c = '*'
                else:
                    raise AssertionError
            else:
                s, c = k
            if input_stats is None:
                input_norm = nh.layers.InputNorm()
            else:
                stats = input_stats.get((s, c), None)
                if stats is None:
                    input_norm = nh.layers.InputNorm()
                else:
                    input_norm = nh.layers.InputNorm(
                        **(ub.udict(stats) & {'mean', 'std'}))

            # key = sanitize_key(str((s, c)))
            key = f'{s}:{c}'
            self.sensor_channel_tokenizers[key] = nn.Sequential(
                input_norm,
            )

    def forward(self, x):
        return x

    def shared_step(self, batch, batch_idx=None, with_loss=True):
        outputs = {
            "change_probs": [
                [
                    0.5 * torch.ones(*frame["output_dims"])
                    for frame in example["frames"]
                    if frame["change"] is not None
                ]
                for example in batch
            ],
            "saliency_probs": [
                [
                    torch.ones(*frame["output_dims"], 2).sigmoid()
                    for frame in example["frames"]
                ]
                for example in batch
            ],
            "class_probs": [
                [
                    torch.ones(*frame["output_dims"], self.num_classes).softmax(dim=-1)
                    for frame in example["frames"]
                ]
                for example in batch
            ],
        }

        if with_loss:
            outputs["loss"] = self.dummy_param

        return outputs

    training_step = shared_step
    # this is a special thing for the predict step
    forward_step = shared_step

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def save_package(self, package_path, verbose=1):
        """

        CommandLine:
            xdoctest -m geowatch.tasks.fusion.methods.noop_model NoopModel.save_package

        Example:
            >>> # Test without datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> #from geowatch.tasks.fusion.methods.noop_model import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = join(dpath, 'my_package.pt')

            >>> # Use one of our fusion.architectures in a test
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> model = self = methods.NoopModel(
            >>>     input_sensorchan=5,)

            >>> # Save the model (TODO: need to save datamodule as well)
            >>> model.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> #recon = methods.NoopModel.load_package(package_path)
            >>> from geowatch.tasks.fusion.utils import load_model_from_package
            >>> recon = load_model_from_package(package_path)
            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = model.state_dict()
            >>> assert recon is not model
            >>> assert set(recon_state) == set(recon_state)
            >>> for key in recon_state.keys():
            >>>     assert (model_state[key] == recon_state[key]).all()
            >>>     assert model_state[key] is not recon_state[key]

            >>> # Check what's inside of the package
            >>> import zipfile
            >>> import json
            >>> zfile = zipfile.ZipFile(package_path)
            >>> header_file = zfile.open('my_package/package_header/package_header.json')
            >>> package_header = json.loads(header_file.read())
            >>> print('package_header = {}'.format(ub.urepr(package_header, nl=1)))
            >>> assert 'version' in package_header
            >>> assert 'arch_name' in package_header
            >>> assert 'module_name' in package_header
            >>> assert 'packaging_time' in package_header
            >>> assert 'git_hash' in package_header
            >>> assert 'module_path' in package_header

        Example:
            >>> # Test with datamodule
            >>> import ubelt as ub
            >>> from os.path import join
            >>> from geowatch.tasks.fusion import datamodules
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion.methods.noop_model import *  # NOQA
            >>> dpath = ub.Path.appdir('geowatch/tests/package').ensuredir()
            >>> package_path = dpath / 'my_package.pt'

            >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
            >>>     train_dataset='special:vidshapes8-multispectral-multisensor', chip_size=32,
            >>>     batch_size=1, time_steps=2, num_workers=2, normalize_inputs=10)
            >>> datamodule.setup('fit')
            >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
            >>> classes = datamodule.torch_datasets['train'].classes

            >>> # Use one of our fusion.architectures in a test
            >>> self = methods.NoopModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats, input_sensorchan=datamodule.input_sensorchan)

            >>> # We have to run an input through the module because it is lazy
            >>> batch = ub.peek(iter(datamodule.train_dataloader()))
            >>> outputs = self.training_step(batch)

            >>> trainer = pl.Trainer(max_steps=0)
            >>> trainer.fit(model=self, datamodule=datamodule)

            >>> # Save the self
            >>> self.save_package(package_path)

            >>> # Test that the package can be reloaded
            >>> recon = methods.NoopModel.load_package(package_path)

            >>> # Check consistency and data is actually different
            >>> recon_state = recon.state_dict()
            >>> model_state = self.state_dict()
            >>> assert recon is not self
            >>> assert set(recon_state) == set(recon_state)
            >>> from geowatch.utils.util_kwarray import torch_array_equal
            >>> for key in recon_state.keys():
            >>>     v1 = model_state[key]
            >>>     v2 = recon_state[key]
            >>>     if not torch.allclose(v1, v2, equal_nan=True):
            >>>         print('v1 = {}'.format(ub.urepr(v1, nl=1)))
            >>>         print('v2 = {}'.format(ub.urepr(v2, nl=1)))
            >>>         raise AssertionError(f'Difference in key={key}')
            >>>     assert v1 is not v2, 'should be distinct copies'
        """
        self._save_package(package_path, verbose=verbose)
