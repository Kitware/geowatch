import kwcoco
import ubelt as ub
import torch
import numpy as np
from geowatch.tasks.fusion.methods.network_modules import _class_weights_from_freq


class WatchModuleMixins:
    """
    Mixin methods for geowatch lightning modules
    """

    def reset_weights(self):
        for name, mod in self.named_modules():
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()

    def _device_dict(self):
        return {key: item.device for key, item in self.state_dict().items()}

    def devices(self):
        """
        Returns all devices this module state is mounted on

        Returns:
            Set[torch.device]: set of devices used by this model
        """
        state_devices = self._device_dict()
        devices = set(state_devices.values())
        if hasattr(self, 'device_ids'):
            # Handle data parallel
            for _id in self.device_ids:
                devices.add(torch.device(_id))
        return devices

    @property
    def main_device(self):
        """
        The main/src torch device used by this model
        """
        if hasattr(self, 'src_device_obj'):
            return self.src_device_obj
        else:
            devices = self.devices()
            if len(devices) > 1:
                raise NotImplementedError('no information maintained on which device is primary')
            else:
                return list(devices)[0]

    @classmethod
    def demo_dataset_stats(cls):
        """
        Mock data that mimiks a dataset summary a kwcoco dataloader could
        provide.
        """
        channels = kwcoco.ChannelSpec.coerce('pan,red|green|blue,nir|swir16|swir22')
        unique_sensor_modes = {
            ('sensor1', 'pan'),
            ('sensor1', 'red|green|blue'),
            ('sensor1', 'nir|swir16|swir22'),
        }
        input_stats = {k: {
            'mean': np.random.rand(len(k[1].split('|')), 1, 1),
            'std': np.random.rand(len(k[1].split('|')), 1, 1),
        } for k in unique_sensor_modes}

        classes = kwcoco.CategoryTree.coerce(3)
        dataset_stats = {
            'unique_sensor_modes': unique_sensor_modes,
            'input_stats': input_stats,
            'class_freq': {c: np.random.randint(0, 10000) for c in classes},
        }
        return channels, classes, dataset_stats

    def demo_batch(self, batch_size=1, num_timesteps=3, width=8, height=8, nans=0, rng=None, new_mode_sample=0):
        """
        Example:
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> batch = self.demo_batch()
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> result = self.forward_step(batch)
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(result))

        Example:
            >>> # With nans
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> batch = self.demo_batch(nans=0.5, num_timesteps=2)
            >>> item = batch[0]
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(batch))
            >>> result1 = self.forward_step(batch)
            >>> result2 = self.forward_step(batch, with_loss=0)
            >>> if 1:
            >>>   print(nh.data.collate._debug_inbatch_shapes(result1))
            >>>   print(nh.data.collate._debug_inbatch_shapes(result2))

        Example:
            >>> from geowatch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
            >>> channels, clases, dataset_stats = MultimodalTransformer.demo_dataset_stats()
            >>> self = MultimodalTransformer(
            >>>     arch_name='smt_it_stm_p1', tokenizer='linconv',
            >>>     decoder='mlp', classes=clases, global_saliency_weight=1,
            >>>     dataset_stats=dataset_stats, input_sensorchan=channels)
            >>> batch = self.demo_batch(new_mode_sample=1)
            >>> print(nh.data.collate._debug_inbatch_shapes(batch))
        """
        import kwarray
        from kwarray import distributions

        def _specific_coerce(val, rng=None):
            # Coerce for what we want to do here,
            import numbers
            if isinstance(val, numbers.Integral):
                distri = distributions.Constant(val, rng=rng)
            elif isinstance(val, (tuple, list)) and len(val) == 2:
                low, high = val
                distri = distributions.DiscreteUniform(low, high, rng=rng)
            else:
                raise TypeError(val)
            return distri

        rng = kwarray.ensure_rng(rng)

        B = batch_size
        C = len(self.classes)
        T = num_timesteps
        batch = []

        width_distri = _specific_coerce(width, rng=rng)
        height_distri = _specific_coerce(height, rng=rng)

        sensor_to_modes = ub.ddict(list)
        for sensor, mode in self.dataset_stats['unique_sensor_modes']:
            sensor_to_modes[sensor].append(mode)

        import itertools as it
        import kwcoco
        # sensor_mode_iter = it.cycle(self.dataset_stats['unique_sensor_modes'])
        sensor_iter = it.cycle(sensor_to_modes.keys())

        # OLD_MODES = 0
        for bx in range(B):
            modes = []
            frames = []
            for time_index in range(T):

                # Sample output target shape
                H0 = height_distri.sample()
                W0 = width_distri.sample()

                # Sample input shapes
                H1 = height_distri.sample()
                W1 = width_distri.sample()

                H2 = height_distri.sample()
                W2 = width_distri.sample()

                if new_mode_sample:
                    sensor = next(sensor_iter)
                    modes = {}
                    H, W = H0, W0
                    for mode in sensor_to_modes[sensor]:
                        size = kwcoco.FusedChannelSpec.coerce(mode).numel()
                        modes[mode] = rng.rand(size, H, W).astype("float32")
                        H = height_distri.sample()
                        W = width_distri.sample()

                else:
                    sensor = 'sensor1'
                    modes = {
                        'pan': rng.rand(1, H0, W0).astype("float32"),
                        'red|green|blue': rng.rand(3, H1, W1).astype("float32"),
                        'nir|swir16|swir22': rng.rand(3, H2, W2).astype("float32"),
                    }

                    # Add in channels the model exepcts
                    for stream in self.input_sensorchan.streams():
                        C = stream.chans.numel()
                        modes[stream.chans.spec] = rng.rand(C, H1, W1).astype("float32")

                frame = {}
                if time_index == 0:
                    frame['change'] = None
                    frame['change_weights'] = None
                    frame['change_output_dims'] = None
                else:
                    frame['change'] = rng.randint(low=0, high=1, size=(H0, W0))
                    frame['change_weights'] = rng.rand(H0, W0)
                    frame['change_output_dims'] = (H0, W0)

                # TODO: allow class-ohe xor class-idxs to be missing
                frame['class_idxs'] = rng.randint(low=0, high=C - 1, size=(H0, W0))
                frame['class_ohe'] = np.eye(C)[np.random.choice(C, H0 * W0)].reshape(H0, W0, C)
                frame['class_weights'] = rng.rand(H0, W0)
                frame['class_output_dims'] = (H0, W0)

                frame['saliency'] = rng.randint(low=0, high=1, size=(H0, W0))
                frame['saliency_weights'] = rng.rand(H0, W0)
                frame['saliency_output_dims'] = (H0, W0)

                frame['date_captured'] = '',
                frame['gid'] = bx
                frame['sensor'] = sensor
                frame['time_index'] = bx
                frame['time_offset'] = np.array([1]),
                frame['timestamp'] = 1
                frame['modes'] = modes
                # specify the desired predicted output size for this frame
                frame['output_dims'] = (H0, W0)

                if nans:
                    for v in modes.values():
                        flags = rng.rand(*v.shape[1:]) < nans
                        v[:, flags] = float('nan')

                        if time_index > 0:
                            frame['change_weights'][flags] = 0
                        frame['class_weights'][flags] = 0
                        frame['saliency_weights'][flags] = 0

                for k in ['change', 'change_weights', 'class_idxs', 'class_ohe',
                          'class_weights', 'saliency', 'saliency_weights']:
                    v = frame[k]
                    if v is not None:
                        frame[k] = torch.from_numpy(v)

                for k in modes.keys():
                    v = modes[k]
                    if v is not None:
                        modes[k] = torch.from_numpy(v)

                frames.append(frame)

            positional_tensors = {
                'mode_tensor': torch.rand(T, 16),
                'sensor': torch.rand(T, 16),
                'time_index': torch.rand(T, 8),
                'time_offset': torch.rand(T, 1),
            }
            target = {
                'gids': list(range(T)),
                'space_slice': [
                    slice(0, H0),
                    slice(0, W0),
                ]
            }
            item = {
                'video_id': 3,
                'video_name': 'toy_video_3',
                'frames': frames,
                'positional_tensors': positional_tensors,
                'target': target,
            }
            batch.append(item)
        return batch

    @property
    def has_trainer(self):
        try:
            # Lightning 1.7 raises an attribute error if not attached
            return self.trainer is not None
        except RuntimeError:
            return False

    @classmethod
    def load_package(cls, package_path, verbose=1):
        """
        DEPRECATE IN FAVOR OF geowatch.tasks.fusion.utils.load_model_from_package

        TODO:
            - [ ] Make the logic that defines the save_package and load_package
                methods with appropriate package header data a lightning
                abstraction.
        """
        # NOTE: there is no gaurentee that this loads an instance of THIS
        # model, the model is defined by the package and the tool that loads it
        # is agnostic to the model contained in said package.
        # This classmethod existing is a convinience more than anything else
        from geowatch.tasks.fusion.utils import load_model_from_package
        self = load_model_from_package(package_path)
        return self

    def _coerce_class_weights(self, class_weights):
        """
        Handles automatic class weighting based on dataset stats.

        Args:
            class_weights (str | FloatTensor):
                If already a tensor does nothing. If the string "auto" then
                class frequency weighting is used. The string "auto" can be
                suffixed with a "class modulation code".

        Note:
            A class modulate code is a a special syntax that lets the user
            modulate automatically computed class weights. Should be a comma
            separated list of name*weight or name*weight+offset. E.g.
            `auto:negative*0,background*0.001,No Activity*0.1+1`

        Example:
            >>> # xdoctest: +IGNORE_WANT
            >>> from geowatch.tasks.fusion.methods.watch_module_mixins import *  # NOQA
            >>> self = WatchModuleMixins()
            >>> self.classes = ['a', 'b', 'c', 'd', 'e']
            >>> self.class_freq = {
            >>>     'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100,
            >>> }
            >>> self._coerce_class_weights('auto')
            tensor([1., 1., 1., 1., 1.])
            >>> self.class_freq = {
            >>>     'a': 100, 'b': 100, 'c': 200, 'd': 300, 'e': 500, 'f': 800,
            >>> }
            >>> self._coerce_class_weights('auto')
            tensor([1.0000, 1.0000, 0.5000, 0.3333, 0.2000])
            >>> self._coerce_class_weights('auto:a+1,b*2,c*0+31415')
            tensor([2.0000e+00, 2.0000e+00, 3.1415e+04, 3.3333e-01, 2.0000e-01])
        """
        from geowatch import heuristics
        hueristic_ignore_keys = heuristics.IGNORE_CLASSNAMES
        if isinstance(class_weights, str):
            if class_weights.startswith('auto'):
                if ':' in class_weights:
                    class_weights, modulate_class_weights = class_weights.split(':')
                else:
                    modulate_class_weights = None
                class_weights.split(':')
                if self.class_freq is None:
                    heuristic_weights = {}
                else:
                    class_freq = ub.udict(self.class_freq) - hueristic_ignore_keys
                    total_freq = np.array(list(class_freq.values()))
                    cat_weights = _class_weights_from_freq(total_freq)
                    catnames = list(class_freq.keys())
                    print('total_freq = {!r}'.format(total_freq))
                    print('cat_weights = {!r}'.format(cat_weights))
                    print('catnames = {!r}'.format(catnames))
                    heuristic_weights = ub.dzip(catnames, cat_weights)
                #print('heuristic_weights = {}'.format(ub.urepr(heuristic_weights, nl=1)))

                heuristic_weights.update({k: 0 for k in hueristic_ignore_keys})
                print('heuristic_weights = {}'.format(ub.urepr(heuristic_weights, nl=1, align=':')))
                class_weights = []
                for catname in self.classes:
                    w = heuristic_weights.get(catname, 1.0)
                    class_weights.append(w)
                using_class_weights = ub.dzip(self.classes, class_weights)

                # Add in user-specific modulation of the weights
                if modulate_class_weights:
                    import re
                    parts = [p.strip() for p in modulate_class_weights.split(',')]
                    parts = [p for p in parts if p]
                    for part in parts:
                        toks = re.split('([+*])', part)
                        catname = toks[0]
                        rest_iter = iter(toks[1:])
                        weight = using_class_weights[catname]
                        nrhtoks = len(toks) - 1
                        assert nrhtoks % 2 == 0
                        nstmts = nrhtoks // 2
                        for _ in range(nstmts):
                            opcode = next(rest_iter)
                            arg = float(next(rest_iter))
                            if opcode == '*':
                                weight = weight * arg
                            elif opcode == '+':
                                weight = weight + arg
                            else:
                                raise KeyError(opcode)
                        # Modulate
                        using_class_weights[catname] = weight

                print('using_class_weights = {}'.format(ub.urepr(using_class_weights, nl=1, align=':')))
                class_weights = [
                    using_class_weights.get(catname, 1.0)
                    for catname in self.classes
                    if catname not in hueristic_ignore_keys
                ]
                class_weights = torch.FloatTensor(class_weights)
            else:
                raise KeyError(class_weights)
        else:
            raise NotImplementedError(f'{class_weights!r}')
        return class_weights

    def _coerce_saliency_weights(self, saliency_weights):
        """
        Finds weights to balance saliency forward / background classes.

        Args:
            saliency_weights (Tensor | str | None):
                Can be None, a raw tensor, "auto", or a string "<bg>:<fg>".
                Can also accept a YAML mapping from the keys "bg" and "fg" to
                their respective float weights.

        Returns:
            Tensor

        Example:
            >>> # xdoctest: +IGNORE_WANT
            >>> from geowatch.tasks.fusion.methods.watch_module_mixins import *  # NOQA
            >>> self = WatchModuleMixins()
            >>> self.saliency_num_classes = 2
            >>> self.background_classes = ['a', 'b', 'c']
            >>> self.foreground_classes = ['d', 'e']
            >>> self.class_freq = {
            >>>     'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100,
            >>> }
            >>> self._coerce_saliency_weights('auto')
            tensor([1.0000, 1.4925])
            >>> self.background_classes = ['a', 'b', 'c']
            >>> self.foreground_classes = []
            >>> self._coerce_saliency_weights('auto')
            tensor([  1., 300.])
            >>> self.background_classes = []
            >>> self.foreground_classes = []
            >>> self._coerce_saliency_weights('auto')
            tensor([1., 0.])
            >>> self._coerce_saliency_weights('2:1')
            tensor([2., 1.])
            >>> self._coerce_saliency_weights('70:20')
            tensor([70., 20.])
            >>> self._coerce_saliency_weights('{fg: 1, bg: 2}')
            tensor([2., 1.])
        """
        if saliency_weights is None:
            bg_weight = 1.0
            fg_weight = 1.0
        elif isinstance(saliency_weights, str):
            if saliency_weights == 'auto':
                class_freq = self.class_freq
                if class_freq is not None:
                    print(f'class_freq={class_freq}')
                    bg_freq = sum(class_freq.get(k, 0) for k in self.background_classes)
                    fg_freq = sum(class_freq.get(k, 0) for k in self.foreground_classes)
                    bg_weight = 1.
                    fg_weight = bg_freq / (fg_freq + 1)
                else:
                    bg_weight = 1.0
                    fg_weight = 1.0
            elif saliency_weights.lower() in {'null', 'none'}:
                bg_weight = 1.0
                fg_weight = 1.0
            else:
                try:
                    bg_weight, fg_weight = saliency_weights.split(':')
                    fg_weight = float(fg_weight.strip())
                    bg_weight = float(bg_weight.strip())
                except Exception:
                    from kwutil.util_yaml import Yaml
                    saliency_weights_ = Yaml.coerce(saliency_weights)
                    try:
                        bg_weight = float(saliency_weights_['bg'])
                        fg_weight = float(saliency_weights_['fg'])
                    except Exception:
                        # TODO better error message
                        raise
        else:
            raise TypeError(f'saliency_weights : {type(saliency_weights)} = {saliency_weights!r}')
        print(f'bg_weight={bg_weight}')
        print(f'fg_weight={fg_weight}')
        bg_fg_weights = [bg_weight, fg_weight]
        # What is the motivation for having "saliency_num_classes" be not 2?
        _n = self.saliency_num_classes
        _w = bg_fg_weights + ([0.0] * (_n - len(bg_fg_weights)))
        saliency_weights = torch.Tensor(_w)
        return saliency_weights

    def set_dataset_specific_attributes(self, input_sensorchan, dataset_stats):
        """
        Set module attributes based on dataset stats it will be trained on.

        Args:
            input_sensorchan (str | kwcoco.SensorchanSpec | None):
                The input sensor channels the model should expect

            dataset_stats (Dict | None):
                See :func:`demo_dataset_stats` for an example of this structure

        Returns:
            None | Dict: input_stats

        The following attributes will be set after calling this method.

            * self.class_freq

            * self.dataset_stats

            * self.input_sensorchan

            * self.unique_sensor_modes

        We also return an ``input_stats`` variable which should be used for
        setting model-dependent handling of input normalization.

        The handling of dataset_stats and input_sensorchan are weirdly coupled
        for legacy reasons and duplicated across several modules. This is a
        common location for that code to allow it to be more easily refactored
        and simplified at a later date.
        """
        if dataset_stats is not None:
            input_stats = dataset_stats['input_stats']
            class_freq = dataset_stats['class_freq']
            if input_sensorchan is None:
                input_sensorchan = ','.join(
                    [f'{s}:{c}' for s, c in dataset_stats['unique_sensor_modes']])
        else:
            class_freq = None
            input_stats = None

        # Handle channel-wise input mean/std in the network (This is in
        # contrast to common practice where it is done in the dataloader)
        if input_sensorchan is None:
            raise Exception(
                'need to specify input_sensorchan at least as the number of '
                'input channels')
        input_sensorchan = kwcoco.SensorChanSpec.coerce(input_sensorchan)

        if dataset_stats is None:
            # Handle the case where we know what the input streams are, but not
            # what their statistics are.
            input_stats = None
            unique_sensor_modes = {
                (s.sensor.spec, s.chans.spec)
                for s in input_sensorchan.streams()
            }
        else:
            unique_sensor_modes = dataset_stats['unique_sensor_modes']

        self.class_freq = class_freq
        self.dataset_stats = dataset_stats
        self.unique_sensor_modes = unique_sensor_modes
        self.input_sensorchan = input_sensorchan
        return input_stats

    def overfit(self, batch):
        """
        Overfit script and demo

        CommandLine:
            python -m xdoctest -m geowatch.tasks.fusion.methods.channelwise_transformer MultimodalTransformer.overfit --overfit-demo

        Example:
            >>> # xdoctest: +REQUIRES(--overfit-demo)
            >>> # ============
            >>> # DEMO OVERFIT:
            >>> # ============
            >>> from geowatch.tasks.fusion.methods.heterogeneous import *  # NOQA
            >>> from geowatch.tasks.fusion import methods
            >>> from geowatch.tasks.fusion import datamodules
            >>> from geowatch.utils.util_data import find_dvc_dpath
            >>> import geowatch
            >>> import kwcoco
            >>> from os.path import join
            >>> import os
            >>> if 0:
            >>>     '''
            >>>     # Generate toy datasets
            >>>     DATA_DPATH=$HOME/data/work/toy_change
            >>>     TRAIN_FPATH=$DATA_DPATH/vidshapes_msi_train/data.kwcoco.json
            >>>     mkdir -p "$DATA_DPATH"
            >>>     kwcoco toydata --key=vidshapes-videos8-frames5-randgsize-speed0.2-msi-multisensor --bundle_dpath "$DATA_DPATH/vidshapes_msi_train" --verbose=5
            >>>     '''
            >>>     coco_fpath = ub.expandpath('$HOME/data/work/toy_change/vidshapes_msi_train/data.kwcoco.json')
            >>>     coco_dset = kwcoco.CocoDataset.coerce(coco_fpath)
            >>>     channels="B11,r|g|b,B1|B8|B11"
            >>> if 1:
            >>>     dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>>     coco_dset = (dvc_dpath / 'Drop6') / 'imganns-KR_R001.kwcoco.zip'
            >>>     channels='blue|green|red|nir'
            >>> if 0:
            >>>     coco_dset = geowatch.demo.demo_kwcoco_multisensor(max_speed=0.5)
            >>>     # coco_dset = 'special:vidshapes8-frames9-speed0.5-multispectral'
            >>>     #channels='B1|B11|B8|r|g|b|gauss'
            >>>     channels='X.2|Y:2:6,B1|B8|B8a|B10|B11,r|g|b,disparity|gauss,flowx|flowy|distri'
            >>> coco_dset = kwcoco.CocoDataset.coerce(coco_dset)
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset=coco_dset,
            >>>     chip_size=128, batch_size=1, time_steps=5,
            >>>     channels=channels,
            >>>     normalize_peritem='blue|green|red|nir',
            >>>     normalize_inputs=32, neg_to_pos_ratio=0,
            >>>     num_workers='avail/2',
            >>>     mask_low_quality=True,
            >>>     observable_threshold=0.6,
            >>>     use_grid_positives=False, use_centered_positives=True,
            >>> )
            >>> datamodule.setup('fit')
            >>> dataset = torch_dset = datamodule.torch_datasets['train']
            >>> torch_dset.disable_augmenter = True
            >>> dataset_stats = datamodule.dataset_stats
            >>> input_sensorchan = datamodule.input_sensorchan
            >>> classes = datamodule.classes
            >>> print('dataset_stats = {}'.format(ub.urepr(dataset_stats, nl=3)))
            >>> print('input_sensorchan = {}'.format(input_sensorchan))
            >>> print('classes = {}'.format(classes))
            >>> # Choose subclass to test this with (does not cover all cases)
            >>> self = methods.HeterogeneousModel(
            >>>     classes=classes,
            >>>     dataset_stats=dataset_stats,
            >>>     input_sensorchan=channels,
            >>>     #token_dim=708,
            >>>     #token_dim=768 - 60,
            >>>     #backbone='vit_B_16_imagenet1k',
            >>>     token_dim=208,
            >>>     backbone='sits-former',
            >>>     position_encoder=position_encoder,
            >>>     )
            >>> self.datamodule = datamodule
            >>> datamodule._notify_about_tasks(model=self)
            >>> # Run one visualization
            >>> loader = datamodule.train_dataloader()
            >>> # Load one batch and show it before we do anything
            >>> batch = next(iter(loader))
            >>> print(ub.urepr(dataset.summarize_item(batch[0]), nl=3))
            >>> import kwplot
            >>> plt = kwplot.autoplt(force='Qt5Agg')
            >>> plt.ion()
            >>> canvas = datamodule.draw_batch(batch, max_channels=5, overlay_on_image=0)
            >>> kwplot.imshow(canvas, fnum=1)
            >>> # Run overfit
            >>> device = 0
            >>> self.overfit(batch)

        nh.initializers.KaimingNormal()(self)
        nh.initializers.Orthogonal()(self)
        """
        import kwplot
        # import torch_optimizer
        import xdev
        import kwimage
        import pandas as pd
        # import netharn as nh
        from kwutil.slugify_ext import smart_truncate
        from kwplot.mpl_make import render_figure_to_image

        sns = kwplot.autosns()
        datamodule = self.datamodule
        device = 0
        self = self.to(device)
        # loader = datamodule.train_dataloader()
        # batch = next(iter(loader))
        walker = ub.IndexableWalker(batch)
        for path, val in walker:
            if isinstance(val, torch.Tensor):
                walker[path] = val.to(device)
        outputs = self.training_step(batch)
        max_channels = 3
        canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=max_channels, overlay_on_image=0)
        kwplot.imshow(canvas)

        loss_records = []
        loss_records = [g[0] for g in ub.group_items(loss_records, lambda x: x['step']).values()]
        step = 0
        _frame_idx = 0
        # dpath = ub.ensuredir('_overfit_viz09')

        # optim_cls, optim_kw = nh.api.Optimizer.coerce(
        #     optim='RAdam', lr=1e-3, weight_decay=0,
        #     params=self.parameters())
        try:
            [optim], [sched] = self.configure_optimizers()
        except Exception:
            # optim = torch.optim.SGD(self.parameters(), lr=1e-4)
            optim = torch.optim.AdamW(self.parameters(), lr=1e-4)

        # optim = torch_optimizer.RAdam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        fnum = 2
        fig = kwplot.figure(fnum=fnum, doclf=True)
        fig.set_size_inches(15, 6)
        fig.subplots_adjust(left=0.05, top=0.9)
        prev = None
        for _frame_idx in xdev.InteractiveIter(list(range(_frame_idx + 1, 1000))):
            # for _frame_idx in list(range(_frame_idx, 1000)):
            num_steps = 20
            ex = None
            for _i in ub.ProgIter(range(num_steps), desc='overfit'):
                optim.zero_grad()
                outputs = self.training_step(batch)
                # outputs['item_losses']
                loss = outputs['loss']
                if torch.any(torch.isnan(loss)):
                    print('NAN OUTPUT!!!')
                    print('loss = {!r}'.format(loss))
                    print('prev = {!r}'.format(prev))
                    ex = Exception('prev = {!r}'.format(prev))
                    break
                # elif loss > 1e4:
                #     # Turn down the learning rate when loss gets huge
                #     scale = (loss / 1e4).detach()
                #     loss /= scale
                prev = loss
                # item_losses_ = nh.data.collate.default_collate(outputs['item_losses'])
                # item_losses = ub.map_vals(lambda x: sum(x).item(), item_losses_)
                loss.backward()
                item_losses = {'loss': loss.detach().cpu().numpy().ravel().mean()}
                loss_records.extend([{'part': key, 'val': val, 'step': step} for key, val in item_losses.items()])
                optim.step()
                step += 1
            canvas = datamodule.draw_batch(batch, outputs=outputs, max_channels=max_channels, overlay_on_image=0, max_items=4)
            kwplot.imshow(canvas, pnum=(1, 2, 1), fnum=fnum)
            fig = kwplot.figure(fnum=fnum, pnum=(1, 2, 2))
            #kwplot.imshow(canvas, pnum=(1, 2, 1))
            ax = sns.lineplot(data=pd.DataFrame(loss_records), x='step', y='val', hue='part')
            try:
                ax.set_yscale('logit')
            except Exception:
                ...
            fig.suptitle(smart_truncate(str(optim).replace('\n', ''), max_length=64))
            img = render_figure_to_image(fig)
            img = kwimage.convert_colorspace(img, src_space='bgr', dst_space='rgb')
            # fpath = join(dpath, 'frame_{:04d}.png'.format(_frame_idx))
            #kwimage.imwrite(fpath, img)
            xdev.InteractiveIter.draw()
            if ex:
                raise ex
        # TODO: can we get this batch to update in real time?
        # TODO: start a server process that listens for new images
        # as it gets new images, it starts playing through the animation
        # looping as needed

    def _save_package(self, package_path, verbose=1):
        """
        We define this as a protected method to allow modules to reuse the core
        code, but force each module to define the ``save_package`` method
        themselves with a doctest. In the future if this logic is general we
        may remove that restriction and refactor tests to be part of unit
        tests.
        """
        # import copy
        import json
        import torch.package

        # Fix an issue on 3.10 with torch 1.12
        from geowatch.monkey import monkey_torch
        monkey_torch.fix_package_modules()

        # shallow copy of self, to apply attribute hacks to
        # model = copy.copy(self)
        model = self

        backup_attributes = {}
        # Remove attributes we don't want to pickle before we serialize
        # then restore them
        unsaved_attributes = [
            'trainer',
            'train_dataloader',
            'val_dataloader',
            'test_dataloader',
            '_load_state_dict_pre_hooks',  # lightning 1.5
            '_trainer',  # lightning 1.7
        ]
        for key in unsaved_attributes:
            try:
                val = getattr(model, key, None)
            except Exception:
                val = None
            if val is not None:
                backup_attributes[key] = val

        train_dpath_hint = getattr(model, 'train_dpath_hint', None)
        if model.has_trainer:
            if train_dpath_hint is None:
                train_dpath_hint = model.trainer.log_dir
            datamodule = model.trainer.datamodule
            if datamodule is not None:
                model.datamodule_hparams = datamodule.hparams

        metadata_fpaths = []
        if train_dpath_hint is not None:
            train_dpath_hint = ub.Path(train_dpath_hint)
            metadata_fpaths += list(train_dpath_hint.glob('hparams.yaml'))
            metadata_fpaths += list(train_dpath_hint.glob('fit_config.yaml'))
            metadata_fpaths += list(train_dpath_hint.glob('config.yaml'))

        try:
            for key in backup_attributes.keys():
                setattr(model, key, None)
            arch_name = 'model.pkl'
            module_name = 'watch_tasks_fusion'
            """
            exp = torch.package.PackageExporter(package_path, debug=True)
            """
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
            # with torch.package.PackageExporter(package_path) as exp:
            with torch.package.PackageExporter(package_path) as exp:
                # if True:
                # TODO: this is not a problem yet, but some package types (mainly
                # binaries) will need to be excluded and added as mocks
                exp.extern('**', exclude=[
                    'geowatch.tasks.fusion.**',
                    'geowatch.tasks.fusion.methods.*'
                ])
                # exp.intern('geowatch.tasks.fusion.methods.*', allow_empty=False)
                exp.intern('geowatch.tasks.fusion.**', allow_empty=False)

                # Attempt to standardize some form of package metadata that can
                # allow for model importing with fewer hard-coding requirements

                # TODO:
                # Add information about how this was trained, and what epoch it
                # was saved at.
                package_header = {
                    'version': '0.3.0',
                    'arch_name': arch_name,
                    'module_name': module_name,
                    'packaging_time': ub.timestamp(),
                    'git_hash': None,
                    'module_path': None,
                }

                # Encode a git hash if we can identify that we are in a git
                # repository
                try:
                    import os
                    module_path = ub.Path(ub.modname_to_modpath(self.__class__.__module__)).absolute()
                    package_header['module_path'] = os.fspath(module_path)
                    info = ub.cmd('git rev-parse --short HEAD', cwd=module_path.parent)
                    if info.returncode == 0:
                        package_header['git_hash'] = info.stdout.strip()
                except Exception:
                    ...

                exp.save_text(
                    'package_header', 'package_header.json',
                    json.dumps(package_header)
                )
                exp.save_pickle(module_name, arch_name, model)

                # Save metadata
                for meta_fpath in metadata_fpaths:
                    with open(meta_fpath, 'r') as file:
                        text = file.read()
                    exp.save_text('package_header', meta_fpath.name, text)
        finally:
            # restore attributes
            for key, val in backup_attributes.items():
                setattr(model, key, val)

    def configure_optimizers(self):
        """
        Note: this is only a fallback for testing purposes. This should be
        overwrriten in your module or done via lightning CLI.
        """
        import netharn as nh
        from torch.optim import lr_scheduler

        # Netharn api will convert a string code into a type/class and
        # keyword-arguments to create an instance.
        optim_cls, optim_kw = nh.api.Optimizer.coerce(
            optimizer='adamw', lr=3e-4, weight_decay=3e-6)
        optim_kw['params'] = self.parameters()
        optimizer = optim_cls(**optim_kw)
        max_epochs = 160
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs)
        return [optimizer], [scheduler]
