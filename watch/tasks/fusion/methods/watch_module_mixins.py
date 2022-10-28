import kwcoco
import torch
import numpy as np


class WatchModuleMixins:
    """
    Mixin methods for watch lightning modules
    """

    def reset_weights(self):
        for name, mod in self.named_modules():
            if hasattr(mod, 'reset_parameters'):
                mod.reset_parameters()

    @classmethod
    def demo_dataset_stats(cls):
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
        # 'sensor_mode_hist': {('sensor3', 'B10|B11|r|g|b|flowx|flowy|distri'): 1,
        #  ('sensor2', 'B8|B11|r|g|b|disparity|gauss'): 2,
        #  ('sensor0', 'B1|B8|B8a|B10|B11'): 1},
        # 'class_freq': {'star': 0,
        #  'superstar': 5822,
        #  'eff': 0,
        #  'negative': 0,
        #  'ignore': 9216,
        #  'background': 21826}}
        return channels, classes, dataset_stats

    def demo_batch(self, batch_size=1, num_timesteps=3, width=8, height=8, nans=0, rng=None):
        """
        Example:
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
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
            >>> from watch.tasks.fusion.methods.channelwise_transformer import *  # NOQA
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
        C = len(self.hparams.classes)
        T = num_timesteps
        batch = []

        width_distri = _specific_coerce(width, rng=rng)
        height_distri = _specific_coerce(height, rng=rng)

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

                modes = {
                    'pan': rng.rand(1, H0, W0).astype("float32"),
                    'red|green|blue': rng.rand(3, H1, W1).astype("float32"),
                    'nir|swir16|swir22': rng.rand(3, H2, W2).astype("float32"),
                }
                frame = {}
                if time_index == 0:
                    frame['change'] = None
                    frame['change_weights'] = None
                    frame['change_output_dims'] = None
                else:
                    frame['change'] = rng.randint(low=0, high=1, size=(H0, W0))
                    frame['change_weights'] = rng.rand(H0, W0)
                    frame['change_output_dims'] = (H0, W0)

                frame['class_idxs'] = rng.randint(low=0, high=C - 1, size=(H0, W0))
                frame['class_weights'] = rng.rand(H0, W0)
                frame['class_output_dims'] = (H0, W0)

                frame['saliency'] = rng.randint(low=0, high=1, size=(H0, W0))
                frame['saliency_weights'] = rng.rand(H0, W0)
                frame['saliency_output_dims'] = (H0, W0)

                frame['date_captured'] = '',
                frame['gid'] = bx
                frame['sensor'] = 'sensor1'
                frame['time_index'] = bx
                frame['time_offset'] = np.array([1]),
                frame['timestamp'] = 1
                frame['modes'] = modes
                # specify the desired predicted output size for this frame
                frame['output_dims'] = (H0, W0)

                if nans:
                    for v in modes.values():
                        flags = rng.rand(*v.shape) < nans
                        v[flags] = float('nan')

                for k in ['change', 'change_weights', 'class_idxs',
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
        DEPRECATE IN FAVOR OF watch.tasks.fusion.utils.load_model_from_package

        TODO:
            - [ ] Make the logic that defines the save_package and load_package
                methods with appropriate package header data a lightning
                abstraction.
        """
        # NOTE: there is no gaurentee that this loads an instance of THIS
        # model, the model is defined by the package and the tool that loads it
        # is agnostic to the model contained in said package.
        # This classmethod existing is a convinience more than anything else
        from watch.tasks.fusion.utils import load_model_from_package
        self = load_model_from_package(package_path)
        return self
