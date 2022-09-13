"""
Defines a torch Dataset for kwcoco video data.

The parameters to each are handled by scriptconfig objects, which prevents us
from needing to specify what the available options are in multiple places.
"""
import einops
import warnings
import kwarray
import kwcoco
import kwimage
import numpy as np
import pandas as pd
import torch
import ubelt as ub
from torch.utils import data
from typing import Dict
from typing import Tuple
import scriptconfig as scfg


from watch import heuristics
from watch.utils import kwcoco_extensions
from watch.utils import util_bands
from watch.utils import util_iter
from watch.utils import util_kwimage
from watch.utils import util_time
from watch.tasks.fusion import utils
from watch.tasks.fusion.datamodules import data_utils
from watch.tasks.fusion.datamodules.data_augment import SpacetimeAugmentMixin
from watch.tasks.fusion.datamodules.smart_mixins import SMARTDataMixin
from watch.tasks.fusion.datamodules.spacetime_grid_builder import sample_video_spacetime_targets

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


class KWCocoVideoDatasetConfig(scfg.Config):
    """
    This is the configuration for a single dataset that could be used for
    train, test, or validation.

    In the future this might be convertable to, or handled by omegaconfig
    """
    default = {
        'time_steps': scfg.Value(2, help='number of temporal sampler per batch'),

        'chip_size': scfg.Value(128, help='spatial width and height per batch. DEPRECATED. Use chip_dims instead.'),

        'chip_dims': scfg.Value(None, help=ub.paragraph(
            '''
            spatial height/width per batch. If given as a single number, used
            as both width and height. Default is currently taken from
            deprecated chip_size, but in the future will be 128.
            '''), alias=['window_space_dims'], nargs='+'),

        'window_space_scale': scfg.Value(None, help=ub.paragraph(
            '''
            Change the "scale" or resolution of the video space used by the
            sliding window.
            Note: this modifies the GSD BEFORE the sample window has been
            selected, so the extent and resolution of the data changes.

            If specified as a numeric value then this is applied to as a scale
            factor. (E.g.  setting this to 2 is equivalent to scaling video
            space by 2). For geospatial data where each video has a
            "target_gsd", then this can be set to as an absolute by including
            the "GSD" suffix. (e.g. If this is set to "10GSD", then video space
            will be scaled to match).
            ''')),

        'space_scale': scfg.Value(None, help=ub.paragraph(
            '''
            Change the "scale" or resolution of the sampled video space.
            Note: this modifies the GSD AFTER the sample window has been
            selected, so the extend of the data does NOT change, but the resolution does.

            If specified as a
            numeric value then this is applied to as a scale factor. (E.g.
            setting this to 2 is equivalent to scaling video space by 2). For
            geospatial data where each video has a "target_gsd", then this can
            be set to as an absolute by including the "GSD" suffix. (e.g. If
            this is set to "10GSD", then video space will be scaled to match).

            This can also be set to "native" to use heterogeneous sampling.
            '''), alias=['data_space_scale']),

        # 'time_overlap': scfg.Value(0.0, help='fraction of time steps to overlap'),
        'chip_overlap': scfg.Value(
            0.0, help=ub.paragraph(
                '''
                Fraction of the spatial sliding window that will overlap.
                Only applies to training dataset when used in the data module.
                '''),
            alias=['window_space_overlap'],
        ),

        'channels': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            channels to use should be SensorChanSpec coercable
            ''')),

        'dist_weights': scfg.Value(0, help=ub.paragraph(
            '''
            To use distance-transform based weights on annotations or
            not
            ''')),

        'exclude_sensors': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            comma delimited list of sensors to avoid, such as S2 or L8
            ''')),

        'ignore_dilate': scfg.Value(0, help='Dilation applied to ignore masks.'),

        'max_epoch_length': scfg.Value(None, help=ub.paragraph(
            '''
            If specified, restricts number of steps per epoch
            ''')),

        'min_spacetime_weight': scfg.Value(0.5, help='Minimum space-time dilation weight'),

        'normalize_perframe': scfg.Value(False, help='undocumented - ignored'),

        'resample_invalid_frames': scfg.Value(True, help=ub.paragraph(
            '''
            if True, will attempt to resample any frame without valid
            data
            ''')),

        'set_cover_algo': scfg.Value(None, choices=[None, 'approx', 'exact'], help=ub.paragraph(
            '''
            Set cover algorithm to remove redundant gids when building space
            time targets. Options are 'approx' (a greedy solution) or 'exact'
            (an ILP solution). If None is passed, set cover is not computed.
            The 'exact' method requires the pulp package (and can be very slow
            so it is generally not recommended).
            ''')),

        'temporal_dropout': scfg.Value(0.0, type=float, help=ub.paragraph(
            '''
            Drops frames in a fraction of training batches
            ''')),

        'time_sampling': scfg.Value('contiguous', type=str, help=ub.paragraph(
            '''
            Strategy for expanding the time window across non-contiguous
            frames. Can be auto, contiguous, hard+distribute, or
            dilate_affinity
            ''')),

        'time_span': scfg.Value('2y', type=str, help=ub.paragraph(
            '''
            how long a time window should roughly span by default
            ''')),

        'use_centered_positives': scfg.Value(False, help=ub.paragraph(
            '''
            Use centers of annotations as window centers
            Only applies to training dataset when used in the data module.
            Validation/test dataset defaults to False.
            ''')),

        'upweight_centers': scfg.Value(True, help=ub.paragraph(
            '''
            Applies a weighting such that the center of the frame incurs more
            loss.
            ''')),

        'use_cloudmask': scfg.Value(1, type=int, help=ub.paragraph(
            '''
            Allow the dataloader to use the quality band to skip frames.
            ''')),

        'use_grid_positives': scfg.Value(True, help=ub.paragraph(
            '''
            Use annotation overlaps with grid as positives.
            Only applies to training dataset when used in the data module.
            Validation/test dataset defaults to True.
            ''')),

        # Overwritten for non-train
        'neg_to_pos_ratio': scfg.Value(1.0, type=float, help=ub.paragraph(
            '''
            maximum ratio of samples with no annotations to samples with
            annots.
            Only applies to training dataset when used in the data module.
            Validation/test dataset defaults to zero.
            ''')),
    }

    def normalize(self):
        if isinstance(self['exclude_sensors'], str):
            self['exclude_sensors'] = [s.strip() for s in self['exclude_sensors'].split(',')]
        self['time_steps'] = int(self['time_steps'])

        if self['chip_dims'] is None:
            d = int(self['chip_size'])
            self['chip_dims'] = [d, d]  # has to be a list not a tuple for yaml

        self['chip_size'] = None

        if self['space_scale'] is None:
            self['space_scale'] = self['window_space_scale']


class KWCocoVideoDataset(data.Dataset, SpacetimeAugmentMixin, SMARTDataMixin):
    """
    Accepted keyword arguments are specified in
    :class:`KWCocoVideoDatasetConfig`

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=10)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> channels = 'B10|B8a|B1|B8'
        >>> sample_shape = (3, 256, 256)
        >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape,
        >>>                           channels=channels,
        >>>                           time_sampling='soft2+distribute',
        >>>                           temporal_dropout=0.5)
        >>> index = len(self) // 4
        >>> item = self[index]
        >>> canvas = self.draw_item(item, overlay_on_image=1)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
    """
    def __init__(self, sampler, sample_shape=None, window_overlap=None,
                 mode='fit', **kwargs):

        config = KWCocoVideoDatasetConfig(cmdline=0, data=kwargs)
        BACKWARDS_COMPATIBILITY = True
        if BACKWARDS_COMPATIBILITY:
            if window_overlap is not None:
                config['chip_overlap'] = window_overlap
            if sample_shape is not None:
                config['time_steps'] = sample_shape[0]
                config['chip_dims'] = sample_shape[1:3]

        chip_dims = config['chip_dims']
        if not ub.iterable(chip_dims):
            chip_dims = (chip_dims, chip_dims)
        chip_h, chip_w = chip_dims
        window_dims = (config['time_steps'], chip_h, chip_w)
        window_overlap = config['chip_overlap']

        self.config = config
        # TODO: maintain instance variables xor items in the config, not both.
        self.__dict__.update(self.config.to_dict())
        self.sampler = sampler

        # Add extra categories if we need to and construct a new classes object
        graph = self.sampler.classes.graph

        # Update with heuristics
        # HACK: Overwrite kwcoco data
        for _catinfo in heuristics.CATEGORIES:
            name = _catinfo['name']
            exists_flag = name in graph.nodes
            if not exists_flag and _catinfo.get('required'):
                graph.add_node(name, **_catinfo)
            if exists_flag:
                graph.nodes[name].update(**_catinfo)

        self.classes = kwcoco.CategoryTree(graph)
        self.background_classes = set(heuristics.BACKGROUND_CLASSES) & set(graph.nodes)
        self.negative_classes = set(heuristics.NEGATIVE_CLASSES) & set(graph.nodes)
        self.ignore_classes = set(heuristics.IGNORE_CLASSNAMES) & set(graph.nodes)
        self.undistinguished_classes = set(heuristics.UNDISTINGUISHED_CLASSES) & set(graph.nodes)

        # construct composite classes
        # the idea is that these specific definitions will be configurable in the future
        self.non_salient_classes = self.background_classes | self.negative_classes
        self.salient_ignore_classes = self.ignore_classes
        # should we remove the ignore classes from salient_classes in the future?
        self.salient_classes = set(self.classes) - self.non_salient_classes

        # define foreground classes for the class activity head
        self.class_foreground_classes = set(self.classes) - self.background_classes - self.ignore_classes - self.undistinguished_classes

        channels = config['channels']
        time_sampling = config['time_sampling']
        exclude_sensors = config['exclude_sensors']
        use_centered_positives = config['use_centered_positives']
        use_grid_positives = config['use_grid_positives']
        set_cover_algo = config['set_cover_algo']
        time_span = config['time_span']
        neg_to_pos_ratio = config['neg_to_pos_ratio']
        max_epoch_length = config['max_epoch_length']
        window_space_scale = self.config['window_space_scale']

        if time_sampling == 'auto':
            time_sampling = 'hard+distribute'

        if mode == 'custom':
            new_sample_grid = None
            self.length = 1
        elif mode == 'test':
            # In test mode we have to sample everything for BAS
            # (TODO: for activity clf, we should only focus on candidate regions)
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=window_dims,
                window_overlap=window_overlap,
                keepbound=True,
                use_annot_info=False,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
                time_span=time_span,
                window_space_scale=window_space_scale,
                set_cover_algo=set_cover_algo,
                # workers='max(avail, 8)',  # could configure this
            )
            self.length = len(new_sample_grid['targets'])
        else:
            negative_classes = (
                self.ignore_classes | self.background_classes | self.negative_classes)
            new_sample_grid = sample_video_spacetime_targets(
                sampler.dset, window_dims=window_dims,
                window_overlap=window_overlap,
                negative_classes=negative_classes,
                keepbound=False,
                use_annot_info=True,
                exclude_sensors=exclude_sensors,
                time_sampling=time_sampling,
                time_span=time_span,
                use_centered_positives=use_centered_positives,
                use_grid_positives=use_grid_positives,
                window_space_scale=window_space_scale,
                set_cover_algo=set_cover_algo,
                # workers='max(avail, 8)',   # could configure this
            )

            n_pos = len(new_sample_grid['positives_indexes'])
            n_neg = len(new_sample_grid['negatives_indexes'])

            target_vidids = [v['video_id'] for v in new_sample_grid['targets']]

            # Hack: determine if videos should be grouped together
            target_posbit = kwarray.boolmask(
                new_sample_grid['positives_indexes'],
                len(new_sample_grid['targets']))

            if 1:
                vidnames = self.sampler.dset.videos(target_vidids).lookup('name')
                df = pd.DataFrame({
                    'vidid': target_vidids,
                    'vidname': vidnames,
                    'is_positive': target_posbit,
                }).reset_index(drop=False)

                # Hack, because we didn't encode the region in the cropped site
                # (rookie move)
                import watch
                pat = watch.utils.util_pattern.Pattern.coerce(r'\w+_R\d+_\d+', 'regex')
                vidname_to_region_name = {}
                for vidname in set(vidnames):
                    if pat.match(vidname):
                        vidname_to_region_name[vidname] = vidname.rsplit('_', 1)[0]

                self.vidname_to_region_name = vidname_to_region_name

                if vidname_to_region_name:
                    df['region'] = df['vidname'].apply(vidname_to_region_name.__getitem__)
                else:
                    df['region'] = df['vidname']

                key_to_group = dict(list(df.groupby(['region', 'is_positive'])))
                vidname_to_pool = {}
                for key, group in key_to_group.items():
                    vidname, flag = key
                    if flag:
                        pos_vid_idxs = group['index']
                        other_key = (vidname, False)
                        if other_key in key_to_group:
                            other = key_to_group[other_key]
                            neg_vid_idxs = other['index']
                        else:
                            neg_vid_idxs = []
                            other = []
                        n_pos = len(group)
                        n_neg = len(other)
                        max_neg = min(int(max(1, (neg_to_pos_ratio * n_pos))), n_neg)

                        if 0:
                            # TODO: dataloader logger
                            print(f'restrict to {max_neg=} in {vidname=} with {n_pos=} and {n_neg=}')

                        # neg_vid_idxs = posneg_groups[False]['index'].values
                        neg_vid_pool_ = list(util_iter.chunks(neg_vid_idxs, nchunks=max_neg))
                        pos_vid_pool_ = list(util_iter.chunks(pos_vid_idxs, nchunks=n_pos))
                        vid_pool = pos_vid_pool_ + neg_vid_pool_
                        vidname_to_pool[vidname] = [p for p in vid_pool if p]

                freqs = list(map(len, vidname_to_pool.values()))
                if len(freqs) == 0:
                    max_per_vid = 100
                    warnings.warn('Warning: no video pool')
                else:
                    max_per_vid = int(np.median(freqs))
                all_chunks = []
                for vidname, vid_pool in vidname_to_pool.items():
                    # print(len(vid_pool[0]))
                    # print(len(vid_pool[-1]))
                    rechunked_video_pool = list(util_iter.chunks(vid_pool, nchunks=max_per_vid))
                    all_chunks.extend(rechunked_video_pool)

                self.nested_pool = data_utils.NestedPool(all_chunks)

            self.length = len(self.nested_pool)

            if max_epoch_length is not None:
                self.length = min(self.length, max_epoch_length)

        self.new_sample_grid = new_sample_grid

        bg_catname = ub.peek(sorted(self.background_classes))
        self.bg_idx = self.classes.node_to_idx[bg_catname]

        # bg_node = graph.nodes['background']
        # if 'color' not in bg_node:
        #     bg_node['color'] = (0., 0., 0.)
        utils.category_tree_ensure_color(self.classes)

        self.special_inputs = {}

        if channels is None:
            # If channels is not specified, attempt to determine a something
            # sensible from the dataset statistics
            sensorchan_hist = kwcoco_extensions.coco_channel_stats(sampler.dset)['sensorchan_hist']
            sensorchans = ','.join(sorted([f'{sensor}:{chans}' for sensor, chan_hist in sensorchan_hist.items() for chans in chan_hist.keys()]))
            sensorchans = kwcoco.SensorChanSpec.coerce(sensorchans)
            if len(sensorchan_hist) > 0:
                warnings.warn(
                    'Channels are unspecified, but the dataset has a complex '
                    'set of channels with multiple sensors. '
                    'Passing an explicit sensorchan spec (via the `channels` '
                    'argument would be better.')
        else:
            # hack
            sensorchan_hist = None
            sensorchans = channels

        self.sensorchan = kwcoco.SensorChanSpec.coerce(sensorchans).normalize()

        # handle generic * sensors, the idea is that we find matches
        # in the dataset that can support the requested channels.
        if '*' in [s.sensor.spec for s in self.sensorchan.streams()]:
            # handle * sensor in a way that works with previous models
            # This code is a little messy and should be cleaned up
            if sensorchan_hist is None:
                sensorchan_hist = kwcoco_extensions.coco_channel_stats(sampler.dset)['sensorchan_hist']

            expanded_input_sensorchan_streams = []
            for fused_sensorchan in self.sensorchan.streams():
                sensor = fused_sensorchan.sensor
                chans = fused_sensorchan.chans
                if sensor.spec == '*':
                    for cand_sensor, cand_chans in sensorchan_hist.items():
                        valid_chan_cands = []
                        for cand_chan_group in cand_chans:
                            cand_chan_group = kwcoco.FusedChannelSpec.coerce(cand_chan_group)
                            chan_isect = chans & cand_chan_group
                            if chan_isect.spec == chans.spec:
                                valid_chan_cands.append(valid_chan_cands)
                                expanded_input_sensorchan_streams.append(cand_sensor + ':' + chans.spec)
                                break
                else:
                    expanded_input_sensorchan_streams.append('{}:{}'.format(sensor, chans))

            if not expanded_input_sensorchan_streams:
                print('sensorchan_hist = {}'.format(ub.repr2(sensorchan_hist, nl=1)))
                raise ValueError('The generic sensor * was given, but no data in the kwcoco file matched')

            self.sensorchan = kwcoco.SensorChanSpec.coerce(','.join(list(ub.unique(expanded_input_sensorchan_streams)))).normalize()

        # TODO: Clean up this code.
        _input_channels = []
        _sample_channels = []
        _input_sensorchans = []
        _sample_sensorchans = []
        for fused_sensorchan in self.sensorchan.streams():
            sensor = fused_sensorchan.sensor
            chans = fused_sensorchan.chans
            _stream = chans.as_oset()
            _sample_stream = _stream.copy()
            special_bands = _stream & util_bands.SPECIALIZED_BANDS
            if special_bands:
                raise NotImplementedError('This is broken ATM')
                # TODO: introspect which extra bands are needed for to compute
                # the sample, but hard code for now
                _sample_stream -= special_bands
                _sample_stream = _sample_stream | ub.oset('blue|green|red|nir|swir16|swir22'.split('|'))
                self.special_inputs[key] = special_bands
                _stream = [s + p for p in _stream for s in ['', 'D']]
            _input_sensorchans.append(sensor.spec + ':' + '|'.join(_stream))
            _sample_sensorchans.append(sensor.spec + ':' + '|'.join(_sample_stream))
            _input_channels.append('|'.join(_stream))
            _sample_channels.append('|'.join(_sample_stream))

            #### New: input_sensorchan will replace input_channels
            self.sample_sensorchan = kwcoco.SensorChanSpec(
                ','.join(_sample_sensorchans)
            )

            self.input_sensorchan = kwcoco.SensorChanSpec.coerce(
                ','.join(_input_sensorchans)
            )

        self.mode = mode

        self.disable_augmenter = False

        # hidden option for now (todo: expose this)
        self.inference_only = False
        self.with_change = True
        self.requested_tasks = {
            'change': True,
            'class': True,
            'saliency': True,
        }

        # Hacks: combinable channels can be visualized as RGB images.
        # The only reason this is a hack is because of the hardcoded names
        # otherwise it is a cool feature.
        self.default_combinable_channels = [
            ub.oset(['red', 'green', 'blue']),
            ub.oset(['Dred', 'Dgreen', 'Dblue']),
            ub.oset(['r', 'g', 'b']),
        ] + heuristics.HUERISTIC_COMBINABLE_CHANNELS

    def __len__(self):
        return self.length

    def _notify_about_tasks(self, requested_tasks=None, model=None):
        """
        Hacky method. Given the multimodal model, tell all the datasets which
        tasks they will need to generate data for. (This helps make the
        visualizations cleaner).
        """
        if model is not None:
            assert requested_tasks is None
            requested_tasks = {k: w > 0 for k, w in model.global_head_weights.items()}
        print(f'dataset notified: requested_tasks={requested_tasks}')
        assert requested_tasks is not None
        self.requested_tasks = requested_tasks

    @profile
    def _sample_one_frame(self, gid, sampler, coco_dset, target_, with_annots,
                          gid_to_isbad, gid_to_sample):
        # helper that was previously a nested function moved out for profiling
        coco_img = coco_dset.coco_image(gid)
        sensor_coarse = coco_img.img.get('sensor_coarse', '*')
        matching_sensorchan = self.sample_sensorchan.matching_sensor(sensor_coarse)
        sensor_channels = matching_sensorchan.chans
        # Require
        REPLACE_SAMECOLOR_REGIONS_WITH_NAN = target_.get('REPLACE_SAMECOLOR_REGIONS_WITH_NAN', 1)

        # sensor_channels = (self.sample_channels & coco_img.channels).normalize()
        tr_frame = target_.copy()
        tr_frame['gids'] = [gid]
        sample_streams = {}

        # TODO: separate ndsampler annotation loading function
        first_with_annot = with_annots

        # Flag will be set to true if any heuristic on any channel stream
        # forces us to mark this image as bad.
        force_bad = False

        if self.use_cloudmask:
            # Skip if quality mask indicates more than 50% clouds.
            is_cloud_iffy = self._interpret_quality_mask(
                sampler, coco_img, tr_frame)
            if is_cloud_iffy is not None:
                cloud_frac = is_cloud_iffy.mean()
                if cloud_frac > 0.5:
                    force_bad = 'too cloudy'

        if sensor_channels.numel() == 0:
            force_bad = 'Missing requested channels'

        for stream in sensor_channels.streams():
            if force_bad:
                break
            tr_frame['channels'] = stream
            tr_frame['padkw' ] = {'constant_values': np.nan}
            tr_frame['nodata' ] = 'float'
            sample = sampler.load_sample(
                tr_frame, with_annots=first_with_annot,
                dtype=np.float32
            )

            if REPLACE_SAMECOLOR_REGIONS_WITH_NAN:
                # This should be a better heuristic than the others we were
                # using

                # Process the bands in groups of 3
                hwc = sample['im'][0]
                # band_slider = kwarray.SlidingWindow((int(np.ceil(hwc.shape[2] / 3) * 3),), window=(3,))
                band_slider = kwarray.SlidingWindow((hwc.shape[2],), window=(1,))
                flag_stack = []
                for b_sl in band_slider:
                    bands = hwc[:, :, b_sl[0]]
                    bands = np.ascontiguousarray(bands)
                    is_samecolor = util_kwimage.find_samecolor_regions(bands)
                    flag_stack.append(is_samecolor)
                is_samecolor = np.stack(flag_stack, axis=2)
                samecolor_flags = is_samecolor[None, :] > 0
                num_samecolor = samecolor_flags.sum()
                if num_samecolor > 0:
                    # print(f'stream={stream}')
                    # print(f'num_samecolor={num_samecolor}')
                    sample['im'][samecolor_flags] = np.nan

            invalid_mask = np.isnan(sample['im'])

            any_invalid = np.any(invalid_mask)
            none_invalid = not any_invalid
            if none_invalid:
                all_invalid = False
            else:
                all_invalid = np.all(invalid_mask)

            if any_invalid:
                sample['invalid_mask'] = invalid_mask
            else:
                sample['invalid_mask'] = None

            if not all_invalid:
                sample_streams[stream.spec] = sample
                if 'annots' in sample:
                    # dont ask for annotations multiple times
                    first_with_annot = False
            else:
                # HACK: if the red channel is all bad, discard the frame
                # This can be removed once nodata is correctly propogated
                # in the team features. OR we can add a feature where we
                # keep track of an image wide observation mask and use that
                # instead of using red as a proxy for it.
                if 'red' in set(stream):
                    force_bad = 'invalid red channel'
                    break

        if not force_bad:
            if len(sample_streams) == 0:
                force_bad = 'no-streams'

        gid_to_isbad[gid] = force_bad
        gid_to_sample[gid] = sample_streams

    def summarize_item(self, item):
        """
        Return debugging stats about the item

        Args:
            item (dict): an item returned by __getitem__

        Returns:
            dict : a summary of the item
        """
        item_summary = {}
        item_summary['frame_summaries'] = []
        timestamps = []
        for frame in item['frames']:
            frame_summary = {}
            for mode_key, im_mode in frame['modes'].items():
                frame_summary[frame['sensor'] + ':' + mode_key] = im_mode.shape
            label_keys = [
                'class_idxs', 'saliency', 'change'
                'class_weights', 'saliency_weights', 'change_weights'
            ]
            for key in label_keys:
                if frame.get(key, None) is not None:
                    frame_summary[key] = frame[key].shape
            item_summary['frame_summaries'].append(frame_summary)
            if frame['date_captured']:
                timestamps.append(ub.timeparse(frame['date_captured']))
            frame_summary['num_annots'] = len(frame['ann_aids'])

        item_summary['video_name'] = item['video_name']
        if timestamps:
            deltas = np.diff(timestamps)
            deltas = [d.total_seconds() for d in deltas]
            item_summary['min_time'] = ub.timestamp(min(timestamps))
            item_summary['max_time'] = ub.timestamp(max(timestamps))
            item_summary['min_delta'] = min(deltas)
            item_summary['max_delta'] = max(deltas)
            item_summary['mean_delta'] = np.mean(deltas)
        item_summary['sample_gsd'] = item['sample_gsd']
        return item_summary

    @profile
    def __getitem__(self, index):
        """
        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> import watch
            >>> coco_dset = watch.demo.demo_kwcoco_multisensor()
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> # Each sensor uses all of its own channels
            >>> channels = None
            >>> self = KWCocoVideoDataset(sampler, sample_shape=(5, 256, 256), channels=channels, normalize_perframe=False)
            >>> self.disable_augmenter = False
            >>> index = 0
            >>> index = target = self.new_sample_grid['targets'][0]
            >>> item = self[index]
            >>> canvas = self.draw_item(item)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        # The index can be specified as either
        # * directly as a target (target) dictionary, or
        # * an integer index
        if isinstance(index, dict):
            target = index
            index = 'given-as-dictionary'
        else:
            if self.mode == 'test':
                # In test-mode the index directly determines the grid location.
                target = self.new_sample_grid['targets'][index]
            else:
                # In non-test-mode we discard the user index and randomly
                # sample a grid location to achive balanced sampling.
                try:
                    tr_idx = self.nested_pool.sample()
                except Exception as ex:
                    print(f'Failed to sample grid location: {ex=}')
                    target = None
                else:
                    target = self.new_sample_grid['targets'][tr_idx]

        if target is None:
            # Return None to indicate a failed sampling of a grid location
            return None

        target_ = target.copy()

        # get positive sample definition
        # collect sample
        sampler = self.sampler
        coco_dset = self.sampler.dset
        target_['as_xarray'] = False
        target_['legacy_annots'] = False
        target_['legacy_targets'] = False

        if 'video_id' not in target_:
            _gid = ub.peek(target_['gids'])
            target_['video_id'] = sampler.dset.imgs[_gid]['video_id']

        vidid = target_['video_id']
        video = coco_dset.index.videos[vidid]

        # Compute scale if we are doing that
        # This should live somewhere else, but lets just get it hooked up
        space_scale = self.config['space_scale']
        if target_.get('space_scale', None) is not None:
            # The target is allowed to overload the spatial scale
            space_scale = target_['space_scale']
        else:
            target_['space_scale'] = space_scale

        # Resolve spatial scale code
        vidspace_gsd = video.get('target_gsd', None)
        resolved_scale = data_utils.resolve_scale_request(
            request=space_scale, data_gsd=vidspace_gsd)
        sample_scale = resolved_scale['scale']
        sample_gsd = resolved_scale['gsd']
        target_['scale'] = sample_scale

        if isinstance(sample_scale, str) and sample_scale == 'native':
            target_.pop('scale')
            # native scales will only work in late-fused modes
            target_['use_native_scale'] = True
            target_['realign_native'] = 'largest'

        allow_augment = target_.get('allow_augment', True)
        if allow_augment:
            target_ = self._augment_spacetime_target(target_)

        if self.inference_only:
            with_annots = []
        else:
            with_annots = ['boxes', 'segmentation']

        ALLOW_RESAMPLE = self.resample_invalid_frames
        ALLOW_FEWER_FRAMES = 1

        # New true-multimodal data items
        gid_to_sample: Dict[str, Dict] = {}
        gid_to_isbad: Dict[str, bool] = {}

        for gid in target_['gids']:
            self._sample_one_frame(gid, sampler, coco_dset, target_, with_annots,
                                   gid_to_isbad, gid_to_sample)

        time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
        video_gids = time_sampler.video_gids

        # If we skipped the main gid, record why
        main_gid = target.get('main_gid', None)
        if main_gid is not None and gid_to_isbad[main_gid]:
            main_skip_reason = gid_to_isbad[main_gid]
        else:
            main_skip_reason = None

        error_level = 0 if ALLOW_FEWER_FRAMES else 1
        if ALLOW_RESAMPLE:
            # If any image is junk allow for a resample
            if any(gid_to_isbad.values()):
                vidid = target_['video_id']
                time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
                max_tries = 30  # parameterize
                for iter_idx in range(max_tries):
                    good_gids = np.array([gid for gid, flag in gid_to_isbad.items() if not flag])
                    if len(good_gids) == len(target['gids']):
                        break
                    bad_gids = np.array([gid for gid, flag in gid_to_isbad.items() if flag])
                    # print('resampling: {}'.format(index))
                    include_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, good_gids))[0]
                    exclude_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, bad_gids))[0]
                    try:
                        chosen = time_sampler.sample(include=include_idxs,
                                                     exclude=exclude_idxs,
                                                     error_level=error_level,
                                                     return_info=False)
                    except Exception:
                        if ALLOW_FEWER_FRAMES:
                            break
                        else:
                            raise
                    new_idxs = np.setdiff1d(chosen, include_idxs)
                    new_gids = video_gids[new_idxs]
                    # print('new_gids = {!r}'.format(new_gids))
                    if not len(new_gids):
                        print('exhausted resample possibilities')
                        # Exhausted all possibilities
                        break
                    for gid in new_gids:
                        self._sample_one_frame(gid, sampler, coco_dset, target_,
                                               with_annots, gid_to_isbad,
                                               gid_to_sample)

        good_gids = [gid for gid, flag in gid_to_isbad.items() if not flag]
        if len(good_gids) == 0:
            # Cannot force any good sample, try and return None
            return None

        final_gids = ub.oset(video_gids) & good_gids
        num_frames = len(final_gids)
        if num_frames == 0:
            raise Exception('0 frames')

        # coco_dset.images(final_gids).lookup('date_captured')
        target_['gids'] = final_gids

        if not self.inference_only:
            # build up info about the tracks
            dset = self.sampler.dset
            gid_to_dets: Dict[int, kwimage.Detections] = {}
            gid_to_det_window_dsize: Dict[int, Tuple[int, int]] = {}
            tid_to_aids = ub.ddict(list)
            tid_to_cids = ub.ddict(list)
            # tid_to_catnames = ub.ddict(list)
            for gid in final_gids:
                stream_sample = gid_to_sample[gid]
                frame_dets = None
                for mode_sample in stream_sample.values():
                    if 'annots' in mode_sample:
                        frame_dets: kwimage.Detections = mode_sample['annots']['frame_dets'][0]
                        break
                if frame_dets is None:
                    raise AssertionError(ub.paragraph(
                        f'''
                        Did not sample correctly.
                        Please send this info to jon.crall@kitware.com:
                        {dset=!r}
                        {gid=!r}
                        {target=!r}
                        {target_=!r}
                        '''
                    ))
                # The detections live in the space of their sample (i.e. video
                # or image space).  We can grab that info from ndsampler
                # (the naming could be better though)
                sample_tlbr = mode_sample['params']['sample_tlbr']
                dets_dsize = (
                    sample_tlbr.width.ravel()[0],
                    sample_tlbr.height.ravel()[0]
                )
                gid_to_det_window_dsize[gid] = dets_dsize
                gid_to_dets[gid] = frame_dets

            for gid, frame_dets in gid_to_dets.items():
                aids = frame_dets.data['aids']
                cids = frame_dets.data['cids']
                tids = dset.annots(aids).lookup('track_id', None)
                frame_dets.data['tids'] = tids
                for tid, aid, cid in zip(tids, aids, cids):
                    tid_to_aids[tid].append(aid)
                    tid_to_cids[tid].append(cid)

            tid_to_frame_cids = ub.ddict(list)
            for gid, frame_dets in gid_to_dets.items():
                cids = frame_dets.data['cids']
                tids = frame_dets.data['tids']
                frame_tid_to_cid = ub.dzip(tids, cids)
                for tid in tid_to_aids.keys():
                    cid = frame_tid_to_cid.get(tid, None)
                    tid_to_frame_cids[tid].append(cid)

            # TODO: be more efficient at this
            tid_to_frame_cnames = ub.map_vals(
                lambda cids: list(ub.take(self.classes.id_to_node, cids, None)),
                tid_to_frame_cids
            )

            task_tid_to_cnames = {
                'saliency': {},
                'class': {},
            }
            for tid, cnames in tid_to_frame_cnames.items():
                task_tid_to_cnames['class'][tid] = heuristics.hack_track_categories(cnames, 'class')
                task_tid_to_cnames['saliency'][tid] = heuristics.hack_track_categories(cnames, 'saliency')

            if self.upweight_centers:
                # Learn more from the center of the space-time patch
                time_weights = kwimage.gaussian_patch((1, num_frames))[0]
                time_weights = time_weights / time_weights.max()
                time_weights = time_weights.clip(0, 1)
                time_weights = np.maximum(time_weights, self.min_spacetime_weight)

        input_dsize = None

        # TODO: handle all augmentation before we construct any labels
        frame_items = []
        for time_idx, gid in enumerate(final_gids):
            img = coco_dset.index.imgs[gid]

            stream_sample = gid_to_sample[gid]
            assert len(stream_sample) > 0

            # Collect image data from all modes within this frame
            mode_to_imdata = {}
            mode_to_invalid_mask = {}
            mode_to_dsize = {}
            for mode_key, mode_sample in stream_sample.items():

                mode_imdata = mode_sample['im'][0]
                mode_invalid_mask = mode_sample.get('invalid_mask', None)
                if mode_invalid_mask is not None:
                    mode_invalid_mask = mode_invalid_mask[0]

                if input_dsize is not None:
                    # OI! This is very likely NOT the right thing to do here.
                    # We spent all this effort on robustly sampling the data.
                    # Let's not throw it away with a rando scale factor.
                    # ... but we do still need to solve the issue where different
                    # windows sizes are returned.
                    mode_imdata, _resize_info = kwimage.imresize(
                        mode_imdata, dsize=input_dsize, interpolation='linear',
                        antialias=True, return_info=True)

                    # TODO: need to handle any potential offset if letterbox is
                    # ever true, which currently it is not. For now (and maybe
                    # forever?) we can ignore this.
                    if mode_invalid_mask is not None:
                        mode_invalid_mask = kwimage.imresize(
                            mode_invalid_mask.astype(np.uint8),
                            dsize=input_dsize,
                            interpolation='nearest')
                else:
                    _resize_info = None
                    _resize_info

                mode_imdata = np.asarray(mode_imdata, dtype=np.float32)
                # ensure channel dim is not squeezed
                mode_hwc = kwarray.atleast_nd(mode_imdata, 3)
                # rearrange image axes for pytorch
                mode_chw = einops.rearrange(mode_hwc, 'h w c -> c h w')
                mode_to_imdata[mode_key] = mode_chw
                mode_to_invalid_mask[mode_key] = mode_invalid_mask
                h, w = mode_hwc.shape[0:2]
                mode_to_dsize[mode_key] = (w, h)

            # For each frame we need to choose a resolution for the truth.
            # Using the maximum resolution mode should be decent choise.
            # We could choose this to be arbitrary or independent of the input
            # dimensions, but it makes sense to pin it to the input data
            # in most cases.
            frame_target_dsize = max(mode_to_dsize.values(), key=np.prod)
            target_dims = frame_target_dsize[::-1]  # the size we want to predict
            # frame_target_dsize = (180, 180)

            dt_captured = img.get('date_captured', None)
            if dt_captured:
                dt_captured = util_time.coerce_datetime(dt_captured)
                timestamp = dt_captured.timestamp()
            else:
                timestamp = np.nan

            sensor = img.get('sensor_coarse', '*')

            frame_item = {
                'gid': gid,
                'date_captured': img.get('date_captured', ''),
                'timestamp': timestamp,
                'time_index': time_idx,
                'sensor': sensor,
                'modes': mode_to_imdata,
                'change': None,
                'class_idxs': None,
                'saliency': None,
                'change_weights': None,
                'class_weights': None,
                'saliency_weights': None,
                'target_dims': target_dims,
                'ann_aids': None,
            }

            if not self.inference_only:

                # The frame detections will be in a scaled videos space the
                # constant scale case.  TODO: will need special handling for
                # "native" resolutions on a per-mode / frame basis, we will
                # need the concept of an annotation window (where ndsampler
                # lets us assume the corners of each window are in
                # correspondence)
                _target_dsize = np.array(frame_target_dsize)
                _dets_dsize = np.array(gid_to_det_window_dsize[gid])
                dets_scale = (_target_dsize / _dets_dsize)

                frame_dets = gid_to_dets[gid]
                if frame_dets is None:
                    raise AssertionError('frame_dets = {!r}'.format(frame_dets))

                # Remember to apply any transform to the dets as well
                # TODO: the info scale is on a per-mode basis, need to
                # normalize it first or compute a mode-to-truth transform.

                # Annotations are returned relative to a some window, which
                # might not be the same as the final target space. Rescale the
                # annotations to put them into the final target output space.
                dets = frame_dets.scale(dets_scale)

                # TODO: if we ever letterbox, we may need a translation factor
                # right now we can ignore this.
                # dets = dets.translate(_resize_info['offset'])

                # Create truth masks
                bg_idx = self.bg_idx
                frame_target_shape = frame_target_dsize[::-1]
                space_shape = frame_target_shape
                frame_cidxs = np.full(space_shape, dtype=np.int32,
                                      fill_value=bg_idx)

                class_ohe_shape = (len(self.classes),) + space_shape
                salient_shape = space_shape

                # A "Salient" class is anything that is a foreground class
                # Not sure if this should be a dataloader thing or not
                frame_saliency = np.zeros(salient_shape, dtype=np.uint8)

                frame_class_ohe = np.zeros(class_ohe_shape, dtype=np.uint8)
                saliency_ignore = np.zeros(space_shape, dtype=np.uint8)
                frame_class_ignore = np.zeros(space_shape, dtype=np.uint8)

                task_target_ohe = {}
                task_target_ohe['saliency'] = frame_saliency
                task_target_ohe['class'] = frame_class_ohe

                task_target_ignore = {}
                task_target_ignore['saliency'] = saliency_ignore
                task_target_ignore['class'] = frame_class_ignore

                # Rasterize frame targets
                ann_polys = dets.data['segmentations'].to_polygon_list()
                ann_aids = dets.data['aids']
                ann_cids = dets.data['cids']
                ann_tids = dets.data['tids']
                frame_item['ann_aids'] = ann_aids

                frame_poly_weights = np.ones(space_shape, dtype=np.float32)

                # Note: it is important to respect class indexes, ids, and
                # name mappings
                # TODO: layer ordering? Multiclass prediction?
                for poly, aid, cid, tid in zip(ann_polys, ann_aids, ann_cids, ann_tids):  # NOQA

                    flag_poly_filled = False
                    if self.requested_tasks['saliency']:
                        # orig_cname = self.classes.id_to_node[cid]
                        new_salient_catname = task_tid_to_cnames['saliency'][tid][time_idx]
                        if new_salient_catname in self.salient_classes:
                            poly.fill(frame_saliency, value=1)
                            flag_poly_filled = True
                        if new_salient_catname in self.salient_ignore_classes:
                            poly.fill(saliency_ignore, value=1)

                    if self.requested_tasks['class']:
                        new_class_catname = task_tid_to_cnames['class'][tid][time_idx]
                        new_class_cidx = self.classes.node_to_idx[new_class_catname]
                        orig_cidx = self.classes.id_to_idx[cid]
                        if new_class_catname in self.ignore_classes:
                            poly.fill(frame_class_ignore, value=1)
                            poly.fill(frame_class_ohe[orig_cidx], value=1)
                        elif new_class_catname in self.class_foreground_classes:
                            poly.fill(frame_class_ohe[new_class_cidx], value=1)
                            flag_poly_filled = True

                    if self.dist_weights and flag_poly_filled:
                        # New feature where we encode that we care much more about
                        # segmenting the inside of the object than the outside.
                        # Effectively boundaries become uncertain.
                        shape = frame_class_ohe[0].shape
                        dtype = frame_class_ohe[0].dtype
                        dist, poly_mask = util_kwimage.polygon_distance_transform(
                            poly, shape=shape, dtype=dtype)
                        max_dist = dist.max()
                        if max_dist > 0:
                            dist_weight = dist / max_dist
                            weight_mask = dist_weight + (1 - poly_mask)
                            frame_poly_weights = frame_poly_weights * weight_mask

                frame_poly_weights = np.maximum(frame_poly_weights, self.min_spacetime_weight)

                # Postprocess (Dilate?) the truth map
                for cidx, class_map in enumerate(frame_class_ohe):
                    # class_map = kwimage.morphology(class_map, 'dilate', kernel=5)
                    frame_cidxs[class_map > 0] = cidx

                if self.upweight_centers:
                    sigma = (
                        (4.8 * ((space_shape[1] - 1) * 0.5 - 1) + 0.8),
                        (4.8 * ((space_shape[0] - 1) * 0.5 - 1) + 0.8),
                    )
                    space_weights = kwimage.normalize(kwimage.gaussian_patch(space_shape, sigma=sigma))
                    # space_weights = util_kwimage.upweight_center_mask(space_shape)
                    space_weights = np.maximum(space_weights, self.min_spacetime_weight)
                    frame_weights = space_weights * time_weights[time_idx] * frame_poly_weights
                else:
                    frame_weights = frame_poly_weights

                # Note: ensure this is resampled into target output space
                # Module the pixelwise weights by the 1 - the fraction of modes
                # that have nodata.
                DOWNWEIGHT_NAN_REGIONS = 1
                if DOWNWEIGHT_NAN_REGIONS:
                    nodata_total = 0.0
                    for mask in mode_to_invalid_mask.values():
                        if mask is None:
                            nodata_total += 0
                        else:
                            if len(mask.shape) == 3:
                                mask_ = ((mask.sum(axis=2) / mask.shape[2])).astype(float)
                            else:
                                mask_ = mask.astype(float)
                            mask_ = kwimage.imresize(mask_, dsize=frame_target_dsize)
                            nodata_total += mask_
                    # nodata_total = np.add.reduce([0 if mask is None else mask.sum(axis=2) / mask.shape[2] for mask in mode_to_invalid_mask.values()])
                    total_bands = len(mode_to_invalid_mask)
                    nodata_frac = nodata_total / total_bands
                    nodata_weight = 1 - nodata_frac
                    frame_weights = frame_weights * nodata_weight

                # Dilate ignore masks (dont care about the surrounding area # either)
                # frame_saliency = kwimage.morphology(frame_saliency, 'dilate', kernel=ignore_dilate)
                if self.ignore_dilate > 0:
                    saliency_ignore = kwimage.morphology(saliency_ignore, 'dilate', kernel=self.ignore_dilate)
                    frame_class_ignore = kwimage.morphology(frame_class_ignore, 'dilate', kernel=self.ignore_dilate)

                saliency_weights = frame_weights * (1 - saliency_ignore)
                class_weights = frame_weights * (1 - frame_class_ignore)
                saliency_weights = saliency_weights.clip(0, 1)
                frame_weights = frame_weights.clip(0, 1)

            if not self.inference_only:
                if self.requested_tasks['class'] or self.requested_tasks['change']:
                    frame_item['class_idxs'] = frame_cidxs
                    frame_item['class_weights'] = class_weights
                if self.requested_tasks['saliency']:
                    frame_item['saliency'] = frame_saliency
                    frame_item['saliency_weights'] = saliency_weights

            frame_items.append(frame_item)

        if self.normalize_perframe:
            for frame_item in frame_items:
                frame_modes = frame_item['modes']
                for mode_key in list(frame_modes.keys()):
                    mode_data = frame_modes[mode_key]
                    to_restack = []
                    for item in mode_data:
                        # TODO: use real nodata values? Ideally they have
                        # already been converted into nans
                        mask = (item != 0) & np.isfinite(item)
                        norm_item = kwimage.normalize_intensity(item, params={
                            'high': 0.90,
                            'mid': 0.5,
                            'low': 0.01,
                            'mode': 'linear',
                        }, mask=mask)
                        to_restack.append(norm_item)
                    mode_data_normed = np.stack(to_restack, axis=0)
                    frame_modes[mode_key] = mode_data_normed

        # Add in change truth
        if not self.inference_only:
            if self.requested_tasks['change']:
                if frame_items:
                    frame1 = frame_items[0]
                for frame1, frame2 in ub.iter_window(frame_items, 2):
                    class_weights1 = frame1['class_weights']
                    class_weights2 = frame2['class_weights']
                    class_idxs1 = frame1['class_idxs']
                    class_idxs2 = frame2['class_idxs']
                    if class_idxs2.shape != class_idxs1.shape:
                        class_idxs1 = kwimage.imresize(
                            class_idxs1, dsize=class_idxs2.shape[0:2][::-1],
                            interpolation='nearest')
                        class_weights1 = kwimage.imresize(
                            class_weights1, dsize=class_weights2.shape[0:2][::-1],
                            interpolation='nearest')
                    frame_change = (class_idxs1 != class_idxs2).astype(np.uint8)
                    # ToDO: configure kernel size here
                    frame_change = kwimage.morphology(frame_change, 'open', kernel=3)
                    change_weights = class_weights1 * class_weights2
                    frame2['change'] = frame_change
                    frame2['change_weights'] = change_weights.clip(0, 1)

        truth_keys = [
            'change', 'class_idxs',
            'saliency', 'class_weights',
            'saliency_weights', 'change_weights'
        ]

        # If we are augmenting
        fliprot_params = target_.get('fliprot_params', None)
        if fliprot_params is not None:
            for frame_item in frame_items:
                frame_modes = frame_item['modes']
                for mode_key in list(frame_modes.keys()):
                    mode_data = frame_modes[mode_key]
                    frame_modes[mode_key] = data_utils.fliprot(mode_data, **fliprot_params, axes=[1, 2])
                for key in truth_keys:
                    data = frame_item.get(key, None)
                    if data is not None:
                        frame_item[key] = data_utils.fliprot(data, **fliprot_params, axes=[-2, -1])

        # Convert data to torch
        for frame_item in frame_items:
            frame_modes = frame_item['modes']
            for mode_key in list(frame_modes.keys()):
                mode_data = frame_modes[mode_key]
                frame_modes[mode_key] = kwarray.ArrayAPI.tensor(mode_data)
            for key in truth_keys:
                data = frame_item.get(key, None)
                if data is not None:
                    frame_item[key] = kwarray.ArrayAPI.tensor(data)

        positional_tensors = None

        if True:
            # TODO: what is the standard way to do the learned embedding
            # "input vector"?

            # TODO: preprocess any auxiliary learnable information into a
            # Tensor. It is likely ideal to pre-stack whenever possible, but we
            # need to keep the row-form data to make visualization
            # straight-forward. We could use a flag to toggle it depending on
            # if we need to visualize or not.
            permode_datas = ub.ddict(list)
            prev_timestamp = None

            time_index_encoding = utils.ordinal_position_encoding(len(frame_items), 8).numpy()

            for frame_item in frame_items:

                k = 'timestamp'
                frame_timestamp = np.array([frame_item[k]]).astype(np.float32)

                for mode_code in frame_item['modes'].keys():
                    # Maybe this should be a model responsibility.
                    # I dont like defining the positional encoding in the
                    # dataset
                    key_tensor = data_utils._string_to_hashvec(mode_code)
                    permode_datas['mode_tensor'].append(key_tensor)
                    #
                    k = 'time_index'
                    time_index = frame_item[k]
                    # v = np.array([frame_item[k]]).astype(np.float32)
                    v = time_index_encoding[time_index]
                    permode_datas[k].append(v)

                    if prev_timestamp is None:
                        time_offset = np.array([0]).astype(np.float32)
                    else:
                        time_offset = frame_timestamp - prev_timestamp

                    # TODO: add seasonal positional encoding

                    permode_datas['time_offset'].append(time_offset)

                    k = 'sensor'
                    key_tensor = data_utils._string_to_hashvec(k)
                    permode_datas[k].append(key_tensor)

                frame_item['time_offset'] = time_offset
                prev_timestamp = frame_timestamp

            positional_arrays = ub.map_vals(np.stack, permode_datas)
            time_offset = positional_arrays.pop('time_offset', None)
            if time_offset is not None:
                scaled_time_offset = data_utils.abslog_scaling(time_offset)
                positional_arrays['time_offset'] = scaled_time_offset
            else:
                print('NONE TIME OFFSET: {}'.format(list(permode_datas.keys())))

            # This is flattened for each frame for each mode.
            # A bit hacky, not in love with it.
            positional_tensors = ub.map_vals(torch.from_numpy, positional_arrays)

        # Only pass back some of the metadata (because I think torch
        # multiprocessing makes a new file descriptor for every Python object
        # or something like that)
        tr_subset = ub.dict_isect(target_, {
            'gids', 'space_slice', 'video_id', 'fliprot_params', 'scale',
        })
        if main_skip_reason:
            tr_subset['main_skip_reason'] = main_skip_reason
        item = {
            # TODO: breakup modes into different items
            'index': index,
            'frames': frame_items,
            'positional_tensors': positional_tensors,
            'video_id': vidid,
            'video_name': video['name'],
            'sample_gsd': sample_gsd,
            'target': tr_subset
        }
        return item

    def cached_dataset_stats(self, num=None, num_workers=0, batch_size=2,
                             with_intensity=True, with_class=True):
        """
        Compute the normalization stats, and caches them

        TODO:
            - [ ] Does this dataset have access to the workdir?
            - [ ] Cacher needs to depend on config of this dataset
        """
        # Get stats on the dataset (todo: nice way to disable augmentation temporarilly for this)
        depends = ub.odict([
            ('num', num),
            ('hashid', self.sampler.dset._cached_hashid()),
            ('sensorchan', self.input_sensorchan.concise().spec),
            ('normalize_perframe', self.normalize_perframe),
            ('with_intensity', with_intensity),
            ('with_class', with_class),
            ('depends_version', 16),  # bump if `compute_dataset_stats` changes
        ])
        workdir = None
        cacher = ub.Cacher('dset_mean', dpath=workdir, depends=depends)
        dataset_stats = cacher.tryload()
        if dataset_stats is None or ub.argflag('--force-recompute-stats'):
            dataset_stats = self.compute_dataset_stats(
                num, num_workers=num_workers, batch_size=batch_size)
            cacher.save(dataset_stats)
        return dataset_stats

    def compute_dataset_stats(self, num=None, num_workers=0, batch_size=2,
                              with_intensity=True, with_class=True,
                              with_vidid=True):
        """
        Args:
            num (int | None): number of input items to compute stats for

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> dct_dset = coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=3)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 256, 256)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> self.compute_dataset_stats(num_workers=2)

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2')
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> sample_shape = (2, 256, 256)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=sample_shape, channels=None)
            >>> stats = self.compute_dataset_stats()
            >>> assert stats['class_freq']['star'] > 0 or stats['class_freq']['superstar'] > 0 or stats['class_freq']['eff'] > 0
            >>> assert stats['class_freq']['background'] > 0

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import watch
            >>> from watch.tasks.fusion import datamodules
            >>> num = 10
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='vidshapes-watch', chip_size=64, time_steps=3,
            >>>     num_workers=0, batch_size=3,
            >>>     normalize_inputs=num)
            >>> datamodule.setup('fit')
            >>> self = datamodule.torch_datasets['train']
            >>> coco_dset = self.sampler.dset
            >>> print({c.get('sensor_coarse') for c in coco_dset.images().coco_images})
            >>> print({c.channels.spec for c in coco_dset.images().coco_images})
            >>> num_workers = 0
            >>> batch_size = 6
            >>> s = (self.compute_dataset_stats(num=num))
            >>> print('s = {}'.format(ub.repr2(s, nl=3)))
            >>> self.compute_dataset_stats(num=num, with_intensity=False)
            >>> self.compute_dataset_stats(num=num, with_class=False)
            >>> self.compute_dataset_stats(num=num, with_class=False, with_intensity=False)
        """
        num = num if isinstance(num, int) and num is not True else 1000
        if not with_class and not with_intensity:
            num = 1  # efficiency hack
        stats_idxs = kwarray.shuffle(np.arange(len(self)), rng=0)[0:min(num, len(self))]
        stats_subset = torch.utils.data.Subset(self, stats_idxs)

        # Hack: disable augmentation if we are doing that
        self.disable_augmenter = True

        loader = self.make_loader(subset=stats_subset, num_workers=num_workers,
                                  shuffle=True, batch_size=batch_size)

        # Track moving average of each fused channel stream
        channel_stats = ub.AutoDict()

        timer = ub.Timer().tic()
        timer.first = 1

        classes = self.classes
        num_classes = len(classes)
        bins = np.arange(num_classes + 1)
        total_freq = np.zeros(num_classes, dtype=np.int64)

        sensor_mode_hist = ub.ddict(lambda: 0)

        video_id_histogram = {}

        # TODO: we should ensure instance level frequency data as well
        # as pixel level frequency data.

        # TODO: we should ensure we include at least one sample from each type
        # of modality.
        # Note: the requested order of the channels could be different that
        # what is registered in the dataset. Need to find a good way to account
        # for this.

        # Make a list of all unique modes in the dataset.
        # User specifies all of this explicitly now
        unique_sensor_modes = set(
            (s.sensor.spec, s.chans.spec)
            for s in self.input_sensorchan.streams())

        is_native = self.config['space_scale'] == 'native'

        print('unique_sensor_modes = {}'.format(ub.repr2(unique_sensor_modes, nl=1)))
        # TODO: we can compute the intensity histogram much more efficiently by
        # only doing it for unique channels (which might be duplicated)
        prog = ub.ProgIter(loader, desc='estimate dataset stats')
        for batch_items in prog:
            for item in batch_items:
                if item is None:
                    continue
                if with_vidid:
                    vidid = item['video_id']
                    if vidid not in set(video_id_histogram.keys()):
                        video_id_histogram[vidid] = 0
                    video_id_histogram[vidid] += 1
                for frame_item in item['frames']:
                    if with_class:
                        class_idxs = frame_item['class_idxs']
                        if class_idxs is not None:
                            # print(np.unique(class_idxs))
                            item_freq = np.histogram(class_idxs.ravel(), bins=bins)[0]
                            total_freq += item_freq
                    if with_intensity:
                        sensor_code = frame_item['sensor']
                        modes = frame_item['modes']

                        for mode_code, mode_val in modes.items():
                            sensor_mode_hist[(sensor_code, mode_code)] += 1
                            running = channel_stats[sensor_code][mode_code]
                            if not running:
                                running = kwarray.RunningStats()
                                channel_stats[sensor_code][mode_code] = running
                            dtype = np.float64
                            val = mode_val.numpy().astype(dtype)
                            weights = np.isfinite(val).astype(dtype)
                            # kwarray can handle nans now
                            if is_native:
                                # Put channels last so we can update multiple at once
                                flat_vals = val.transpose(1, 2, 0).reshape(-1, val.shape[0])
                                flat_weights = weights.transpose(1, 2, 0).reshape(-1, weights.shape[0])
                                data_utils.update_many(running, flat_vals, weights=flat_weights)
                            else:
                                running.update(val, weights=weights)

            if timer.first or timer.toc() > 5:
                from watch.utils.slugify_ext import smart_truncate
                if with_class:
                    intermediate = ub.sorted_vals(ub.dzip(classes, total_freq), reverse=True)
                    intermediate_text = ub.repr2(intermediate, compact=1)
                    intermediate_text = smart_truncate(intermediate_text, max_length=40, trunc_loc=0.8)
                else:
                    intermediate_text = ''

                if with_intensity:
                    curr = ub.udict(running.summarize(keepdims=False))
                    curr = curr & {'mean', 'std', 'max', 'min'}
                    curr = curr.map_values(float)
                    text = ub.repr2(curr, compact=1, precision=1, nl=0) + ' ' + intermediate_text
                else:
                    text = intermediate_text
                prog.set_postfix_str(text)
                timer.first = 0
                timer.tic()
        self.disable_augmenter = False

        channel_stats = channel_stats.to_dict()

        # Return the raw counts and let the model choose how to handle it
        if with_class:
            class_freq = ub.dzip(classes, total_freq)
        else:
            class_freq = None

        if with_intensity:
            input_stats = {}
            for sensor, submodes in channel_stats.items():
                for chan_key, running in submodes.items():
                    if is_native:
                        # ensure we have the expected shape
                        perchan_stats = running.summarize(axis=ub.NoParam, keepdims=True)
                        chan_mean = perchan_stats['mean'][:, None, None]
                        chan_std = perchan_stats['std'][:, None, None]
                    else:
                        perchan_stats = running.summarize(axis=(1, 2))
                        chan_mean = perchan_stats['mean']
                        chan_std = perchan_stats['std']

                    # For nans, set the mean to zero and set the std to a huge
                    # number if we dont have any data on it. That will prevent
                    # the network from doing much with it which is really the
                    # best we can do here.
                    chan_mean[np.isnan(chan_mean)] = 0
                    chan_std[np.isnan(chan_std)] = 1e8

                    chan_mean = chan_mean.round(6)
                    chan_std = chan_std.round(6)
                    # print('perchan_stats = {}'.format(ub.repr2(perchan_stats, nl=1)))
                    input_stats[(sensor, chan_key)] = {
                        'mean': chan_mean,
                        'std': chan_std,
                    }
        else:
            input_stats = None

        dataset_stats = {
            'unique_sensor_modes': unique_sensor_modes,
            'sensor_mode_hist': dict(sensor_mode_hist),
            'input_stats': input_stats,
            'class_freq': class_freq,
            'video_id_histogram': video_id_histogram,
        }
        return dataset_stats

    def draw_item(self, item, item_output=None, combinable_extra=None,
                  max_channels=5, max_dim=224, norm_over_time=0,
                  overlay_on_image=False, draw_weights=True, rescale='auto'):
        """
        Visualize an item produced by this DataSet.

        Each channel will be a row, and each column will be a timestep.

        Args:
            item (Dict): An item returned from the torch Dataset.

            overlay_on_image (bool):
                if True, the truth and prediction is drawn on top of
                an image, otherwise it is drawn on a black image.

            max_dim (int):
                max dimension to resize each grid cell to.

            max_channels (int) :
                maximum number of channel rows to draw

            item_output (Dict):
                Special task keys that we know how to plot.
                These should be some sort of binary or class prediction from
                the network. I'm not sure how best to pass the details
                of how they should be interpreted.

                Known keys:
                    change_probs
                    class_probs

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> channels = 'B10|B8a|B1|B8|B11'
            >>> combinable_extra = [['B10', 'B8', 'B8a']]  # special behavior
            >>> # combinable_extra = None  # uncomment for raw behavior
            >>> self = KWCocoVideoDataset(sampler, sample_shape=(5, 530, 610), channels=channels)
            >>> index = len(self) // 4
            >>> item = self[index]
            >>> fliprot_params = item['target'].get('fliprot_params', None)
            >>> # Calculate the probability of change for each frame
            >>> item_output = {}
            >>> change_prob_list = []
            >>> for frame in item['frames'][1:]:
            >>>     change_prob = kwimage.Heatmap.random(
            >>>         dims=frame['target_dims'], classes=1).data['class_probs'][0]
            >>>     if fliprot_params:
            >>>         change_prob = data_utils.fliprot(change_prob, **fliprot_params)
            >>>     change_prob_list += [change_prob]
            >>> change_probs = np.stack(change_prob_list)
            >>> item_output['change_probs'] = change_probs  # first frame does not have change
            >>> #
            >>> # Probability of each class for each frame
            >>> class_prob_list = []
            >>> for frame in item['frames']:
            >>>     class_prob = kwimage.Heatmap.random(
            >>>         dims=frame['target_dims'], classes=list(sampler.classes)).data['class_probs']
            >>>     class_prob_ = einops.rearrange(class_prob, 'c h w -> h w c')
            >>>     if fliprot_params:
            >>>         class_prob_ = data_utils.fliprot(class_prob_, **fliprot_params)
            >>>     class_prob_list += [class_prob_]
            >>> class_probs = np.stack(class_prob_list)
            >>> item_output['class_probs'] = class_probs  # first frame does not have change
            >>> #binprobs[0][:] = 0  # first change prob should be all zeros
            >>> print(ub.repr2(self.summarize_item(item), nl=-1))
            >>> canvas = self.draw_item(item, item_output, combinable_extra=combinable_extra, overlay_on_image=1)
            >>> canvas2 = self.draw_item(item, item_output, combinable_extra=combinable_extra, max_channels=3, overlay_on_image=0)
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas, fnum=1, pnum=(1, 2, 1))
            >>> kwplot.imshow(canvas2, fnum=1, pnum=(1, 2, 2))
            >>> kwplot.show_if_requested()
        """
        if rescale == 'auto':
            rescale = self.config['space_scale'] != 'native'

        if item is None:
            # BIG RED X
            # h, w = vertical_stack[-1].shape[0:2]
            h = w = (max_dim or 224)
            bad_canvas = kwimage.draw_text_on_image(
                {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                valign='center', halign='center', fontScale=10,
                color='red')
            return bad_canvas

        from watch.tasks.fusion.datamodules.batch_visualization import BatchVisualizationBuilder
        builder = BatchVisualizationBuilder(
            item=item, item_output=item_output,
            default_combinable_channels=self.default_combinable_channels,
            norm_over_time=norm_over_time, max_dim=max_dim,
            max_channels=max_channels, overlay_on_image=overlay_on_image,
            draw_weights=draw_weights, combinable_extra=combinable_extra,
            classes=self.classes, requested_tasks=self.requested_tasks,
            rescale=rescale)
        canvas = builder.build()
        return canvas

    def make_loader(self, subset=None, batch_size=1, num_workers=0, shuffle=False,
                    pin_memory=False):
        """
        Use this to make the dataloader so we ensure that we have the right
        worker init function.

        Args:
            subset (None | Dataset): if specified, the loader is made for
                this dataset instead of ``self``.

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import ndsampler
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> sampler = ndsampler.CocoSampler(coco_dset)
            >>> self = KWCocoVideoDataset(sampler, sample_shape=(3, 530, 610))
            >>> loader = self.make_loader(batch_size=2)
            >>> batch = next(iter(loader))
        """
        if subset is None:
            dataset = self
        else:
            dataset = subset
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory,
            worker_init_fn=worker_init_fn,
            collate_fn=ub.identity,  # disable collation
        )
        return loader


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()  # TODO
    self = worker_info.dataset

    if hasattr(self, 'sampler'):
        if hasattr(self.sampler.dset, 'connect'):
            # Reconnect to the backend if we are using SQL
            self.sampler.dset.connect(readonly=True)
