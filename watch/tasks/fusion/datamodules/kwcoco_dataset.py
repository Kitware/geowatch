"""
Defines a torch Dataset for kwcoco video data.

The parameters to each are handled by scriptconfig objects, which prevents us
from needing to specify what the available options are in multiple places.

Example:
    >>> # Demo with toy data
    >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
    >>> import kwcoco
    >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=10)
    >>> channels = 'B10|B8a|B1|B8'
    >>> self = KWCocoVideoDataset(coco_dset, time_dims=3, window_dims=(300, 300),
    >>>                           channels=channels,
    >>>                           input_space_scale='native',
    >>>                           output_space_scale=None,
    >>>                           window_space_scale=1.2,
    >>>                           time_sampling='soft2+distribute',
    >>>                           time_kernel='-1y,0,1y',
    >>>                           temporal_dropout=0.5)
    >>> self.disable_augmenter = True
    >>> index = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][3]]
    >>> item = self[index]
    >>> print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
    >>> canvas = self.draw_item(item, overlay_on_image=0, rescale=0)
    >>> # xdoctest: +REQUIRES(--show)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(canvas)
    >>> kwplot.show_if_requested()

Example:
    >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
    >>> # Demo with real data
    >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
    >>> import watch
    >>> import kwcoco
    >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    >>> #coco_fpath = dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'
    >>> coco_fpath = dvc_dpath / 'Drop4-BAS/data_train.kwcoco.json'
    >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
    >>> ##'red|green|blue',
    >>> self = KWCocoVideoDataset(
    >>>     coco_dset,
    >>>     time_dims=11, window_dims=(196, 196),
    >>>     window_overlap=0,
    >>>     channels="(S2,L8):blue|green|red|nir",
    >>>     input_space_scale='10GSD',
    >>>     window_space_scale='10GSD',
    >>>     output_space_scale='10GSD',
    >>>     #normalize_peritem='nir',
    >>>     dist_weights=1,
    >>>     quality_threshold=0,
    >>>     neg_to_pos_ratio=0, time_sampling='soft2',
    >>> )
    >>> self.requested_tasks['change'] = 1
    >>> self.requested_tasks['saliency'] = 1
    >>> self.requested_tasks['class'] = 0
    >>> self.requested_tasks['boxes'] = 1
    >>> index = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][0]]
    >>> index['allow_augment'] = False

    >>> target = item['target']
    >>> print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
    >>> # xdoctest: +REQUIRES(--show)
    >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0, rescale=0)
    >>> import kwplot
    >>> kwplot.autompl()
    >>> kwplot.imshow(canvas, fnum=1)
    >>> kwplot.show_if_requested()

Ignore:
    >>> self.disable_augmenter = True
    >>> self.normalize_peritem = None
    >>> self.config['mask_low_quality'] = True
    >>> self.config['force_bad_frames'] = True
    >>> self.config['resample_invalid_frames'] = 0
    >>> index = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][int((2.5 * 17594) // 3)]]
    >>> item1 = self[index]
    >>> self.normalize_peritem = kwcoco.FusedChannelSpec.coerce('red|green|blue|nir')
    >>> item2 = self[index]
    >>> canvas1 = self.draw_item(item1, max_channels=10, overlay_on_image=0, rescale=0, draw_weights=0, draw_truth=0)
    >>> canvas2 = self.draw_item(item2, max_channels=10, overlay_on_image=0, rescale=0, draw_weights=0, draw_truth=0)
    >>> kwplot.imshow(canvas1, fnum=3, pnum=(2, 1, 1), title='no norm (per-frame normalized for viz purposes only)')
    >>> kwplot.imshow(canvas2, fnum=3, pnum=(2, 1, 2), title='per-item normalization (across time)')
"""
import einops
import warnings
import kwarray
import kwcoco
import kwimage
import ndsampler
import numpy as np
import pandas as pd
import torch
import ubelt as ub
from shapely.ops import unary_union
from torch.utils import data
from typing import Dict
import functools
import scriptconfig as scfg
from os import getenv

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
from watch.tasks.fusion.datamodules import spacetime_grid_builder

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity

# See ~/code/watch/docs/coding_oddities.rst
# For notes on Spaces


class KWCocoVideoDatasetConfig(scfg.Config):
    """
    This is the configuration for a single dataset that could be used for
    train, test, or validation.

    In the future this might be convertable to, or handled by omegaconfig

    The core spacetime parameters are:

        * window_space_scale
        * input_space_scale
        * output_space_scale
        * time_steps
        * time_sampling
        * chip_dims / window_space_dims

    Also:
        * set_cover_algo

    """
    default = {
        ###############
        # SPACE OPTIONS
        ###############

        'chip_dims': scfg.Value(128, help=ub.paragraph(
            '''
            spatial height/width per batch. If given as a single number, used
            as both width and height. Default is currently taken from
            deprecated chip_size, but in the future will be 128.
            '''), alias=['window_space_dims', 'window_dims', 'chip_size'], nargs='+'),

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
            '''), alias=['window_resolution']),

        'input_space_scale': scfg.Value(None, help=ub.paragraph(
            '''
            Change the "scale" or resolution of the sampled video space.
            Note: this modifies the GSD AFTER the sample window has been
            selected, so the extend of the data does NOT change, but the resolution does.

            If specified as a numeric value then this is applied to as a scale
            factor. (E.g.  setting this to 2 is equivalent to scaling video
            space by 2). For geospatial data where each video has a
            "target_gsd", then this can be set to as an absolute by including
            the "GSD" suffix. (e.g. If this is set to "10GSD", then video space
            will be scaled to match).

            This can also be set to "native" to use heterogeneous sampling.
            '''), alias=['space_scale', 'data_space_scale', 'input_resolution']),

        'output_space_scale': scfg.Value(None, help=ub.paragraph(
            '''
            Change the "scale" or resolution of the desired target resolution.

            Follows other GSD / scale semantics.
            '''), alias=['target_space_scale', 'output_resolution']),

        # 'time_overlap': scfg.Value(0.0, help='fraction of time steps to overlap'),
        'chip_overlap': scfg.Value(
            0.0, help=ub.paragraph(
                '''
                Fraction of the spatial sliding window that will overlap.
                Only applies to training dataset when used in the data module.
                '''),
            alias=['window_space_overlap', 'window_overlap'],
        ),

        ##############
        # TIME OPTIONS
        ##############

        'time_steps': scfg.Value(2, help='number of temporal sampler per batch', alias=['time_dims']),


        'time_sampling': scfg.Value('contiguous', type=str, help=ub.paragraph(
            '''
            Strategy for expanding the time window across non-contiguous
            frames. Can be auto, contiguous, hard+distribute, or
            dilate_affinity
            ''')),

        'time_span': scfg.Value('2y', help=ub.paragraph(
            '''
            how long a time window should roughly span by default
            ''')),

        'time_kernel': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Mutually exclusive with time_span.
            ''')),

        ##############
        # MODE OPTIONS
        ##############

        'channels': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            channels to use should be SensorChanSpec coercable
            ''')),

        'include_sensors': scfg.Value(None, help=ub.paragraph(
            '''
            if specified can be comma separated valid sensors. NOTE: this
            should be specified via a sensorchan speci in channels instead
            ''')),

        'exclude_sensors': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            comma delimited list of sensors to avoid, such as S2 or L8
            ''')),

        ##############
        # SIZE OPTIONS
        ##############

        'select_images': scfg.Value(
            None, type=str, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which images
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.images[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.id < 3' will select all image ids less than 3.
                '.file_name | test(".*png")' will select only images with
                file names that end with png.
                '.file_name | test(".*png") | not' will select only images
                with file names that do not end with png.
                '.myattr == "foo"' will select only image dictionaries
                where the value of myattr is "foo".
                '.id < 3 and (.file_name | test(".*png"))' will select only
                images with id less than 3 that are also pngs.
                .myattr | in({"val1": 1, "val4": 1}) will take images
                where myattr is either val1 or val4.

                Requries the "jq" python library is installed.
                ''')),

        'select_videos': scfg.Value(
            None, help=ub.paragraph(
                '''
                A json query (via the jq spec) that specifies which videos
                belong in the subset. Note, this is a passed as the body of
                the following jq query format string to filter valid ids
                '.videos[] | select({select_images}) | .id'.

                Examples for this argument are as follows:
                '.name | startswith("foo")' will select only videos
                where the name starts with foo.

                Only applicable for dataset that contain videos.

                Requries the "jq" python library is installed.
                ''')),

        'max_epoch_length': scfg.Value(None, help=ub.paragraph(
            '''
            If specified, restricts number of steps per epoch
            ''')),

        #######################
        # SAMPLING GRID OPTIONS
        #######################

        'set_cover_algo': scfg.Value(None, choices=[None, 'approx', 'exact'], help=ub.paragraph(
            '''
            Set cover algorithm to remove redundant gids when building space
            time targets. Options are 'approx' (a greedy solution) or 'exact'
            (an ILP solution). If None is passed, set cover is not computed.
            The 'exact' method requires the pulp package (and can be very slow
            so it is generally not recommended).
            ''')),

        'use_centered_positives': scfg.Value(False, help=ub.paragraph(
            '''
            Use centers of annotations as window centers
            Only applies to training dataset when used in the data module.
            Validation/test dataset defaults to False.
            ''')),

        'use_grid_positives': scfg.Value(True, help=ub.paragraph(
            '''
            Use annotation overlaps with grid as positives.
            Only applies to training dataset when used in the data module.
            Validation/test dataset defaults to True.
            ''')),

        'use_grid_valid_regions': scfg.Value(True, help=ub.paragraph(
            '''
            If True, the initial grid will only place windows in valid regions.
            ''')),

        # Overwritten for non-train
        'neg_to_pos_ratio': scfg.Value(1.0, type=float, help=ub.paragraph(
            '''
            maximum ratio of samples with no annotations to samples with
            annots.
            Only applies to training dataset when used in the data module.
            Validation/test dataset defaults to zero.
            ''')),

        'use_grid_cache': scfg.Value(True, help=ub.paragraph(
            '''
            If true, will cache the spacetime grid to make multiple
            runs quicker.
            ''')),

        ###################
        # WEIGHTING OPTIONS
        ###################

        'ignore_dilate': scfg.Value(0, help='Dilation applied to ignore masks.'),
        'weight_dilate': scfg.Value(0, help='Dilation applied to weight masks.'),

        'normalize_perframe': scfg.Value(False, help='undocumented - ignored'),

        'normalize_peritem': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            if specified normalize these bands/channels on a per-batch-item
            level across time. if True normalize all bands.
            ''')),

        'min_spacetime_weight': scfg.Value(0.9, help=ub.paragraph(
            '''
            Minimum space-time dilation weight. Used in conjunction with
            ''')),

        'upweight_centers': scfg.Value(True, help=ub.paragraph(
            '''
            Applies a weighting such that the center of the frame incurs more
            loss.
            ''')),

        'upweight_time': scfg.Value(None, help=ub.paragraph(
            '''
            A number between 0.0 and 1.0 representing where to upweight time
            the most (1.0 is last frame 0.0 is the first frame).
            ''')),

        'dist_weights': scfg.Value(0, help=ub.paragraph(
            '''
            To use distance-transform based weights on annotations or
            not
            ''')),

        'balance_areas': scfg.Value(False, help='if True balance the weight of small and large polygons'),

        ##################################
        # DYNAMIC FILTER / MASKING OPTIONS
        ##################################

        'use_cloudmask': scfg.Value(None, help=ub.paragraph(
            '''
            Allow the dataloader to use the quality band to skip frames.
            DEPRECATED: set quality_threshold=0 to disable the cloudmask.
            Set to a positive value to use it, up to that threshold.
            ''')),

        'quality_threshold': scfg.Value(0.0, help=ub.paragraph(
            '''
            The minimum fraction of usable pixels required in a frame sample.
            If a frame has fewer than this fraction of usable pixels (i.e. not
            clouds or other quality flags), it is marked for resampling as a
            "bad" frame.
            ''')),

        'mask_low_quality': scfg.Value(False, help='if True, mask low quality pixels with nans'),

        'mask_samecolor_method': scfg.Value('histogram', help=ub.paragraph(
            '''
            If enabled, set as method to use for SAMECOLOR_QUALITY_HEURISTIC.
            Can be histogram or region.
            ''')),

        'force_bad_frames': scfg.Value(False, help='if True, force loading, even if data is nan / missing'),

        'observable_threshold': scfg.Value(0.0, help=ub.paragraph(
            '''
            The minimum fraction of non-nan pixels required in a frame sample.
            If a frame has fewer than this fraction of usable pixels (i.e. not
            clouds or other quality flags), it is marked for resampling as a
            "bad" frame.
            ''')),

        'resample_invalid_frames': scfg.Value(3, help=ub.paragraph(
            '''
            Number of attempts to resample any frame marked as invalid via
            quality or nodata checks.
            '''), alias=['resample_max_tries']),

        'downweight_nan_regions': scfg.Value(True, help='if True, unobservable (i.e. nan) pixels are downweighted'),

        ######################
        # AUGMENTATION OPTIONS
        ######################
        ### TODO: these should likely become a nested jsonargparse
        ### style config for a more general "augmentation scheme".

        'augment_space_shift_rate': scfg.Value(0.9, help=ub.paragraph(
            '''
            In fit mode, perform translation augmentations this fraction of the
            time.
            ''')),

        'augment_space_xflip': scfg.Value(True, help=ub.paragraph(
            '''In fit mode, if true, perform random x-flips''')),

        'augment_space_yflip': scfg.Value(True, help=ub.paragraph(
            '''In fit mode, if true, perform random y-flips''')),

        'augment_space_rot': scfg.Value(True, help=ub.paragraph(
            '''In fit mode, if true, perform random 90 degree rotations''')),

        'augment_time_resample_rate': scfg.Value(0.8, help=ub.paragraph(
            '''
            In fit mode, perform temporal jitter this fraction of the time.
            ''')),

        'temporal_dropout': scfg.Value(0.0, type=float, help=ub.paragraph(
            '''
            Drops frames in a fraction of training batches
            ''')),
    }

    def normalize(self):
        if isinstance(self['exclude_sensors'], str):
            self['exclude_sensors'] = [s.strip() for s in self['exclude_sensors'].split(',')]
        self['time_steps'] = int(self['time_steps'])

        # if self['chip_dims'] is None:
        #     d = int(self['chip_size'])
        #     self['chip_dims'] = [d, d]  # has to be a list not a tuple for yaml
        # self['chip_size'] = None

        if self['input_space_scale'] in {None, 'None', 'window'}:
            self['input_space_scale'] = self['window_space_scale']

        if self['output_space_scale'] is {None, 'None', 'input'}:
            self['output_space_scale'] = self['input_space_scale']

        if self['output_space_scale'] == 'window':
            self['output_space_scale'] = self['window_space_scale']

        if self['time_sampling'] == 'auto':
            self['time_sampling'] = 'hard+distribute'

        if self['use_cloudmask'] is not None:
            if not self['use_cloudmask']:
                self['quality_threshold'] = 0


class KWCocoVideoDataset(data.Dataset, SpacetimeAugmentMixin, SMARTDataMixin):
    """
    Accepted keyword arguments are specified in
    :class:`KWCocoVideoDatasetConfig`

    Example:
        >>> # Native Data Sampling
        >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> import watch
        >>> coco_dset = watch.demo.coerce_kwcoco('watch-multisensor-msi', geodata=True)
        >>> print({c.get('sensor_coarse') for c in coco_dset.images().coco_images})
        >>> print({c.channels.spec for c in coco_dset.images().coco_images})
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> self = KWCocoVideoDataset(sampler, time_dims=5, window_dims=(200, 300),
        >>>                           input_space_scale='native',
        >>>                           window_space_scale='0.05GSD',
        >>>                           output_space_scale='native',
        >>>                           channels='auto',
        >>> )
        >>> self.disable_augmenter = True
        >>> target = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][3]]
        >>> Box = util_kwimage.Box
        >>> #target['space_slice'] = Box.from_slice(target['space_slice']).translate((30, 0)).quantize().to_slice()
        >>> target['verbose_ndsample'] = True
        >>> item = self[target]
        >>> #print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
        >>> canvas = self.draw_item(item, overlay_on_image=0, rescale=0, max_channels=3)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()

    Example:
        >>> # Target GSD Data Sampling
        >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
        >>> import ndsampler
        >>> import kwcoco
        >>> import watch
        >>> coco_dset = watch.demo.coerce_kwcoco('watch', geodata=True)
        >>> print({c.get('sensor_coarse') for c in coco_dset.images().coco_images})
        >>> print({c.channels.spec for c in coco_dset.images().coco_images})
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> self = KWCocoVideoDataset(sampler, window_dims=(100, 100), time_dims=5,
        >>>                           input_space_scale='0.35GSD',
        >>>                           window_space_scale='0.7GSD',
        >>>                           output_space_scale='0.2GSD',
        >>>                           channels='auto',
        >>> )
        >>> self.disable_augmenter = True
        >>> index = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][3]]
        >>> Box = util_kwimage.Box
        >>> index['space_slice'] = Box.from_slice(index['space_slice']).translate((30, 0)).quantize().to_slice()
        >>> item = self[index]
        >>> #print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
        >>> canvas = self.draw_item(item, overlay_on_image=1, rescale=0, max_channels=3)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(canvas)
        >>> kwplot.show_if_requested()
    """

    def __init__(self, sampler, mode='fit', **kwargs):
        """
        Args:
            sampler (kwcoco.CocoDataset | ndsampler.CocoSampler): kwcoco dataset
            mode (str): fit or predict
            **kwargs: see :class:`KWCocoVideoDatasetConfig` for valid options
        """
        # note: sampler can be a ndsampler.CocoSampler or a kwcoco.CocoDataset
        sampler = ndsampler.CocoSampler.coerce(sampler)
        config = KWCocoVideoDatasetConfig(cmdline=0, data=kwargs)
        chip_dims = config['chip_dims']
        if isinstance(chip_dims, str):
            window_dims = chip_dims
        else:
            if not ub.iterable(chip_dims):
                chip_dims = (chip_dims, chip_dims)
            chip_h, chip_w = chip_dims
            window_dims = (chip_h, chip_w)
        time_dims = config['time_steps']
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
        self.class_foreground_classes = set(self.classes) - (
            self.background_classes |
            self.ignore_classes |
            self.undistinguished_classes)

        channels = config['channels']
        neg_to_pos_ratio = config['neg_to_pos_ratio']
        max_epoch_length = config['max_epoch_length']

        import os
        grid_workers = int(os.environ.get('WATCH_GRID_WORKERS', 0))
        common_grid_kw = dict(
            time_dims=time_dims,
            window_dims=window_dims,
            window_overlap=window_overlap,
            exclude_sensors=config['exclude_sensors'],
            include_sensors=config['include_sensors'],
            select_images=config['select_images'],
            select_videos=config['select_videos'],
            time_sampling=config['time_sampling'],
            time_span=config['time_span'],
            time_kernel=config['time_kernel'],
            window_space_scale=self.config['window_space_scale'],
            set_cover_algo=config['set_cover_algo'],
            workers=grid_workers,  # could configure this
            use_cache=self.config['use_grid_cache'],
            respect_valid_regions=self.config['use_grid_valid_regions'],
        )
        # print('common_grid_kw = {}'.format(ub.urepr(common_grid_kw, nl=1)))

        if mode == 'custom':
            new_sample_grid = None
            self.length = 1
        elif mode == 'test':
            # FIXME: something is wrong with the cache when using an sqlview.
            # In test mode we have to sample everything for BAS
            # (TODO: for activity clf, we should only focus on candidate regions)
            builder = spacetime_grid_builder.SpacetimeGridBuilder(
                dset=sampler.dset,
                keepbound=True,
                use_annot_info=False,
                **common_grid_kw
            )
            new_sample_grid = builder.build()
            self.length = len(new_sample_grid['targets'])
        else:
            negative_classes = (
                self.ignore_classes | self.background_classes | self.negative_classes)
            builder = spacetime_grid_builder.SpacetimeGridBuilder(
                sampler.dset,
                negative_classes=negative_classes,
                keepbound=False,
                use_annot_info=True,
                use_centered_positives=config['use_centered_positives'],
                use_grid_positives=config['use_grid_positives'],
                **common_grid_kw
            )
            new_sample_grid = builder.build()

            # Train time data balancing
            n_pos = len(new_sample_grid['positives_indexes'])
            n_neg = len(new_sample_grid['negatives_indexes'])

            target_vidids = [v['video_id'] for v in new_sample_grid['targets']]

            # Hack: determine if videos should be grouped together
            target_posbit = kwarray.boolmask(
                new_sample_grid['positives_indexes'],
                len(new_sample_grid['targets']))

            if 1:
                # TODO: each video should be able to have some sort of group
                # attribute we can use to balance over similar videos.
                print('Balancing over videos')

                # Do this for unique video ids otherwise SQLviews will take forever
                unique_vidids, _idx_to_unique_idx = np.unique(target_vidids, return_inverse=True)
                unique_vidnames = self.sampler.dset.videos(unique_vidids).lookup('name')
                vidnames = list(ub.take(unique_vidnames, _idx_to_unique_idx))

                # if 0:
                #     # DEBUG postgres
                #     # all_vidids = self.sampler.dset.videos()
                #     # all_vidids = set(all_vidids)
                #     # len(set(target_vidids) & all_vidids)
                #     # len(set(target_vidids) - all_vidids)
                #     # len(all_vidids - set(target_vidids))
                #     vid_table = self.sampler.dset.raw_table('videos')

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

        # x = list(ub.flatten(ub.flatten(all_chunks)))
        # import networkx as nx
        # def nested_tree(tree, nested, name='root'):
        #     tree.add_node(name)
        #     for idx, child in enumerate(nested):
        #         key = f'{name}.{idx}'
        #         if ub.iterable(child):
        #             child = nested_tree(tree, child, key)
        #         else:
        #             tree.add_node(key)
        #         tree.add_edge(name, key)
        # nested = self.nested_pool
        # tree = nx.DiGraph()
        # xdev.fix_embed_globals()
        # node = nested_tree(tree, nested)
        # ub.dict_hist(list(map(len, nested)))

        self.new_sample_grid = new_sample_grid

        bg_catname = ub.peek(sorted(self.background_classes))
        self.bg_idx = self.classes.node_to_idx[bg_catname]

        utils.category_tree_ensure_color(self.classes)

        self.special_inputs = {}

        if self.config['normalize_peritem']:
            self.normalize_peritem = kwcoco.ChannelSpec.coerce(self.config['normalize_peritem']).fuse()
        else:
            self.normalize_peritem = None

        if channels is None or channels == 'auto':
            # Find reasonable channel defaults if channels is not specified.
            # Use dataset stats to determine something sensible.
            sensorchan_hist = kwcoco_extensions.coco_channel_stats(sampler.dset)['sensorchan_hist']
            parts = []
            for sensor, chan_hist in sensorchan_hist.items():
                for c in chan_hist.keys():
                    chancode = kwcoco.ChannelSpec.coerce(c).fuse().spec
                    parts.append(f'{sensor}:{chancode}')
            sensorchans = ','.join(sorted(parts))
            sensorchans = kwcoco.SensorChanSpec.coerce(sensorchans)
            if len(sensorchan_hist) > 0 and channels is None:
                # Only warn if not explicitly in auto mode
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
                sensorchan_stats = kwcoco_extensions.coco_channel_stats(sampler.dset)
                sensorchan_hist = sensorchan_stats['sensorchan_hist']

            expanded_input_sensorchan_streams = []
            for fused_sensorchan in self.sensorchan.streams():
                sensor = fused_sensorchan.sensor
                chans = fused_sensorchan.chans
                if sensor.spec == '*':
                    for cand_sensor, cand_chans in sensorchan_hist.items():
                        valid_chan_cands = []
                        for cand_chan_group in cand_chans:
                            cand_chan_group = kwcoco.ChannelSpec.coerce(cand_chan_group).fuse()
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

            self.sensorchan = kwcoco.SensorChanSpec.coerce(','.join(
                list(ub.unique(expanded_input_sensorchan_streams)))).normalize()

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
        self.requested_tasks = {
            'change': True,
            'class': True,
            'saliency': True,
            'boxes': True,
        }

        # Hacks: combinable channels can be visualized as RGB images.
        # The only reason this is a hack is because of the hardcoded names
        # otherwise it is a cool feature.
        self.default_combinable_channels = [
            ub.oset(['red', 'green', 'blue']),
            ub.oset(['Dred', 'Dgreen', 'Dblue']),
            ub.oset(['r', 'g', 'b']),
        ] + heuristics.HUERISTIC_COMBINABLE_CHANNELS

    @property
    def coco_dset(self):
        return self.sampler.dset

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
        self.requested_tasks.update(requested_tasks)

    @profile
    def _sample_one_frame(self, gid, sampler, coco_dset, target_, with_annots,
                          gid_to_isbad, gid_to_sample):
        # helper that was previously a nested function moved out for profiling
        coco_img = coco_dset.coco_image(gid)
        sensor_coarse = coco_img.img.get('sensor_coarse', '*')
        matching_sensorchan = self.sample_sensorchan.matching_sensor(sensor_coarse)
        sensor_channels = matching_sensorchan.chans

        # TODO: disable the samecolor quality heuristic by defaults.  We should
        # use a preprocessing step to nan-out these regions more robustly.

        # SAMECOLOR_QUALITY_HEURISTIC = target_.get('SAMECOLOR_QUALITY_HEURISTIC', 'region')
        SAMECOLOR_VALUES = {0}
        SAMECOLOR_QUALITY_HEURISTIC = target_.get('SAMECOLOR_QUALITY_HEURISTIC', self.config['mask_samecolor_method'])
        # SAMECOLOR_QUALITY_HEURISTIC = target_.get('SAMECOLOR_QUALITY_HEURISTIC', None)
        use_samecolor_region_method = SAMECOLOR_QUALITY_HEURISTIC == 'region'
        # There are only some values that we care about for the samecolor
        # metric. It turns out in our data only zeros are confused for NODATA.
        # so we can hack this metric in. Setting it to none would generalize it
        # to allow any value in a large homogenous region to be considered as
        # nodata.

        force_bad_frames = target_.get('force_bad_frames', self.config['force_bad_frames'])
        stop_on_bad_image = not force_bad_frames
        quality_threshold = target_.get('quality_threshold', self.config['quality_threshold'])
        observable_threshold = target_.get('observable_threshold', self.config['observable_threshold'])
        mask_low_quality = target_.get('mask_low_quality', self.config['mask_low_quality'])

        # These bands propogate their nans to other bands / streams
        PROPOGATE_NAN_BANDS = target_.get('PROPOGATE_NAN_BANDS', {'red'})

        # We are only going to compute the same color quality heuristic on a
        # single band.
        valid_bands_for_samecolor_quality_heuristic = {
            'red', 'green', 'blue', 'nir', 'swir16', 'swir22'}

        # sensor_channels = (self.sample_channels & coco_img.channels).normalize()
        tr_frame = target_.copy()
        tr_frame['gids'] = [gid]

        # TODO: separate ndsampler annotation loading function
        first_with_annot = with_annots

        # Flag will be set to true if any heuristic on any channel stream
        # forces us to mark this image as bad.
        force_bad = False

        # Track pixel positions we will force to nan
        unobservable_mask = data_utils.MultiscaleMask()

        # Handle a special quality band channel.
        if quality_threshold > 0 or mask_low_quality:
            # Skip if quality mask indicates more than 50% clouds.
            is_low_quality = self._interpret_quality_mask(
                sampler, coco_img, tr_frame)
            if is_low_quality is not None:
                is_low_quality = is_low_quality[0]  # just first frame
                cloud_threshold = (1 - quality_threshold)
                # TODO: account for nodata values here.
                # such that quality threshold is over the valid data
                # observations.
                cloud_frac = is_low_quality.mean()
                if cloud_frac > cloud_threshold:
                    force_bad = 'too cloudy'
                if mask_low_quality:
                    unobservable_mask.update(is_low_quality)
        else:
            is_low_quality = None

        if sensor_channels.numel() == 0:
            force_bad = 'Missing requested channels'

        # Sample information from each stream (each stream is a separate mode)
        sample_streams = {}
        for stream in sensor_channels.streams():
            if stop_on_bad_image and force_bad:
                break
            tr_frame['channels'] = stream
            tr_frame['padkw' ] = {'constant_values': np.nan}
            tr_frame['nodata' ] = 'float'
            # FIXME: each kwcoco asset should be able to control its own
            # interpolation as a function of its role.
            sample = sampler.load_sample(
                tr_frame, with_annots=first_with_annot,
                dtype=np.float32,
            )

            stream_oset = ub.oset(stream)
            if SAMECOLOR_QUALITY_HEURISTIC:
                # Update our observable mask based on bands heuristically
                # marked as valid or observable (i.e. rgb bands)
                relevant_bands = stream_oset & valid_bands_for_samecolor_quality_heuristic
                if relevant_bands:
                    samecolor_mask = data_utils.samecolor_nodata_mask(
                        stream, sample['im'][0], relevant_bands,
                        use_regions=use_samecolor_region_method,
                        samecolor_values=SAMECOLOR_VALUES)
                    unobservable_mask.update(samecolor_mask)

            relevant_bands = stream_oset & PROPOGATE_NAN_BANDS
            for band in relevant_bands:
                # Marke the nans in these bands as unobservable.
                bx = stream_oset.index('red')
                band = sample['im'][0][:, :, bx]
                nodata_mask = np.isnan(band)
                unobservable_mask.update(nodata_mask)

            if unobservable_mask.masked_fraction == 1.0:
                force_bad = 'unobservable sample'
                if stop_on_bad_image:
                    break

            if observable_threshold:
                invalid_frac = unobservable_mask.masked_fraction
                observable_frac = 1 - invalid_frac
                if observable_frac < observable_threshold:
                    force_bad = 'failed observable threshold'
                    if stop_on_bad_image:
                        break

            sample_streams[stream.spec] = sample
            if 'annots' in sample:
                # dont ask for annotations multiple times
                first_with_annot = False

        # After all channels are sampled, apply final invalid mask.
        for stream, sample in sample_streams.items():
            unobservable_mask.apply(sample['im'][0], np.nan)
            invalid_mask = np.isnan(sample['im'])
            any_invalid = np.any(invalid_mask)
            if any_invalid:
                sample['invalid_mask'] = invalid_mask
            else:
                sample['invalid_mask'] = None

        if not force_bad:
            if len(sample_streams) == 0:
                force_bad = 'no-streams'

        gid_to_isbad[gid] = force_bad
        gid_to_sample[gid] = sample_streams

        HACK_FIX_NATIVE_ANNOT_SIZE = getenv("HACK_FIX_NATIVE_ANNOT_SIZE", "True").lower() in ('true', '1', 't')
        if HACK_FIX_NATIVE_ANNOT_SIZE:
            # When sampling in native resolution, the annotations will be
            # sampled at that resolution. However, when there are multiple
            # modes for a input frame, it becomes unclear which native scale is
            # the right one to sample the annotations in. Thus we find the
            # maximum dimension over all the modes, and then upscale the
            # annotations to match that.
            annot_mode_dims = None
            all_mode_dims = []
            frame_dets = None
            for sample in sample_streams.values():
                mode_dims = sample['im'].shape[1:3]
                if 'annots' in sample:
                    frame_dets = sample['annots']['frame_dets'][0]
                    annot_mode_dims = mode_dims
                all_mode_dims.append(mode_dims)
            if all_mode_dims:
                max_mode_dims = np.array(max(all_mode_dims, key=np.prod))
                if frame_dets is not None:
                    fixup_scale = (max_mode_dims / annot_mode_dims)[::-1]
                    frame_dets.scale(fixup_scale, inplace=True)
                    # Save the input dimensions we scaled to.
                    # We will need to transform this to the output dims later.
                    frame_dets.meta['input_dims'] = max_mode_dims

    def __getitem__(self, index):
        """
        Example:
            >>> # Native sampling project data doctest
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import watch
            >>> import kwcoco
            >>> coco_dset = watch.coerce_kwcoco('watch-msi-geodata-dates')
            >>> self = KWCocoVideoDataset(
            >>>     coco_dset,
            >>>     time_dims=5, window_dims=(320, 320),
            >>>     window_overlap=0,
            >>>     input_space_scale='native',
            >>>     window_space_scale='0.3GSD',
            >>>     output_space_scale='0.6GSD',
            >>>     dist_weights=1,
            >>>     quality_threshold=0,
            >>>     neg_to_pos_ratio=0, time_sampling='soft2',
            >>> )
            >>> self.requested_tasks['change'] = False
            >>> # Find a sample with S2 and L8 images in it.
            >>> for target in self.new_sample_grid['targets']:
            ...     sensors = coco_dset.images(target['gids']).lookup('sensor_coarse')
            ...     shist = ub.dict_hist(sensors)
            ...     if len(shist) > 1 and all(v > 1 for v in shist.values()):
            ...         break
            >>> target['allow_augment'] = False
            >>> index = target
            >>> item = self[index]
            >>> print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
            >>> # xdoctest: +REQUIRES(--show)
            >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0, rescale=0)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DATA_DPATH)
            >>> # Native sampling project data doctest
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import watch
            >>> import kwcoco
            >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>> coco_fpath = dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> self = KWCocoVideoDataset(
            >>>     coco_dset,
            >>>     time_dims=5, window_dims=(320, 320),
            >>>     window_overlap=0,
            >>>     channels="(S2,L8):blue|green|red|nir",
            >>>     input_space_scale='native',
            >>>     window_space_scale='10GSD',
            >>>     output_space_scale='native',
            >>>     #output_space_scale='10GSD',
            >>>     dist_weights=1,
            >>>     quality_threshold=0,
            >>>     neg_to_pos_ratio=0, time_sampling='soft2',
            >>> )
            >>> self.requested_tasks['change'] = False
            >>> # Find a sample with S2 and L8 images in it.
            >>> for target in self.new_sample_grid['targets']:
            ...     sensors = coco_dset.images(target['gids']).lookup('sensor_coarse')
            ...     shist = ub.dict_hist(sensors)
            ...     if len(shist) > 1 and all(v > 1 for v in shist.values()):
            ...         break
            >>> target['allow_augment'] = False
            >>> index = target
            >>> item = self[index]
            >>> print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
            >>> # xdoctest: +REQUIRES(--show)
            >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0, rescale=0)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Example:
            >>> # xdoctest: +REQUIRES(env:DVC_DATA_DPATH)
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import watch
            >>> import kwcoco
            >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
            >>> coco_fpath = dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'
            >>> coco_dset = kwcoco.CocoDataset(coco_fpath)
            >>> self = KWCocoVideoDataset(
            >>>     coco_dset,
            >>>     time_dims=5, window_dims=(320, 320),
            >>>     window_overlap=0,
            >>>     channels="(S2,L8):blue|green|red|nir",
            >>>     input_space_scale='10GSD',
            >>>     window_space_scale='10GSD',
            >>>     output_space_scale='10GSD',
            >>>     dist_weights=1,
            >>>     quality_threshold=0,
            >>>     neg_to_pos_ratio=0, time_sampling='soft2',
            >>> )
            >>> self.requested_tasks['change'] = False
            >>> index = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][0]]
            >>> index['allow_augment'] = False
            >>> item = self[index]
            >>> target = item['target']
            >>> print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
            >>> # xdoctest: +REQUIRES(--show)
            >>> canvas = self.draw_item(item, max_channels=10, overlay_on_image=0, rescale=0)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()
        """
        try:
            return self.getitem(index)
        except FailedSample:
            return None

    @profile
    def getitem(self, index):
        """
        This is just the same thing as `__getitem__` but it raises an error
        when it fails, which is handled by `__getitem__`.
        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import kwcoco
            >>> import watch
            >>> coco_dset = watch.coerce_kwcoco('watch-msi-dates-geodata-heatmap', num_frames=5, image_size=(256, 256), num_videos=1)
            >>> # Remove two annotations to test new time weights
            >>> aids = coco_dset.images().take([0]).annots[0].lookup('id')
            >>> coco_dset.remove_annotations(aids)
            >>> #
            >>> # Each sensor uses all of its own channels
            >>> channels = 'auto'
            >>> self = KWCocoVideoDataset(coco_dset, time_dims=5,
            >>>                           window_resolution='0.09GSD',
            >>>                           input_resolution='0.09GSD',
            >>>                           window_dims=(256, 256),
            >>>                           channels=channels,
            >>>                           balance_areas=True,
            >>>                           weight_dilate=3,
            >>>                           normalize_perframe=False)
            >>> self.disable_augmenter = True
            >>> # Pretend that some external object has given us information about desired class weights
            >>> from watch.tasks.fusion.methods import watch_module_mixins
            >>> dataset_stats = self.cached_dataset_stats()
            >>> from watch.tasks.fusion.methods.network_modules import _class_weights_from_freq
            >>> class_keys = dataset_stats['class_freq']
            >>> total_freq = np.array(list(dataset_stats['class_freq'].values()))
            >>> class_importance_weights = _class_weights_from_freq(total_freq)
            >>> catname_to_weight = ub.dzip(class_keys, class_importance_weights)
            >>> catname_to_weight['star'] = 2.0
            >>> self.catname_to_weight = catname_to_weight
            >>> #
            >>> index = 0
            >>> index = target = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][4]]
            >>> item = self[index]
            >>> # xdoctest: +REQUIRES(--show)
            >>> canvas = self.draw_item(item)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(canvas)
            >>> kwplot.show_if_requested()

        Ignore:
            import xdev
            _ = xdev.profile_now(self.getitem)(target)
        """
        target = self._coerce_target(index)

        sampler = self.sampler
        coco_dset = self.sampler.dset

        ###
        # Handle details about the sampling target
        ###
        target_ = target.copy()

        target_['as_xarray'] = False
        target_['legacy_annots'] = False
        target_['legacy_targets'] = False

        if 'video_id' not in target_:
            _gid = ub.peek(target_['gids'])
            target_['video_id'] = sampler.dset.imgs[_gid]['video_id']

        vidid = target_['video_id']
        video = coco_dset.index.videos[vidid]

        resolution_info = self._resolve_resolution(target_, video)

        allow_augment = target_.get('allow_augment', True)
        if allow_augment:
            target_ = self._augment_spacetime_target(target_)

        vidspace_box = resolution_info['vidspace_box']
        final_gids, gid_to_sample = self._sample_from_target(target_, vidspace_box)

        num_frames = len(final_gids)
        if num_frames == 0:
            raise Exception('0 frames')

        ###
        # Process sampled data
        if not self.inference_only:
            truth_info = self._prepare_truth_info(final_gids, gid_to_sample,
                                                  num_frames, target, target_)
        else:
            truth_info = None

        frame_items = self._build_frame_items(final_gids, gid_to_sample,
                                              truth_info, resolution_info)

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

        if self.normalize_peritem is not None:
            # Gather items that need normalization
            needs_norm = ub.ddict(list)
            for frame_item in frame_items:
                sensor = frame_item['sensor']
                frame_modes = frame_item['modes']
                for mode_key in list(frame_modes.keys()):
                    mode_chan = kwcoco.FusedChannelSpec.coerce(mode_key)
                    common_key = mode_chan.intersection(self.normalize_peritem)
                    if common_key:
                        parent_data = frame_modes[mode_key]
                        for chan_name, chan_sl in mode_chan.component_indices(axis=0).items():
                            if chan_name in common_key:
                                chan_data = parent_data[chan_sl]
                                valid_mask = np.isfinite(chan_data)
                                needs_norm[(sensor, chan_name)].append((chan_data, valid_mask, parent_data, chan_sl))

            peritem_normalizer_params = {
                'high': 0.95,
                # 'mid': 0.5,
                'mid': 0.5,
                'low': 0.00,
                # 'mode': 'sigmoid',
                'mode': 'linear',
            }
            # print('DO NORM')
            for key, norm_items in needs_norm.items():
                raw_datas = np.concatenate([t[0].ravel() for t in norm_items], axis=0)
                valid_mask = np.concatenate([t[1].ravel() for t in norm_items], axis=0)
                valid_raw_datas = raw_datas[valid_mask]
                # Compute normalizers over the entire temporal range per-sensor
                normalizer = kwimage.find_robust_normalizers(valid_raw_datas,
                                                             params=peritem_normalizer_params)
                # Postprocess / regularize the normalizer
                prior_min = min(0, normalizer['min_val'])
                alpha = 0.5
                normalizer['min_val'] * alpha + (1 - alpha) * prior_min
                # normalizer['min_val'] = 0  # keep min
                # print(f'normalizer={normalizer}')
                # Apply the normalize to the original data
                for chan_data, valid_mask, parent_data, chan_sl in norm_items:
                    valid_data = chan_data[valid_mask]
                    # Apply normalizer (todo: use kwimage variant)
                    imdata_normalized = apply_robust_normalizer(
                        normalizer, chan_data, valid_data, valid_mask,
                        dtype=np.float32, copy=True)
                    # Overwrite original data with new normalized variants
                    parent_data[chan_sl] = imdata_normalized

        # Add in change truth
        if not self.inference_only:
            # Build multi-frame truth
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

        pixelwise_truth_keys = [
            'change', 'class_idxs',
            'saliency', 'class_weights',
            'saliency_weights', 'change_weights'
        ]
        annotwise_truth_keys = [
            'box_ltrb', 'box_tids', 'box_cidx', 'box_weight',
        ]
        coord_truth_keys = [
            'box_ltrb',
        ]
        truth_keys = pixelwise_truth_keys + annotwise_truth_keys

        # If we are augmenting
        fliprot_params = target_.get('fliprot_params', None)
        if fliprot_params is not None:
            for frame_item in frame_items:
                frame_modes = frame_item['modes']
                for mode_key in list(frame_modes.keys()):
                    # Augment the underlying data
                    mode_data = frame_modes[mode_key]
                    frame_modes[mode_key] = data_utils.fliprot(mode_data, **fliprot_params, axes=[1, 2])
                for key in pixelwise_truth_keys:
                    # Augment the truth rasters in the same way
                    data = frame_item.get(key, None)
                    if data is not None:
                        frame_item[key] = data_utils.fliprot(data, **fliprot_params, axes=[-2, -1])
                for key in coord_truth_keys:
                    # Augment the truth coordinates in the same way
                    data = frame_item.get(key, None)
                    if data is not None:
                        output_dims = frame_item['output_dims']
                        frame_item[key] = data_utils.fliprot_annot(
                            kwimage.Boxes(data, 'ltrb'), **fliprot_params, axes=[-2, -1], canvas_dsize=output_dims).data

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
            'gids', 'space_slice', 'video_id', 'fliprot_params',
            'main_idx', 'scale', 'main_skip_reason'
        })
        # dont allow augmenting on resample by default
        tr_subset['allow_augment'] = False

        resolved_input_scale = resolution_info['resolved_input_scale']
        resolved_output_scale = resolution_info['resolved_output_scale']
        item = {
            'index': index,
            'frames': frame_items,
            'positional_tensors': positional_tensors,
            'video_id': vidid,
            'video_name': video['name'],
            'input_gsd': resolved_input_scale.get('gsd', None),
            'output_gsd': resolved_output_scale.get('gsd', None),
            'target': tr_subset
        }
        return item

    def _coerce_target(self, index):
        """
        Returns a target dictionary given an index or an explicit target dictionary
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
                    raise FailedSample(f'Failed to sample grid location: {ex=}')
                else:
                    target = self.new_sample_grid['targets'][tr_idx]

        if target is None:
            raise FailedSample('no target')
        return target

    @profile
    def _sample_from_target(self, target_, vidspace_box):
        """
        Given a space-time target, samples frame rasters and annotation vectors.

        This includes
            * rejection sampling
            * quality masking
            * dynamic resolution
        """
        ###
        # Execute data sampling
        ###
        sampler = self.sampler
        coco_dset = self.sampler.dset

        vidid = target_['video_id']
        video = coco_dset.index.videos[vidid]

        with_annots = [] if self.inference_only else ['boxes', 'segmentation']

        # New true-multimodal data items
        gid_to_sample: Dict[int, Dict] = {}
        gid_to_isbad: Dict[int, bool] = {}

        for gid in target_['gids']:
            ...
            self._sample_one_frame(gid, sampler, coco_dset, target_, with_annots,
                                   gid_to_isbad, gid_to_sample)

        time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
        video_gids = time_sampler.video_gids

        # If we skipped the main gid, record why
        main_gid = target_.get('main_gid', None)
        if main_gid is not None and gid_to_isbad[main_gid]:
            main_skip_reason = gid_to_isbad[main_gid]
        else:
            main_skip_reason = None

        resample_invalid = target_.get('resample_invalid_frames', self.config['resample_invalid_frames'])
        num_images_wanted = len(target_['gids'])
        if resample_invalid:
            if resample_invalid is True:
                max_tries = 3
            else:
                max_tries = int(resample_invalid)
            # print(f'max_tries={max_tries}')
            vidname = video['name']
            self._resample_bad_images(
                video_gids, gid_to_isbad, sampler, coco_dset, target_,
                num_images_wanted,
                with_annots, gid_to_sample, vidspace_box, vidname, max_tries)

        good_gids = [gid for gid, flag in gid_to_isbad.items() if not flag]
        if len(good_gids) == 0:
            raise FailedSample('Cannot force a good sample')

        final_gids = ub.oset(video_gids) & good_gids
        force_bad_frames = target_.get('force_bad_frames', 0)
        if force_bad_frames:
            final_gids = ub.oset(video_gids) & set(gid_to_isbad.keys())
            print('gid_to_isbad = {}'.format(ub.repr2(gid_to_isbad, nl=1)))

        # coco_dset.images(final_gids).lookup('date_captured')
        target_['gids'] = final_gids

        if main_skip_reason:
            target_['main_skip_reason'] = main_skip_reason

        return final_gids, gid_to_sample

    def _resolve_resolution(self, target_, video):
        # Compute scale if we are doing that
        # This should live somewhere else, but lets just get it hooked up
        vidspace_gsd = video.get('target_gsd', None)

        # The target is allowed to overload the scales
        if target_.get('input_space_scale', None) is None:
            target_['input_space_scale'] = self.config['input_space_scale']
        if target_.get('output_space_scale', None) is None:
            target_['output_space_scale'] = self.config['output_space_scale']
        # Resolve spatial scale code
        resolved_input_scale = data_utils.resolve_scale_request(
            request=target_['input_space_scale'], data_gsd=vidspace_gsd)

        resolved_output_scale = data_utils.resolve_scale_request(
            request=target_['output_space_scale'], data_gsd=vidspace_gsd)

        common_input_scale = resolved_input_scale['scale']
        common_output_scale = resolved_output_scale['scale']
        target_['scale'] = common_input_scale

        # Put the target slice in video space.
        vidspace_box = util_kwimage.Box.from_slice(target_['space_slice'])
        vidspace_dsize = np.array([vidspace_box.width, vidspace_box.height])

        # Size of the video the target is embedded in.
        video_dsize = np.array([video['width'], video['height']])

        if isinstance(common_input_scale, str) and common_input_scale == 'native':
            target_.pop('scale')
            # native scales will only work in late-fused modes
            target_['use_native_scale'] = True
            target_['realign_native'] = 'largest'
        else:
            if isinstance(common_output_scale, str) and common_output_scale == 'native':
                raise Exception(
                    'output scale can only be native when input scale is native')

        if isinstance(common_output_scale, str) and common_output_scale == 'native':
            common_outspace_box = None
        else:
            # Compute where this output chip should live in its output space canvas.
            common_output_scale = resolved_output_scale['scale']
            common_outspace_box = vidspace_box.scale(common_output_scale)
            common_outspace_box = common_outspace_box.quantize()

        # fixme: giant tuple returns are error prone
        resolution_info = {
            'common_outspace_box': common_outspace_box,
            'vidspace_box': vidspace_box,
            'video_dsize': video_dsize,
            'vidspace_dsize': vidspace_dsize,
            'resolved_input_scale': resolved_input_scale,
            'resolved_output_scale': resolved_output_scale,
            'common_input_scale': common_input_scale,
            'common_output_scale': common_output_scale,

        }
        return resolution_info

    @profile
    def _resample_bad_images(self, video_gids, gid_to_isbad, sampler,
                             coco_dset, target_, num_images_wanted, with_annots,
                             gid_to_sample, vidspace_box, vidname, max_tries):
        """
        If the initial sample has marked any of the images as "bad", then we
        attempt to find replacements by reusing the temporal sampler, but with
        extra arguments to exclude the bad frames.
        """
        # If any image is junk allow for a resample
        if any(gid_to_isbad.values()):
            vidid = target_['video_id']
            time_sampler = self.new_sample_grid['vidid_to_time_sampler'][vidid]
            for iter_idx in range(max_tries):
                # print(f'resample try iter_idx={iter_idx}')
                good_gids = np.array([gid for gid, flag in gid_to_isbad.items() if not flag])
                if len(good_gids) == num_images_wanted:
                    break
                bad_gids = np.array([gid for gid, flag in gid_to_isbad.items() if flag])
                include_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, good_gids))[0]
                exclude_idxs = np.where(kwarray.isect_flags(time_sampler.video_gids, bad_gids))[0]
                try:
                    chosen = time_sampler.sample(include=include_idxs,
                                                 exclude=exclude_idxs,
                                                 error_level=0,
                                                 return_info=False)
                except Exception:
                    break
                new_idxs = np.setdiff1d(chosen, include_idxs)
                new_gids = video_gids[new_idxs]
                # print('new_gids = {!r}'.format(new_gids))
                if not len(new_gids):
                    # import warnings
                    # warnings.warn('exhausted resample possibilities')
                    _bad_reasons = repr({k: v for k, v in gid_to_isbad.items() if v})
                    vidspace_box_str = str(vidspace_box)
                    if 0:
                        print(f'exhausted resample possibilities: {vidname} {vidspace_box_str} {_bad_reasons}')
                    # Exhausted all possibilities
                    break
                for gid in new_gids:
                    self._sample_one_frame(gid, sampler, coco_dset, target_,
                                           with_annots, gid_to_isbad,
                                           gid_to_sample)

    def _prepare_truth_info(self, final_gids, gid_to_sample, num_frames, target, target_):
        """
        Helper used to construct information about the truth before we start
        constructing the frames.
        """
        # build up info about the tracks
        dset = self.sampler.dset
        gid_to_dets: Dict[int, kwimage.Detections] = {}
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
            # The returne detections will live in the "input/data" space
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

        if self.upweight_centers or self.upweight_time is not None:

            if self.upweight_time is None:
                upweight_time = 0.5
            else:
                upweight_time = self.upweight_time

            # Learn more from the center of the space-time patch
            time_weights = biased_1d_weights(upweight_time, num_frames)

            time_weights = time_weights / time_weights.max()
            time_weights = time_weights.clip(0, 1)
            time_weights = np.maximum(time_weights, self.min_spacetime_weight)

        truth_info = {
            'task_tid_to_cnames': task_tid_to_cnames,
            'gid_to_dets': gid_to_dets,
            'time_weights': time_weights,
        }
        return truth_info

    @profile
    def _build_frame_items(self, final_gids, gid_to_sample,
                           truth_info, resolution_info):

        common_outspace_box = resolution_info['common_outspace_box']
        vidspace_dsize = resolution_info['vidspace_dsize']
        vidspace_box = resolution_info['vidspace_box']
        video_dsize = resolution_info['video_dsize']

        coco_dset = self.sampler.dset
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
            if common_outspace_box is None:
                # In the native case, we use the size of the largest mode for
                # each frame.
                max_mode_dsize = np.array(max(mode_to_dsize.values(), key=np.prod))
                # Compute the scale factor for this frame wrt video space
                scale_inspace_from_vid = max_mode_dsize / vidspace_dsize
                frame_outspace_box = vidspace_box.scale(scale_inspace_from_vid).quantize()
            else:
                frame_outspace_box = common_outspace_box

            output_dsize = (frame_outspace_box.width, frame_outspace_box.height)
            scale_outspace_from_vid = output_dsize / vidspace_dsize
            output_dims = output_dsize[::-1]  # the size we want to predict

            dt_captured = img.get('date_captured', None)
            if dt_captured:
                dt_captured = util_time.coerce_datetime(dt_captured)
                timestamp = dt_captured.timestamp()
            else:
                timestamp = np.nan

            sensor = img.get('sensor_coarse', '*')

            # The size of the larger image this output is expected to be
            # embedded in.
            outimg_dsize = video_dsize * scale_outspace_from_vid
            outimg_box = util_kwimage.Box.from_dsize(outimg_dsize).quantize()

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
                # Could group these into head and input/head specific dictionaries?
                # info for how to construct the output.
                'change_output_dims': None if time_idx == 0 else output_dims,
                'class_output_dims': output_dims,
                'saliency_output_dims': output_dims,
                #
                'output_dims': output_dims,
                'output_space_slice': frame_outspace_box.to_slice(),
                'output_image_dsize': outimg_box.dsize,
                'scale_outspace_from_vid': scale_outspace_from_vid,
                'ann_aids': None,
            }

            if not self.inference_only:
                # Build single-frame truth
                self._populate_frame_labels(
                    frame_item, gid, output_dsize, time_idx,
                    mode_to_invalid_mask, resolution_info, truth_info)

            frame_items.append(frame_item)
        return frame_items

    def _populate_frame_labels(self, frame_item, gid, output_dsize, time_idx,
                               mode_to_invalid_mask, resolution_info, truth_info):
        """
        Build single-frame truth-labels.

        Helper function to populate truth labels for a frame in a video
        sequence. This was factored out of the original getitem, and
        could use work to reduce the number of input params.
        """

        common_input_scale = resolution_info['common_input_scale']
        common_output_scale = resolution_info['common_output_scale']

        # The frame detections will be in a scaled videos space the
        # constant scale case.
        # TODO: will need special handling for "native" resolutions on
        # a per-mode / frame basis, we will need the concept of an
        # annotation window (where ndsampler lets us assume the corners
        # of each window are in correspondence)

        task_tid_to_cnames = truth_info['task_tid_to_cnames']
        time_weights = truth_info['time_weights']
        gid_to_dets = truth_info['gid_to_dets']

        input_is_native = (isinstance(common_input_scale, str) and common_input_scale == 'native')
        output_is_native = (isinstance(common_output_scale, str) and common_output_scale == 'native')

        frame_dets = gid_to_dets[gid]
        if frame_dets is None:
            raise AssertionError('frame_dets = {!r}'.format(frame_dets))

        # As of ndsampler >= 0.7.1 the dets are sampled in the input space
        if input_is_native:
            if output_is_native:
                # Both scales are native, use detections as-is.
                dets = frame_dets.copy()
            else:
                # Input scale is native, but output scale is given,
                # Need to resize. We enriched the dets with metadata
                # to do this earlier.
                annot_input_dsize = frame_dets.meta['input_dims'][::-1]
                dets_scale = output_dsize / annot_input_dsize
                dets = frame_dets.scale(dets_scale)
        else:
            if output_is_native:
                raise NotImplementedError(
                    'input scale is constant and output scale is native. '
                    'no logic for this case yet.'
                )
            else:
                # Simple case where input/output scales are constant
                dets_scale = common_output_scale / common_input_scale
                dets = frame_dets.scale(dets_scale)

        # Create truth masks
        bg_idx = self.bg_idx
        frame_target_shape = output_dsize[::-1]
        space_shape = frame_target_shape
        frame_cidxs = np.full(space_shape, dtype=np.int32,
                              fill_value=bg_idx)

        # A "Salient" class is anything that is a foreground class
        task_target_ohe = {}
        task_target_ignore = {}
        task_target_weight = {}

        # Rasterize frame targets into semantic segmentation masks
        ann_polys = dets.data['segmentations'].to_polygon_list()
        ann_aids = dets.data['aids']
        ann_cids = dets.data['cids']
        ann_tids = dets.data['tids']
        ann_ltrb = dets.data['boxes'].to_ltrb().data

        # frame_poly_saliency_weights = np.ones(space_shape, dtype=np.float32)
        # frame_poly_class_weights = np.ones(space_shape, dtype=np.float32)

        wants_saliency = self.requested_tasks['saliency']
        wants_class = self.requested_tasks['class']
        wants_change = self.requested_tasks['change']
        wants_boxes = self.requested_tasks['boxes']

        wants_class_sseg = wants_class or wants_change
        wants_saliency_sseg = wants_saliency

        frame_box = kwimage.Box.from_dsize(space_shape[::-1])
        frame_box = frame_box.to_shapely()

        # catname_to_weight = getattr(self, 'catname_to_weight', None)

        # Note: it is important to respect class indexes, ids, and
        # name mappings
        if wants_boxes:
            box_labels = {
                'box_ltrb': [],
                'box_tids': [],
                'box_cidxs': [],
                'box_class_weights': [],
                'box_saliency_weights': [],
            }
            # Do we want saliency boxes and class boxes?
            for ltrb, cid, tid in zip(ann_ltrb, ann_cids, ann_tids):
                new_salient_catname = task_tid_to_cnames['saliency'][tid][time_idx]
                new_class_catname = task_tid_to_cnames['class'][tid][time_idx]
                new_class_cidx = self.classes.node_to_idx[new_class_catname]
                box_labels['box_ltrb'].append(ltrb)
                box_labels['box_tids'].append(-1 if tid is None else tid)
                box_labels['box_cidxs'].append(new_class_cidx)
                box_labels['box_saliency_weights'].append(
                    float(new_salient_catname in self.salient_classes))
                box_labels['box_class_weights'].append(
                    float(new_class_catname in self.class_foreground_classes))
            box_labels['box_ltrb'] = np.array(box_labels['box_ltrb']).astype(np.float32)
            box_labels['box_tids'] = np.array(box_labels['box_tids']).astype(np.int64)
            box_labels['box_cidxs'] = np.array(box_labels['box_cidxs']).astype(np.int64)
            box_labels['box_class_weights'] = np.array(box_labels['box_class_weights']).astype(np.float32)
            box_labels['box_saliency_weights'] = np.array(box_labels['box_saliency_weights']).astype(np.float32)
            frame_item.update(box_labels)

        if wants_saliency:
            ### Build single frame SALIENCY target labels and weights
            task_target_ohe['saliency'] = np.zeros(space_shape, dtype=np.uint8)
            task_target_ignore['saliency'] = np.zeros(space_shape, dtype=np.uint8)
            task_target_weight['saliency'] = np.empty(space_shape, dtype=np.float32)

            # Group polygons into foreground / background for the saliency task
            saliency_sseg_groups = {
                'foreground': [],
                'background': [],
                'ignore': [],
            }
            for poly, tid in zip(ann_polys, ann_tids):
                new_salient_catname = task_tid_to_cnames['saliency'][tid][time_idx]
                if new_salient_catname in self.salient_classes:
                    saliency_sseg_groups['foreground'].append(poly)
                elif new_salient_catname in self.salient_ignore_classes:
                    saliency_sseg_groups['ignore'].append(poly)
                elif new_salient_catname in self.non_salient_classes:
                    saliency_sseg_groups['background'].append(poly)
                else:
                    raise AssertionError

            if self.config['balance_areas']:
                # num_fg_polys = len(saliency_sseg_groups['foreground'])
                big_poly_fg = unary_union([p.to_shapely() for p in saliency_sseg_groups['foreground']])
                big_poly_ignore = unary_union([p.to_shapely() for p in saliency_sseg_groups['ignore']])
                big_poly_bg = (frame_box - big_poly_fg) - big_poly_ignore
                #unit_area_share = fg_polys.area / len(fg_polys)
                total_area = frame_box.area
                bg_cover_frac = big_poly_bg.area / (total_area + 1)
                # fg_cover_frac = big_poly_fg.area / (total_area + 1)
                bg_weight_share = (1 - bg_cover_frac)
                task_target_weight['saliency'][:] = bg_weight_share ** 0.5
            else:
                task_target_weight['saliency'][:] = 1

            for poly in saliency_sseg_groups['background']:
                if self.config['balance_areas']:
                    weight = (1 - (poly.area / (total_area + 1)))
                    poly.fill(task_target_weight['saliency'], value=weight, assert_inplace=True)

            for poly in saliency_sseg_groups['foreground']:
                task_target_ohe['saliency'] = poly.fill(task_target_ohe['saliency'], value=1, assert_inplace=True)
                if self.config['balance_areas']:
                    weight = (1 - (poly.area / (total_area + 1)))
                    poly.fill(task_target_weight['saliency'], value=weight, assert_inplace=True)

                if self.dist_weights:
                    # New feature where we encode that we care much more about
                    # segmenting the inside of the object than the outside.
                    # Effectively boundaries become uncertain.
                    dist, poly_mask = util_kwimage.polygon_distance_transform(
                        poly, shape=space_shape)
                    max_dist = dist.max()
                    if max_dist > 0:
                        dist_weight = dist / max_dist
                        weight_mask = dist_weight + (1 - poly_mask)
                        task_target_weight['saliency'] = task_target_weight['saliency'] * weight_mask

            for poly in saliency_sseg_groups['ignore']:
                poly.fill(task_target_ohe['saliency'], value=1, assert_inplace=True)
                poly.fill(task_target_ignore['saliency'], value=1, assert_inplace=True)

            max_weight = task_target_weight['saliency'].max()
            if max_weight > 0:
                task_target_weight['saliency'] /= max_weight

        if wants_class_sseg:
            ### Build single frame CLASS target labels and weights

            task_target_ohe['class'] = np.zeros((len(self.classes),) + space_shape, dtype=np.uint8)
            task_target_ignore['class'] = np.zeros(space_shape, dtype=np.uint8)
            task_target_weight['class'] = np.ones(space_shape, dtype=np.float32)

            # Group polygons into foreground / background for the class task
            class_sseg_groups = {
                'foreground': [],
                'background': [],
                'ignore': [],
                'undistinguished': [],
            }
            for poly, cid, tid in zip(ann_polys, ann_cids, ann_tids):
                new_class_catname = task_tid_to_cnames['class'][tid][time_idx]
                new_class_cidx = self.classes.node_to_idx[new_class_catname]
                orig_cidx = self.classes.id_to_idx[cid]
                poly.meta['new_class_cidx'] = new_class_cidx
                poly.meta['orig_cidx'] = orig_cidx
                if new_class_catname in self.ignore_classes:
                    class_sseg_groups['ignore'].append(poly)
                elif new_class_catname in self.class_foreground_classes:
                    class_sseg_groups['foreground'].append(poly)
                elif new_class_catname in self.background_classes:
                    class_sseg_groups['background'].append(poly)
                elif new_class_catname in self.undistinguished_classes:
                    class_sseg_groups['undistinguished'].append(poly)

            if self.config['balance_areas']:
                big_poly_fg = unary_union([p.to_shapely() for p in class_sseg_groups['foreground']])
                big_poly_ignore = unary_union([p.to_shapely() for p in class_sseg_groups['ignore']])
                big_poly_undistinguished = unary_union([p.to_shapely() for p in class_sseg_groups['undistinguished']])
                big_poly_bg = ((frame_box - big_poly_fg) - big_poly_ignore) - big_poly_undistinguished
                total_area = frame_box.area
                bg_cover_frac = big_poly_bg.area / (total_area + 1)
                # fg_cover_frac = big_poly_fg.area / (total_area + 1)
                bg_weight_share = (1 - bg_cover_frac)
                task_target_weight['class'][:] = bg_weight_share ** 0.5
            else:
                task_target_weight['class'][:] = 1

            for poly in class_sseg_groups['ignore']:
                poly.fill(task_target_ignore['class'], value=1, assert_inplace=True)
                poly.fill(task_target_ohe['class'][poly.meta['orig_cidx']], value=1, assert_inplace=True)

            for poly in class_sseg_groups['background']:
                # task_target_ignore['class'] = poly.fill(task_target_ignore['class'], value=1, assert_inplace=True)
                poly.fill(task_target_ohe['class'][poly.meta['orig_cidx']], value=1, assert_inplace=True)

            for poly in class_sseg_groups['undistinguished']:
                task_target_ignore['class'] = poly.fill(task_target_ignore['class'], value=1, assert_inplace=True)
                poly.fill(task_target_ohe['class'][poly.meta['orig_cidx']], value=1, assert_inplace=True)

            for poly in class_sseg_groups['foreground']:
                poly.fill(task_target_ohe['class'][poly.meta['new_class_cidx']], value=1, assert_inplace=True)

                if self.config['balance_areas']:
                    weight = (1 - (poly.area / (total_area + 1)))
                    poly.fill(task_target_weight['class'], value=weight, assert_inplace=True)

                if self.dist_weights:
                    # New feature where we encode that we care much more about
                    # segmenting the inside of the object than the outside.
                    # Effectively boundaries become uncertain.
                    dist, poly_mask = util_kwimage.polygon_distance_transform(
                        poly, shape=space_shape)
                    max_dist = dist.max()
                    if max_dist > 0:
                        dist_weight = dist / max_dist
                        weight_mask = dist_weight + (1 - poly_mask)
                        task_target_weight['class'] = task_target_weight['class'] * weight_mask

            max_weight = task_target_weight['class'].max()
            if max_weight > 0:
                task_target_weight['class'] /= max_weight

        # frame_poly_weights = np.maximum(frame_poly_weights, self.min_spacetime_weight)

        if self.upweight_centers:
            sigma = (
                (4.8 * ((space_shape[1] - 1) * 0.5 - 1) + 0.8),
                (4.8 * ((space_shape[0] - 1) * 0.5 - 1) + 0.8),
            )
            space_weights = kwarray.normalize(kwimage.gaussian_patch(space_shape, sigma=sigma))
            # space_weights = util_kwimage.upweight_center_mask(space_shape)
            space_weights = np.maximum(space_weights, self.min_spacetime_weight)
            spacetime_weights = space_weights * time_weights[time_idx]
        else:
            spacetime_weights = 1

        # Note: ensure this is resampled into target output space
        # Module the pixelwise weights by the 1 - the fraction of modes
        # that have nodata.
        if self.config['downweight_nan_regions']:
            nodata_total = 0.0
            for mask in mode_to_invalid_mask.values():
                if mask is None:
                    nodata_total += 0
                else:
                    if len(mask.shape) == 3:
                        mask_ = ((mask.sum(axis=2) / mask.shape[2])).astype(float)
                    else:
                        mask_ = mask.astype(float)
                    mask_ = kwimage.imresize(mask_, dsize=output_dsize)
                    nodata_total += mask_
            total_bands = len(mode_to_invalid_mask)
            nodata_frac = nodata_total / total_bands
            nodata_weight = 1 - nodata_frac
        else:
            nodata_weight = 1
            # frame_weights = frame_weights * nodata_weight

        generic_frame_weight = nodata_weight * spacetime_weights

        # Dilate ignore masks (dont care about the surrounding area # either)
        # frame_saliency = kwimage.morphology(frame_saliency, 'dilate', kernel=ignore_dilate)
        if self.ignore_dilate > 0:
            for k, v in task_target_ignore.items():
                task_target_ignore[k] = kwimage.morphology(v, 'dilate', kernel=self.ignore_dilate)

        if self.weight_dilate > 0:
            for k, v in task_target_weight.items():
                task_target_weight[k] = kwimage.morphology(v, 'dilate', kernel=self.weight_dilate)

        frame_item['ann_aids'] = ann_aids
        if wants_class_sseg:
            # Postprocess (Dilate?) the truth map
            # TODO: it would be better if the network accepted indicator vector
            # style labels rather than integer style labels.
            for cidx, class_map in enumerate(task_target_ohe['class']):
                # class_map = kwimage.morphology(class_map, 'dilate', kernel=5)
                frame_cidxs[class_map > 0] = cidx
            task_frame_weight = (
                (1 - task_target_ignore['class']) *
                task_target_weight['class'] *
                generic_frame_weight
            )
            frame_item['class_idxs'] = frame_cidxs
            frame_item['class_weights'] = np.clip(task_frame_weight, 0, 1)
        if wants_saliency_sseg:
            task_frame_weight = (
                (1 - task_target_ignore['saliency']) *
                task_target_weight['saliency'] *
                generic_frame_weight
            )
            frame_item['saliency'] = task_target_ohe['saliency']
            frame_item['saliency_weights'] = np.clip(task_frame_weight, 0, 1)

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
            ('normalize_perframe', self.config['normalize_perframe']),
            ('with_intensity', with_intensity),
            ('with_class', with_class),
            ('depends_version', 16),  # bump if `compute_dataset_stats` changes
        ])
        if self.config['normalize_peritem']:
            depends['normalize_peritem'] = self.config['normalize_peritem']
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

        CommandLine:
            xdoctest -m watch.tasks.fusion.datamodules.kwcoco_dataset KWCocoVideoDataset.compute_dataset_stats:2

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import kwcoco
            >>> dct_dset = coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=3)
            >>> self = KWCocoVideoDataset(dct_dset, time_dims=2, window_dims=(256, 256), channels='auto')
            >>> self.compute_dataset_stats(num_workers=2)

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2')
            >>> self = KWCocoVideoDataset(coco_dset, time_dims=2, window_dims=(256, 256), channels='auto')
            >>> stats = self.compute_dataset_stats()
            >>> assert stats['class_freq']['star'] > 0 or stats['class_freq']['superstar'] > 0 or stats['class_freq']['eff'] > 0
            >>> assert stats['class_freq']['background'] > 0

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import watch
            >>> from watch.tasks.fusion import datamodules
            >>> num = 1
            >>> datamodule = datamodules.KWCocoVideoDataModule(
            >>>     train_dataset='vidshapes-watch', window_dims=64, time_steps=3,
            >>>     num_workers=0, batch_size=3, channels='auto',
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
            >>> stats1 = self.compute_dataset_stats(num=num, with_intensity=False)
            >>> stats2 = self.compute_dataset_stats(num=num, with_class=False)
            >>> stats3 = self.compute_dataset_stats(num=num, with_class=False, with_intensity=False)

        Ignore:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2')
            >>> for img in coco_dset.imgs.values():
            ...     img['sensor_coarse'] = 'demo'  # hack in a sensor
            >>> self = KWCocoVideoDataset(coco_dset, time_dims=1, window_dims=(32, 32), channels='demo:(r|g,b,n)')
            >>> self.input_sensorchan
            >>> stats = self.compute_dataset_stats(batch_size=1)

        """
        from watch.utils.slugify_ext import smart_truncate
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
        channel_stats = ub.ddict(lambda: ub.ddict(kwarray.RunningStats))

        classes = self.classes
        num_classes = len(classes)
        bins = np.arange(num_classes + 1)
        total_freq = np.zeros(num_classes, dtype=np.int64)

        sensor_mode_hist = ub.ddict(lambda: 0)
        video_id_histogram = ub.ddict(lambda: 0)
        image_id_histogram = ub.ddict(lambda: 0)

        # Make a list of all unique modes in the dataset.
        # User specifies all of this explicitly now
        unique_sensor_modes = set(
            (s.sensor.spec, s.chans.spec)
            for s in self.input_sensorchan.streams())

        is_native = self.config['input_space_scale'] == 'native'

        print('unique_sensor_modes = {}'.format(ub.repr2(unique_sensor_modes, nl=1)))
        intensity_dtype = np.float64

        # Ensure instance level frequency data in addition to pixel level
        USE_INSTANCE_LEVEL_CLASS_STATS = 1
        if USE_INSTANCE_LEVEL_CLASS_STATS:
            annots = self.sampler.dset.annots()
            track_ids = annots.lookup('track_id', None)
            cnames = annots.cnames
            trackid_to_cnames = ub.udict(ub.group_items(cnames, track_ids))
            trackid_to_cnames = trackid_to_cnames.map_values(set)
            track_classes = list(ub.flatten(trackid_to_cnames.values()))
            annot_class_freq = ub.udict(ub.dict_hist(cnames)).sorted_keys()
            track_class_freq = ub.udict(ub.dict_hist(track_classes)).sorted_keys()
            print('annot_class_freq = {}'.format(ub.urepr(annot_class_freq, nl=1)))
            print('track_class_freq = {}'.format(ub.urepr(track_class_freq, nl=1)))
        else:
            track_class_freq = None
            annot_class_freq = None

        WITH_PROG_POSTFIX_TEXT = 1
        if WITH_PROG_POSTFIX_TEXT:
            # Create timer to periodically summarize intermediate results while
            # full dataset stats are accumulating
            timer = ub.Timer().tic()
            timer._first = True
            postfix_update_threshold = 5  # seconds

        def current_input_stats():
            """
            Summarizes current stats estimates either for display or for the
            final output.
            """
            input_stats = {}
            for sensor, submodes in channel_stats.items():
                for chan_key, running in submodes.items():
                    if is_native:
                        # ensure we have the expected shape
                        try:
                            perchan_stats = running.summarize(axis=ub.NoParam, keepdims=True)
                        except RuntimeError:
                            perchan_stats = {'mean': np.array([np.nan]), 'std': np.array([np.nan])}
                        chan_mean = perchan_stats['mean'][:, None, None]
                        chan_std = perchan_stats['std'][:, None, None]
                        chan_min = perchan_stats['min'][:, None, None]
                        chan_max = perchan_stats['max'][:, None, None]
                    else:
                        try:
                            perchan_stats = running.summarize(axis=(1, 2))
                        except RuntimeError:
                            perchan_stats = {'mean': np.array([[[np.nan]]]), 'std': np.array([[[np.nan]]])}
                        chan_mean = perchan_stats['mean']
                        chan_std = perchan_stats['std']
                        chan_min = perchan_stats['min']
                        chan_max = perchan_stats['max']

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
                        'min': chan_min,
                        'max': chan_max,
                    }
            return input_stats

        from watch.utils import util_progress
        from watch.utils import util_environ
        USE_RICH_UPDATES = util_environ.envflag('USE_RICH_UPDATES', 1)
        pman = util_progress.ProgressManager(
            backend='rich' if USE_RICH_UPDATES else 'progiter')

        def update_displayed_estimates():
            """
            Build an intermediate summary to display to the user while this is
            running.
            """
            if USE_RICH_UPDATES:
                stat_lines = ['Current Estimated Dataset Statistics: ']
                if with_intensity:
                    input_stats = current_input_stats()
                    input_stats2 = {sc: {k: v.ravel() for k, v in stats.items()} for sc, stats in input_stats.items()}
                    intensity_info_text = 'Spectra Stats: ' + ub.urepr(input_stats2, with_dtype=False, precision=4)
                    stat_lines.append(intensity_info_text)
                if with_class:
                    class_stats = ub.sorted_vals(ub.dzip(classes, total_freq), reverse=True)
                    class_info_text = 'Class Stats: ' + ub.urepr(class_stats)
                    stat_lines.append(class_info_text)
                stat_lines.append('Unique Image Samples: {}'.format(len(image_id_histogram)))
                stat_lines.append('Unique Video Samples: {}'.format(len(video_id_histogram)))
                info_text = '\n'.join(stat_lines).strip()
                if info_text:
                    pman.update_info(info_text)
            else:
                if with_class:
                    intermediate = ub.sorted_vals(ub.dzip(classes, total_freq), reverse=True)
                    intermediate_text = ub.repr2(intermediate, compact=1)
                    intermediate_text_trunc = smart_truncate(intermediate_text, max_length=40, trunc_loc=0.8)
                else:
                    intermediate_text = ''
                    intermediate_text_trunc = ''

                if with_intensity:
                    try:
                        # Broken in general, only looks at one sensorchan, but
                        # not sure how to do better. This is off by default
                        # anyway.
                        input_stats = current_input_stats()
                        input_stats2 = {sc: {k: v.ravel() for k, v in stats.items()} for sc, stats in input_stats.items()}
                        curr = ub.peek(input_stats2.values())
                    except RuntimeError:
                        curr = {}
                    else:
                        curr = curr & {'mean', 'std', 'max', 'min'}
                        curr = curr.map_values(float)
                    text = ub.repr2(curr, compact=1, precision=1, nl=0) + ' ' + intermediate_text_trunc
                else:
                    text = intermediate_text_trunc
                prog.set_postfix_str(text)

        def update_intensity_estimates(frame_item):
            # Update pixel-level intensity histogram
            sensor_code = frame_item['sensor']
            modes = frame_item['modes']

            for mode_code, mode_val in modes.items():
                sensor_mode_hist[(sensor_code, mode_code)] += 1
                running = channel_stats[sensor_code][mode_code]
                val = mode_val.numpy().astype(intensity_dtype)
                weights = np.isfinite(val).astype(intensity_dtype)
                # kwarray can handle nans now
                if is_native:
                    # Put channels last so we can update multiple at once
                    flat_vals = val.transpose(1, 2, 0).reshape(-1, val.shape[0])
                    flat_weights = weights.transpose(1, 2, 0).reshape(-1, weights.shape[0])
                    running.update_many(flat_vals, weights=flat_weights)
                else:
                    running.update(val, weights=weights)

        # TODO: we can compute the intensity histogram more efficiently by
        # only doing it for unique channels (which might be duplicated)
        with pman:
            prog = pman.progiter(loader, desc='estimate dataset stats', verbose=1)
            iter_ = iter(prog)

            for batch_items in iter_:
                for item in batch_items:
                    if item is None:
                        continue

                    if with_vidid:
                        vidid = item['video_id']
                        video_id_histogram[vidid] += 1

                    for frame_item in item['frames']:
                        image_id_histogram[frame_item['gid']] += 1
                        if with_class:
                            # Update pixel-level class histogram
                            class_idxs = frame_item['class_idxs']
                            if class_idxs is not None:
                                item_freq = np.histogram(class_idxs.ravel(), bins=bins)[0]
                                total_freq += item_freq
                        if with_intensity:
                            update_intensity_estimates(frame_item)

                if WITH_PROG_POSTFIX_TEXT and timer._first or timer.toc() > postfix_update_threshold:
                    update_displayed_estimates()
                    timer._first = 0
                    timer.tic()

            if WITH_PROG_POSTFIX_TEXT:
                update_displayed_estimates()

        # TODO: we should ensure we include at least one sample from each type
        # of modality.  Note: the requested order of the channels could be
        # different that what is registered in the dataset. Need to find a good
        # way to account for this.
        MISSING_SENSOR_FALLBACK = util_environ.envflag('MISSING_SENSOR_FALLBACK', 1)
        if MISSING_SENSOR_FALLBACK and with_intensity:
            missing_sensor_modes = set(unique_sensor_modes) - set(sensor_mode_hist)
            # Try to find a few examples with these missing modes
            if missing_sensor_modes:
                print(f'Warning: we are missing stats for {missing_sensor_modes}. '
                      'We will try to force something for them')
                coco_images = self.sampler.dset.images().coco_images
                sensor_to_images = ub.group_items(coco_images, key=lambda x: x.img.get('sensor_coarse', None))
                extra_sample_groups = []
                for sensor, mode in missing_sensor_modes:
                    candidate_images = sensor_to_images.get(sensor, [])
                    if len(candidate_images) == 0:
                        print(f'sensor warning: unable to sample data for {sensor}:{mode}')
                    else:
                        filtered = []
                        for img in candidate_images:
                            if (img.channels & mode).numel():
                                filtered.append(img)
                        if not filtered:
                            print(f'mode warning: unable to sample data for {sensor}:{mode}')
                        extra_sample_groups.append(filtered)

                # Build extra fallback samples
                for group in ub.ProgIter(extra_sample_groups, desc='process fallbacks'):
                    image_ids = [g.img['id'] for g in group]
                    images = self.sampler.dset.images(image_ids)
                    vidid_to_gids = ub.group_items(image_ids, images.lookup('video_id'))
                    for vidid, gids in vidid_to_gids.items():
                        video = self.sampler.dset.index.videos[vidid]
                        # Hack: just use the entire video, if that fails we should
                        # implement windowing here.
                        space_slice = (
                            slice(0, video['height']),
                            slice(0, video['width']),
                        )
                        sample = {'video_id': vidid, 'gids': gids[0:1],
                                  'space_slice': space_slice}
                        item = self[sample]

        self.disable_augmenter = False

        # Return the raw counts and let the model choose how to handle it
        if with_class:
            class_freq = ub.dzip(classes, total_freq)
        else:
            class_freq = None

        if with_intensity:
            input_stats = current_input_stats()
        else:
            input_stats = None

        dataset_stats = {
            'unique_sensor_modes': unique_sensor_modes,
            'sensor_mode_hist': dict(sensor_mode_hist),
            'input_stats': input_stats,
            'class_freq': class_freq,  # pixelwise

            'annot_class_freq': annot_class_freq,
            'track_class_freq': track_class_freq,
            # 'video_id_histogram': dict(video_id_histogram),
        }
        return dataset_stats

    def draw_item(self, item, item_output=None, combinable_extra=None,
                  max_channels=5, max_dim=224, norm_over_time=0,
                  overlay_on_image=False, draw_weights=True, rescale='auto', **kw):
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
                    saliency_probs
                    class_probs
                    pred_ltrb

        Note:
            The ``self.requested_tasks`` controls the task labels returned by
            getitem, and hence what can be visualized here.

        Example:
            >>> from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA
            >>> import kwcoco
            >>> import kwarray
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> channels = 'B10|B8a|B1|B8|B11'
            >>> combinable_extra = [['B10', 'B8', 'B8a']]  # special behavior
            >>> # combinable_extra = None  # uncomment for raw behavior
            >>> self = KWCocoVideoDataset(coco_dset, time_dims=5, window_dims=(530, 610), channels=channels)
            >>> #index = len(self) // 4
            >>> index = self.new_sample_grid['targets'][self.new_sample_grid['positives_indexes'][5]]
            >>> if 1:
            >>>     # More controlled settings for debug
            >>>     self.disable_augmenter = True
            >>> item = self[index]
            >>> print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
            >>> fliprot_params = item['target'].get('fliprot_params', None)
            >>> rng = kwarray.ensure_rng(None)
            >>> #
            >>> # Generate random predicted change probabilities for each frame
            >>> item_output = {}
            >>> change_prob_list = []
            >>> for frame in item['frames'][1:]:  # first frame does not have change
            >>>     change_prob = kwimage.Heatmap.random(
            >>>         dims=frame['output_dims'], classes=1, rng=rng).data['class_probs'][0]
            >>>     if fliprot_params:
            >>>         change_prob = data_utils.fliprot(change_prob, **fliprot_params)
            >>>     change_prob_list += [change_prob]
            >>> change_probs = np.stack(change_prob_list)
            >>> item_output['change_probs'] = change_probs
            >>> #
            >>> # Generate random predicted class probabilities for each frame
            >>> class_prob_list = []
            >>> frame_pred_ltrb_list = []
            >>> for frame in item['frames']:
            >>>     class_prob = kwimage.Heatmap.random(
            >>>         dims=frame['output_dims'], classes=list(self.classes), rng=rng).data['class_probs']
            >>>     class_prob_ = einops.rearrange(class_prob, 'c h w -> h w c')
            >>>     if fliprot_params:
            >>>         class_prob_ = data_utils.fliprot(class_prob_, **fliprot_params)
            >>>     class_prob_list += [class_prob_]
            >>>     # Also generate a predicted box for each frame
            >>>     frame_output_dsize = frame['output_dims'][::-1]
            >>>     num_pred_boxes = rng.randint(0, 8)
            >>>     pred_boxes = kwimage.Boxes.random(num_pred_boxes).scale(frame_output_dsize)
            >>>     frame_pred_ltrb_list.append(pred_boxes.to_ltrb().data)
            >>> class_probs = np.stack(class_prob_list)
            >>> item_output['class_probs'] = class_probs
            >>> item_output['pred_ltrb'] = frame_pred_ltrb_list
            >>> #binprobs[0][:] = 0  # first change prob should be all zeros
            >>> print('item summary: ' + ub.repr2(self.summarize_item(item), nl=3))
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
            rescale = self.config['input_space_scale'] != 'native'

        if item is None:
            # BIG RED X
            # h, w = vertical_stack[-1].shape[0:2]
            h = w = (max_dim or 224)
            bad_canvas = kwimage.draw_text_on_image(
                {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                valign='center', halign='center', fontScale=10,
                color='red')
            return bad_canvas

        default_combinable_channels = self.default_combinable_channels

        from watch.tasks.fusion.datamodules.batch_visualization import BatchVisualizationBuilder
        builder = BatchVisualizationBuilder(
            item=item, item_output=item_output,
            default_combinable_channels=default_combinable_channels,
            norm_over_time=norm_over_time, max_dim=max_dim,
            max_channels=max_channels, overlay_on_image=overlay_on_image,
            draw_weights=draw_weights, combinable_extra=combinable_extra,
            classes=self.classes, requested_tasks=self.requested_tasks,
            rescale=rescale, **kw)
        canvas = builder.build()
        return canvas

    def summarize_item(self, item):
        """
        Return debugging stats about the item

        Args:
            item (dict): an item returned by __getitem__

        Returns:
            dict : a summary of the item
        """
        if item is None:
            raise Exception('Cant summarize a failed sample item=None')
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
                'box_ltrb',
                # 'box_weights', 'box_tids', 'box_cidxs',
            ]
            for key in label_keys:
                if frame.get(key, None) is not None:
                    frame_summary[key] = frame[key].shape
            item_summary['frame_summaries'].append(frame_summary)
            if frame['date_captured']:
                timestamps.append(ub.timeparse(frame['date_captured']))

            annots = self.sampler.dset.annots(frame['ann_aids'])
            cids = annots.lookup('category_id')
            class_hist = ub.dict_hist(ub.udict(self.classes.id_to_node).take(cids))
            frame_summary['class_hist'] = class_hist
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
        item_summary['input_gsd'] = item['input_gsd']
        item_summary['output_gsd'] = item['output_gsd']
        return item_summary

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
            >>> import kwcoco
            >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral', num_frames=5)
            >>> self = KWCocoVideoDataset(coco_dset, time_dims=3, window_dims=(530, 610), channels='auto')
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


@functools.cache
def biased_1d_weights(upweight_time, num_frames):
    """
    import kwplot
    plt = kwplot.autoplt()

    kwplot.figure()
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/watch'))
    from watch.tasks.fusion.datamodules.kwcoco_dataset import *  # NOQA

    kwplot.figure(fnum=1, doclf=1)
    num_frames = 5
    values = biased_1d_weights(0.5, num_frames)
    plt.plot(values)
    values = biased_1d_weights(0.1, num_frames)
    plt.plot(values)
    values = biased_1d_weights(0.0, num_frames)
    plt.plot(values)
    values = biased_1d_weights(0.9, num_frames)
    plt.plot(values)
    values = biased_1d_weights(1.0, num_frames)
    plt.plot(values)
    """
    # from kwarray.distributions import TruncNormal
    from scipy.stats import norm
    # from kwarray.distributions import TruncNormal
    sigma = kwimage.im_cv2._auto_kernel_sigma(kernel=((num_frames, 1)))[1][0]
    mean = upweight_time * (num_frames - 1) + 0.5
    # rv = TruncNormal(mean=mean, std=sigma, low=0.0, high=num_frames).rv
    rv = norm(mean, sigma)
    locs = np.arange(num_frames) + 0.5
    values = rv.pdf(locs)
    return values


# Backwards compat
sample_video_spacetime_targets = spacetime_grid_builder.sample_video_spacetime_targets


class FailedSample(Exception):
    ...


def apply_robust_normalizer(normalizer, imdata, imdata_valid, mask, dtype, copy=True):
    """
        data = [self.dataset[idx] for idx in possibly_batched_index]
      File "/home/joncrall/code/watch/watch/tasks/fusion/datamodules/kwcoco_dataset.py", line 1004, in __getitem__
        return self.getitem(index)
      File "/home/joncrall/code/watch/watch/tasks/fusion/datamodules/kwcoco_dataset.py", line 1375, in getitem
        imdata_normalized = apply_robust_normalizer(
      File "/home/joncrall/code/watch/watch/tasks/fusion/datamodules/kwcoco_dataset.py", line 2513, in apply_robust_normalizer
        imdata_valid_normalized = kwarray.normalize(
      File "/home/joncrall/code/kwarray/kwarray/util_numpy.py", line 760, in normalize
        old_min = np.nanmin(float_out)
      File "<__array_function__ internals>", line 5, in nanmin
      File "/home/joncrall/.pyenv/versions/3.10.5/envs/pyenv3.10.5/lib/python3.10/site-packages/numpy/lib/nanfunctions.py", line 319, in nanmin
        res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
    """
    import kwarray
    if normalizer['type'] is None:
        imdata_normalized = imdata.astype(dtype, copy=copy)
    elif normalizer['type'] == 'normalize':
        # Note: we are using kwarray normalize, the one in kwimage is deprecated
        arr = imdata_valid.astype(dtype, copy=copy)
        imdata_valid_normalized = kwarray.normalize(
            arr, mode=normalizer['mode'],
            beta=normalizer['beta'], alpha=normalizer['alpha'],
        )
        if mask is None:
            imdata_normalized = imdata_valid_normalized
        else:
            imdata_normalized = imdata.copy() if copy else imdata
            imdata_normalized[mask] = imdata_valid_normalized
    else:
        raise KeyError(normalizer['type'])
    return imdata_normalized
