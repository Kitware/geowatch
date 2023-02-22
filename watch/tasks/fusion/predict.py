#!/usr/bin/env python
"""
Fusion prediction script.

TODO:
    - [ ] Prediction caching?
    - [ ] Reduce memory usage?
    - [ ] Pseudo Live.
"""
import torch
import ubelt as ub
import numpy as np
import kwimage
import kwarray
import kwcoco
from watch.tasks.fusion import datamodules
from watch.tasks.fusion import utils
from watch.utils import util_path
from watch.utils import util_parallel
from watch.tasks.fusion.datamodules import data_utils
from watch.tasks.fusion.coco_stitcher import CocoStitchingManager
from watch.tasks.fusion.coco_stitcher import quantize_float01  # NOQA
# APPLY Monkey Patches
from watch.monkey import monkey_torch
from watch.monkey import monkey_torchmetrics
monkey_torchmetrics.fix_torchmetrics_compatability()

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def make_predict_config(cmdline=False, **kwargs):
    """
    Configuration for fusion prediction
    """
    # TODO: switch to jsonargparse / scriptconfig + jsonargparse
    from watch.utils import configargparse_ext
    from scriptconfig.smartcast import smartcast

    parser = configargparse_ext.ArgumentParser(
        add_config_file_help=False,
        description='Prediction script for the fusion task',
        auto_env_var_prefix='WATCH_FUSION_PREDICT_',
        add_env_var_help=True,
        formatter_class='raw',
        config_file_parser_class='yaml',
        args_for_setting_config_path=['--config'],
        args_for_writing_out_config_file=['--dump'],
    )
    parser.add_argument('--datamodule', default='KWCocoVideoDataModule')
    parser.add_argument('--pred_dataset', default=None, dest='pred_dataset', help='path to the output dataset (note: test_dataset is the input dataset)')

    # parser.add_argument('--pred_dpath', dest='pred_dpath', type=pathlib.Path, help='path to dump results. Deprecated, do not use.')

    parser.add_argument('--package_fpath', type=str)
    parser.add_argument('--devices', default=None, help='lightning devices')  # TODO accelerator and whatever
    parser.add_argument('--thresh', type=smartcast, default=0.01)

    parser.add_argument('--with_change', type=smartcast, default='auto')
    parser.add_argument('--with_class', type=smartcast, default='auto')
    parser.add_argument('--with_saliency', type=smartcast, default='auto')

    parser.add_argument('--compress', type=str, default='DEFLATE', help='type of compression for prob images')
    parser.add_argument('--track_emissions', type=smartcast, default=True, help='set to false to disable emission tracking')

    parser.add_argument('--quantize', type=smartcast, default=True, help='quantize outputs')

    parser.add_argument('--tta_fliprot', type=smartcast, default=0, help='number of times to flip/rotate the frame, can be in [0,7]')
    parser.add_argument('--tta_time', type=smartcast, default=0, help='number of times to expand the temporal sample for a frame'),

    parser.add_argument('--clear_annots', type=smartcast, default=1, help='Clear existing annotations in output file. Otherwise keep them')
    parser.add_argument('--drop_unused_frames', type=smartcast, default=0, help='if True, remove any images that were not predicted on')
    parser.add_argument('--write_workers', type=smartcast, default='datamodule', help='workers to use for writing results. If unspecified uses the datamodule num_workers')

    # TODO:
    # parser.add_argument('--cache', type=smartcast, default=0, help='if True, dont rerun prediction on images where predictions exist'),

    parser.add_argument(
        '--write_preds', default=False, type=smartcast, help=ub.paragraph(
            '''
            If True, convert probability maps into raw "hard" predictions and
            write them as annotations to the prediction kwcoco file.
            '''))

    parser.add_argument(
        '--write_probs', default=True, type=smartcast, help=ub.paragraph(
            '''
            If True, write raw "soft" probability maps into the kwcoco file as
            a new auxiliary channel.  The channel name is currently hard-coded
            based on expected output heads. This may change in the future.
            '''))

    parser.set_defaults(**kwargs)
    # parse the datamodule and method strings
    default_args = None if cmdline else []
    temp_args, _ = parser.parse_known_args(
        default_args, ignore_help_args=True, ignore_write_args=True)

    # get the datamodule and method classes
    datamodule_class = getattr(datamodules, temp_args.datamodule)

    # add the appropriate args to the parse
    # for dataset, method, and trainer
    # Note: Adds '--test_dataset' to argparse (
    # may want to modify behavior to only expose non-training params)
    overloadable_datamodule_keys = [
        'channels',
        'normalize_peritem',
        'chip_size',
        'chip_dims',
        'time_steps',
        'time_sampling',
        'time_span',
        'time_kernel',
        'input_space_scale',
        'window_space_scale',
        'output_space_scale',
        'use_cloudmask',
        'mask_low_quality',
        'observable_threshold',
        'quality_threshold',
        'resample_invalid_frames',
        'set_cover_algo',
    ]
    parser = datamodule_class.add_argparse_args(parser)
    datamodule_defaults = {k: parser.get_default(k) for k in overloadable_datamodule_keys}
    parser.set_defaults(**{
        'batch_size': 1,
        'chip_overlap': 0.3,
    })
    parser.set_defaults(**{k: 'auto' for k in overloadable_datamodule_keys})

    # parse and pass to main
    parser.set_defaults(**kwargs)
    # args, _ = parser.parse_known_args(default_args)
    args = parser.parse_args(default_args)
    args.datamodule_defaults = datamodule_defaults
    # assert args.batch_size == 1
    return args


def build_stitching_managers(config, method, result_dataset, writer_queue=None):
    # could be torch on-device stitching
    stitch_managers = {}
    stitch_device = 'numpy'

    ignore_classes = {
        'not_salient', 'ignore', 'background', 'Unknown'}
    # hack, not general
    ignore_classes.update({'negative', 'positive'})

    stitcher_common_kw = dict(
        stiching_space='video',
        device=stitch_device,
        thresh=config['thresh'],
        write_probs=config['write_probs'],
        write_preds=config['write_preds'],
        prob_compress=config['compress'],
        quantize=config['quantize'],
        writer_queue=writer_queue,
    )

    # If we only care about some predictions from the model, then keep track of
    # the class indices we need to take.
    task_keep_indices = {}
    if config['with_change']:
        task_name = 'change'
        head_classes = ['change']
        head_keep_idxs = [
            idx for idx, catname in enumerate(head_classes)
            if catname not in ignore_classes]
        head_keep_classes = list(ub.take(head_classes, head_keep_idxs))
        chan_code = '|'.join(head_keep_classes)
        task_keep_indices[task_name] = head_keep_idxs
        print('task_name = {!r}'.format(task_name))
        print('head_classes = {!r}'.format(head_classes))
        print('head_keep_classes = {!r}'.format(head_keep_classes))
        print('chan_code = {!r}'.format(chan_code))
        print('head_keep_idxs = {!r}'.format(head_keep_idxs))
        stitch_managers[task_name] = CocoStitchingManager(
            result_dataset,
            chan_code=chan_code,
            short_code='pred_' + task_name,
            num_bands=len(head_keep_classes),
            **stitcher_common_kw,
        )
        stitch_managers[task_name].head_keep_idxs = head_keep_idxs
        result_dataset.ensure_category('change')

    if config['with_class']:
        task_name = 'class'
        if hasattr(method, 'foreground_classes'):
            foreground_classes = method.foreground_classes
        else:
            from watch import heuristics
            not_foreground = (heuristics.BACKGROUND_CLASSES |
                              heuristics.IGNORE_CLASSNAMES |
                              heuristics.NEGATIVE_CLASSES)
            foreground_classes = ub.oset(method.classes) - not_foreground
        head_classes = method.classes
        head_keep_idxs = [
            idx for idx, catname in enumerate(head_classes)
            if catname not in ignore_classes]
        head_keep_classes = list(ub.take(head_classes, head_keep_idxs))
        task_keep_indices[task_name] = head_keep_idxs
        chan_code = '|'.join(list(head_keep_classes))
        print('task_name = {!r}'.format(task_name))
        print('head_classes = {!r}'.format(head_classes))
        print('head_keep_classes = {!r}'.format(head_keep_classes))
        print('chan_code = {!r}'.format(chan_code))
        print('head_keep_idxs = {!r}'.format(head_keep_idxs))
        stitch_managers[task_name] = CocoStitchingManager(
            result_dataset,
            chan_code=chan_code,
            short_code='pred_' + task_name,
            polygon_categories=foreground_classes,
            num_bands=len(head_keep_classes),
            **stitcher_common_kw,
        )
        stitch_managers[task_name].head_keep_idxs = head_keep_idxs

    if config['with_saliency']:
        # hack: the model should tell us what the shape of its head is
        task_name = 'saliency'
        head_classes = ['not_salient', 'salient']
        head_keep_idxs = [
            idx for idx, catname in enumerate(head_classes)
            if catname not in ignore_classes]
        head_keep_classes = list(ub.take(head_classes, head_keep_idxs))
        task_keep_indices[task_name] = head_keep_idxs
        chan_code = '|'.join(head_keep_classes)
        print('task_name = {!r}'.format(task_name))
        print('head_classes = {!r}'.format(head_classes))
        print('head_keep_classes = {!r}'.format(head_keep_classes))
        print('chan_code = {!r}'.format(chan_code))
        print('head_keep_idxs = {!r}'.format(head_keep_idxs))
        stitch_managers[task_name] = CocoStitchingManager(
            result_dataset,
            chan_code=chan_code,
            short_code='pred_' + task_name,
            polygon_categories=['salient'],
            num_bands=len(head_keep_classes),
            **stitcher_common_kw,
        )
        stitch_managers[task_name].head_keep_idxs = head_keep_idxs
    return stitch_managers


def resolve_datamodule(config, method, datamodule_defaults):
    """
    TODO: refactor / cleanup.

    Breakup the sections that handle getting the traintime params, resolving
    the datamodule args, and building the datamodule.
    """
    # init datamodule from args
    datamodule_class = getattr(datamodules, config['datamodule'])
    datamodule_vars = datamodule_class.compatible(config)

    parsetime_vals = ub.udict(datamodule_vars) & datamodule_defaults
    need_infer = ub.udict({
        k: v for k, v in parsetime_vals.items() if v == 'auto' or v == ['auto']})
    # Try and infer what data we were given at train time
    if hasattr(method, 'config_cli_yaml'):
        traintime_params = method.config_cli_yaml["data"]
    elif hasattr(method, 'fit_config'):
        traintime_params = method.fit_config
    elif hasattr(method, 'datamodule_hparams'):
        traintime_params = method.datamodule_hparams
    else:
        traintime_params = {}
        if datamodule_vars['channels'] in {None, 'auto'}:
            # import xdev
            # xdev.embed()
            print('Warning have to make assumptions. Might not always work')
            raise NotImplementedError('TODO: needs to be sensorchan if we do this')
            if hasattr(method, 'input_channels'):
                # note input_channels are sometimes different than the channels the
                # datamodule expects. Depending on special keys and such.
                traintime_params['channels'] = method.input_channels.spec
            else:
                traintime_params['channels'] = list(method.input_norms.keys())[0]

    def get_scriptconfig_compatible(config_cls, other):
        """
        TODO: add to scriptconfig. Get the set of keys that we will accept.
        """
        acceptable_keys = set(config_cls.default.keys())
        for val in config_cls.default.values():
            if val.alias:
                acceptable_keys.update(val.alias)

        common = ub.udict(other) & acceptable_keys
        resolved = dict(config_cls(cmdline=0, data=common))
        return resolved

    config_cls = datamodules.kwcoco_dataset.KWCocoVideoDatasetConfig
    other = traintime_params
    traintime_datavars = get_scriptconfig_compatible(
        config_cls,
        other
    )

    # FIXME: Some of the inferred args seem to not have the right type here.
    able_to_infer = traintime_datavars & need_infer
    if able_to_infer.get('channels', None) is not None:
        # do this before smartcast breaks the spec
        able_to_infer['channels'] = kwcoco.SensorChanSpec.coerce(able_to_infer['channels'])
    from scriptconfig.smartcast import smartcast
    able_to_infer = ub.udict(able_to_infer).map_values(smartcast)
    unable_to_infer = need_infer - traintime_datavars
    # Use defaults when we can't infer
    overloads = able_to_infer.copy()
    overloads.update(datamodule_defaults & unable_to_infer)
    datamodule_vars.update(overloads)
    config.update(datamodule_vars)
    print('able_to_infer = {}'.format(ub.repr2(able_to_infer, nl=1)))
    print('unable_to_infer = {}'.format(ub.repr2(unable_to_infer, nl=1)))
    print('overloads = {}'.format(ub.repr2(overloads, nl=1)))

    # Look at the difference between predict and train time settings
    print('deviation from fit->predict settings:')
    for key in (traintime_datavars.keys() & datamodule_vars.keys()):
        f_val = traintime_datavars[key]  # fit-time value
        p_val = datamodule_vars[key]  # pred-time value
        if f_val != p_val:
            print(f'    {key!r}: {f_val!r} -> {p_val!r}')

    HACK_FIX_MODELS_WITH_BAD_CHANNEL_SPEC = True
    if HACK_FIX_MODELS_WITH_BAD_CHANNEL_SPEC:
        # There was an issue where we trained models and specified
        # r|g|b|mat:0.3 but we only passed data with r|g|b. At train time
        # current logic (whch we need to fix) will happilly just take a subset
        # of those channels, which means the recorded channels disagree with
        # what the model was actually trained with.
        if hasattr(method, 'sensor_channel_tokenizers'):
            from watch.tasks.fusion.methods.network_modules import RobustModuleDict
            datamodule_sensorchan_spec = datamodule_vars['channels']
            unique_channel_streams = ub.oset()
            model_sensorchan_stem_parts = []
            for sensor, tokenizers in method.sensor_channel_tokenizers.items():
                sensor = RobustModuleDict._unnormalize_key(sensor)
                if ':' in sensor:
                    # full sensorchan already exists (sequence aware model)
                    sensorchan = sensor
                    model_sensorchan_stem_parts.append(sensorchan)
                else:
                    # dict is nested sensor channel code (older model)
                    for code in tokenizers.keys():
                        code = RobustModuleDict._unnormalize_key(code)
                        unique_channel_streams.add(code)
                        sensorchan = f'{sensor}:{code}'
                        model_sensorchan_stem_parts.append(sensorchan)

            hack_model_sensorchan_spec = kwcoco.SensorChanSpec.coerce(','.join(model_sensorchan_stem_parts))
            # hack_model_spec = kwcoco.ChannelSpec.coerce(','.join(unique_channel_streams))
            if datamodule_sensorchan_spec is not None:
                datamodule_sensorchan_spec = kwcoco.SensorChanSpec.coerce(datamodule_sensorchan_spec)
                hack_model_sensorchan_spec = hack_model_sensorchan_spec.normalize()
                datamodule_sensorchan_spec = datamodule_sensorchan_spec.normalize()
                hack_sensorchan_set = set(hack_model_sensorchan_spec.normalize().spec.split(","))
                datamodule_sensorchan_set = set(datamodule_sensorchan_spec.normalize().spec.split(","))
                if hack_sensorchan_set != datamodule_sensorchan_set:
                    print('Warning: reported model channels may be incorrect '
                          'due to bad train hyperparams',
                          hack_model_sensorchan_spec.normalize().concise().spec,
                          'versus',
                          datamodule_sensorchan_spec.normalize().concise().spec)

                    compat_parts = []
                    for model_part in hack_model_sensorchan_spec.streams():
                        data_part = datamodule_sensorchan_spec.matching_sensor(model_part.sensor.spec)
                        if not data_part.chans.spec:
                            # Try the generic sensor
                            data_part = datamodule_sensorchan_spec.matching_sensor('*')
                        isect_part = model_part.chans.fuse().intersection(data_part.chans.fuse())
                        # Stems required chunked channels, cant take subsets of them
                        if isect_part.spec == model_part.chans.spec:
                            compat_parts.append(model_part)

                    if len(compat_parts) == 0:
                        print(f'datamodule_sensorchan_spec={datamodule_sensorchan_spec}')
                        print(f'hack_model_sensorchan_spec={hack_model_sensorchan_spec}')
                        raise ValueError('no compatible channels between model and data')
                    hack_common = sum(compat_parts)
                    # hack_common = hack_model_sensorchan_spec.intersection(datamodule_sensorchan_spec)
                    datamodule_vars['channels'] = hack_common.spec

    DZYNE_MODEL_HACK = 1
    if DZYNE_MODEL_HACK:
        package_fpath = ub.Path(config['package_fpath'])
        if package_fpath.stem == 'lc_rgb_fusion_model_package':
            # This model has an issue with the L8 features it was trained on
            datamodule_vars['exclude_sensors'] = ['L8']

    datamodule = datamodule_class(
        **datamodule_vars
    )
    return config, traintime_params, datamodule
    #### real diff part


@profile
def predict(cmdline=False, **kwargs):
    """
    Example:
        >>> # Train a demo model (in the future grab a pretrained demo model)
        >>> from watch.tasks.fusion.fit import fit_model  # NOQA
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> from watch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> disable_lightning_hardware_warnings()
        >>> args = None
        >>> cmdline = False
        >>> devices = None
        >>> test_dpath = ub.Path.appdir('watch/test/fusion/').ensuredir()
        >>> results_path = (test_dpath / 'predict').ensuredir()
        >>> results_path.delete()
        >>> results_path.ensuredir()
        >>> package_fpath = test_dpath / 'my_test_package.pt'
        >>> import kwcoco
        >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes4-multispectral', num_frames=5, image_size=(128, 128))
        >>> test_dset = kwcoco.CocoDataset.demo('special:vidshapes2-multispectral', num_frames=5, image_size=(128, 128))
        >>> fit_kwargs = kwargs = {
        ...     'train_dataset': train_dset.fpath,
        ...     'datamodule': 'KWCocoVideoDataModule',
        ...     'workdir': ub.ensuredir((test_dpath, 'train')),
        ...     'package_fpath': package_fpath,
        ...     #'channels': 'auto',
        ...     'max_epochs': 1,
        ...     'time_steps': 2,
        ...     'time_span': "2m",
        ...     'chip_size': 64,
        ...     'time_sampling': 'hardish3',
        ...     'global_change_weight': 1.0,
        ...     'global_class_weight': 1.0,
        ...     'global_saliency_weight': 1.0,
        ...     'max_steps': 1,
        ...     'learning_rate': 1e-5,
        ...     'num_workers': 0,
        ...     'devices': devices,
        ... }
        >>> package_fpath = fit_model(**fit_kwargs)
        >>> assert ub.Path(package_fpath).exists()
        >>> # Predict via that model
        >>> predict_kwargs = kwargs = {
        >>>     'package_fpath': package_fpath,
        >>>     'pred_dataset': ub.Path(results_path) / 'pred.kwcoco.json',
        >>>     'test_dataset': test_dset.fpath,
        >>>     'datamodule': 'KWCocoVideoDataModule',
        >>>     'batch_size': 1,
        >>>     'num_workers': 0,
        >>>     'devices': devices,
        >>> }
        >>> result_dataset = predict(**kwargs)
        >>> dset = result_dataset
        >>> dset.dataset['info'][-1]['properties']['config']['time_sampling']
        >>> # Check that the result format looks correct
        >>> for vidid in dset.index.videos.keys():
        >>>     # Note: only some of the images in the pred sequence will get
        >>>     # a change predictoion, depending on the temporal sampling.
        >>>     images = dset.images(dset.index.vidid_to_gids[1])
        >>>     pred_chans = [[a['channels'] for a in aux] for aux in images.lookup('auxiliary')]
        >>>     assert any('change' in cs for cs in pred_chans), 'some frames should have change'
        >>>     assert not all('change' in cs for cs in pred_chans), 'some frames should not have change'
        >>>     # Test number of annots in each frame
        >>>     frame_to_cathist = {
        >>>         img['frame_index']: ub.dict_hist(annots.cnames, labels=result_dataset.object_categories())
        >>>         for img, annots in zip(images.objs, images.annots)
        >>>     }
        >>>     assert frame_to_cathist[0]['change'] == 0, 'first frame should have no change polygons'
        >>>     # This test may fail with very low probability, so warn
        >>>     import warnings
        >>>     if sum(d['change'] for d in frame_to_cathist.values()) == 0:
        >>>         warnings.warn('should have some change predictions elsewhere')
        >>> coco_img = dset.images().coco_images[1]
        >>> # Test that new quantization does not existing APIs
        >>> pred1 = coco_img.delay('salient', nodata_method='float').finalize()
        >>> assert pred1.max() <= 1
        >>> # new delayed image does not make it easy to remove dequantization
        >>> # add test back in if we add support for that.
        >>> # pred2 = coco_img.delay('salient').finalize(nodata_method='float', dequantize=False)
        >>> # assert pred2.max() > 1

    Example:
        >>> # Train a demo model (in the future grab a pretrained demo model)
        >>> from watch.tasks.fusion.fit import fit_model  # NOQA
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> from watch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> disable_lightning_hardware_warnings()

        >>> args = None
        >>> cmdline = False
        >>> devices = None
        >>> test_dpath = ub.Path.appdir('watch/test/fusion/').ensuredir()
        >>> results_path = ub.ensuredir((test_dpath, 'predict'))
        >>> ub.delete(results_path)
        >>> ub.ensuredir(results_path)

        >>> import kwcoco
        >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes8-multispectral-multisensor', num_frames=5, image_size=(128, 128))
        >>> test_dset = kwcoco.CocoDataset.demo('special:vidshapes8-multispectral-multisensor', num_frames=5, image_size=(128, 128))
        >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        >>>     train_dataset=train_dset, #'special:vidshapes8-multispectral-multisensor',
        >>>     test_dataset=test_dset, #'special:vidshapes8-multispectral-multisensor',
        >>>     chip_size=32,
        >>>     channels="r|g|b",
        >>>     batch_size=1, time_steps=3, num_workers=2, normalize_inputs=10)
        >>> datamodule.setup('fit')
        >>> datamodule.setup('test')
        >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
        >>> classes = datamodule.torch_datasets['train'].classes
        >>> print("classes = ", classes)

        >>> from watch.tasks.fusion import methods
        >>> from watch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
        >>> position_encoder = methods.heterogeneous.ScaleAgnostictPositionalEncoder(3)
        >>> backbone = TransformerEncoderDecoder(
        >>>     encoder_depth=1,
        >>>     decoder_depth=1,
        >>>     dim=position_encoder.output_dim + 16,
        >>>     queries_dim=position_encoder.output_dim,
        >>>     logits_dim=16,
        >>>     cross_heads=1,
        >>>     latent_heads=1,
        >>>     cross_dim_head=1,
        >>>     latent_dim_head=1,
        >>> )
        >>> model = methods.HeterogeneousModel(
        >>>     classes=classes,
        >>>     position_encoder=position_encoder,
        >>>     backbone=backbone,
        >>>     decoder="trans_conv",
        >>>     token_width=16,
        >>>     global_change_weight=1, global_class_weight=1, global_saliency_weight=1,
        >>>     dataset_stats=dataset_stats, input_sensorchan=datamodule.input_sensorchan)
        >>> print("model.heads.keys = ", model.heads.keys())

        >>> # Save the self
        >>> package_fpath = test_dpath / 'my_test_package.pt'
        >>> model.save_package(package_fpath)
        >>> # package_fpath = fit_model(**fit_kwargs)
        >>> assert ub.Path(package_fpath).exists()

        >>> # Predict via that model
        >>> test_dset = datamodule.train_dataset
        >>> predict_kwargs = kwargs = {
        >>>     'package_fpath': package_fpath,
        >>>     'pred_dataset': ub.Path(results_path) / 'pred.kwcoco.json',
        >>>     'test_dataset': test_dset.sampler.dset.fpath,
        >>>     'datamodule': 'KWCocoVideoDataModule',
        >>>     'channels': 'r|g|b',
        >>>     'batch_size': 1,
        >>>     'num_workers': 0,
        >>>     'devices': devices,
        >>> }
        >>> result_dataset = predict(**kwargs)

        >>> dset = result_dataset
        >>> dset.dataset['info'][-1]['properties']['config']['time_sampling']
        >>> # Check that the result format looks correct
        >>> for vidid in dset.index.videos.keys():
        >>>     # Note: only some of the images in the pred sequence will get
        >>>     # a change predictoion, depending on the temporal sampling.
        >>>     images = dset.images(dset.index.vidid_to_gids[1])
        >>>     pred_chans = [[a['channels'] for a in aux] for aux in images.lookup('auxiliary')]
        >>>     print("pred_chans = ", pred_chans)
        >>>     assert any('change' in cs for cs in pred_chans), 'some frames should have change'
        >>>     assert not all('change' in cs for cs in pred_chans), 'some frames should not have change'
        >>>     # Test number of annots in each frame
        >>>     frame_to_cathist = {
        >>>         img['frame_index']: ub.dict_hist(annots.cnames, labels=result_dataset.object_categories())
        >>>         for img, annots in zip(images.objs, images.annots)
        >>>     }
        >>>     assert frame_to_cathist[0]['change'] == 0, 'first frame should have no change polygons'
        >>>     # This test may fail with very low probability, so warn
        >>>     import warnings
        >>>     if sum(d['change'] for d in frame_to_cathist.values()) == 0:
        >>>         warnings.warn('should have some change predictions elsewhere')
        >>> coco_img = dset.images().coco_images[1]
        >>> # Test that new quantization does not existing APIs
        >>> pred1 = coco_img.delay('salient', nodata_method='float').finalize()
        >>> assert pred1.max() <= 1
        >>> # new delayed image does not make it easy to remove dequantization
        >>> # add test back in if we add support for that.
        >>> # pred2 = coco_img.delay('salient').finalize(nodata_method='float', dequantize=False)
        >>> # assert pred2.max() > 1
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    config = args.__dict__.copy()
    datamodule_defaults = args.datamodule_defaults
    print('kwargs = {}'.format(ub.repr2(kwargs, nl=1)))
    print('config = {}'.format(ub.repr2(config, nl=2)))

    package_fpath = ub.Path(config['package_fpath']).expand()

    # try:
    # Ideally we have a package, everything is defined there
    method = utils.load_model_from_package(package_fpath)
    # fix einops bug
    for _name, mod in method.named_modules():
        if 'Rearrange' in mod.__class__.__name__:
            try:
                mod._recipe = mod.recipe()
            except AttributeError:
                pass
    # hack: dont load the metrics
    method.class_metrics = None
    method.saliency_metrics = None
    method.change_metrics = None
    method.head_metrics = None
    # except Exception as ex:
    #     print('ex = {!r}'.format(ex))
    #     print(f'Failed to read {package_fpath=!r} attempting workaround')
    #     # If we have a checkpoint path we can load it if we make assumptions
    #     # init method from checkpoint.
    #     raise

    # Hack to fix GELU issue
    monkey_torch.fix_gelu_issue(method)

    method.eval()
    method.freeze()

    # TODO: perhaps we should enforce that that packaged model
    # knows how to construct the appropriate test dataset?
    config, traintime_params, datamodule = resolve_datamodule(config, method, datamodule_defaults)

    # TODO: if TTA=True, disable determenistic time sampling
    datamodule.setup('test')
    print('Finished dataset setup')

    if config['tta_time']:
        print('Expanding time samples')
        # Expand targets to include time augmented samples
        n_time_expands = config['tta_time']
        test_torch_dset = datamodule.torch_datasets['test']
        test_torch_dset._expand_targets_time(n_time_expands)

    if config['tta_fliprot']:
        print('Expanding fliprot samples')
        n_fliprot = config['tta_fliprot']
        test_torch_dset = datamodule.torch_datasets['test']
        test_torch_dset._expand_targets_fliprot(n_fliprot)

    if ub.argflag('--debug-timesample'):
        import kwplot
        plt = kwplot.autoplt()
        # TODO Could
        test_torch_dset = datamodule.torch_datasets['test']
        vidid_to_time_sampler = test_torch_dset.new_sample_grid['vidid_to_time_sampler']
        vidid = ub.peek(vidid_to_time_sampler.keys())
        time_sampler = vidid_to_time_sampler[vidid]
        time_sampler.show_summary()
        plt.show()

    print('Construct dataloader')
    test_coco_dataset = datamodule.coco_datasets['test']

    test_torch_dataset = datamodule.torch_datasets['test']
    # hack this setting
    test_torch_dataset.inference_only = True
    test_dataloader = datamodule.test_dataloader()

    # T, H, W = test_torch_dataset.window_dims

    # Create the results dataset as a copy of the test CocoDataset
    print('Populate result dataset')
    result_dataset: kwcoco.CocoDataset = test_coco_dataset.copy()

    # Remove all annotations in the results copy
    if config['clear_annots']:
        result_dataset.clear_annotations()

    # Change all paths to be absolute paths
    result_dataset.reroot(absolute=True)
    # Set the filepath for the prediction coco file
    # (modifies the bundle_dpath)
    # if config['pred_dataset'] is None:
    #     pred_dpath = util_path.coercepath(config['pred_dpath'])
    #     result_dataset.fpath = str(pred_dpath / 'pred.kwcoco.json')
    # else:
    if not config['pred_dataset']:
        raise ValueError(
            f'Must specify path to the output (predicted) kwcoco file. '
            f'Got {config["pred_dataset"]=}')
    result_dataset.fpath = str(ub.Path(config['pred_dataset']).expand())
    result_fpath = util_path.coercepath(result_dataset.fpath)

    from watch.utils.lightning_ext import util_device
    print('devices = {!r}'.format(config['devices']))
    devices = util_device.coerce_devices(config['devices'])
    print('devices = {!r}'.format(devices))
    if len(devices) > 1:
        raise NotImplementedError('TODO: handle multiple devices')
    device = devices[0]

    print('Predict on device = {!r}'.format(device))

    UNPACKAGE_METHOD_HACK = 0
    if UNPACKAGE_METHOD_HACK:
        # unpackage method hack
        from watch.tasks.fusion import methods
        unpackged_method = methods.MultimodalTransformer(**method.hparams)
        unpackged_method.load_state_dict(method.state_dict())
        method = unpackged_method

    method = method.to(device)

    # Resolve what tasks are requested by looking at what heads are available.
    global_head_weights = getattr(method, 'global_head_weights', {})
    if config['with_change'] == 'auto':
        config['with_change'] = getattr(method, 'global_change_weight', 1.0) or global_head_weights.get('change', 1)
    if config['with_class'] == 'auto':
        config['with_class'] = getattr(method, 'global_class_weight', 1.0) or global_head_weights.get('class', 1)
    if config['with_saliency'] == 'auto':
        config['with_saliency'] = getattr(method, 'global_saliency_weight', 0.0) or global_head_weights.get('saliency', 1)

    # Start background procs before we make threads
    batch_iter = iter(test_dataloader)

    from watch.utils import util_progress
    pman = util_progress.ProgressManager(backend='rich')

    # prog = ub.ProgIter(batch_iter, desc='predicting', verbose=1, freq=1)

    # Make threads after starting background proces.
    if args.write_workers == 'datamodule':
        args.write_workers = datamodule.num_workers
    writer_queue = util_parallel.BlockingJobQueue(
        mode='thread',
        # mode='serial',
        max_workers=args.write_workers
    )

    result_fpath.parent.ensuredir()
    print('result_fpath = {!r}'.format(result_fpath))

    stitch_managers = build_stitching_managers(
        config, method, result_dataset,
        writer_queue=writer_queue
    )

    expected_outputs = set(stitch_managers.keys())
    got_outputs = None
    writable_outputs = None

    print('Expected outputs: ' + str(expected_outputs))

    head_key_mapping = {
        'saliency_probs': 'saliency',
        'class_probs': 'class',
        'change_probs': 'change',
    }

    DEBUG_GRID = 0
    if DEBUG_GRID:
        # Check to see if the grid will cover all images
        image_id_to_target_space_slices = ub.ddict(list)
        seen_gids = set()
        primary_gids = set()
        for target in test_dataloader.dataset.new_sample_grid['targets']:
            target['video_id']
            # Denote we have seen this vidspace slice in this image.
            for gid in target['gids']:
                image_id_to_target_space_slices[gid].append(target['space_slice'])
            primary_gids.add(target['main_gid'])
            seen_gids.update(target['gids'])
        all_gids = list(test_dataloader.dataset.sampler.dset.images())
        from xdev import set_overlaps
        img_overlaps = set_overlaps(all_gids, seen_gids)
        print('img_overlaps = {}'.format(ub.repr2(img_overlaps, nl=1)))
        # primary_img_overlaps = set_overlaps(all_gids, primary_gids)
        # print('primary_img_overlaps = {}'.format(ub.repr2(primary_img_overlaps, nl=1)))

        # Check to see how much of each image is covered in video space
        # import kwimage
        coco_dset = test_dataloader.dataset.sampler.dset
        gid_to_iou = {}
        print('image_id_to_target_space_slices = {}'.format(ub.repr2(image_id_to_target_space_slices, nl=2)))
        for gid, slices in image_id_to_target_space_slices.items():
            vidid = coco_dset.index.imgs[gid]['video_id']
            video = coco_dset.index.videos[vidid]
            video_poly = kwimage.Box.from_dsize((video['width'], video['height'])).to_polygon()
            boxes = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(sl) for sl in slices])
            polys = boxes.to_polygons()
            covered = polys.unary_union().simplify(0.01)
            iou = covered.iou(video_poly)
            gid_to_iou[gid] = iou
        ious = list(gid_to_iou.values())
        iou_stats = kwarray.stats_dict(ious, n_extreme=True)
        print('iou_stats = {}'.format(ub.repr2(iou_stats, nl=1)))

    DEBUG_PRED_SPATIAL_COVERAGE = 0
    if DEBUG_PRED_SPATIAL_COVERAGE:
        # Enable debugging to ensure the dataloader actually passed
        # us the targets that cover the entire image.
        image_id_to_video_space_slices = ub.ddict(list)
        image_id_to_output_space_slices = ub.ddict(list)

    # add hyperparam info to "info" section
    info = result_dataset.dataset.get('info', [])

    def jsonify(data):
        # This will be serailized in kwcoco, so make sure it can be coerced to json
        from kwcoco.util import util_json
        jsonified = util_json.ensure_json_serializable(data)
        walker = ub.IndexableWalker(jsonified)
        for problem in util_json.find_json_unserializable(jsonified):
            bad_data = problem['data']
            if hasattr(bad_data, 'spec'):
                walker[problem['loc']] = bad_data.spec
            if isinstance(bad_data, kwcoco.CocoDataset):
                fixed_fpath = getattr(bad_data, 'fpath', None)
                if fixed_fpath is not None:
                    walker[problem['loc']] = fixed_fpath
                else:
                    walker[problem['loc']] = '<IN_MEMORY_DATASET: {}>'.format(
                        bad_data._build_hashid())
        return jsonified

    jsonified_args = jsonify(args.__dict__)
    config_resolved = jsonify(config)
    traintime_params = jsonify(traintime_params)

    from kwcoco.util import util_json
    assert not list(util_json.find_json_unserializable(jsonified_args))
    assert not list(util_json.find_json_unserializable(config_resolved))
    assert not list(util_json.find_json_unserializable(traintime_params))

    from watch.utils import process_context
    proc_context = process_context.ProcessContext(
        name='watch.tasks.fusion.predict',
        type='process',
        config=config_resolved,
        track_emissions=config['track_emissions'],
        extra={'fit_config': traintime_params}
    )

    assert not list(util_json.find_json_unserializable(proc_context.obj))

    info.append(proc_context.obj)
    proc_context.start()
    proc_context.add_disk_info(test_coco_dataset.fpath)

    with torch.set_grad_enabled(False), pman:
        # FIXME: that data loader should not be producing incorrect sensor/mode
        # pairs in the first place!
        EMERGENCY_INPUT_AGREEMENT_HACK = 1 and hasattr(method, 'input_norms')
        # prog.set_extra(' <will populate stats after first video>')
        # pman.start()
        prog = pman.progiter(batch_iter, desc='predicting')
        _batch_iter = iter(prog)
        if 0:
            item = test_dataloader.dataset[0]

            orig_batch = next(_batch_iter)
            item = orig_batch[0]
            item['target']
            frame = item['frames'][0]
            ub.peek(frame['modes'].values()).shape
        for orig_batch in _batch_iter:
            batch_trs = []
            # Move data onto the prediction device, grab spacetime region info
            fixed_batch = []
            for item in orig_batch:
                if item is None:
                    continue
                item = item.copy()
                batch_gids = [frame['gid'] for frame in item['frames']]
                frame_infos = [ub.udict(f) & {
                    'gid',
                    'output_space_slice',
                    'output_image_dsize',
                    'scale_outspace_from_vid',
                } for f in item['frames']]
                batch_trs.append({
                    'space_slice': tuple(item['target']['space_slice']),
                    # 'scale': item['target']['scale'],
                    'scale': item['target'].get('scale', None),
                    'gids': batch_gids,
                    'frame_infos': frame_infos,
                    'fliprot_params': item['target'].get('fliprot_params', None)
                })
                position_tensors = item.get('positional_tensors', None)
                if position_tensors is not None:
                    for k, v in position_tensors.items():
                        position_tensors[k] = v.to(device)

                filtered_frames = []
                for frame in item['frames']:
                    frame = frame.copy()
                    sensor = frame['sensor']
                    if EMERGENCY_INPUT_AGREEMENT_HACK:
                        try:
                            known_sensor_modes = method.input_norms[sensor]
                        except KeyError:
                            known_sensor_modes = None
                            continue
                    filtered_modes = {}
                    modes = frame['modes']
                    for key, mode in modes.items():
                        if EMERGENCY_INPUT_AGREEMENT_HACK:
                            if key not in known_sensor_modes:
                                continue
                        filtered_modes[key] = mode.to(device)
                    frame['modes'] = filtered_modes
                    filtered_frames.append(frame)
                item['frames'] = filtered_frames
                fixed_batch.append(item)

            if len(fixed_batch) == 0:
                continue

            batch = fixed_batch

            if 0:
                import netharn as nh
                print(nh.data.collate._debug_inbatch_shapes(batch))

            # Predict on the batch: todo: rename to predict_step
            try:
                outputs = method.forward_step(batch, with_loss=False)
            except RuntimeError as ex:
                msg = ('A predict batch failed ex = {}'.format(ub.repr2(ex, nl=1)))
                print(msg)
                import warnings
                warnings.warn(msg)
                continue

            outputs = {head_key_mapping.get(k, k): v for k, v in outputs.items()}

            if got_outputs is None:
                got_outputs = list(outputs.keys())
                prog.ensure_newline()
                writable_outputs = set(got_outputs) & expected_outputs
                print('got_outputs = {!r}'.format(got_outputs))
                print('writable_outputs = {!r}'.format(writable_outputs))

            # For each item in the batch, process the results
            for head_key in writable_outputs:
                head_probs = outputs[head_key]
                head_stitcher = stitch_managers[head_key]
                chan_keep_idxs = head_stitcher.head_keep_idxs

                # HACK: FIXME: WE ARE HARD CODING THAT CHANGE IS GIVEN TO
                # ALL FRAMES EXECPT THE FIRST IN MULTIPLE PLACES.
                if head_key == 'change':
                    predicted_frame_slice = slice(1, None)
                else:
                    predicted_frame_slice = slice(None)

                # TODO: if the predictions are downsampled wrt to the input
                # images, we need to determine what that transform is so we can
                # correctly (i.e with crops) warp the predictions back into
                # image space.

                num_batches = len(batch_trs)

                for bx in range(num_batches):
                    target: dict = batch_trs[bx]
                    item_head_probs: list[torch.Tensor] | torch.Tensor = head_probs[bx]
                    # Keep only the channels we want to write to disk
                    item_head_relevant_probs = [p[..., chan_keep_idxs] for p in item_head_probs]
                    bin_probs = [p.detach().cpu().numpy() for p in item_head_relevant_probs]

                    # Get the spatio-temporal subregion this prediction belongs to
                    # out_gids: list[int] = target['gids'][predicted_frame_slice]
                    # space_slice: tuple[slice, slice] = target['space_slice']
                    frame_infos: list[dict] = target['frame_infos'][predicted_frame_slice]

                    fliprot_params: dict = target['fliprot_params']
                    # Update the stitcher with this windowed prediction
                    for probs, frame_info in zip(bin_probs, frame_infos):
                        if fliprot_params is not None:
                            # Undo fliprot TTA
                            probs = data_utils.inv_fliprot(probs, **fliprot_params)

                        gid = frame_info['gid']
                        output_image_dsize = frame_info['output_image_dsize']
                        output_space_slice = frame_info['output_space_slice']
                        scale_outspace_from_vid = frame_info['scale_outspace_from_vid']
                        # print(f'output_image_dsize={output_image_dsize}')
                        # print(f'output_space_slice={output_space_slice}')

                        if DEBUG_PRED_SPATIAL_COVERAGE:
                            image_id_to_video_space_slices[gid].append(target['space_slice'])
                            image_id_to_output_space_slices[gid].append(output_space_slice)

                        # print(f'output_space_slice={output_space_slice}')
                        # print(f'gid={gid}')
                        # print(f'output_image_dsize={output_image_dsize}')
                        # print(f'scale_outspace_from_vid={scale_outspace_from_vid}')
                        head_stitcher.accumulate_image(
                            gid, output_space_slice, probs,
                            dsize=output_image_dsize,
                            scale=scale_outspace_from_vid)

                # Free up space for any images that have been completed
                for gid in head_stitcher.ready_image_ids():
                    head_stitcher._ready_gids.difference_update({gid})  # avoid race condition
                    head_stitcher.submit_finalize_image(gid)

        writer_queue.wait_until_finished()  # hack to avoid race condition

        # Prediction is completed, finalize all remaining images.
        for _head_key, head_stitcher in stitch_managers.items():
            for gid in head_stitcher.managed_image_ids():
                head_stitcher.submit_finalize_image(gid)
        writer_queue.wait_until_finished()
        # pman.stop()

    if DEBUG_PRED_SPATIAL_COVERAGE:
        coco_dset = test_dataloader.dataset.sampler.dset
        gid_to_vidspace_iou = {}
        gid_to_vidspace_iooa = {}
        for gid, slices in image_id_to_video_space_slices.items():
            vidid = coco_dset.index.imgs[gid]['video_id']
            video = coco_dset.index.videos[vidid]
            vidspace_gsd = video['target_gsd']
            video_poly = kwimage.Box.from_dsize((video['width'], video['height'])).to_polygon()
            # output_poly = video_poly.scale(scale)
            boxes = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(sl) for sl in slices])
            polys = boxes.to_polygons()
            covered = polys.unary_union().simplify(0.01)
            gid_to_vidspace_iooa[gid] = covered.iooa(video_poly)
            gid_to_vidspace_iou[gid] = covered.iou(video_poly)

        outspace_areas = []
        gid_to_outspace_iou = {}
        gid_to_outspace_iooa = {}
        for gid, slices in image_id_to_output_space_slices.items():
            boxes = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(sl) for sl in slices])
            polys = boxes.to_polygons()
            covered = polys.unary_union().simplify(0.01)
            outspace_areas.append(covered.area)

            output_space_scale = datamodule.config['output_space_scale']
            if output_space_scale != 'native':
                vidid = coco_dset.index.imgs[gid]['video_id']
                video = coco_dset.index.videos[vidid]
                vidspace_gsd = video['target_gsd']
                resolved_scale = data_utils.resolve_scale_request(
                    request=output_space_scale, data_gsd=vidspace_gsd)
                scale = resolved_scale['scale']
                video_poly = kwimage.Box.from_dsize((video['width'], video['height'])).to_polygon()
                output_poly = video_poly.scale(scale)
                boxes = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(sl) for sl in slices])
                polys = boxes.to_polygons()
                covered = polys.unary_union().simplify(0.01)
                gid_to_outspace_iooa[gid] = covered.iooa(output_poly)
                gid_to_outspace_iou[gid] = covered.iou(output_poly)

        print('outspace_rt_areas ' + repr(ub.dict_hist(np.sqrt(np.array(outspace_areas)))))
        vidspace_iou_stats = kwarray.stats_dict(
            list(gid_to_vidspace_iou.values()), n_extreme=True)
        vidspace_iooa_stats = kwarray.stats_dict(
            list(gid_to_vidspace_iooa.values()), n_extreme=True)

        outspace_iou_stats = kwarray.stats_dict(
            list(gid_to_outspace_iou.values()), n_extreme=True)
        outspace_iooa_stats = kwarray.stats_dict(
            list(gid_to_outspace_iooa.values()), n_extreme=True)
        print('vidspace_iou_stats = {}'.format(ub.repr2(vidspace_iou_stats, nl=1)))
        print('vidspace_iooa_stats = {}'.format(ub.repr2(vidspace_iooa_stats, nl=1)))
        print('outspace_iou_stats = {}'.format(ub.repr2(outspace_iou_stats, nl=1)))
        print('outspace_iooa_stats = {}'.format(ub.repr2(outspace_iooa_stats, nl=1)))

    proc_context.add_device_info(device)
    proc_context.stop()

    # Print logs about what we predicted on
    all_video_ids = list(result_dataset.videos())
    print(f'Requested predictions for {len(all_video_ids)} videos')
    stitched_video_histogram = ub.ddict(lambda: 0)
    stitched_video_patch_histogram = ub.ddict(lambda: 0)
    for _head_key, head_stitcher in stitch_managers.items():
        _histo = ub.dict_hist(result_dataset.images(head_stitcher._seen_gids).lookup('video_id'))
        print(f'stitched videos for {_head_key}={ub.urepr(_histo)}')

        for gid, v in head_stitcher._stitched_gid_patch_histograms.items():
            vidid = result_dataset.index.imgs[gid]['video_id']
            stitched_video_patch_histogram[vidid] += v

        for k, v in _histo.items():
            stitched_video_histogram[k] += v

    print('stitched_video_histogram = {}'.format(ub.urepr(stitched_video_histogram, nl=1)))
    print('stitched_video_patch_histogram = {}'.format(ub.urepr(stitched_video_patch_histogram, nl=1)))
    missing_vidids = set(all_video_ids) - set(stitched_video_histogram)
    if missing_vidids:
        print(f'missing_vidids={missing_vidids}')
    else:
        print('Made at least one prediction on each video')

    if args.drop_unused_frames:
        keep_gids = set()
        for manager in stitch_managers.values():
            keep_gids.update(manager.seen_image_ids)
        drop_gids = set(result_dataset.images()) - keep_gids
        print(f'Dropping {len(drop_gids)} unused frames')
        result_dataset.remove_images(drop_gids)

    # validate and save results
    if 0:
        print(result_dataset.validate())
    print('dump result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    result_dataset.dump(result_dataset.fpath)
    print('return result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    return result_dataset


def main(cmdline=True, **kwargs):
    predict(cmdline=cmdline, **kwargs)


if __name__ == '__main__':
    r"""
    Test old model:

    python -m watch.tasks.fusion.predict \
        --write_probs=True \
        --with_class=auto \
        --with_saliency=auto \
        --with_change=False \
        --package_fpath=/localdisk0/SCRATCH/watch/ben/smart_watch_dvc/training/raven/brodie/uky_invariants/features_22_03_14/runs/BASELINE_EXPERIMENT_V001/package.pt \
        --pred_dataset=/localdisk0/SCRATCH/watch/ben/smart_watch_dvc/training/raven/brodie/uky_invariants/features_22_03_14/runs/BASELINE_EXPERIMENT_V001/pred.kwcoco.json \
        --test_dataset=/localdisk0/SCRATCH/watch/ben/smart_watch_dvc/Drop2-Aligned-TA1-2022-02-15/data_nowv_vali.kwcoco.json \
        --num_workers=5 \
        --devices=0, \
        --batch_size=1

    Develop TTA:

    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc)
    (cd $DVC_DPATH && dvc pull -r aws $DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=19-step=13659-v1.pt.dvc)

    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc)
    MODEL_FNAME=models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=18-step=12976.pt
    MODEL_FPATH=$DVC_DPATH/$MODEL_FNAME
    smartwatch model_info $MODEL_FPATH
    (cd $DVC_DPATH && dvc pull -r aws $MODEL_FNAME)

    # Small datset for testing
    kwcoco subset \
        --src $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json \
        --dst $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1_small.kwcoco.json \
        --select_images '.frame_index < 100' \
        --select_videos '.name == "KR_R001"'

    # Small datset for testing
    kwcoco subset \
        --src $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali.kwcoco.json \
        --dst $DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1.kwcoco.json \
        --select_videos '.name == "KR_R001"'

    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc)
    TEST_DATASET=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1_small.kwcoco.json
    python -m watch.tasks.fusion.predict \
        --write_probs=True \
        --with_class=auto \
        --with_saliency=auto \
        --with_change=False \
        --package_fpath=$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=19-step=13659-v1.pt \
        --num_workers=5 \
        --devices=0, \
        --batch_size=1 \
        --exclude_sensors=L8 \
        --pred_dataset=$PRED_DATASET \
        --test_dataset=$TEST_DATASET \
        --tta_fliprot=0 \
        --tta_time=0 --dump=$DVC_DPATH/_tmp/test_pred_config.yaml


    # Testing first heterogeneous model

    DVC_EXPT_DPATH=$(WATCH_PREIMPORT=none smartwatch_dvc --tags='phase2_expt')
    DVC_DATA_DPATH=$(WATCH_PREIMPORT=none smartwatch_dvc --tags=phase2_data --hardware=ssd)
    PACKAGE_FPATH=$DVC_EXPT_DPATH/package_epoch10_step200000.pt
    TEST_DATASET=$DVC_DATA_DPATH/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/KR_R001.kwcoco.json
    PRED_DATASET=$DVC_EXPT_DPATH/_testing/hg_kr1/pred.kwcoco.json
    echo "
    DVC_EXPT_DPATH = $DVC_EXPT_DPATH
    DVC_DATA_DPATH = $DVC_DATA_DPATH
    PACKAGE_FPATH = $PACKAGE_FPATH
    TEST_DATASET = $TEST_DATASET
    PRED_DATASET = $PRED_DATASET
    "

    smartwatch model_stats "$PACKAGE_FPATH"

    python -m watch.tasks.fusion.predict \
        --with_class=auto \
        --with_saliency=auto \
        --package_fpath=$PACKAGE_FPATH \
        --num_workers=5 \
        --devices=0, \
        --batch_size=1 \
        --pred_dataset=$PRED_DATASET \
        --test_dataset=$TEST_DATASET

    """
    main()
