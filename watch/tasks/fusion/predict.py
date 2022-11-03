#!/usr/bin/env python
"""
Fusion prediction script.

TODO:
    - [ ] Prediction caching?
    - [ ] Reduce memory usage?
    - [ ] Pseudo Live.
"""
import torch  # NOQA
# import pathlib
import ubelt as ub
import numpy as np
from os.path import join
from os.path import relpath
import kwimage
import kwarray
import kwcoco
from watch.tasks.fusion import datamodules
from watch.tasks.fusion import utils
from watch.tasks.tracking.utils import mask_to_polygons
from watch.utils import util_path
from watch.utils import util_parallel
from watch.utils import util_kwimage
from watch.tasks.fusion.datamodules import data_utils
# APPLY Monkey Patches
from watch.tasks.fusion import monkey  # NOQA

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def make_predict_config(cmdline=False, **kwargs):
    """
    Configuration for fusion prediction
    """
    # TODO: switch to jsonargparse
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
    parser.add_argument('--devices', default=None, help='lightning devices')
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
        'chip_size',
        'chip_dims',
        'time_steps',
        'channels',
        'time_sampling',
        'time_span',
        'input_space_scale',
        'window_space_scale',
        'output_space_scale',
        'use_cloudmask',
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
    args, _ = parser.parse_known_args(default_args)
    args.datamodule_defaults = datamodule_defaults
    # assert args.batch_size == 1
    return args


def build_stitching_managers(config, method, result_dataset):
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
    need_infer = ub.udict({k: v for k, v in parsetime_vals.items() if v == 'auto' or v == ['auto']})
    # Try and infer what data we were given at train time
    if hasattr(method, 'fit_config'):
        traintime_params = method.fit_config
    elif hasattr(method, 'datamodule_hparams'):
        traintime_params = method.datamodule_hparams
    else:
        traintime_params = {}
        if datamodule_vars['channels'] in {None, 'auto'}:
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
                          hack_model_sensorchan_spec.normalize().spec,
                          'versus',
                          datamodule_sensorchan_spec.normalize().spec)

                    compat_parts = []
                    for model_part in hack_model_sensorchan_spec.streams():
                        data_part = datamodule_sensorchan_spec.matching_sensor(model_part.sensor.spec)
                        if not data_part.chans.spec:
                            # Try the generic sensor
                            data_part = datamodule_sensorchan_spec.matching_sensor('*')
                        isect_part = model_part.chans.intersection(data_part.chans)
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
        >>> args = None
        >>> cmdline = False
        >>> devices = None
        >>> test_dpath = ub.Path.appdir('watch/test/fusion/').ensuredir()
        >>> results_path = ub.ensuredir((test_dpath, 'predict'))
        >>> ub.delete(results_path)
        >>> ub.ensuredir(results_path)
        >>> package_fpath = join(test_dpath, 'my_test_package.pt')
        >>> import kwcoco
        >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes4-multispectral', num_frames=5, gsize=(128, 128))
        >>> test_dset = kwcoco.CocoDataset.demo('special:vidshapes2-multispectral', num_frames=5, gsize=(128, 128))
        >>> fit_kwargs = kwargs = {
        ...     'train_dataset': train_dset.fpath,
        ...     'datamodule': 'KWCocoVideoDataModule',
        ...     'workdir': ub.ensuredir((test_dpath, 'train')),
        ...     'package_fpath': package_fpath,
        ...     'max_epochs': 1,
        ...     'time_steps': 2,
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
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    config = args.__dict__.copy()
    datamodule_defaults = args.datamodule_defaults
    print('kwargs = {}'.format(ub.repr2(kwargs, nl=1)))
    print('config = {}'.format(ub.repr2(config, nl=2)))

    package_fpath = ub.Path(config['package_fpath'])

    try:
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
    except Exception as ex:
        print('ex = {!r}'.format(ex))
        print(f'Failed to read {package_fpath=!r} attempting workaround')
        # If we have a checkpoint path we can load it if we make assumptions
        # init method from checkpoint.
        raise

    # Hack to fix GELU issue
    monkey.fix_gelu_issue(method)

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
    result_dataset.fpath = str(config['pred_dataset'])
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

    if config['with_change'] == 'auto':
        config['with_change'] = getattr(method, 'global_change_weight', 1.0)
    if config['with_class'] == 'auto':
        config['with_class'] = getattr(method, 'global_class_weight', 1.0)
    if config['with_saliency'] == 'auto':
        config['with_saliency'] = getattr(method, 'global_saliency_weight', 0.0)

    stitch_managers = build_stitching_managers(config, method, result_dataset)

    expected_outputs = set(stitch_managers.keys())
    got_outputs = None
    writable_outputs = None

    print('Expected outputs: ' + str(expected_outputs))

    head_key_mapping = {
        'saliency_probs': 'saliency',
        'class_probs': 'class',
        'change_probs': 'change',
    }

    batch_iter = iter(test_dataloader)
    prog = ub.ProgIter(batch_iter, desc='predicting', verbose=1)

    # Start background procs before we make threads
    writer_queue = util_parallel.BlockingJobQueue(
        mode='thread',
        # mode='serial',
        max_workers=datamodule.num_workers)

    result_fpath.parent.ensuredir()
    print('result_fpath = {!r}'.format(result_fpath))

    CHECK_GRID = 0
    if CHECK_GRID:
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
        for gid, slices in image_id_to_target_space_slices.items():
            vidid = coco_dset.index.imgs[gid]['video_id']
            video = coco_dset.index.videos[vidid]
            video_poly = util_kwimage.Box.from_dsize((video['width'], video['height'])).to_polygon()
            boxes = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(sl) for sl in slices])
            polys = boxes.to_polygons()
            covered = polys.unary_union().simplify(0.01)
            iou = covered.iou(video_poly)
            gid_to_iou[gid] = iou
        ious = list(gid_to_iou.values())
        iou_stats = kwarray.stats_dict(ious, n_extreme=True)
        print('iou_stats = {}'.format(ub.repr2(iou_stats, nl=1)))

    CHECK_PRED_SPATIAL_COVERAGE = 0
    if CHECK_PRED_SPATIAL_COVERAGE:
        # Enable debugging to ensure the dataloader actually passed
        # us the targets that cover the entire image.
        image_id_to_input_space_slices = ub.ddict(list)
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
        args=jsonified_args,
        config=config_resolved,
        track_emissions=config['track_emissions'],
        extra={'fit_config': traintime_params}
    )

    assert not list(util_json.find_json_unserializable(proc_context.obj))

    info.append(proc_context.obj)
    proc_context.start()
    proc_context.add_disk_info(test_coco_dataset.fpath)

    with torch.set_grad_enabled(False):
        # FIXME: that data loader should not be producing incorrect sensor/mode
        # pairs in the first place!
        EMERGENCY_INPUT_AGREEMENT_HACK = 1 and hasattr(method, 'input_norms')
        # prog.set_extra(' <will populate stats after first video>')
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
            outputs = method.forward_step(batch, with_loss=False)

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

                        if CHECK_PRED_SPATIAL_COVERAGE:
                            image_id_to_input_space_slices[gid].append(target['space_slice'])
                            image_id_to_output_space_slices[gid].append(output_space_slice)

                        head_stitcher.accumulate_image(
                            gid, output_space_slice, probs,
                            dsize=output_image_dsize,
                            scale=scale_outspace_from_vid)

                # Free up space for any images that have been completed
                for gid in head_stitcher.ready_image_ids():
                    head_stitcher._ready_gids.difference_update({gid})  # avoid race condition
                    writer_queue.submit(head_stitcher.finalize_image, gid)

        writer_queue.wait_until_finished()  # hack to avoid race condition

        # Prediction is completed, finalize all remaining images.
        for _head_key, head_stitcher in stitch_managers.items():
            for gid in head_stitcher.managed_image_ids():
                writer_queue.submit(head_stitcher.finalize_image, gid)
        writer_queue.wait_until_finished()

    if CHECK_PRED_SPATIAL_COVERAGE:
        coco_dset = test_dataloader.dataset.sampler.dset
        gid_to_iou = {}
        for gid, slices in image_id_to_input_space_slices.items():
            vidid = coco_dset.index.imgs[gid]['video_id']
            video = coco_dset.index.videos[vidid]
            video_poly = util_kwimage.Box.from_dsize((video['width'], video['height'])).to_polygon()
            boxes = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(sl) for sl in slices])
            polys = boxes.to_polygons()
            covered = polys.unary_union().simplify(0.01)
            iou = covered.iou(video_poly)
            gid_to_iou[gid] = iou

        outspace_areas = []
        for gid, slices in image_id_to_output_space_slices.items():
            boxes = kwimage.Boxes.concatenate([kwimage.Boxes.from_slice(sl) for sl in slices])
            polys = boxes.to_polygons()
            covered = polys.unary_union().simplify(0.01)
            outspace_areas.append(covered.area)

        print('outspace_rt_areas ' + repr(ub.dict_hist(np.sqrt(np.array(outspace_areas)))))
        ious = list(gid_to_iou.values())
        iou_stats = kwarray.stats_dict(ious, n_extreme=True)
        print('input_space iou_stats = {}'.format(ub.repr2(iou_stats, nl=1)))

    proc_context.add_device_info(device)
    proc_context.stop()

    # validate and save results
    print(result_dataset.validate())
    print('dump result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    result_dataset.dump(result_dataset.fpath)
    print('return result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    return result_dataset


class CocoStitchingManager(object):
    """
    Manage stitching for multiple images / videos in a CocoDataset.

    This is done in a memory-efficient way where after all sub-regions in an
    image or video have been completed, it is finalized, written to the kwcoco
    manifest / disk, and the memory used for stitching is freed.

    Args:
        result_dataset (CocoDataset):
            The CocoDataset that is being predicted on. This will be modified
            when an image prediction is finalized.

        short_code (str):
            short identifier used for directory names.

        chan_code (str):
            If saving the stitched features, this is the channel code to use.

        stiching_space (str):
            Indicates if the results are given in image or video space.

        device ('numpy' | torch.device):
            Device to stitch on.

        thresh (float):
            if making hard decisions, determines the threshold for converting a
            soft mask into a hard mask, which can be converted into a polygon.

        prob_compress (str):
            Compression algorithm to use when writing probabilities to disk.
            Can be any GDAL compression code, e.g LZW, DEFLATE, RAW, etc.

        polygon_categories (List[str] | None):
            These are the list of channels that should be transformed into
            polygons. If not set, all are used.

        quantize (bool):
            if True quantize heatmaps before writing them to disk

    TODO:
        - [ ] Handle the case where the input space is related to the output
              space by an affine transform.

        - [ ] Handle stitching in image space

        - [ ] Handle the case where we are only stitching over images

        - [ ] Handle the case where iteration is non-contiguous, i.e. define
              a robust criterion to determine when an image is "done" being
              stitched.

        - [ ] Perhaps separate the "soft-probability" prediction stitcher
              from (a) the code that converts soft-to-hard predictions (b)
              the code that adds hard predictions to the kwcoco file and (c)
              the code that adds soft predictions to the kwcoco file?
    """

    def __init__(self, result_dataset, short_code=None, chan_code=None,
                 stiching_space='video', device='numpy', thresh=0.5,
                 write_probs=True, write_preds=True, num_bands='auto',
                 prob_compress='DEFLATE', polygon_categories=None,
                 quantize=True):
        self.short_code = short_code
        self.result_dataset = result_dataset
        self.device = device
        self.chan_code = chan_code
        self.thresh = thresh
        self.num_bands = num_bands
        self.prob_compress = prob_compress
        self.polygon_categories = polygon_categories
        self.quantize = quantize

        self.suffix_code = (
            self.chan_code if '|' not in self.chan_code else
            ub.hash_data(self.chan_code)[0:16]
        )

        self.stiching_space = stiching_space
        if stiching_space != 'video':
            raise NotImplementedError(stiching_space)

        # Setup a dictionary that we will use to make a stitcher for each image
        # as needed.  We use the fact that videos are iterated over
        # sequentially so free up memory of a video after it completes.
        self.image_stitchers = {}
        self.image_scales = {}  # TODO: should be a more general transform
        self._seen_gids = set()
        self._last_vidid = None
        self._ready_gids = set()

        # TODO: writing predictions and probabilities needs robustness work
        self.write_probs = write_probs
        self.write_preds = write_preds

        if self.write_preds:
            from kwcoco import channel_spec
            chan_spec = channel_spec.FusedChannelSpec.coerce(chan_code)
            if self.polygon_categories is None:
                self.polygon_categories = chan_spec.parsed
            # Determine the indexes that we will use for polygon extraction
            _idx_lut = {c: idx for idx, c in enumerate(chan_spec.parsed)}
            self.polygon_idxs = [_idx_lut[c] for c in self.polygon_categories]

        if self.write_probs:
            bundle_dpath = self.result_dataset.bundle_dpath
            prob_subdir = f'_assets/{self.short_code}'
            self.prob_dpath = join(bundle_dpath, prob_subdir)
            ub.ensuredir(self.prob_dpath)

    def accumulate_image(self, gid, space_slice, data, dsize=None, scale=None):
        """
        Stitches a result into the appropriate image stitcher.

        Args:
            gid (int):
                the image id to stitch into

            space_slice (int):
                the slice (in "stitching-space") the data corresponds to.

            data (ndarray | Tensor): the feature or probability data

            dsize (Tuple): the w/h of outputspace

            scale (float | None): the scale to the outspace from from the vidspace
        """
        data = kwarray.atleast_nd(data, 3)
        dset = self.result_dataset
        if self.stiching_space == 'video':
            vidid = dset.index.imgs[gid]['video_id']
            # Create the stitcher if it does not exist
            if gid not in self.image_stitchers:
                if dsize is None:
                    video = dset.index.videos[vidid]
                    height, width = video['height'], video['width']
                else:
                    width, height = dsize
                if self.num_bands == 'auto':
                    if len(data.shape) == 3:
                        self.num_bands = data.shape[2]
                    else:
                        raise NotImplementedError
                # stitch_dims = (width, height, self.num_bands)
                stitch_dims = (height, width, self.num_bands)
                self.image_stitchers[gid] = kwarray.Stitcher(
                    stitch_dims, device=self.device)
                self.image_scales[gid] = scale

            if self._last_vidid is not None and vidid != self._last_vidid:
                # We assume sequential video iteration, thus when we see a new
                # video, we know the images from the previous video are ready.
                video_gids = set(dset.index.vidid_to_gids[self._last_vidid])
                ready_gids = video_gids & set(self.image_stitchers)

                # TODO
                # do something clever to know if frames are ready early?
                # might be tricky in general if we run over multiple
                # times per image with different frame samplings.
                # .
                # TODO: we know if an image is done if all of the samples that
                # contain it have been processed. (although that does not
                # account for dynamic resampling)
                self._ready_gids.update(ready_gids)

            self._last_vidid = vidid
        else:
            raise NotImplementedError(self.stiching_space)

        stitcher: kwarray.Stitcher = self.image_stitchers[gid]

        self._stitcher_center_weighted_add(stitcher, space_slice, data)

    @staticmethod
    def _stitcher_center_weighted_add(stitcher, space_slice, data):
        """
        TODO: refactor
        """
        weights = util_kwimage.upweight_center_mask(data.shape[0:2])

        is_2d = len(data.shape) == 2
        is_3d = len(data.shape) == 3

        if is_3d:
            weights = weights[..., None]

        if stitcher.shape[0] < space_slice[0].stop or stitcher.shape[1] < space_slice[1].stop:
            # By embedding the space slice in the stitcher dimensions we can get a
            # slice corresponding to the valid region in the stitcher, and the extra
            # padding encode the valid region of the data we are trying to stitch into.
            subslice, padding = kwarray.embed_slice(space_slice[0:2], stitcher.shape[0:2])
            output_slice = (
                slice(padding[0][0], data.shape[0] - padding[0][1]),
                slice(padding[1][0], data.shape[1] - padding[1][1]),
            )
            subdata = data[output_slice]
            subweights = weights[output_slice]

            stitch_slice = subslice
            stitch_data = subdata
            stitch_weights = subweights
        else:
            # Normal case
            stitch_slice = space_slice
            stitch_data = data
            stitch_weights = weights

        # Handle stitching nan values
        invalid_output_mask = np.isnan(stitch_data)
        if np.any(invalid_output_mask):
            if is_3d:
                spatial_valid_mask = (1 - invalid_output_mask.any(axis=2, keepdims=True))
            else:
                assert is_2d
                spatial_valid_mask = (1 - invalid_output_mask)
            stitch_weights = stitch_weights * spatial_valid_mask
            stitch_data[invalid_output_mask] = 0

        stitch_slice = fix_slice(stitch_slice)

        HACK_FIX_SHAPE = 1
        if HACK_FIX_SHAPE:
            # Something is causing an off by one error, not sure what it is
            # this hack just forces the slice to agree.
            dh, dw = stitch_data.shape[0:2]
            box = util_kwimage.Box.from_slice(stitch_slice)
            sw, sh = box.dsize
            if sw > dw:
                box = box.resize(width=dw)
            if sh > dh:
                box = box.resize(height=dh)
            if sw < dw:
                stitch_data = stitch_data[:, 0:sw]
                stitch_weights = stitch_weights[:, 0:sw]
            if sh < dh:
                stitch_data = stitch_data[0:sh]
                stitch_weights = stitch_weights[0:sh]
            stitch_slice = box.to_slice()

        try:
            stitcher.add(stitch_slice, stitch_data, weight=stitch_weights)
        except IndexError:
            print(f'stitch_slice={stitch_slice}')
            print(f'stitch_weights.shape={stitch_weights.shape}')
            print(f'stitch_data.shape={stitch_data.shape}')
            raise

    def managed_image_ids(self):
        """
        Return all image ids that are being managed and may be completed or in
        the process of stitching.

        Returns:
            List[int]: image ids
        """
        return list(self.image_stitchers.keys())

    def ready_image_ids(self):
        """
        Returns all image-ids that are known to be ready to finalize.

        Returns:
            List[int]: image ids
        """
        return list(self._ready_gids)

    def finalize_image(self, gid):
        """
        Finalizes the stitcher for this image, deletes it, and adds
        its hard and/or soft predictions to the CocoDataset.

        Args:
            gid (int): the image-id to finalize
        """
        # Remove this image from the managed set.
        img = self.result_dataset.index.imgs[gid]

        self._ready_gids.difference_update({gid})

        try:
            stitcher = self.image_stitchers.pop(gid)
        except KeyError:
            if gid in self._seen_gids:
                raise KeyError((
                    'Attempted to finalize image gid={}, but we already '
                    'finalized it').format(gid))
            else:
                raise KeyError('Attempted to finalize image gid={}, but no data was ever accumulated for it'.format(gid))
                raise KeyError((
                    'Attempted to finalize image gid={}, but no data '
                    'was ever accumulated for it ').format(gid))

        self._seen_gids.add(gid)

        scale_asset_from_vidspace = self.image_scales.pop(gid)

        # Get the final stitched feature for this image
        final_probs = stitcher.finalize()
        final_probs = kwarray.atleast_nd(final_probs, 3)
        # is_nodata = np.isnan(final_probs)
        # final_probs = np.nan_to_num(final_probs)

        final_weights = kwarray.atleast_nd(stitcher.weights, 3)
        is_predicted_pixel = final_weights.any(axis=2).astype('uint8')

        # NOTE: could find and record the valid prediction regions.
        # Given a (rectilinear) non-convex multipolygon where we are guarenteed
        # that all of the angles in the polygon are right angles, what is an
        # efficient algorithm to decompose it into a minimal set of disjoint
        # rectangles?
        # https://stackoverflow.com/questions/5919298/algorithm-for-finding-the-fewest-rectangles-to-cover-a-set-of-rectangles-without/6634668#6634668
        # Or... just write out a polygon... KISS
        _mask = kwimage.Mask(is_predicted_pixel, 'c_mask')
        _poly = _mask.to_multi_polygon()
        predicted_region = _poly.to_geojson()
        # Mark that we made a prediction on this image.
        img['prediction_region'] = predicted_region
        img['has_predictions'] = ub.dict_union(img.get('has_predictions', {}), {self.chan_code: True})

        # Get spatial relationship between the image and the video
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        img_from_vid = vid_from_img.inv()

        n_anns = 0
        total_prob = 0

        if self.write_probs:
            # This currently exists as an example to demonstrate how a
            # prediction script can write a pre-fusion TA-2 feature to disk and
            # register it with the kwcoco file.
            #
            # Save probabilities (or feature maps) as a new auxiliary image
            bundle_dpath = self.result_dataset.bundle_dpath
            new_fname = img.get('name', str(img['id'])) + f'_{self.suffix_code}.tif'  # FIXME
            new_fpath = join(self.prob_dpath, new_fname)
            assert final_probs.shape[2] == (self.chan_code.count('|') + 1)
            vid_from_asset = kwimage.Affine.coerce(scale=scale_asset_from_vidspace).inv()
            img_from_asset = img_from_vid @ vid_from_asset
            aux = {
                'file_name': relpath(new_fpath, bundle_dpath),
                'channels': self.chan_code,
                'height': final_probs.shape[0],
                'width': final_probs.shape[1],
                'num_bands': final_probs.shape[2],
                'warp_aux_to_img': img_from_asset.concise(),
            }
            auxiliary = img.setdefault('auxiliary', [])
            auxiliary.append(aux)

            # Save the prediction to disk
            total_prob += np.nansum(final_probs)

            write_kwargs = {}
            write_kwargs['blocksize'] = 128
            write_kwargs['compress'] = self.prob_compress

            if 'wld_crs_info' in img:
                from osgeo import osr
                # TODO: would be nice to have an easy to use mechanism to get
                # the gdal crs, probably one exists in pyproj.
                auth = img['wld_crs_info']['auth']
                assert auth[0] == 'EPSG', 'unhandled auth'
                epsg = auth[1]
                axis_strat = getattr(osr, img['wld_crs_info']['axis_mapping'])
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(int(epsg))
                srs.SetAxisMappingStrategy(axis_strat)
                img_from_wld = kwimage.Affine.coerce(img['wld_to_pxl'])
                wld_from_img = img_from_wld.inv()
                wld_from_asset = wld_from_img @ img_from_asset
                write_kwargs['crs'] = srs.ExportToWkt()
                write_kwargs['transform'] = wld_from_asset
                write_kwargs['overviews'] = 2

            if self.quantize:
                # Quantize
                quant_probs, quantization = quantize_float01(final_probs)
                aux['quantization'] = quantization

                kwimage.imwrite(
                    str(new_fpath), quant_probs, space=None, backend='gdal',
                    nodata=quantization['nodata'], **write_kwargs,
                )
            else:
                kwimage.imwrite(
                    str(new_fpath), final_probs, space=None, backend='gdal',
                    **write_kwargs,
                )

        if self.write_preds:
            # NOTE: The typical pipeline will never do this.
            # This is generally reserved for a subsequent tracking stage.

            # This is the final step where we convert soft-probabilities to
            # hard-polygons, we need to choose an good operating point here.

            # HACK: We happen to know this is the category atm.
            # Should have a better way to determine it via metadata

            for catname, band_idx in zip(self.polygon_categories, self.polygon_idxs):
                cid = self.result_dataset.ensure_category(catname)

                band_probs = final_probs[..., band_idx]
                # Threshold scores (todo: could be per class)
                thresh = self.thresh
                # Convert to polygons
                scored_polys = list(mask_to_polygons(
                    probs=band_probs, thresh=thresh, scored=True,
                    use_rasterio=False))
                n_anns = len(scored_polys)
                for score, vid_poly in scored_polys:
                    # Transform the video polygon into image space
                    img_poly = vid_poly.warp(img_from_vid)
                    bbox = list(img_poly.bounding_box().to_coco())[0]
                    # Add the polygon as an annotation on the image
                    self.result_dataset.add_annotation(
                        image_id=gid, category_id=cid,
                        bbox=bbox, segmentation=img_poly, score=score)

        info = {
            'n_anns': n_anns,
            'total_prob': total_prob,
        }
        return info


def write_new_asset(img, new_raster):
    """
    Given a new raster and an coco image it belongs to, write the raster to
    disk and update that image's asset / auxiliary list.

    Args:
        img (dict): the coco image dictionary
        new_raster (ndarray): the data to write
    """
    pass


def quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.int16):
    """
    Note:
        Setting old_min / old_max indicates the possible extend of the input
        data (and it will be clipped to it). It does not mean that the input
        data has to have those min and max values, but it should be between
        them.

    Example:
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> from delayed_image.helpers import dequantize
        >>> # Test error when input is not nicely between 0 and 1
        >>> imdata = (np.random.randn(32, 32, 3) - 1.) * 2.5
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))
        >>> #
        >>> for i in range(1, 20):
        >>>     print('i = {!r}'.format(i))
        >>>     quant2, quantization2 = quantize_float01(imdata, old_min=-i, old_max=i)
        >>>     recon2 = dequantize(quant2, quantization2)
        >>>     error2 = np.abs((recon2 - imdata)).sum()
        >>>     print('error2 = {!r}'.format(error2))

    Example:
        >>> # Test dequantize with uint8
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> from delayed_image.helpers import dequantize
        >>> imdata = np.random.randn(32, 32, 3)
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.uint8)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))

    Example:
        >>> # Test quantization with different signed / unsigned combos
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> print(quantize_float01(None, 0, 1, np.int16))
        >>> print(quantize_float01(None, 0, 1, np.int8))
        >>> print(quantize_float01(None, 0, 1, np.uint8))
        >>> print(quantize_float01(None, 0, 1, np.uint16))

    """
    # old_min = 0
    # old_max = 1
    quantize_iinfo = np.iinfo(quantize_dtype)
    quantize_max = quantize_iinfo.max
    if quantize_iinfo.kind == 'u':
        # Unsigned quantize
        quantize_nan = 0
        quantize_min = 1
    elif quantize_iinfo.kind == 'i':
        # Signed quantize
        quantize_min = 0
        quantize_nan = max(-9999, quantize_iinfo.min)

    quantization = {
        'orig_min': old_min,
        'orig_max': old_max,
        'quant_min': quantize_min,
        'quant_max': quantize_max,
        'nodata': quantize_nan,
    }

    old_extent = (old_max - old_min)
    new_extent = (quantize_max - quantize_min)
    quant_factor = new_extent / old_extent

    if imdata is not None:
        invalid_mask = np.isnan(imdata)
        new_imdata = (imdata.clip(old_min, old_max) - old_min) * quant_factor + quantize_min
        new_imdata = new_imdata.astype(quantize_dtype)
        new_imdata[invalid_mask] = quantize_nan
    else:
        new_imdata = None

    return new_imdata, quantization


def main(cmdline=True, **kwargs):
    predict(cmdline=cmdline, **kwargs)


def _fix_int(d):
    return None if d is None else int(d)


def _fix_slice(d):
    return slice(_fix_int(d.start), _fix_int(d.stop), _fix_int(d.step))


def _fix_slice_tup(sl):
    return tuple(map(_fix_slice, sl))


def fix_slice(sl):
    if isinstance(sl, slice):
        return _fix_slice(sl)
    elif isinstance(sl, (tuple, list)) and isinstance(ub.peek(sl), slice):
        return _fix_slice_tup(sl)
    else:
        raise TypeError(repr(sl))


if __name__ == '__main__':
    r"""
    Test old model:

    python -m watch.tasks.fusion.predict \
        --write_probs=True \
        --write_preds=False \
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
        --write_preds=False \
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

    """
    main()
