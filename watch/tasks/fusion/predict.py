#!/usr/bin/env python
"""
Fusion prediction script.

TODO:
    - [ ] Prediction caching?
"""
import torch  # NOQA
import pathlib
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
from watch.tasks.fusion.datamodules.kwcoco_video_data import inv_fliprot

try:
    from xdev import profile
except Exception:
    profile = ub.identity


def make_predict_config(cmdline=False, **kwargs):
    """
    Configuration for fusion prediction
    """
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
    parser.add_argument('--pred_dataset', default=None, dest='pred_dataset')

    # parser.add_argument('--pred_dpath', dest='pred_dpath', type=pathlib.Path, help='path to dump results. Deprecated, do not use.')

    parser.add_argument('--package_fpath', type=str)
    parser.add_argument('--gpus', default=None, help='todo: hook up to lightning')
    parser.add_argument('--thresh', type=smartcast, default=0.01)

    parser.add_argument('--with_change', type=smartcast, default='auto')
    parser.add_argument('--with_class', type=smartcast, default='auto')
    parser.add_argument('--with_saliency', type=smartcast, default='auto')

    parser.add_argument('--compress', type=str, default='DEFLATE', help='type of compression for prob images')
    parser.add_argument('--track_emissions', type=smartcast, default=True, help='set to false to disable emission tracking')

    parser.add_argument('--quantize', type=smartcast, default=True, help='quantize outputs')

    parser.add_argument('--tta_fliprot', type=smartcast, default=0, help='number of times to flip/rotate the frame, can be in [0,7]')
    parser.add_argument('--tta_time', type=smartcast, default=0, help='number of times to expand the temporal sample for a frame'),

    # TODO
    # parser.add_argument('--test_time_augmentation', default=False, help='')

    parser.add_argument(
        '--write_preds', default=True, type=smartcast, help=ub.paragraph(
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
        'time_steps',
        'channels',
        'time_sampling',
        'time_span',
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


@profile
def predict(cmdline=False, **kwargs):
    """
    Example:
        >>> # Train a demo model (in the future grab a pretrained demo model)
        >>> from watch.tasks.fusion.fit import fit_model  # NOQA
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> gpus = None
        >>> test_dpath = ub.ensure_app_cache_dir('watch/test/fusion/')
        >>> results_path = ub.ensuredir((test_dpath, 'predict'))
        >>> ub.delete(results_path)
        >>> ub.ensuredir(results_path)
        >>> package_fpath = join(test_dpath, 'my_test_package.pt')
        >>> import kwcoco
        >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes4-multispectral', num_frames=5, gsize=(128, 128))
        >>> test_dset = kwcoco.CocoDataset.demo('special:vidshapes2-multispectral', num_frames=5, gsize=(128, 128))
        >>> fit_kwargs = kwargs = {
        ...     'train_dataset': test_dset.fpath,
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
        ...     'gpus': gpus,
        ... }
        >>> package_fpath = fit_model(**fit_kwargs)
        >>> # Predict via that model
        >>> predict_kwargs = kwargs = {
        >>>     'package_fpath': package_fpath,
        >>>     'pred_dataset': results_path / 'pred.kwcoco.json',
        >>>     'test_dataset': test_dset.fpath,
        >>>     'datamodule': 'KWCocoVideoDataModule',
        >>>     'batch_size': 1,
        >>>     'num_workers': 0,
        >>>     'gpus': gpus,
        >>> }
        >>> result_dataset = predict(**kwargs)
        >>> dset = result_dataset
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
        >>> pred1 = coco_img.delay('salient').finalize(nodata='float')
        >>> pred2 = coco_img.delay('salient').finalize(nodata='float', dequantize=False)
        >>> assert pred1.max() <= 1
        >>> assert pred2.max() > 1
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    config = args.__dict__
    print('config = {}'.format(ub.repr2(config, nl=2)))

    package_fpath = ub.Path(args.package_fpath)

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

        checkpoint = torch.load(package_fpath)
        print(list(checkpoint.keys()))
        from watch.tasks.fusion import methods
        hparams = checkpoint['hyper_parameters']
        if 'input_channels' in hparams:
            from kwcoco.channel_spec import ChannelSpec
            # Hack for strange pickle issue
            chan = hparams['input_channels']
            if not hasattr(chan, '_spec') and hasattr(chan, '_info'):
                chan = ChannelSpec.coerce(chan._info['spec'])
                hparams['input_channels'] = chan
            else:
                hparams['input_channels'] = ChannelSpec.coerce(chan.spec)

        method = methods.MultimodalTransformer(**hparams)
        state_dict = checkpoint['state_dict']
        method.load_state_dict(state_dict)

    method.eval()
    method.freeze()

    # TODO: perhaps we should enforce that that packaged model
    # knows how to construct the appropriate test dataset?

    # init datamodule from args
    datamodule_class = getattr(datamodules, args.datamodule)
    datamodule_vars = ub.compatible(
        vars(args),
        datamodule_class.__init__,
    )

    parsetime_vals = ub.dict_isect(datamodule_vars, args.datamodule_defaults)
    need_infer = {k: v for k, v in parsetime_vals.items() if v == 'auto'}
    # Try and infer what data we were given at train time
    if hasattr(method, 'fit_config'):
        traintime_params = method.fit_config
    elif hasattr(method, 'datamodule_hparams'):
        traintime_params = method.datamodule_hparams
    else:
        traintime_params = {}
        if datamodule_vars['channels'] in {None, 'auto'}:
            print('Warning have to make assumptions. Might not always work')
            if hasattr(method, 'input_channels'):
                # note input_channels are sometimes different than the channels the
                # datamodule expects. Depending on special keys and such.
                traintime_params['channels'] = method.input_channels.spec
            else:
                traintime_params['channels'] = list(method.input_norms.keys())[0]

    # FIXME: Some of the inferred args seem to not have the right type here.
    able_to_infer = ub.dict_isect(traintime_params, need_infer)
    if able_to_infer.get('channels', None) is not None:
        # do this before smartcast breaks the spec
        able_to_infer['channels'] = kwcoco.ChannelSpec.coerce(able_to_infer['channels'])
    from scriptconfig.smartcast import smartcast
    able_to_infer = ub.map_vals(smartcast, able_to_infer)
    unable_to_infer = ub.dict_diff(need_infer, traintime_params)
    # Use defaults when we can't infer
    overloads = able_to_infer.copy()
    overloads.update(ub.dict_isect(args.datamodule_defaults, unable_to_infer))
    datamodule_vars.update(overloads)
    print('able_to_infer = {}'.format(ub.repr2(able_to_infer, nl=1)))
    print('unable_to_infer = {}'.format(ub.repr2(unable_to_infer, nl=1)))
    print('overloads = {}'.format(ub.repr2(overloads, nl=1)))

    deviation = ub.varied_values([
        ub.dict_isect(traintime_params, datamodule_vars),
        ub.dict_isect(datamodule_vars, traintime_params),
    ], min_variations=1)
    print('deviation from fit->predict settings = {}'.format(ub.repr2(deviation, nl=1)))

    HACK_FIX_MODELS_WITH_BAD_CHANNEL_SPEC = True
    if HACK_FIX_MODELS_WITH_BAD_CHANNEL_SPEC:
        # There was an issue where we trained models and specified
        # r|g|b|mat:0.3 but we only passed data with r|g|b. At train time
        # current logic (whch we need to fix) will happilly just take a subset
        # of those channels, which means the recorded channels disagree with
        # what the model was actually trained with.
        if hasattr(method, 'sensor_channel_tokenizers'):
            datamodule_channel_spec = datamodule_vars['channels']
            unique_channel_streams = ub.oset()
            for sensor, tokenizers in method.sensor_channel_tokenizers.items():
                for code in tokenizers.keys():
                    unique_channel_streams.add(code)
            hack_model_spec = kwcoco.ChannelSpec.coerce(','.join(unique_channel_streams))
            if datamodule_channel_spec is not None:
                if hack_model_spec != datamodule_channel_spec:
                    print('Warning: reported model channels may be incorrect '
                          'due to bad train hyperparams')
                    hack_common = hack_model_spec.intersection(datamodule_channel_spec)
                    datamodule_vars['channels'] = hack_common

    DZYNE_MODEL_HACK = 1
    if DZYNE_MODEL_HACK:
        if package_fpath.stem == 'lc_rgb_fusion_model_package':
            # This model has an issue with the L8 features it was trained on
            datamodule_vars['exclude_sensors'] = ['L8']

    datamodule = datamodule_class(
        **datamodule_vars
    )
    # TODO: if TTA=True, disable determenistic time sampling
    datamodule.setup('test')

    if config['tta_time']:
        # Expand targets to include time augmented samples
        n_time_expands = config['tta_time']
        test_torch_dset = datamodule.torch_datasets['test']
        test_torch_dset._expand_targets_time(n_time_expands)

    if config['tta_fliprot']:
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

    test_coco_dataset = datamodule.coco_datasets['test']

    test_torch_dataset = datamodule.torch_datasets['test']
    # hack this setting
    test_torch_dataset.inference_only = True
    test_dataloader = datamodule.test_dataloader()

    T, H, W = test_torch_dataset.sample_shape

    # Create the results dataset as a copy of the test CocoDataset
    result_dataset: kwcoco.CocoDataset = test_coco_dataset.copy()
    # Remove all annotations in the results copy
    result_dataset.clear_annotations()
    # Change all paths to be absolute paths
    result_dataset.reroot(absolute=True)
    # Set the filepath for the prediction coco file
    # (modifies the bundle_dpath)
    # if args.pred_dataset is None:
    #     pred_dpath = util_path.coercepath(args.pred_dpath)
    #     result_dataset.fpath = str(pred_dpath / 'pred.kwcoco.json')
    # else:
    if not args.pred_dataset:
        raise ValueError(
            f'Must specify path to the output (predicted) kwcoco file. '
            f'Got {args.pred_dataset=}')
    result_dataset.fpath = str(args.pred_dataset)
    result_fpath = util_path.coercepath(result_dataset.fpath)

    # add hyperparam info to "info" section
    info = result_dataset.dataset.get('info', [])

    from kwcoco.util import util_json
    import os
    import socket
    jsonified_args = util_json.ensure_json_serializable(args.__dict__)
    # This will be serailized in kwcoco, so make sure it can be coerced to json
    walker = ub.IndexableWalker(jsonified_args)
    for problem in util_json.find_json_unserializable(jsonified_args):
        bad_data = problem['data']
        if isinstance(bad_data, kwcoco.CocoDataset):
            fixed_fpath = getattr(bad_data, 'fpath', None)
            if fixed_fpath is not None:
                walker[problem['loc']] = fixed_fpath
            else:
                walker[problem['loc']] = '<IN_MEMORY_DATASET: {}>'.format(
                    bad_data._build_hashid())

    start_timestamp = ub.timestamp()

    info.append({
        'type': 'process',
        'properties': {
            'name': 'watch.tasks.fusion.predict',
            'args': jsonified_args,
            'hostname': socket.gethostname(),
            'cwd': os.getcwd(),
            'userhome': ub.userhome(),
            'timestamp': start_timestamp,
            'fit_config': traintime_params,
        }
    })

    result_fpath.parent.mkdir(parents=True, exist_ok=True)

    from watch.utils.lightning_ext import util_device
    print('args.gpus = {!r}'.format(args.gpus))
    devices = util_device.coerce_devices(args.gpus)
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

    stitch_managers = {}

    if args.with_change == 'auto':
        args.with_change = getattr(method, 'global_change_weight', 1.0)
    if args.with_class == 'auto':
        args.with_class = getattr(method, 'global_class_weight', 1.0)
    if args.with_saliency == 'auto':
        args.with_saliency = getattr(method, 'global_saliency_weight', 0.0)

    # could be torch on-device stitching
    stitch_device = 'numpy'

    ignore_classes = {
        'not_salient', 'ignore', 'background', 'Unknown'}
    # hack, not general
    ignore_classes.update({'negative', 'positive'})

    stitcher_common_kw = dict(
        stiching_space='video',
        device=stitch_device,
        thresh=args.thresh,
        write_probs=args.write_probs,
        write_preds=args.write_preds,
        prob_compress=args.compress,
        quantize=args.quantize,
    )

    # If we only care about some predictions from the model, then keep track of
    # the class indices we need to take.
    task_keep_indices = {}
    if args.with_change:
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
        result_dataset.ensure_category('change')

    if args.with_class:
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

    if args.with_saliency:
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

    expected_outputs = set(stitch_managers.keys())
    got_outputs = None
    writable_outputs = None

    print('Expected outputs: ' + str(expected_outputs))

    head_key_mapping = {
        'saliency_probs': 'saliency',
        'class_probs': 'class',
        'change_probs': 'change',
    }

    # Start background procs before we make threads
    batch_iter = iter(test_dataloader)
    writer_queue = util_parallel.BlockingJobQueue(
        mode='thread',
        # mode='serial',
        max_workers=datamodule.num_workers)

    prog = ub.ProgIter(batch_iter, desc='predicting', verbose=1)

    try:
        if args.track_emissions:
            from codecarbon import EmissionsTracker
            emissions_tracker = EmissionsTracker()
            emissions_tracker.start()
        else:
            emissions_tracker = None
    except Exception as ex:
        if args.track_emissions:
            print('ex = {!r}'.format(ex))
        emissions_tracker = None

    with torch.set_grad_enabled(False):

        # FIXME: that data loader should not be producing incorrect sensor/mode
        # pairs in the first place!
        EMERGENCY_INPUT_AGREEMENT_HACK = True

        # prog.set_extra(' <will populate stats after first video>')
        _batch_iter = iter(prog)
        for orig_batch in _batch_iter:
            batch_regions = []
            # Move data onto the prediction device, grab spacetime region info
            fixed_batch = []
            for item in orig_batch:
                if item is None:
                    continue
                batch_regions.append({
                    'space_slice': tuple(item['tr']['space_slice']),
                    'in_gids': [frame['gid'] for frame in item['frames']],
                    'fliprot_params': item['tr'].get('fliprot_params', None)
                })
                position_tensors = item.get('positional_tensors', None)
                if position_tensors is not None:
                    for k, v in position_tensors.items():
                        position_tensors[k] = v.to(device)

                filtered_frames = []
                for frame in item['frames']:
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
                pass

            # self = method
            # with_loss = 0
            # item = batch[0]
            # import xdev
            # xdev.embed()

            # Predict on the batch
            # import xdev
            # with xdev.embed_on_exception_context:
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
                head_stitcher = stitch_managers[head_key]
                head_probs = outputs[head_key]
                chan_keep_idxs = task_keep_indices[head_key]

                # HACK: FIXME: WE ARE HARD CODING THAT CHANGE IS GIVEN TO
                # ALL FRAMES EXECPT THE FIRST IN MULTIPLE PLACES.
                if head_key == 'change':
                    predicted_frame_slice = slice(1, None)
                else:
                    predicted_frame_slice = slice(None)

                for bx, region_info in enumerate(batch_regions):
                    # TODO: if the predictions are downsampled wrt to the input
                    # images, we need to determine what that transform is so we can
                    # correctly warp the predictions back into image space.

                    item_head_probs = head_probs[bx]
                    # Keep only the channels we want to write to disk
                    item_head_relevant_probs = item_head_probs[..., chan_keep_idxs]
                    bin_probs = item_head_relevant_probs.detach().cpu().numpy()

                    # Get the spatio-temporal subregion this prediction belongs to
                    out_gids = region_info['in_gids'][predicted_frame_slice]
                    space_slice = region_info['space_slice']

                    fliprot_params = region_info['fliprot_params']
                    # Update the stitcher with this windowed prediction
                    for gid, probs in zip(out_gids, bin_probs):
                        if fliprot_params is not None:
                            # Undo fliprot TTA
                            probs = inv_fliprot(probs, **fliprot_params)
                        head_stitcher.accumulate_image(gid, space_slice, probs)

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

    try:
        device_info = {
            'total_vram': torch.cuda.get_device_properties(device).total_memory,
            'reserved_vram': torch.cuda.memory_reserved(device),
            'allocated_vram': torch.cuda.memory_allocated(device),
            'device_index': device.index,
            'device_type': device.type,
        }
    except Exception:
        device_info = None

    if emissions_tracker is not None:
        co2_kg = emissions_tracker.stop()
        emissions = {
            'co2_kg': co2_kg,
        }
        try:
            import pint
        except Exception as ex:
            print('ex = {!r}'.format(ex))
        else:
            reg = pint.UnitRegistry()
            co2_ton = (co2_kg * reg.kg).to(reg.metric_ton)
            dollar_per_ton = 15 / reg.metric_ton  # cotap rate
            emissions['co2_ton'] = co2_ton.m
            emissions['est_dollar_to_offset'] = (co2_ton * dollar_per_ton).m
        print('emissions = {}'.format(ub.repr2(emissions, nl=1)))
    else:
        emissions = None

    info.append({
        'type': 'measure',
        'properties': {
            'iters_per_second': prog._iters_per_second,
            'start_timestamp': start_timestamp,
            'end_timestamp': ub.timestamp(),
            'device_info': device_info,
            'emissions': emissions,
        }
    })

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

    def __init__(self, result_dataset, short_code=None, chan_code=None, stiching_space='video',
                 device='numpy', thresh=0.5, write_probs=True,
                 write_preds=True, num_bands='auto', prob_compress='DEFLATE',
                 polygon_categories=None, quantize=True):
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

    def accumulate_image(self, gid, space_slice, data):
        """
        Stitches a result into the appropriate image stitcher.

        Args:
            gid (int):
                the image id to stitch into

            space_slice (int):
                the slice (in "stitching-space") the data corresponds to.

            data (ndarray | Tensor): the feature or probability data
        """
        data = kwarray.atleast_nd(data, 3)
        dset = self.result_dataset
        if self.stiching_space == 'video':
            vidid = dset.index.imgs[gid]['video_id']
            # Create the stitcher if it does not exist
            if gid not in self.image_stitchers:
                video = dset.index.videos[vidid]
                if self.num_bands == 'auto':
                    if len(data.shape) == 3:
                        self.num_bands = data.shape[2]
                    else:
                        raise NotImplementedError
                stitch_dims = (video['height'], video['width'], self.num_bands)

                self.image_stitchers[gid] = kwarray.Stitcher(
                    stitch_dims, device=self.device)

            if self._last_vidid is not None and vidid != self._last_vidid:
                # We assume sequential video iteration, thus when we see a new
                # video, we know the images from the previous video are ready.
                video_gids = set(dset.index.vidid_to_gids[self._last_vidid])
                ready_gids = video_gids & set(self.image_stitchers)

                # TODO
                # do something clever to know if frames are ready early?
                # might be tricky in general if we run over multiple
                # times per image with different frame samplings.
                self._ready_gids.update(ready_gids)

            self._last_vidid = vidid
        else:
            raise NotImplementedError(self.stiching_space)

        stitcher: kwarray.Stitcher = self.image_stitchers[gid]

        weights = util_kwimage.upweight_center_mask(data.shape[0:2])[..., None]

        if stitcher.shape[0] < space_slice[0].stop or stitcher.shape[1] < space_slice[1].stop:
            # By embedding the space slice in the stitcher dimensions we can get a
            # slice corresponding to the valid region in the stitcher, and the extra
            # padding encode the valid region of the data we are trying to stitch into.
            subslice, padding = kwarray.embed_slice(space_slice[0:2], stitcher.shape)
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
            spatial_valid_mask = (1 - invalid_output_mask.any(axis=2, keepdims=True))
            stitch_weights = stitch_weights * spatial_valid_mask
            stitch_data[invalid_output_mask] = 0
        stitcher.add(stitch_slice, stitch_data, weight=stitch_weights)

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

        # Get the final stitched feature for this image
        final_probs = stitcher.finalize()
        final_probs = kwarray.atleast_nd(final_probs, 3)
        final_probs = np.nan_to_num(final_probs)

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
            aux = {
                'file_name': relpath(new_fpath, bundle_dpath),
                'channels': self.chan_code,
                'height': final_probs.shape[0],
                'width': final_probs.shape[1],
                'num_bands': final_probs.shape[2],
                'warp_aux_to_img': img_from_vid.concise(),
            }
            auxiliary = img.setdefault('auxiliary', [])
            auxiliary.append(aux)

            # Save the prediction to disk
            total_prob += np.nansum(final_probs)

            if self.quantize:
                # Quantize
                quant_probs, quantization = quantize_float01(final_probs)
                aux['quantization'] = quantization

                kwimage.imwrite(
                    str(new_fpath), quant_probs, space=None, backend='gdal',
                    compress=self.prob_compress, blocksize=128,
                    nodata=quantization['nodata']
                )
            else:
                kwimage.imwrite(
                    str(new_fpath), final_probs, space=None, backend='gdal',
                    compress=self.prob_compress, blocksize=128,
                )

        if self.write_preds:
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


def quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.int16):
    """
    Note:
        Setting old_min / old_max indicates the possible extend of the input
        data (and it will be clipped to it). It does not mean that the input
        data has to have those min and max values, but it should be between
        them.

    Example:
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> from kwcoco.util.util_delayed_poc import dequantize
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
        >>> from kwcoco.util.util_delayed_poc import dequantize
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
        --gpus=0, \
        --batch_size=1

    Develop TTA:

    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc)
    (cd $DVC_DPATH && dvc pull -r aws $DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=19-step=13659-v1.pt.dvc)

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
        --gpus=0, \
        --batch_size=1 \
        --exclude_sensors=L8 \
        --pred_dataset=$PRED_DATASET \
        --test_dataset=$TEST_DATASET \
        --tta_fliprot=0 \
        --tta_time=0 --dump=$DVC_DPATH/_tmp/test_pred_config.yaml

    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc)
    TEST_DATASET=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1_small.kwcoco.json

    PRED_DATASET_00=$DVC_DPATH/_tmp/_tmp_pred_00/pred.kwcoco.json
    PRED_DATASET_10=$DVC_DPATH/_tmp/_tmp_pred_10/pred.kwcoco.json
    PRED_DATASET_01=$DVC_DPATH/_tmp/_tmp_pred_01/pred.kwcoco.json
    PRED_DATASET_11=$DVC_DPATH/_tmp/_tmp_pred_11/pred.kwcoco.json

    export CUDA_VISIBLE_DEVICES=0
    python -m watch.tasks.fusion.predict \
        --config=$DVC_DPATH/_tmp/test_pred_config.yaml \
        --pred_dataset=$PRED_DATASET_00 \
        --test_dataset=$TEST_DATASET \
        --tta_fliprot=0 \
        --tta_time=0

    export CUDA_VISIBLE_DEVICES=1
    python -m watch.tasks.fusion.predict \
        --config=$DVC_DPATH/_tmp/test_pred_config.yaml \
        --pred_dataset=$PRED_DATASET_10 \
        --test_dataset=$TEST_DATASET \
        --tta_fliprot=1 \
        --tta_time=0

    export CUDA_VISIBLE_DEVICES=0
    python -m watch.tasks.fusion.predict \
        --config=$DVC_DPATH/_tmp/test_pred_config.yaml \
        --pred_dataset=$PRED_DATASET_01 \
        --test_dataset=$TEST_DATASET \
        --tta_fliprot=0 \
        --tta_time=1

    export CUDA_VISIBLE_DEVICES=1
    python -m watch.tasks.fusion.predict \
        --config=$DVC_DPATH/_tmp/test_pred_config.yaml \
        --pred_dataset=$PRED_DATASET_11 \
        --test_dataset=$TEST_DATASET \
        --tta_fliprot=1 \
        --tta_time=1

    #####

    python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_DATASET \
        --pred_dataset=$PRED_DATASET_11 \
        --eval_dpath=$DVC_DPATH/_tmp/_tmp_pred_11/eval
        --score_space=video \
        --draw_curves=1 \
        --draw_heatmaps=1 --workers=2

    python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_DATASET \
        --pred_dataset=$PRED_DATASET_00 \
        --eval_dpath=$DVC_DPATH/_tmp/_tmp_pred_00/eval
        --score_space=video \
        --draw_curves=1 \
        --draw_heatmaps=1 --workers=2

    python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_DATASET \
        --pred_dataset=$PRED_DATASET_01 \
        --eval_dpath=$DVC_DPATH/_tmp/_tmp_pred_01/eval
        --score_space=video \
        --draw_curves=1 \
        --draw_heatmaps=1 --workers=2

    python -m watch.tasks.fusion.evaluate \
        --true_dataset=$TEST_DATASET \
        --pred_dataset=$PRED_DATASET_10 \
        --eval_dpath=$DVC_DPATH/_tmp/_tmp_pred_10/eval
        --score_space=video \
        --draw_curves=1 \
        --draw_heatmaps=1 --workers=2


    DVC_DPATH=$(python -m watch.cli.find_dvc)
    TEST_DATASET=$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data_nowv_vali_kr1.kwcoco.json
    EXPT_PATTERN="*"
    python -m watch.tasks.fusion.schedule_evaluation \
            --gpus="0,1" \
            --model_globstr="$DVC_DPATH/models/fusion/eval3_candidates/packages/Drop3_SpotCheck_V323/Drop3_SpotCheck_V323_epoch=19-step=13659-v1.pt" \
            --test_dataset="$TEST_DATASET" \
            --workdir="$DVC_DPATH/_tmp/smalltest" \
            --sidecar2=1 \
            --tta_fliprot=0,1 \
            --tta_time=0,1 \
            --chip_overlap=0,0.3 \
            --run=1 --backend=slurm


    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc)
    MEASURE_GLOBSTR=${DVC_DPATH}/models/fusion/${EXPT_GROUP_CODE}/eval/${EXPT_NAME_PAT}/${MODEL_EPOCH_PAT}/${PRED_DSET_PAT}/${PRED_CFG_PAT}/eval/curves/measures2.json

    python -m watch.tasks.fusion.gather_results \
        --measure_globstr="$DVC_DPATH/_tmp/smalltest" \
        --out_dpath="$DVC_DPATH/_tmp/smalltest/_agg_results" \
        --dset_group_key="*" --show=True \
        --classes_of_interest "Site Preparation" "Active Construction"

    """
    main()
