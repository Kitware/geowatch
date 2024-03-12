#!/usr/bin/env python3
"""
Fusion prediction script.

Given a kwcoco file and a packaged model, run prediction and output a new
kwcoco file where predicted heatmaps are new raster bands.

This is the module that handles heatmap prediction over a kwcoco file.
There are SMART-specific parts, but it's mostly general. It makes heavy use of
:class:`CocoStitchingManager` and :class:`KWCocoVideoDataModule`. The critical
loop is a simple custom for loop over a dataloader. We currently do not
integrate with LightningCLI here, but we may want to in the future (it is
unclear).

TODO:
    - [ ] Prediction caching?
    - [ ] Reduce memory usage?
    - [ ] Pseudo Live.
    - [ ] Investigate benefits of LightningCLI integration?
    - [ ] Option to keep annotations and only loop over relevant areas for
          drawing interesting validation / test batches.
    - [ ] Optimize for the case where we have an image-only dataset.
    - [ ] Integrate debug visualizations to the CLI
"""
import torch
import ubelt as ub
import numpy as np
import kwimage
import kwarray
import kwcoco
from geowatch.tasks.fusion import datamodules
from geowatch.tasks.fusion import utils
from kwutil import util_parallel
from geowatch.tasks.fusion.datamodules import data_utils
from geowatch.tasks.fusion.coco_stitcher import CocoStitchingManager
from geowatch.tasks.fusion.coco_stitcher import quantize_float01  # NOQA
# APPLY Monkey Patches
from geowatch.monkey import monkey_torch
from geowatch.monkey import monkey_torchmetrics
import scriptconfig as scfg

monkey_torchmetrics.fix_torchmetrics_compatability()

try:
    from xdev import profile
except Exception:
    profile = ub.identity


class DataModuleConfigMixin(scfg.DataConfig):
    # Helps extend our custom predict config with datamodule config settings
    __default__ = {
        k: v.copy()
        for k, v in datamodules.kwcoco_datamodule.KWCocoVideoDataModuleConfig.__default__.items()}
    __default__['batch_size'].value = 1
    __default__['chip_overlap'].value = 0.3

    # Handle hacky overloadable stuff
    # Set the default of all of thse to be "auto", but remember the original
    # value in a custom variable
    # TODO: change name from overloadable to inferrable. The idea is that we
    # can infer these from the given model.
    __DATAMODULE_DEFAULTS__ = {}
    __OVERLOADABLE_DATAMODULE_KEYS__ = [
        'channels',
        'normalize_peritem',
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
    for key in __OVERLOADABLE_DATAMODULE_KEYS__:
        __DATAMODULE_DEFAULTS__[key] = __default__[key].value
        __default__[key].value = 'auto'
        ...


class PredictConfig(DataModuleConfigMixin):
    """
    Prediction script for the fusion task
    """
    # config_file = scfg.Value(None, alias=['config'], help='config file path')
    # write_out_config_file_to_this_path = scfg.Value(None, alias=['dump'], help=ub.paragraph(
    #         '''
    #         takes the current command line args and writes them out to a
    #         config file at the given path, then exits
    #         '''))
    datamodule = scfg.Value('KWCocoVideoDataModule', help='This must always be KWCocoVideoDataModule for now')
    pred_dataset = scfg.Value(None, help=ub.paragraph(
            '''
            path to the output dataset (note: test_dataset is the input
            dataset)
            '''))
    package_fpath = scfg.Value(None, type=str, help=None)
    devices = scfg.Value(None, help='lightning devices')
    thresh = scfg.Value(0.01, help=None)
    with_change = scfg.Value('auto', help=None)
    with_class = scfg.Value('auto', help=None)
    with_saliency = scfg.Value('auto', help=None)

    draw_batches = scfg.Value(False, isflag=True, help=ub.paragraph(
        '''
        if True, then draw batch visualizations as they are predicted.

        In this case it is also a good idea to set --test_with_annot_info=True
        which allows the datamodule to use annotations in validation sampling.
        This is a workaround, and ideally we can come up with a way to avoid
        requiring the user to know about this.
        '''))

    track_emissions = scfg.Value(True, isflag=True, help=ub.paragraph(
            '''
            set to false to disable emission tracking
            '''))
    record_context = scfg.Value(True, help='If enabled records process context stats')
    quantize = scfg.Value(True, help='quantize outputs')
    tta_fliprot = scfg.Value(0, help=ub.paragraph(
            '''
            number of times to flip/rotate the frame, can be in [0,7]
            '''))
    tta_time = scfg.Value(0, help=ub.paragraph(
            '''
            number of times to expand the temporal sample for a frame
            '''))
    clear_annots = scfg.Value(1, help=ub.paragraph(
            '''
            Clear existing annotations in output file. Otherwise keep
            them
            '''))
    drop_unused_frames = scfg.Value(0, help=ub.paragraph(
            '''
            if True, remove any images that were not predicted on
            '''))
    write_workers = scfg.Value('datamodule', help=ub.paragraph(
            '''
            workers to use for writing results. If unspecified uses the
            datamodule num_workers
            '''))
    compress = scfg.Value('DEFLATE', type=str, help='type of compression for prob images')
    format = scfg.Value('cog', type=str, help=ub.paragraph(
            '''
            the output format of the predicted images
            '''))
    write_preds = scfg.Value(False, help=ub.paragraph(
            '''
            If True, convert probability maps into raw "hard"
            predictions and write them as annotations to the prediction
            kwcoco file.
            '''))
    write_probs = scfg.Value(True, help=ub.paragraph(
            '''
            If True, write raw "soft" probability maps into the kwcoco
            file as a new auxiliary channel. The channel name is
            currently hard-coded based on expected output heads. This
            may change in the future.
            '''))
    write_workers = scfg.Value('datamodule', help=ub.paragraph(
            '''
            workers to use for writing results. If unspecified uses the
            datamodule num_workers
            '''))

    saliency_chan_code = scfg.Value('salient', help=ub.paragraph(
        '''
        Quick and dirty param to modify the channel name of salient output.
        This probably isn't generally useful and should be refactored later.
        '''))

    downweight_edges = scfg.Value(True, help=ub.paragraph(
        '''
        if True, spatial edges are downweighted in stitching in addition to
        using any weights coming out of the torch dataset.
        '''))


def build_stitching_managers(config, model, result_dataset, writer_queue=None):
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
        prob_format=config['format'],
        quantize=config['quantize'],
        expected_minmax=(0, 1),
        writer_queue=writer_queue,
        assets_dname='_assets',
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
        if 0:
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
        if hasattr(model, 'foreground_classes'):
            foreground_classes = model.foreground_classes
        else:
            from geowatch import heuristics
            not_foreground = (heuristics.BACKGROUND_CLASSES |
                              heuristics.IGNORE_CLASSNAMES |
                              heuristics.NEGATIVE_CLASSES)
            foreground_classes = ub.oset(model.classes) - not_foreground
        head_classes = model.classes
        head_keep_idxs = [
            idx for idx, catname in enumerate(head_classes)
            if catname not in ignore_classes]
        head_keep_classes = list(ub.take(head_classes, head_keep_idxs))
        task_keep_indices[task_name] = head_keep_idxs
        chan_code = '|'.join(list(head_keep_classes))
        if 0:
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
        salient_code = config.saliency_chan_code
        head_classes = ['not_' + salient_code, salient_code]
        head_keep_idxs = [
            idx for idx, catname in enumerate(head_classes)
            if catname not in ignore_classes]
        head_keep_classes = list(ub.take(head_classes, head_keep_idxs))
        task_keep_indices[task_name] = head_keep_idxs
        chan_code = '|'.join(head_keep_classes)
        if 0:
            print('task_name = {!r}'.format(task_name))
            print('head_classes = {!r}'.format(head_classes))
            print('head_keep_classes = {!r}'.format(head_keep_classes))
            print('chan_code = {!r}'.format(chan_code))
            print('head_keep_idxs = {!r}'.format(head_keep_idxs))
        stitch_managers[task_name] = CocoStitchingManager(
            result_dataset,
            chan_code=chan_code,
            short_code='pred_' + task_name,
            polygon_categories=[salient_code],
            num_bands=len(head_keep_classes),
            **stitcher_common_kw,
        )
        stitch_managers[task_name].head_keep_idxs = head_keep_idxs
    return stitch_managers


def resolve_datamodule(config, model, datamodule_defaults, fit_config):
    """
    TODO: refactor / cleanup.

    Breakup the sections that handle getting the traintime params, resolving
    the datamodule args, and building the datamodule.

    Args:
        ...

        fit_config (dict):
            nested train-time configuration provided by the model
            This should have a "data" key for dataset params.
    """
    import rich
    # init datamodule from args
    datamodule_class = getattr(datamodules, config['datamodule'])
    datamodule_vars = datamodule_class.compatible(config)

    parsetime_vals = ub.udict(datamodule_vars) & datamodule_defaults
    need_infer = ub.udict({
        k: v for k, v in parsetime_vals.items() if v == 'auto' or v == ['auto']})

    def get_scriptconfig_compatible(config_cls, other):
        """
        TODO: add to scriptconfig. Get the set of keys that we will accept.
        """
        acceptable_keys = set(config_cls.default.keys())
        for val in config_cls.default.values():
            if val.alias:
                acceptable_keys.update(val.alias)

        common = ub.udict(other) & acceptable_keys
        resolved = dict(config_cls.cli(cmdline=0, data=common))
        return resolved

    config_cls = datamodules.kwcoco_dataset.KWCocoVideoDatasetConfig
    traintime_data_params = fit_config['data']
    # Determine which train-time data options are compatible with predict-time.
    traintime_datavars = get_scriptconfig_compatible(
        config_cls, traintime_data_params)

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
    print('able_to_infer = {}'.format(ub.urepr(able_to_infer, nl=1)))
    print('unable_to_infer = {}'.format(ub.urepr(unable_to_infer, nl=1)))
    print('overloads = {}'.format(ub.urepr(overloads, nl=1)))

    # Look at the difference between predict and train time settings
    print('deviation from fit->predict settings:')
    for key in (traintime_datavars.keys() & datamodule_vars.keys()):
        f_val = traintime_datavars[key]  # fit-time value
        p_val = datamodule_vars[key]  # pred-time value
        if f_val != p_val:
            rich.print(f'    {key!r}: {f_val!r} -> {p_val!r}')

    HACK_FIX_MODELS_WITH_BAD_CHANNEL_SPEC = True
    # TODO: can we remove this or move it out of this function?
    if HACK_FIX_MODELS_WITH_BAD_CHANNEL_SPEC:
        # There was an issue where we trained models and specified
        # r|g|b|mat:0.3 but we only passed data with r|g|b. At train time
        # current logic (whch we need to fix) will happilly just take a subset
        # of those channels, which means the recorded channels disagree with
        # what the model was actually trained with.
        if hasattr(model, 'sensor_channel_tokenizers'):
            from geowatch.tasks.fusion.methods.network_modules import RobustModuleDict
            datamodule_sensorchan_spec = datamodule_vars['channels']
            unique_channel_streams = ub.oset()
            model_sensorchan_stem_parts = []
            for sensor, tokenizers in model.sensor_channel_tokenizers.items():
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
    return config, datamodule


def _prepare_model(config):
    """
    Load the specified (ideally packaged) model
    """
    package_fpath = ub.Path(config['package_fpath']).expand()

    # try:
    # Ideally we have a package, everything is defined there
    model = utils.load_model_from_package(package_fpath)
    # fix einops bug
    for _name, mod in model.named_modules():
        if 'Rearrange' in mod.__class__.__name__:
            try:
                mod._recipe = mod.recipe()
            except AttributeError:
                pass
    # hack: dont load the metrics
    model.class_metrics = None
    model.saliency_metrics = None
    model.change_metrics = None
    model.head_metrics = None
    # except Exception as ex:
    #     print('ex = {!r}'.format(ex))
    #     print(f'Failed to read {package_fpath=!r} attempting workaround')
    #     # If we have a checkpoint path we can load it if we make assumptions
    #     # init model from checkpoint.
    #     raise

    # Hack to fix GELU issue
    monkey_torch.fix_gelu_issue(model)

    # Fix issue with pre-2023-02 heterogeneous models
    if model.__class__.__name__ == 'HeterogeneousModel':
        if not hasattr(model, 'magic_padding_value'):
            from geowatch.tasks.fusion.methods.heterogeneous import HeterogeneousModel
            new_method = HeterogeneousModel(
                **model.hparams,
                position_encoder=model.position_encoder
            )
            old_state = model.state_dict()
            new_method.load_state_dict(old_state)
            new_method.config_cli_yaml = model.config_cli_yaml
            model = new_method

    model.eval()
    model.freeze()

    # Lookup the parameters used to fit the model (these should be stored in
    # the model, if they are not, then the model packaging needs to be
    # updated).
    if hasattr(model, 'config_cli_yaml'):
        # This should be a lightning nested dictionary
        # with keys like "data", "model", "trainer", "optmizer", etc..
        fit_config = model.config_cli_yaml
    else:
        raise AssertionError(
            'model is missing config_cli_yaml, other mechanisms to get fit '
            'params are commented and may need to be re-instanted')
        # elif hasattr(model, 'fit_config'):
        #     traintime_params = model.fit_config
        # elif hasattr(model, 'datamodule_hparams'):
        #     traintime_params = model.datamodule_hparams
        # else:
        #     # Not sure if code after is still needed for older models.
        #     # if we hit this error, then we may need to rework this.
        #     traintime_params = {}
        #     if datamodule_vars['channels'] in {None, 'auto'}:
        #         print('Warning have to make assumptions. Might not always work')
        #         raise NotImplementedError('TODO: needs to be sensorchan if we do this')
        #         if hasattr(model, 'input_channels'):
        #             # note input_channels are sometimes different than the channels the
        #             # datamodule expects. Depending on special keys and such.
        #             traintime_params['channels'] = model.input_channels.spec
        #         else:
        #             traintime_params['channels'] = list(model.input_norms.keys())[0]

    return model, fit_config


def _prepare_predict_modules(config):
    """
    Build the modules (i.e. data / model) needed to run prediction.

    Args:
        config (PredictConfig):
            the config for this script

    Returns:
        Tuple[PredictConfig, LightningModule, LightningDataModule]:
            modified config, network, and dataloader
    """
    config.datamodule_defaults = config.__DATAMODULE_DEFAULTS__
    datamodule_defaults = config.datamodule_defaults

    model, fit_config = _prepare_model(config)
    config.fit_config = fit_config

    # Determine how to construct the datamodule with correct params
    # TODO: we should not be updating the config here
    config, datamodule = resolve_datamodule(
        config, model, datamodule_defaults, fit_config)

    # TODO: if TTA=True, disable deterministic time sampling
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
    test_torch_dataset = datamodule.torch_datasets['test']
    # hack this setting
    if not config.draw_batches:
        test_torch_dataset.inference_only = True

    return config, model, datamodule


def _debug_grid(test_dataloader):
    DEBUG_GRID = 0
    if DEBUG_GRID:
        # Check to see if the grid will cover all images
        image_id_to_space_boxes = ub.ddict(list)
        seen_gids = set()
        primary_gids = set()
        seen_video_ids = set()
        coco_dset = test_dataloader.dataset.sampler.dset

        # Can use this to build a visualization of spacetime coverage
        vid_to_box_to_timesamples = {}

        for target in test_dataloader.dataset.new_sample_grid['targets']:
            video_id = target['video_id']
            seen_video_ids.add(video_id)
            # Denote we have seen this vidspace slice in this image.
            space_slice = target['space_slice']
            space_box = kwimage.Box.from_slice(space_slice)
            for gid in target['gids']:
                image_id_to_space_boxes[gid].append(space_box)
            primary_gids.add(target['main_gid'])
            seen_gids.update(target['gids'])

            coco_box = tuple(space_box.to_coco())
            requested_timestamps = coco_dset.images(target['gids']).lookup('date_captured')
            requested_timestamps = coco_dset.images(target['gids']).lookup('frame_index')
            if video_id not in vid_to_box_to_timesamples:
                vid_to_box_to_timesamples[video_id] = {}
            if coco_box not in vid_to_box_to_timesamples[video_id]:
                vid_to_box_to_timesamples[video_id][coco_box] = []
            vid_to_box_to_timesamples[video_id][coco_box].append(requested_timestamps)

        VIZ_SPACETIME_COV = 0
        if VIZ_SPACETIME_COV:
            import kwplot
            from geowatch.utils.util_kwplot import time_sample_arcplot
            for videoid, box_to_timesample in vid_to_box_to_timesamples.items():
                fig = kwplot.figure(fnum=videoid)
                ax = fig.gca()
                ax.cla()
                yloc = 0
                ytick_labels = []
                for box, time_samples in sorted(box_to_timesample.items()):
                    time_samples = list(map(sorted, time_samples))
                    time_sample_arcplot(time_samples, yloc, ax=ax)
                    yloc += 1
                    ytick_labels.append(box)
                ax.set_yticks(np.arange(len(ytick_labels)))
                ax.set_yticklabels(ytick_labels)
                video = coco_dset.index.videos[videoid]
                ax.set_title(f'Time Sampling For Video {video["name"]}')
                ax.set_ylabel('space location')
                ax.set_xlabel('frame index')

        all_video_ids = list(coco_dset.videos())
        all_gids = list(coco_dset.images())
        from xdev import set_overlaps
        img_overlaps = set_overlaps(all_gids, seen_gids, s1='all_gids', s2='seen_gids')
        print('img_overlaps = {}'.format(ub.urepr(img_overlaps, nl=1)))

        vid_overlaps = set_overlaps(all_video_ids, seen_video_ids, s1='seen_video_ids', s2='seen_video_ids')
        print('vid_overlaps = {}'.format(ub.urepr(vid_overlaps, nl=1)))
        # primary_img_overlaps = set_overlaps(all_gids, primary_gids)
        # print('primary_img_overlaps = {}'.format(ub.urepr(primary_img_overlaps, nl=1)))

        # Check to see how much of each image is covered in video space
        # import kwimage
        gid_to_iou = {}
        print('image_id_to_space_boxes = {}'.format(ub.urepr(image_id_to_space_boxes, nl=2)))
        for gid, space_boxes in image_id_to_space_boxes.items():
            vidid = coco_dset.index.imgs[gid]['video_id']
            video = coco_dset.index.videos[vidid]
            video_poly = kwimage.Box.from_dsize((video['width'], video['height'])).to_polygon()
            boxes = kwimage.Boxes.concatenate(space_boxes)
            polys = boxes.to_polygons()
            covered = polys.unary_union().simplify(0.01)
            iou = covered.iou(video_poly)
            gid_to_iou[gid] = iou
        ious = list(gid_to_iou.values())
        iou_stats = kwarray.stats_dict(ious, n_extreme=True)
        print('iou_stats = {}'.format(ub.urepr(iou_stats, nl=1)))


def _jsonify(data):
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


def _predict_critical_loop(config, model, datamodule, result_dataset, device):
    import rich

    print('Predict on device = {!r}'.format(device))
    downweight_edges = config.downweight_edges

    UNPACKAGE_METHOD_HACK = 0
    if UNPACKAGE_METHOD_HACK:
        # unpackage model hack
        from geowatch.tasks.fusion import methods
        unpackged_method = methods.MultimodalTransformer(**model.hparams)
        unpackged_method.load_state_dict(model.state_dict())
        model = unpackged_method

    model = model.to(device)

    # Resolve what tasks are requested by looking at what heads are available.
    global_head_weights = getattr(model, 'global_head_weights', {})
    if config['with_change'] == 'auto':
        config['with_change'] = getattr(model, 'global_change_weight', 1.0) or global_head_weights.get('change', 1)
    if config['with_class'] == 'auto':
        config['with_class'] = getattr(model, 'global_class_weight', 1.0) or global_head_weights.get('class', 1)
    if config['with_saliency'] == 'auto':
        config['with_saliency'] = getattr(model, 'global_saliency_weight', 0.0) or global_head_weights.get('saliency', 1)

    # Start background procs before we make threads
    test_dataloader = datamodule.test_dataloader()
    batch_iter = iter(test_dataloader)

    from kwutil import util_progress
    pman = util_progress.ProgressManager(backend='rich')

    # prog = ub.ProgIter(batch_iter, desc='fusion predict', verbose=1, freq=1)

    # Make threads after starting background proces.
    if config.write_workers == 'datamodule':
        config.write_workers = datamodule.num_workers
    writer_queue = util_parallel.BlockingJobQueue(
        mode='thread',
        # mode='serial',
        max_workers=config.write_workers
    )

    result_fpath = ub.Path(result_dataset.fpath)
    result_fpath.parent.ensuredir()
    print('result_fpath = {!r}'.format(result_fpath))

    stitch_managers = build_stitching_managers(
        config, model, result_dataset,
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

    _debug_grid(test_dataloader)

    DEBUG_PRED_SPATIAL_COVERAGE = 0
    if DEBUG_PRED_SPATIAL_COVERAGE:
        # Enable debugging to ensure the dataloader actually passed
        # us the targets that cover the entire image.
        image_id_to_video_space_slices = ub.ddict(list)
        image_id_to_output_space_slices = ub.ddict(list)

    # add hyperparam info to "info" section
    info = result_dataset.dataset.get('info', [])

    pred_dpath = ub.Path(result_dataset.fpath).parent
    rich.print(f'Pred Dpath: [link={pred_dpath}]{pred_dpath}[/link]')

    DRAW_BATCHES = config.draw_batches
    if DRAW_BATCHES:
        viz_batch_dpath = (pred_dpath / '_viz_pred_batches').ensuredir()

    config_resolved = _jsonify(config.asdict())
    fit_config = _jsonify(config.fit_config)

    from kwcoco.util import util_json
    assert not list(util_json.find_json_unserializable(config_resolved))

    if config['record_context']:
        from geowatch.utils import process_context
        proc_context = process_context.ProcessContext(
            name='geowatch.tasks.fusion.predict',
            type='process',
            config=config_resolved,
            track_emissions=config['track_emissions'],
            # Extra information was adjusted in 0.15.1 to ensure more relevant
            # fit params are returned here. A script
            # ~/code/geowatch/geowatch/cli/experimental/fixup_predict_kwcoco_metadata.py
            # exist to help update old results to use this new format.
            extra={
                'fit_config': fit_config
            }
        )
        # assert not list(util_json.find_json_unserializable(proc_context.obj))
        info.append(proc_context.obj)
        proc_context.start()
        test_coco_dataset = datamodule.coco_datasets['test']
        proc_context.add_disk_info(test_coco_dataset.fpath)

    memory_monitor_timer = ub.Timer().tic()
    memory_monitor_interval_seconds = 60

    with torch.set_grad_enabled(False), pman:
        # FIXME: that data loader should not be producing incorrect sensor/mode
        # pairs in the first place!
        EMERGENCY_INPUT_AGREEMENT_HACK = 1 and hasattr(model, 'input_norms')

        # prog.set_extra(' <will populate stats after first video>')
        # pman.start()

        prog = pman.progiter(batch_iter, desc='fusion predict')
        _batch_iter = iter(prog)
        if 0:
            item = test_dataloader.dataset[0]

            orig_batch = next(_batch_iter)
            item = orig_batch[0]
            item['target']
            frame = item['frames'][0]
            ub.peek(frame['modes'].values()).shape

        batch_idx = 0
        for orig_batch in _batch_iter:
            batch_idx += 1
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
                    'output_weights',
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
                            known_sensor_modes = model.input_norms[sensor]
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

            MONITOR_MEMORY = 1
            if MONITOR_MEMORY:
                # TODO: encapsulate this in a helper class that runs some
                # user-specified function if the timer interval has ellapsed.
                if memory_monitor_timer.toc() > memory_monitor_interval_seconds:
                    # TODO: monitor memory usage and report if it looks like we
                    # are about to run out of memory, and maybe do something to
                    # handle it.
                    from geowatch.utils import util_hardware
                    mem_info = util_hardware.get_mem_info()
                    print(f'\n\nmem_info = {ub.urepr(mem_info, nl=1)}\n\n')
                    memory_monitor_timer.tic()

            # Predict on the batch: todo: rename to predict_step
            try:
                outputs = model.forward_step(batch, with_loss=False)
            except RuntimeError as ex:
                msg = ('A predict batch failed ex = {}'.format(ub.urepr(ex, nl=1)))
                print(msg)
                import warnings
                warnings.warn(msg)
                from kwutil import util_environ
                # import xdev
                # xdev.embed()
                if util_environ.envflag('WATCH_STRICT_PREDICT'):
                    raise
                continue

            if DRAW_BATCHES:
                fpath = viz_batch_dpath / f'batch_{batch_idx:04d}.jpg'
                canvas = datamodule.draw_batch(batch, stage='test',
                                               outputs=outputs,
                                               classes=model.classes)
                kwimage.imwrite(fpath, canvas)

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

                    if head_key == 'change':
                        # The change output doesnt seem to have have a
                        # channel. Only used in tests, so just hacking it.
                        # It should be fixed in the model output or there
                        # should be some general shape rectification.
                        if len(item_head_probs.shape) == 3:
                            item_head_probs = item_head_probs[:, :, :, None]

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
                        output_weights = frame_info.get('output_weights', None)

                        try:
                            head_stitcher.accumulate_image(
                                gid, output_space_slice, probs,
                                dsize=output_image_dsize,
                                scale=scale_outspace_from_vid,
                                weights=output_weights,
                                downweight_edges=downweight_edges,
                            )
                        except Exception:
                            rich.print('[red]ERROR IN PREDICT! PRINT ITEM DEBUG INFO')
                            rich.print('[red]ERROR IN PREDICT! PRINT ITEM DEBUG INFO')
                            rich.print('[red]ERROR IN PREDICT! PRINT ITEM DEBUG INFO')
                            space_slice_xywh = kwimage.Box.from_slice(output_space_slice).to_xywh()
                            rich.print(f'output_space_slice      = {ub.urepr(output_space_slice, nl=1)}')
                            rich.print(f'space_slice_xywh        = {ub.urepr(space_slice_xywh, nl=1)}')
                            rich.print(f'probs.shape             = {probs.shape}')
                            rich.print(f'output_weights.shape    = {output_weights.shape}')
                            rich.print(f'output_image_dsize      = {output_image_dsize}')
                            rich.print(f'scale_outspace_from_vid = {scale_outspace_from_vid}')
                            item_summary = test_dataloader.dataset.summarize_item(item)
                            print(f'item_summary = {ub.urepr(item_summary, nl=-1)}')
                            raise

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
        print('vidspace_iou_stats = {}'.format(ub.urepr(vidspace_iou_stats, nl=1)))
        print('vidspace_iooa_stats = {}'.format(ub.urepr(vidspace_iooa_stats, nl=1)))
        print('outspace_iou_stats = {}'.format(ub.urepr(outspace_iou_stats, nl=1)))
        print('outspace_iooa_stats = {}'.format(ub.urepr(outspace_iooa_stats, nl=1)))

    if config['record_context']:
        proc_context.add_device_info(device)
        proc_context.stop()

    # Print logs about what we predicted on
    all_video_ids = list(result_dataset.videos())
    print(f'Requested predictions for {len(all_video_ids)} videos')
    stitched_video_histogram = ub.ddict(lambda: 0)
    stitched_video_patch_histogram = ub.ddict(lambda: 0)
    for _head_key, head_stitcher in stitch_managers.items():
        _histo = ub.dict_hist(result_dataset.images(head_stitcher._seen_gids).lookup('video_id', None))
        print(f'stitched videos for {_head_key}={ub.urepr(_histo)}')

        for gid, v in head_stitcher._stitched_gid_patch_histograms.items():
            vidid = result_dataset.index.imgs[gid].get('video_id', None)
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

    if config.drop_unused_frames:
        keep_gids = set()
        for manager in stitch_managers.values():
            keep_gids.update(manager.seen_image_ids)
        drop_gids = set(result_dataset.images()) - keep_gids
        print(f'Dropping {len(drop_gids)} unused frames')
        result_dataset.remove_images(drop_gids)

    # validate and save results
    if 0:
        print(result_dataset.validate())

    rich.print(f'Pred Dpath: [link={pred_dpath}]{pred_dpath}[/link]')
    print('dump result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    result_dataset.dump(result_dataset.fpath)
    print('return result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    return result_dataset


@profile
def predict(cmdline=False, **kwargs):
    """
    Predict entry point and doctests

    CommandLine:
        xdoctest -m geowatch.tasks.fusion.predict predict:0

    Example:
        >>> # Train a demo model (in the future grab a pretrained demo model)
        >>> from geowatch.tasks.fusion.predict import *  # NOQA
        >>> import os
        >>> from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> disable_lightning_hardware_warnings()
        >>> args = None
        >>> cmdline = False
        >>> devices = None
        >>> test_dpath = ub.Path.appdir('geowatch/test/fusion/').ensuredir()
        >>> results_path = (test_dpath / 'predict').ensuredir()
        >>> results_path.delete()
        >>> results_path.ensuredir()
        >>> import kwcoco
        >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes2-gsize64-frames9-speed0.5-multispectral')
        >>> test_dset = kwcoco.CocoDataset.demo('special:vidshapes1-gsize64-frames9-speed0.5-multispectral')
        >>> root_dpath = ub.Path(test_dpath, 'train').ensuredir()
        >>> fit_config = kwargs = {
        ...     'subcommand': 'fit',
        ...     'fit.data.train_dataset': train_dset.fpath,
        ...     'fit.data.time_steps': 2,
        ...     'fit.data.time_span': "2m",
        ...     'fit.data.chip_dims': 64,
        ...     'fit.data.time_sampling': 'hardish3',
        ...     'fit.data.num_workers': 0,
        ...     #'package_fpath': package_fpath,
        ...     'fit.model.class_path': 'geowatch.tasks.fusion.methods.MultimodalTransformer',
        ...     'fit.model.init_args.global_change_weight': 1.0,
        ...     'fit.model.init_args.global_class_weight': 1.0,
        ...     'fit.model.init_args.global_saliency_weight': 1.0,
        ...     'fit.optimizer.class_path': 'torch.optim.SGD',
        ...     'fit.optimizer.init_args.lr': 1e-5,
        ...     'fit.trainer.max_steps': 10,
        ...     'fit.trainer.accelerator': 'cpu',
        ...     'fit.trainer.devices': 1,
        ...     'fit.trainer.max_epochs': 3,
        ...     'fit.trainer.log_every_n_steps': 1,
        ...     'fit.trainer.default_root_dir': os.fspath(root_dpath),
        ... }
        >>> from geowatch.tasks.fusion import fit_lightning
        >>> package_fpath = root_dpath / 'final_package.pt'
        >>> fit_lightning.main(fit_config)
        >>> # Unfortunately, its not as easy to get the package path of
        >>> # this call..
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
        >>>     'draw_batches': 1,
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
        >>> pred1 = coco_img.imdelay('salient', nodata_method='float').finalize()
        >>> assert pred1.max() <= 1
        >>> # new delayed image does not make it easy to remove dequantization
        >>> # add test back in if we add support for that.
        >>> # pred2 = coco_img.imdelay('salient').finalize(nodata_method='float', dequantize=False)
        >>> # assert pred2.max() > 1

    Example:
        >>> # xdoctest: +REQUIRES(env:SLOW_DOCTEST)
        >>> # FIXME: why does this test hang on the strict dashboard?
        >>> # Train a demo model (in the future grab a pretrained demo model)
        >>> from geowatch.tasks.fusion.predict import *  # NOQA
        >>> from geowatch.utils.lightning_ext.monkeypatches import disable_lightning_hardware_warnings
        >>> disable_lightning_hardware_warnings()

        >>> args = None
        >>> cmdline = False
        >>> devices = None
        >>> test_dpath = ub.Path.appdir('geowatch/test/fusion/').ensuredir()
        >>> results_path = ub.ensuredir((test_dpath, 'predict'))
        >>> ub.delete(results_path)
        >>> ub.ensuredir(results_path)

        >>> import kwcoco
        >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes4-multispectral-multisensor', num_frames=5, image_size=(64, 64))
        >>> test_dset = kwcoco.CocoDataset.demo('special:vidshapes2-multispectral-multisensor', num_frames=5, image_size=(64, 64))
        >>> datamodule = datamodules.kwcoco_video_data.KWCocoVideoDataModule(
        >>>     train_dataset=train_dset, #'special:vidshapes8-multispectral-multisensor',
        >>>     test_dataset=test_dset, #'special:vidshapes8-multispectral-multisensor',
        >>>     chip_dims=32,
        >>>     channels="r|g|b",
        >>>     batch_size=1, time_steps=3, num_workers=2, normalize_inputs=10)
        >>> datamodule.setup('fit')
        >>> datamodule.setup('test')
        >>> dataset_stats = datamodule.torch_datasets['train'].cached_dataset_stats(num=3)
        >>> classes = datamodule.torch_datasets['train'].classes
        >>> print("classes = ", classes)

        >>> from geowatch.tasks.fusion import methods
        >>> from geowatch.tasks.fusion.architectures.transformer import TransformerEncoderDecoder
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
        >>> package_fpath = root_dpath / 'final_package.pt'
        >>> model.save_package(package_fpath)
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
        >>> pred1 = coco_img.imdelay('salient', nodata_method='float').finalize()
        >>> assert pred1.max() <= 1
        >>> # new delayed image does not make it easy to remove dequantization
        >>> # add test back in if we add support for that.
        >>> # pred2 = coco_img.imdelay('salient').finalize(nodata_method='float', dequantize=False)
        >>> # assert pred2.max() > 1
    """
    import rich
    config = PredictConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    config.datamodule_defaults = config.__DATAMODULE_DEFAULTS__
    # print('kwargs = {}'.format(ub.urepr(kwargs, nl=1)))
    rich.print('config = {}'.format(ub.urepr(config, nl=2)))

    config, model, datamodule = _prepare_predict_modules(config)

    test_coco_dataset = datamodule.coco_datasets['test']

    # test_torch_dataset = datamodule.torch_datasets['test']
    # T, H, W = test_torch_dataset.window_dims

    # Create the results dataset as a copy of the test CocoDataset
    print('Populate result dataset')
    result_dataset: kwcoco.CocoDataset = test_coco_dataset.copy()

    # Remove all annotations in the results copy
    if config['clear_annots']:
        result_dataset.clear_annotations()

    # Change all paths to be absolute paths
    result_dataset.reroot(absolute=True)
    if not config['pred_dataset']:
        raise ValueError(
            f'Must specify path to the output (predicted) kwcoco file. '
            f'Got {config["pred_dataset"]=}')
    result_dataset.fpath = str(ub.Path(config['pred_dataset']).expand())

    from geowatch.utils.lightning_ext import util_device
    print('devices = {!r}'.format(config['devices']))
    devices = util_device.coerce_devices(config['devices'])
    print('devices = {!r}'.format(devices))
    if len(devices) > 1:
        raise NotImplementedError('TODO: handle multiple devices')
    device = devices[0]

    result_dataset = _predict_critical_loop(config, model, datamodule, result_dataset, device)
    return result_dataset


def main(cmdline=True, **kwargs):
    if ub.argflag('--warntb'):
        import xdev
        xdev.make_warnings_print_tracebacks()
    predict(cmdline=cmdline, **kwargs)


if __name__ == '__main__':
    r"""
    Test old model:

    python -m geowatch.tasks.fusion.predict \
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
    """
    main()
