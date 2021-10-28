# -*- coding: utf-8 -*-
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
from watch.tasks.fusion import datamodules
from watch.tasks.fusion import utils
from watch.tasks.fusion import heuristics
from watch.tasks.tracking import from_heatmap
from watch.utils import util_path
from watch.utils import util_parallel
from watch.utils import util_kwimage

try:
    import xdev
    profile = xdev.profile
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

    parser.add_argument('--pred_dpath', dest='pred_dpath', type=pathlib.Path, help='path to dump results')

    parser.add_argument('--package_fpath', type=pathlib.Path)
    parser.add_argument('--gpus', default=None, help='todo: hook up to lightning')
    parser.add_argument('--thresh', type=smartcast, default=0.01)

    parser.add_argument('--with_change', type=smartcast, default='auto')
    parser.add_argument('--with_class', type=smartcast, default='auto')
    parser.add_argument('--with_saliency', type=smartcast, default='auto')

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
        >>> train_dset = kwcoco.CocoDataset.demo('special:vidshapes4-multispectral', num_frames=3, gsize=(128, 128))
        >>> test_dset = kwcoco.CocoDataset.demo('special:vidshapes2-multispectral', num_frames=3, gsize=(128, 128))
        >>> fit_kwargs = kwargs = {
        ...     'train_dataset': test_dset.fpath,
        ...     'datamodule': 'KWCocoVideoDataModule',
        ...     'workdir': ub.ensuredir((test_dpath, 'train')),
        ...     'package_fpath': package_fpath,
        ...     'max_epochs': 1,
        ...     'time_steps': 2,
        ...     'chip_size': 64,
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
        >>>     'pred_dpath': results_path,
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
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=2)))

    try:
        # Ideally we have a package, everything is defined there
        method = utils.load_model_from_package(args.package_fpath)
        # fix einops bug
        for name, mod in method.named_modules():
            if 'Rearrange' in mod.__class__.__name__:
                mod._recipe = mod.recipe()

    except Exception:
        # If we have a checkpoint path we can load it if we make assumptions
        # init method from checkpoint.
        checkpoint = torch.load(args.package_fpath)
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
        traintime_vals = method.fit_config
    elif hasattr(method, 'datamodule_hparams'):
        traintime_vals = method.datamodule_hparams
    else:
        traintime_vals = {}
        if datamodule_vars['channels'] in {None, 'auto'}:
            print('Warning have to make assumptions. Might not always work')
            if hasattr(method, 'input_channels'):
                # note input_channels are sometimes different than the channels the
                # datamodule expects. Depending on special keys and such.
                traintime_vals['channels'] = method.input_channels.spec
            else:
                traintime_vals['channels'] = list(method.input_norms.keys())[0]
    able_to_infer = ub.dict_isect(traintime_vals, need_infer)
    unable_to_infer = ub.dict_diff(need_infer, traintime_vals)
    # Use defaults when we can't infer
    overloads = able_to_infer.copy()
    overloads.update(ub.dict_isect(args.datamodule_defaults, unable_to_infer))
    datamodule_vars.update(overloads)
    print('able_to_infer = {}'.format(ub.repr2(able_to_infer, nl=1)))
    print('unable_to_infer = {}'.format(ub.repr2(unable_to_infer, nl=1)))
    print('overloads = {}'.format(ub.repr2(overloads, nl=1)))

    deviation = ub.varied_values([
        ub.dict_isect(traintime_vals, datamodule_vars),
        ub.dict_isect(datamodule_vars, traintime_vals),
    ], min_variations=1)
    print('deviation from fit->predict settings = {}'.format(ub.repr2(deviation, nl=1)))

    datamodule = datamodule_class(
        **datamodule_vars
    )
    datamodule.setup('test')

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
    result_dataset = test_coco_dataset.copy()
    # Remove all annotations in the results copy
    result_dataset.clear_annotations()
    # Change all paths to be absolute paths
    result_dataset.reroot(absolute=True)
    # Set the filepath for the prediction coco file
    # (modifies the bundle_dpath)
    if args.pred_dataset is None:
        pred_dpath = util_path.coercepath(args.pred_dpath)
        result_dataset.fpath = str(pred_dpath / 'pred.kwcoco.json')
    else:
        result_dataset.fpath = str(args.pred_dataset)
    result_fpath = util_path.coercepath(result_dataset.fpath)

    # add hyperparam info to "info" section
    info = result_dataset.dataset.get('info', [])

    from kwcoco.util import util_json
    import os
    import socket
    info.append({
        'type': 'process',
        'properties': {
            'name': 'watch.tasks.fusion.predict',
            'args': util_json.ensure_json_serializable(args.__dict__),
            'hostname': socket.gethostname(),
            'cwd': os.getcwd(),
            'timestamp': ub.timestamp(),
        }
    })

    result_fpath.parent.mkdir(parents=True, exist_ok=True)

    from watch.utils.lightning_ext import util_device
    devices = util_device.coerce_devices(args.gpus)
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

    if args.with_change:
        stitch_managers['change'] = CocoStitchingManager(
            result_dataset,
            stiching_space='video',
            device=stitch_device,
            chan_code='change',
            thresh=args.thresh,
            write_probs=args.write_probs,
            write_preds=args.write_preds,
            num_bands=1,
        )
        result_dataset.ensure_category('change')

    if args.with_class:
        if hasattr(method, 'foreground_classes'):
            foreground_classes = method.foreground_classes
        else:
            not_foreground = (heuristics.BACKGROUND_CLASSES | heuristics.IGNORE_CLASSNAMES)
            foreground_classes = ub.oset(method.classes) - not_foreground
            # hueristics.B
            # raise NotImplementedError('old model, need to hack in fg classes')

        class_chan_code = '|'.join(list(method.classes))
        stitch_managers['class'] = CocoStitchingManager(
            result_dataset,
            stiching_space='video',
            device=stitch_device,
            chan_code=class_chan_code,
            thresh=args.thresh,
            write_probs=args.write_probs,
            write_preds=args.write_preds,
            polygon_categories=foreground_classes,
            num_bands=len(method.classes),
        )

    if args.with_saliency:
        stitch_managers['saliency'] = CocoStitchingManager(
            result_dataset,
            stiching_space='video',
            device=stitch_device,
            chan_code='not_salient|salient',
            thresh=args.thresh,
            write_probs=args.write_probs,
            write_preds=args.write_preds,
            polygon_categories=['salient'],
            num_bands=2,
        )

    # result_infos = []
    # total_info = {'n_anns': 0, 'n_imgs': 0, 'total_prob': 0}

    expected_outputs = set(stitch_managers.keys())
    got_outputs = None
    writable_outputs = None

    print('Expected outputs: ' + str(expected_outputs))

    head_key_mapping = {
        'saliency_probs': 'saliency',
        'class_probs': 'class',
        'change_probs': 'change',
    }

    # running_stats = kwarray.RunningStats()  # check if probs are non-zero

    # Start background procs before we make threads
    batch_iter = iter(test_dataloader)
    writer_queue = util_parallel.BlockingJobQueue(
        mode='thread',
        # mode='serial',
        max_workers=datamodule.num_workers)

    prog = ub.ProgIter(batch_iter, desc='predicting', verbose=1)

    with torch.set_grad_enabled(False):
        # prog.set_extra(' <will populate stats after first video>')
        for batch in prog:

            batch_regions = []
            # Move data onto the prediction device, grab spacetime region info
            for item in batch:
                batch_regions.append({
                    'space_slice': tuple(item['tr']['space_slice']),
                    'in_gids': [frame['gid'] for frame in item['frames']],
                })
                for frame in item['frames']:
                    modes = frame['modes']
                    for key, mode in modes.items():
                        modes[key] = mode.to(device)

            # Predict on the batch
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
                    bin_probs = head_probs[bx].detach().cpu().numpy()

                    # Get the spatio-temporal subregion this prediction belongs to
                    out_gids = region_info['in_gids'][predicted_frame_slice]
                    space_slice = region_info['space_slice']

                    # Update the stitcher with this windowed prediction
                    for gid, probs in zip(out_gids, bin_probs):
                        # running_stats.update(probs.mean())
                        head_stitcher.accumulate_image(gid, space_slice, probs)

                # Free up space for any images that have been completed
                for gid in head_stitcher.ready_image_ids():
                    # finalize_ready(head_stitcher, gid)
                    head_stitcher._ready_gids.difference_update({gid})  # avoid race condition
                    writer_queue.submit(head_stitcher.finalize_image, gid)

        writer_queue.wait_until_finished()  # hack to avoid race condition

        # Prediction is completed, finalize all remaining images.
        for head_key, head_stitcher in stitch_managers.items():
            for gid in head_stitcher.managed_image_ids():
                # finalize_ready(head_stitcher, gid)
                writer_queue.submit(head_stitcher.finalize_image, gid)
        writer_queue.wait_until_finished()

    # prog.set_extra('found {}'.format(nanns))
    # print('Predicted total nanns = {!r}'.format(nanns))

    # validate and save results
    print(result_dataset.validate())
    print('dump result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    result_dataset.dump(result_dataset.fpath)
    print('return result_dataset.fpath = {!r}'.format(result_dataset.fpath))
    return result_dataset


# def finalize_ready(head_stitcher, gid, running_stats, total_info):
#     info = head_stitcher.finalize_image(gid)
#     stats = running_stats.summarize(axis=None, keepdims=False)
#     total_info['n_anns'] += info['n_anns']
#     total_info['total_prob'] += info['total_prob']
#     total_info['n_imgs'] += 1

#     report_info = ub.dict_union(
#         ub.dict_isect(stats, {'mean', 'max', 'min', 'std'}),
#         ub.dict_isect(total_info, {'n_imgs', 'n_anns', 'total_prob'}),
#     )
#     report_info_str = ub.repr2(report_info, precision=4, compact=True)
#     prog.set_extra(' {} - '.format(report_info_str))
#     prog.ensure_newline()
#     result_infos.append(info)


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
                 write_preds=True, num_bands='auto', prob_compress='RAW',
                 polygon_categories=None):
        self.short_code = short_code
        self.result_dataset = result_dataset
        self.device = device
        self.chan_code = chan_code
        self.thresh = thresh
        self.num_bands = num_bands
        self.prob_compress = prob_compress
        self.polygon_categories = polygon_categories

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
        stitcher.add(space_slice, data, weight=weights)

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
            img.get('auxiliary', []).append({
                'file_name': relpath(new_fpath, bundle_dpath),
                'channels': self.chan_code,
                'height': final_probs.shape[0],
                'width': final_probs.shape[1],
                'num_bands': final_probs.shape[2],
                'warp_aux_to_img': img_from_vid.concise(),
            })

            # Save the prediction to disk
            total_prob += final_probs.sum()
            kwimage.imwrite(
                str(new_fpath), final_probs, space=None, backend='gdal',
                compress=self.prob_compress, blocksize=64,
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
                scored_polys = list(from_heatmap.mask_to_scored_polygons(
                    band_probs, thresh))
                n_anns = len(scored_polys)
                for vid_poly, score in scored_polys:
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


def main(cmdline=True, **kwargs):
    predict(cmdline=cmdline, **kwargs)


if __name__ == '__main__':
    main()
