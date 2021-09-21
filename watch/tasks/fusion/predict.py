# -*- coding: utf-8 -*-
"""
Fusion prediction script.

TODO:
    - [ ] Prediction caching?
"""
import torch  # NOQA
import pathlib
import ubelt as ub
from os.path import join
from os.path import relpath
import kwimage
import kwarray
from watch.tasks.fusion import datamodules
from watch.tasks.fusion import utils
from watch.tasks.fusion import postprocess
from watch.utils import util_path


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
    parser.add_argument("--datamodule", default='KWCocoVideoDataModule')
    parser.add_argument("--pred_dataset", default=None, dest='pred_dataset')

    parser.add_argument("--pred_dpath", dest='pred_dpath', type=pathlib.Path, help='path to dump results')

    parser.add_argument("--tag", default='change_prob')
    parser.add_argument("--package_fpath", type=pathlib.Path)
    parser.add_argument("--gpus", default=None, help="todo: hook up to lightning")
    parser.add_argument("--thresh", type=float, default=0.01)

    parser.add_argument(
        "--write_preds", default=True, type=smartcast, help=ub.paragraph(
            '''
            If True, convert probability maps into raw "hard" predictions and
            write them as annotations to the prediction kwcoco file.
            '''))

    parser.add_argument(
        "--write_probs", default=True, type=smartcast, help=ub.paragraph(
            '''
            If True, write raw "soft" probability maps into the kwcoco file as
            a new auxiliary channel.  The channel name is currently denoted by
            the tag parameter, but this may change in the future.
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
    parser = datamodule_class.add_argparse_args(parser)
    parser.set_defaults(**{'batch_size': 1})

    # parse and pass to main
    parser.set_defaults(**kwargs)
    args, _ = parser.parse_known_args(default_args)
    assert args.batch_size == 1
    return args


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
        >>>     assert any('change_prob' in cs for cs in pred_chans), 'some frames should have change'
        >>>     assert not all('change_prob' in cs for cs in pred_chans), 'some frames should not have change'
        >>>     # Test number of annots in each frame
        >>>     num_annots = list(map(len, images.annots))
        >>>     assert num_annots[0] == 0, 'first frame should have none'
        >>>     # This test may fail with very low probability, so warn
        >>>     import warnings
        >>>     if sum(num_annots[1:]) == 0:
        >>>         warnings.warn('should be predictions elsewhere')
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)
    print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=2)))

    try:
        # Ideally we have a package, everything is defined there
        method = utils.load_model_from_package(args.package_fpath)
    except Exception:
        # If we have a checkpoint path we can load it if we make assumptions
        # init method from checkpoint.
        checkpoint = torch.load(args.package_fpath)
        print(list(checkpoint.keys()))
        from watch.tasks.fusion import methods
        hparams = checkpoint['hyper_parameters']
        if 'input_channels' in hparams:
            # Hack for strange pickle issue
            chan = hparams['input_channels']
            if not hasattr(chan, '_spec') and hasattr(chan, '_info'):
                chan = chan.__class__.coerce(chan._info['spec'])
                hparams['input_channels'] = chan

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

    # TODO: default to this, but allow the user to overwrite
    if hasattr(method, 'datamodule_hparams'):
        datamodule_vars['chip_size'] = method.datamodule_hparams['chip_size']
        datamodule_vars['time_steps'] = method.datamodule_hparams['time_steps']
        datamodule_vars['channels'] = method.datamodule_hparams['channels']
    else:
        print('Warning have to make assumptions')
        # datamodule_vars['chip_size'] = method.datamodule_hparams['chip_size']
        # datamodule_vars['time_steps'] = method.datamodule_hparams['time_steps']
        datamodule_vars['channels'] = list(method.input_norms.keys())[0]
        # method.datamodule_hparams['channels']

    datamodule = datamodule_class(
        **datamodule_vars
    )
    datamodule.setup("test")

    test_coco_dataset = datamodule.coco_datasets['test']
    test_torch_dataset = datamodule.torch_datasets['test']
    test_dataloader = datamodule.test_dataloader()

    T, H, W = test_torch_dataset.sample_shape

    # Create the results dataset as a copy of the test CocoDataset
    result_dataset = test_coco_dataset.copy()
    # Remove all annotations in the results copy
    result_dataset.clear_annotations()
    # Change all paths to be absolute paths
    result_dataset.reroot(absolute=True)
    result_dataset.ensure_category("change")
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
    method = method.to(device)

    stitch_manager = CocoStitchingManager(
        result_dataset,
        stiching_space='video',
        device='numpy',  # could be torch on-device stitching
        chan_code=args.tag,
        thresh=args.thresh,
        write_probs=args.write_probs,
        write_preds=args.write_preds,
    )

    result_infos = []
    total_info = {'n_anns': 0, 'n_imgs': 0, 'total_prob': 0}

    def finalize_ready(gid):
        info = stitch_manager.finalize_image(gid)
        stats = running_stats.summarize(axis=None, keepdims=False)
        total_info['n_anns'] += info['n_anns']
        total_info['total_prob'] += info['total_prob']
        total_info['n_imgs'] += 1

        report_info = ub.dict_union(
            ub.dict_isect(stats, {'mean', 'max', 'min', 'std'}),
            ub.dict_isect(total_info, {'n_imgs', 'n_anns', 'total_prob'}),
        )
        # TODO: once compact=True is available in ub 0.9.6, the rest can be
        # removed
        report_info_str = ub.repr2(
            report_info, precision=4, compact=True, with_dtype=False, nl=0, nobr=1,
            sk=True, sv=True, kvsep='=', itemsep='')
        prog.set_extra(' {} - '.format(report_info_str))
        # prog.ensure_newline()
        result_infos.append(info)

    import kwarray
    running_stats = kwarray.RunningStats()  # for inspecting probability

    prog = ub.ProgIter(test_dataloader, desc='predicting', verbose=1)
    for batch in prog:
        # Move data onto the prediction device
        for item in batch:
            for frame in item['frames']:
                modes = frame['modes']
                for key, mode in modes.items():
                    modes[key] = mode.to(device)

        # Predict on the batch
        outputs = method.forward_step(batch, with_loss=False)

        # TODO: we will eventually output more than "binary_predictions"
        if 'binary_predictions' in outputs:
            batch_bin_probs = outputs['binary_predictions']

        if 'change_probs' in outputs:
            batch_bin_probs = outputs['change_probs']

        # For each item in the batch, process the results
        for bx, item in enumerate(batch):

            # TODO: if the predictions are downsampled wrt to the input images,
            # we need to determine what that transform is so we can correctly
            # warp the predictions back into image space.
            bin_probs = batch_bin_probs[bx].detach().cpu().numpy()

            # Get the spatio-temporal subregion that this prediction belongs to
            in_gids = [frame['gid'] for frame in item['frames']]
            space_slice = tuple(item['tr']['space_slice'])
            # NOTE: the returned tr space slice seems bugged?
            # tuple(item['tr']['space_slice'])

            # Update the stitcher with this windowed prediction
            for gid, probs in zip(in_gids[1:], bin_probs):
                running_stats.update(probs)
                stitch_manager.accumulate_image(gid, space_slice, probs)

            # Free up space for any images that have been completed
            for gid in stitch_manager.ready_image_ids():
                finalize_ready(gid)

    # Prediction is completed, finalize all remaining images.
    for gid in stitch_manager.managed_image_ids():
        finalize_ready(gid)

    # prog.set_extra('found {}'.format(nanns))
    # print('Predicted total nanns = {!r}'.format(nanns))

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

        chan_code (str):
            If saving the stitched features, this is the channel code to use.

        stiching_space (str):
            Indicates if the results are given in image or video space.

        device ('numpy' | torch.device):
            Device to stitch on.

        thresh (float):
            if making hard decisions, determines the threshold for converting a
            soft mask into a hard mask, which can be converted into a polygon.


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

    def __init__(self, result_dataset, chan_code=None, stiching_space='video',
                 device='numpy', thresh=0.5, write_probs=True,
                 write_preds=True):
        self.result_dataset = result_dataset
        self.device = device
        self.chan_code = chan_code
        self.thresh = thresh

        self.stiching_space = stiching_space
        if stiching_space != 'video':
            raise NotImplementedError(stiching_space)

        # Setup a dictionary that we will use to make a stitcher for each image
        # as needed.  We use the fact that videos are iterated over
        # sequentially so free up memory of a video after it completes.
        self.image_stitchers = {}
        self._last_vidid = None
        self._ready_gids = set()

        # TODO: writing predictions and probabilities needs robustness work
        self.write_probs = write_probs
        self.write_preds = write_preds

        if self.write_probs:
            bundle_dpath = self.result_dataset.bundle_dpath
            prob_subdir = f'_assets/{self.chan_code}'
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
        dset = self.result_dataset
        if self.stiching_space == 'video':
            vidid = dset.index.imgs[gid]['video_id']
            # Create the stitcher if it does not exist
            if gid not in self.image_stitchers:
                video = dset.index.videos[vidid]
                space_dims = (video['height'], video['width'])
                self.image_stitchers[gid] = kwarray.Stitcher(
                    space_dims, device=self.device)

            if self._last_vidid is not None and vidid != self._last_vidid:
                # We assume sequential video iteration, thus when we see a new
                # video, we know the images from the previous video are ready.
                video_gids = set(dset.index.vidid_to_gids[self._last_vidid])
                ready_gids = video_gids & set(self.image_stitchers)
                self._ready_gids.update(ready_gids)

            self._last_vidid = vidid
        else:
            raise NotImplementedError(self.stiching_space)

        stitcher = self.image_stitchers[gid]
        stitcher.add(space_slice, data)

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
        stitcher = self.image_stitchers.pop(gid)

        # Get the final stitched feature for this image
        change_probs = stitcher.finalize()

        # Get spatial relationship between the image and the video
        vid_from_img = kwimage.Affine.coerce(img['warp_img_to_vid'])
        img_from_vid = vid_from_img.inv()

        n_anns = 0
        total_prob = 0

        # TODO: find and record the valid prediction regions
        # Given a (rectilinear) non-convex multipolygon where we are guarenteed
        # that all of the angles in the polygon are right angles, what is an
        # efficient algorithm to decompose it into a minimal set of disjoint
        # rectangles?
        # https://stackoverflow.com/questions/5919298/algorithm-for-finding-the-fewest-rectangles-to-cover-a-set-of-rectangles-without/6634668#6634668
        # Or... just write out a polygon... KISS
        import numpy as np
        is_predicted_pixel = (stitcher.weights > 0).astype(np.uint8)
        predicted_region = kwimage.Mask(is_predicted_pixel, 'c_mask').to_multi_polygon().to_geojson()
        # Mark that we made a prediction on this image.
        img['prediction_region'] = predicted_region
        img['has_predictions'] = True

        if self.write_probs:
            # This currently exists as an example to demonstrate how a
            # prediction script can write a pre-fusion TA-2 feature to disk and
            # register it with the kwcoco file.
            #
            # Save probabilities (or feature maps) as a new auxiliary image
            bundle_dpath = self.result_dataset.bundle_dpath
            new_feature = kwarray.atleast_nd(change_probs, 3)
            new_fname = img.get('name', str(img['id'])) + f'_{self.chan_code}.tiff'  # FIXME
            new_fpath = join(self.prob_dpath, new_fname)
            img.get('auxiliary', []).append({
                'file_name': relpath(new_fpath, bundle_dpath),
                'channels': self.chan_code,
                'height': new_feature.shape[0],
                'width': new_feature.shape[1],
                'num_bands': new_feature.shape[2],
                'warp_aux_to_img': img_from_vid.concise(),
            })

            # Save the prediction to disk
            total_prob += new_feature.sum()
            kwimage.imwrite(
                str(new_fpath), new_feature, space=None, backend='gdal',
                compress='LZW'
            )

        if self.write_preds:
            # This is the final step where we convert soft-probabilities to
            # hard-polygons, we need to choose an good operating point here.

            # HACK: We happen to know this is the category atm.
            # Should have a better way to determine it via metadata
            change_cid = self.result_dataset.index.name_to_cat['change']['id']

            # Threshold scores
            thresh = self.thresh
            # Convert to polygons
            scored_polys = list(postprocess.mask_to_scored_polygons(
                change_probs, thresh))
            n_anns = len(scored_polys)
            for score, vid_poly in scored_polys:
                # Transform the video polygon into image space
                img_poly = vid_poly.warp(img_from_vid)
                bbox = list(img_poly.bounding_box().to_coco())[0]
                # Add the polygon as an annotation on the image
                self.result_dataset.add_annotation(
                    image_id=gid, category_id=change_cid,
                    bbox=bbox, segmentation=img_poly, score=score)

        return {
            'n_anns': n_anns,
            'total_prob': total_prob,
        }


def main(cmdline=True, **kwargs):
    predict(cmdline=cmdline, **kwargs)


if __name__ == "__main__":
    main()
