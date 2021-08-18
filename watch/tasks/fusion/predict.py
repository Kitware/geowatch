# -*- coding: utf-8 -*-
import torch
import pathlib
import ubelt as ub
from os.path import join
from os.path import relpath
import kwimage
import kwarray
from watch.tasks.fusion import datamodules
from watch.tasks.fusion import utils


def make_predict_config(cmdline=False, **kwargs):
    """
    Configuration for fusion prediction
    """
    import argparse
    import configargparse
    class RawDescriptionDefaultsHelpFormatter(
            argparse.RawDescriptionHelpFormatter,
            argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = configargparse.ArgumentParser(
        add_config_file_help=False,
        description='Prediction script for the fusion task',
        formatter_class=RawDescriptionDefaultsHelpFormatter,
    )
    parser.add_argument("--datamodule", default='WatchDataModule')
    parser.add_argument("--pred_dataset", default=None, dest='pred_dataset')

    parser.add_argument("--pred_dpath", dest='pred_dpath', type=pathlib.Path, help='path to dump results')

    parser.add_argument("--tag", default='change_prob')
    parser.add_argument("--package_fpath", type=pathlib.Path)
    parser.add_argument("--gpus", default=None, help="todo: hook up to lightning")
    parser.add_argument("--thresh", type=float, default=0.01)

    parser.set_defaults(**kwargs)
    # parse the datamodule and method strings
    temp_args, _ = parser.parse_known_args(None if cmdline else [])

    # get the datamodule and method classes
    datamodule_class = getattr(datamodules, temp_args.datamodule)

    # add the appropriate args to the parse
    # for dataset, method, and trainer
    parser = datamodule_class.add_data_specific_args(parser)
    parser.set_defaults(**{'batch_size': 1})

    # parse and pass to main
    parser.set_defaults(**kwargs)
    args = parser.parse_args(None if cmdline else [])
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
        ...     'datamodule': 'WatchDataModule',
        ...     'workdir': ub.ensuredir((test_dpath, 'train')),
        ...     'package_fpath': package_fpath,
        ...     'max_epochs': 1,
        ...     'time_steps': 3,
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
        >>>     'datamodule': 'WatchDataModule',
        >>>     'batch_size': 1,
        >>>     'num_workers': 0,
        >>>     'gpus': gpus,
        >>> }
        >>> result_dataset = predict(**kwargs)
        >>> dset = result_dataset
        >>> # Check that the result format looks correct
        >>> for vidid in dset.index.videos.keys():
        >>>     # The first image in each video should not get predictions
        >>>     # (There is no change!)
        >>>     images = dset.images(dset.index.vidid_to_gids[1])
        >>>     aux_per_frame = list(map(len, images.lookup('auxiliary')))
        >>>     # Test number of auxiliary images
        >>>     first, *rest = aux_per_frame
        >>>     assert ub.allsame(rest)
        >>>     assert first == rest[0] - 1
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

    # init method from checkpoint
    method = utils.load_model_from_package(args.package_fpath)
    method.eval()
    method.freeze()

    # TODO: perhaps we should enforce that that packaged model
    # knows how to construct the appropriate test dataset?

    # init datamodule from args
    datamodule_class = getattr(datamodules, args.datamodule)
    datamodule_vars = utils.filter_args(
        vars(args),
        datamodule_class.__init__,
    )

    # TODO: default to this, but allow the user to overwrite
    datamodule_vars['chip_size'] = method.datamodule_hparams['chip_size']
    datamodule_vars['time_steps'] = method.datamodule_hparams['time_steps']
    datamodule_vars['channels'] = method.datamodule_hparams['channels']

    # datamodule_vars["preprocessing_step"] = method.preprocessing_step
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
        result_dataset.fpath = str(args.pred_dpath / 'pred.kwcoco.json')
    else:
        result_dataset.fpath = str(args.pred_dataset)

    # todo: use lightning device magic
    try:
        if int(args.gpus) == 0:
            device = torch.device('cpu')
        elif int(args.gpus) == 1:
            device = torch.device(0)
        else:
            raise ValueError('only 1 gpu for now')
    except Exception:
        device = torch.device('cpu')

    print('Predict on device = {!r}'.format(device))
    method = method.to(device)

    stitch_manager = CocoStitchingManager(
        result_dataset,
        chan_code=args.tag,
        stiching_space='video',
        device='numpy',  # could be torch on-device stitching
        thresh=args.thresh,
    )

    prog = ub.ProgIter(test_dataloader, desc='predicting', verbose=1)
    # nanns = 0

    result_infos = []

    def finalize_ready(gid):
        info = stitch_manager.finalize_image(gid)
        stats = running_stats.summarize(axis=None)
        stats = ub.dict_isect(stats, {'mean', 'max', 'min', 'std', 'n'})
        # extra = ub.repr2(stats, precision=6, nl=0)
        prog.ensure_newline()
        # prog.set_extra(extra)
        print('stats = {}'.format(ub.repr2(stats, nl=0, precision=4)))
        print('pred gid={} info = {}'.format(gid, ub.repr2(info, nl=0, precision=2)))
        result_infos.append(info)

    import kwarray
    running_stats = kwarray.RunningStats()  # for inspecting probability

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
        batch_bin_probs = outputs['binary_predictions']

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
                 device='numpy', thresh=0.5):
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

        # HACKS: SAVE_PREDS currently exists for debugging, and demoing
        self.SAVE_PROBS = 1
        self.SAVE_PREDS = 1

        if self.SAVE_PROBS:
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
            raise NotImplementedError

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
        vid_to_img = kwimage.Affine.coerce(img['warp_img_to_vid']).inv()

        num_pred_anns = 0
        total_prob = 0

        if self.SAVE_PROBS:
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
                'warp_aux_to_img': vid_to_img.inv().concise(),
            })
            # Save the prediction to disk
            total_prob += new_feature.sum()
            kwimage.imwrite(
                str(new_fpath), new_feature, space=None, backend='gdal',
                compress='LZW'
            )

        if self.SAVE_PREDS:
            # This is the final step where we convert soft-probabilities to
            # hard-polygons, we need to choose an good operating point here.

            # HACK: We happen to know this is the category atm.
            # Should have a better way to determine it via metadata
            change_cid = self.result_dataset.index.name_to_cat['change']['id']

            # Threshold scores
            change_pred = change_probs > self.thresh
            # Convert to polygons
            vid_polys = kwimage.Mask(change_pred, 'c_mask').to_multi_polygon()
            num_pred_anns = len(vid_polys)
            for vid_poly in vid_polys:
                # Compute a score for the polygon
                # TODO: This is very inefficient, should fix this
                w = vid_poly.to_mask(change_probs.shape).data
                score = (w * change_probs).sum() / w.sum()

                # Transform the video polygon into image space
                img_poly = vid_poly.warp(vid_to_img)
                bbox = list(img_poly.bounding_box().to_coco())[0]
                # Add the polygon as an annotation on the image
                self.result_dataset.add_annotation(
                    image_id=gid, category_id=change_cid,
                    bbox=bbox, segmentation=img_poly, score=score)

        return {
            'num_pred_anns': num_pred_anns,
            'total_prob': total_prob,
        }


def main(cmdline=True, **kwargs):
    predict(cmdline=cmdline, **kwargs)


if __name__ == "__main__":
    main()
