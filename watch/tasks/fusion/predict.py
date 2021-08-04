import pathlib
import ubelt as ub
import tqdm
import kwimage
import kwarray
from watch.tasks.fusion import datasets
from watch.tasks.fusion import utils


def make_predict_config(cmdline=False, **kwargs):
    """
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
    parser.add_argument("--dataset", default='WatchDataModule')
    parser.add_argument("--tag", default='fusion')
    parser.add_argument("--checkpoint_path", type=pathlib.Path)
    parser.add_argument("--results_dir", type=pathlib.Path, help='path to dump results')
    parser.add_argument("--use_gpu", action="store_true")

    parser.set_defaults(**kwargs)
    # parse the dataset and method strings
    temp_args, _ = parser.parse_known_args(None if cmdline else [])

    # get the dataset and method classes
    dataset_class = getattr(datasets, temp_args.dataset)

    # add the appropriate args to the parse
    # for dataset, method, and trainer
    parser = dataset_class.add_data_specific_args(parser)

    # parse and pass to main
    parser.set_defaults(**kwargs)
    args = parser.parse_args(None if cmdline else [])
    assert args.batch_size == 1
    return args


def main(cmdline=True, **kwargs):
    predict(cmdline=cmdline, **kwargs)


def predict(cmdline=False, **kwargs):
    """
    Example:
        >>> # Train a demo model (in the future grab a pretrained demo model)
        >>> from watch.tasks.fusion.fit import fit_model  # NOQA
        >>> from watch.tasks.fusion.predict import *  # NOQA
        >>> args = None
        >>> cmdline = False
        >>> gpus = 1
        >>> test_dpath = ub.ensure_app_cache_dir('watch/test/fusion/')
        >>> fit_kwargs = kwargs = {
        ...     'train_dataset': 'special:vidshapes8-multispectral',
        ...     'dataset': 'WatchDataModule',
        ...     'workdir': ub.ensuredir((test_dpath, 'train')),
        ...     'max_epochs': 1,
        ...     'max_steps': 1,
        ...     'learning_rate': 1e-5,
        ...     'num_workers': 0,
        ...     'gpus': gpus,
        ... }
        >>> package_fpath = fit_model(**fit_kwargs)
        >>> # Predict via that model
        >>> results_path = ub.ensuredir((test_dpath, 'predict'))
        >>> ub.delete(results_path)
        >>> ub.ensuredir(results_path)
        >>> predict_kwargs = kwargs = {
        >>>     'checkpoint_path': package_fpath,
        >>>     'results_dir': results_path,
        >>>     'test_dataset': 'special:vidshapes2-multispectral',
        >>>     'dataset': 'WatchDataModule',
        >>>     'batch_size': 1,
        >>>     'num_workers': 0,
        >>>     'use_gpu': gpus > 0,
        >>> }
        >>> results_ds = predict(**kwargs)
    """
    args = make_predict_config(cmdline=cmdline, **kwargs)

    # init method from checkpoint
    method = utils.load_model_from_package(args.checkpoint_path)
    method.eval()
    method.freeze()

    # init dataset from args
    dataset_class = getattr(datasets, args.dataset)
    dataset_var_dict = utils.filter_args(
        vars(args),
        dataset_class.__init__,
    )

    # TODO: default to this, but allow the user to overwrite
    dataset_var_dict['chip_size'] = method.datamodule_hparams['chip_size']
    dataset_var_dict['time_steps'] = method.datamodule_hparams['time_steps']

    # dataset_var_dict["preprocessing_step"] = method.preprocessing_step
    dataset = dataset_class(
        **dataset_var_dict
    )
    dataset.setup("test")

    test_coco_dataset = dataset.coco_datasets['test']
    test_dataset = dataset.torch_datasets['test']
    test_dataloader = dataset.test_dataloader()

    T, H, W = test_dataset.sample_shape

    # Create the results dataset as a copy of the test dataset
    results_ds = test_coco_dataset.copy()

    # Remove all annotations in the results copy
    results_ds.clear_annotations()

    # Change all paths to be absolute paths
    results_ds.reroot(absolute=True)
    change_cid = results_ds.ensure_category("change")

    # Set the filepath for the prediction coco file
    # (modifies the bundle_dpath)
    results_ds.fpath = str(args.results_dir / 'pred.kwcoco.json')

    if args.use_gpu:
        device = 0
    else:
        device = 'cpu'
    method = method.to(device)

    # HACKS: SAVE_PREDS currently exists for debugging
    SAVE_PROBS = 1
    SAVE_PREDS = 1

    if SAVE_PROBS:
        prob_dpath = (args.results_dir / f'_assets/{args.tag}')
        ub.ensuredir(str(prob_dpath))

    # Setup a dictionary that we will use to make a stitcher for each video as
    # needed. We use the fact that videos are iterated over sequentially so
    # free up memory of a video after it completes.
    vidid_to_stitchers = {}
    prev_video_id = None
    for batch in ub.ProgIter(test_dataloader, desc='predicting', verbose=0):

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

            # TODO: if the preditions are downsampled wrt to the input images,
            # we need to determine what that transform is so we can correctly
            # warp the predictions back into image space.
            bin_probs = batch_bin_probs[bx].detach().cpu().numpy()

            # Get the spatio-temporal subregion that this prediction belongs to
            video_id = item['video_id']
            in_gids = [frame['gid'] for frame in item['frames']]
            space_slice = tuple(item['tr']['space_slice'])
            # NOTE: the returned tr space slice seems bugged?
            # tuple(item['tr']['space_slice'])

            if video_id not in vidid_to_stitchers:
                # new video-space canvas for each image prediction in the video
                video = results_ds.index.videos[video_id]
                vid_space_dims = (video['height'], video['width'])
                vidid_to_stitchers[video_id] = {
                    gid: kwarray.Stitcher(vid_space_dims, device='numpy')
                    for gid in results_ds.index.vidid_to_gids[video_id]
                }

            # Update the stitcher with this windowed prediction
            stitchers = vidid_to_stitchers[video_id]
            for gid, probs in zip(in_gids[1:], bin_probs):
                stitchers[gid].add(space_slice, probs)

            if prev_video_id != video_id:
                if prev_video_id is not None:
                    # Finalize any stitchers that are complete
                    finalize_video_predictions(args, SAVE_PROBS, SAVE_PREDS,
                                               prob_dpath, results_ds,
                                               change_cid, vidid_to_stitchers,
                                               prev_video_id)
            prev_video_id = video_id

    # Finalize the last video
    if prev_video_id in vidid_to_stitchers:
        finalize_video_predictions(args, SAVE_PROBS, SAVE_PREDS, prob_dpath,
                                   results_ds, change_cid, vidid_to_stitchers,
                                   prev_video_id)

    # validate and save results
    print(results_ds.validate())
    results_ds.dump(results_ds.fpath)
    return results_ds


def finalize_video_predictions(args, SAVE_PROBS, SAVE_PREDS, prob_dpath,
                               results_ds, change_cid, vidid_to_stitchers,
                               prev_video_id):
    """
    TODO: This might be better written as a class that can maintain the
    different stitchers with a better API.
    """
    # If we see a new video id, we have finished stitching the
    # previous video.
    prev_stitchers = vidid_to_stitchers.pop(prev_video_id)
    for gid, stitcher in prev_stitchers.items():

        if stitcher.weights.max() == 0:
            # This will skip the first frame where change is not predicted
            continue

        # Get the final probs for this image in video space
        change_probs = stitcher.finalize()
        img = results_ds.index.imgs[gid]

        # Get spatial relationship between the image and the video
        vid_to_img = kwimage.Affine.coerce(img['warp_img_to_vid']).inv()

        if SAVE_PROBS:
            # This mostly exists as an example, for what pre-fusion TA-2
            # modules should do to add new auxiliary information to a
            # kwcoco file.
            #
            # Save probabilitys (or feature maps) as a new auxiliary image
            new_feature = kwarray.atleast_nd(change_probs, 3)
            new_fname = img['name'] + f'_{args.tag}.tiff'
            new_fpath = prob_dpath / new_fname
            img['auxiliary'].append({
                'file_name': str(new_fpath.relative_to(args.results_dir)),
                'channels': args.tag,
                'height': new_feature.shape[0],
                'width': new_feature.shape[1],
                'num_bands': new_feature.shape[2],
                'warp_aux_to_img': vid_to_img.inv().concise(),
            })
            # Save the prediction to disk
            kwimage.imwrite(
                str(new_fpath), new_feature, space=None, backend='gdal',
                compress='LZW')

        if SAVE_PREDS:
            # This is the final step where we convert soft-probabilities to
            # hard-polygons, we need to choose an appropriate operating
            # point here.
            #
            # Threshold scores
            thresh = 0.05  # HARD CODED, CONFIGURE
            change_pred = change_probs > thresh
            # Convert to polygons
            vid_polys = kwimage.Mask(change_pred, 'c_mask').to_multi_polygon()
            for vid_poly in vid_polys:

                # Compute a score for the polygon
                # TODO: This is very inefficient, should fix this
                w = vid_poly.to_mask(change_probs.shape).data
                score = (w * change_probs).sum() / w.sum()

                # Transform the video polygon into image space
                img_poly = vid_poly.warp(vid_to_img)
                # Add the polygon as an annotatio on the image
                results_ds.add_annotation(image_id=gid, category_id=change_cid,
                                          segmentation=img_poly, score=score)


if __name__ == "__main__":
    main()
