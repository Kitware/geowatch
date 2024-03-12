"""
Defines the :class:`CocoStitchingManager`, which stitches predictions from
subregions of an image or video (or more generally - data cube) back into
rasters corresponding to the original data. This requires that the user use a
sliding window (e.g. perhaps defined by :class:`kwarray.SlidingWindow`) to
iterate over space/time, produce predictions, and know the coordinates those
predictions should be stitched back together at.

The following attempts to provide a minimal example with a visualization.

CommandLine:
    xdoctest -m geowatch.tasks.fusion.coco_stitcher __doc__:0

Example:
    >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
    >>> from geowatch.tasks.fusion.coco_stitcher import demo_coco_stitching_manager
    >>> # See the contents of the function for details, might port it to a full
    >>> # doctest later.
    >>> demo_coco_stitching_manager()
"""


import ubelt as ub
import numpy as np
import kwimage
import kwarray
import warnings
from os.path import relpath


def demo_coco_stitching_manager():
    import kwcoco
    from geowatch.tasks.fusion.datamodules.kwcoco_dataset import KWCocoVideoDataset

    # This test will write to this output directory and we seed our RNG
    out_dpath = ub.Path.appdir('geowatch/kwcoco_stitcher/demo')
    rng = kwarray.ensure_rng(0)

    # Given some kwcoco dataset
    coco_dset = kwcoco.CocoDataset.demo('vidshapes', num_videos=1, num_frames=4, image_size=(128, 128))

    result_dataset = coco_dset.copy()
    result_dataset.reroot(absolute=True)

    # Tell the result dataset where it will live.
    # The stitcher will write into this bundle directory.
    result_dataset.fpath = out_dpath / 'demo_stitched.kwcoco.zip'

    # We are going to predict a new raster feature. We will call it "demofeat"
    # and it will have three channels: demochan.0, demochan.1, demochan.2.
    # These features will be native to video space.
    stitcher = CocoStitchingManager(
        result_dataset=result_dataset,
        short_code='demofeat',
        chan_code='demochan.0|demochan.1|demochan.2',
        stiching_space='video'
    )

    # We will use the :class:`KWCocoVideoDataset` to handle the sliding window
    # You don't have to, but it is handy.
    dataset = KWCocoVideoDataset(
        coco_dset, time_dims=3, window_dims=(96, 96),
        window_overlap=0.3,
        channels='r|g|b',
        mode='test',
        time_sampling='uniform',
    )

    # There needs to be some loop that iterates over spacetime windows where
    # predictions are generated. It is then our job to pass those predictions
    # to the stitcher.
    for idx in range(len(dataset)):
        # Get a single batch item
        item = dataset[idx]

        # For each frame in the batch, accumulate its predictions
        for frame_info in item['frames']:

            # The KWCocoVideoDataset gives us information about where the
            # predicted output should live in output space.
            image_id = frame_info['gid']
            output_image_dsize = frame_info['output_image_dsize']
            output_space_slice = frame_info['output_space_slice']
            scale_outspace_from_vid = frame_info['scale_outspace_from_vid']
            output_weights = frame_info.get('output_weights', None)

            # Generate a fake prediction for this frame
            prob_h, prob_w = frame_info['output_dims']
            fake_prediction = frame_info['modes']['r|g|b'] > 200
            mask = (fake_prediction.all(axis=0).numpy()).astype(np.float32)
            probs = rng.rand(prob_h, prob_w, 3) * mask[:, :, None]
            probs = kwimage.gaussian_blur(probs)
            # Add in noticable edge effects
            probs[:4, :, 0] = 1
            probs[-4:, :, 1] = 1
            probs[:, :4, 2] = 1
            probs[:, -4:, 2:] = 1

            # Tell the stitcher where the probabilities should be placed in the
            # larger context.
            stitcher.accumulate_image(
                image_id, output_space_slice, probs,
                dsize=output_image_dsize,
                scale=scale_outspace_from_vid,
                weights=output_weights,
                downweight_edges=1,
            )

    # The user needs to call finalize when they are done with an image.
    # In this case we just stitch the entire thing and call finalize on
    # everything at the end.
    for image_id in result_dataset.images():
        stitcher.finalize_image(image_id)

    # The stitcher modified the result dataset inplace. Dump it to disk.
    result_dataset.dump()

    if 1:
        # Visualize the stitched predictions.
        ub.cmd(f'geowatch visualize {result_dataset.fpath} --channels="r|g|b,demochan.0:3" --stack=True', system=1)


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
            TODO: rename to prefix? OR or something more indicative that this
            is a directory name?

        chan_code (str):
            If saving the stitched features, this is the channel code to use.

        stiching_space (str):
            Indicates if the results are given in image or video space (up to a
            scale factor).

        device ('numpy' | torch.device):
            Device to stitch on.

        thresh (float):
            if making hard decisions, determines the threshold for converting a
            soft mask into a hard mask, which can be converted into a polygon.

        prob_compress (str):
            Compression algorithm to use when writing probabilities to disk.
            Can be any GDAL compression code, e.g LZW, DEFLATE, RAW, etc.

        prob_blocksize (int):
            tiled blocksize for output predictions. Defaults to 128.

        prob_format (str):
            the format of the output images. (png, tif, cog).

        polygon_categories (List[str] | None):
            These are the list of channels that should be transformed into
            polygons. If not set, all are used.

        quantize (bool):
            if True quantize heatmaps before writing them to disk

        expected_minmax (Tuple[float, float]):
            The expected minimum and maximum values allowed in the output
            to be stitched -- i.e. (0, 1) for probabilities. If unspecified
            this is infered per image.

        writer_queue (None | BlockingJobQueue):
            if specified, uses this shared writer queue, otherwise creates
            its own.

        write_prediction_attrs (bool):
            set to True if you are adding predictions to the kwcoco file,
            otherwise set to False to remove unnecessary attributes.

        dtype (str): the dtype to stitch over. Defaults to 'float32'

        assets_dname (str):
            The name of the top-level directory to write new assets. Defaults
            to _assets

    TODO:
        - [ ] Handle the case where the input space is related to the output
              space by an affine transform.

        - [X] Handle stitching in image space

        - [X] Handle the case where we are only stitching over images

        - [ ] Handle the case where iteration is non-contiguous, i.e. define
              a robust criterion to determine when an image is "done" being
              stitched.

        - [ ] Perhaps separate the "soft-probability" prediction stitcher
              from (a) the code that converts soft-to-hard predictions (b)
              the code that adds hard predictions to the kwcoco file and (c)
              the code that adds soft predictions to the kwcoco file?

        - [ ] TODO: remove polygon "predictions" from this completely.

    Example:
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
        >>> import geowatch
        >>> dset = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True,
        >>>                            multispectral=True)
        >>> result_dataset = dset.copy()
        >>> self = CocoStitchingManager(
        >>>     result_dataset=result_dataset,
        >>>     short_code='demofeat',
        >>>     chan_code='df1|df2',
        >>>     prob_format='png',
        >>>     stiching_space='video')
        >>> coco_img = result_dataset.images().coco_images[0]
        >>> # Compute a feature in 0.5 video space for a subset of an image
        >>> gid = coco_img.img['id']
        >>> hidden = coco_img.imdelay(space='video').finalize().mean(axis=2)
        >>> my_feature = kwimage.imresize(hidden, scale=0.5)
        >>> asset_dsize = my_feature.shape[0:2][::-1]
        >>> space_slice = None
        >>> self.accumulate_image(gid, space_slice, my_feature,
        >>>                       asset_dsize=asset_dsize,
        >>>                       scale_asset_from_stitchspace=0.5)
        >>> self.finalize_image(gid)
        >>> # The new auxiliary image is now in our result dataset
        >>> result_img = result_dataset.coco_image(gid)
        >>> print(ub.urepr(result_img.img, nl=-1))
        >>> assert 'df1' in result_img.channels
        >>> im1 = result_img.imdelay('df1', space='video')
        >>> im2 = result_img.imdelay(channels='df1', space='asset')
        >>> assert im1.shape[0] == hidden.shape[0]
        >>> assert im2.shape[0] == my_feature.shape[0]

    Example:
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
        >>> import geowatch
        >>> dset = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True,
        >>>                            multispectral=True)
        >>> result_dataset = dset.copy()
        >>> self = CocoStitchingManager(
        >>>     result_dataset=result_dataset,
        >>>     short_code='demofeat',
        >>>     chan_code='df1|df2',
        >>>     stiching_space='image')
        >>> coco_img = result_dataset.images().coco_images[0]
        >>> # Compute a feature in 0.5 image space for a subset of an image
        >>> gid = coco_img.img['id']
        >>> hidden = coco_img.imdelay(space='image').finalize().mean(axis=2)
        >>> my_feature = kwimage.imresize(hidden, scale=0.5)
        >>> asset_dsize = my_feature.shape[0:2][::-1]
        >>> space_slice = None
        >>> self.accumulate_image(gid, space_slice, my_feature,
        >>>                       asset_dsize=asset_dsize,
        >>>                       scale_asset_from_stitchspace=0.5)
        >>> self.finalize_image(gid)
        >>> # The new auxiliary image is now in our result dataset
        >>> result_img = result_dataset.coco_image(gid)
        >>> print(ub.urepr(result_img.img, nl=-1))
        >>> assert 'df1' in result_img.channels
        >>> im1 = result_img.imdelay('df1', space='image')
        >>> im2 = result_img.imdelay(channels='df1', space='asset')
        >>> assert im1.shape[0] == 600
        >>> assert im2.shape[0] == 300
    """

    def __init__(self,
                 result_dataset,
                 short_code=None,
                 chan_code=None,
                 stiching_space='video',
                 device='numpy',
                 thresh=0.5,
                 write_probs=True,
                 write_preds=False,
                 num_bands='auto',
                 prob_compress='DEFLATE',
                 prob_blocksize=128,
                 prob_format='cog',
                 polygon_categories=None,
                 expected_minmax=None,
                 quantize=True,
                 writer_queue=None,
                 write_prediction_attrs=True,
                 assets_dname='_assets',
                 dtype='float32'):

        from kwutil import util_parallel
        self.short_code = short_code
        self.result_dataset = result_dataset
        self.device = device
        self.chan_code = chan_code
        self.thresh = thresh
        self.num_bands = num_bands
        self.imwrite_kwargs = {
            'compress': prob_compress,
            'blocksize': prob_blocksize,
        }
        self.prob_format = prob_format
        self.polygon_categories = polygon_categories
        self.quantize = quantize
        self.expected_minmax = expected_minmax
        self.write_prediction_attrs = write_prediction_attrs
        self.dtype = dtype
        self.assets_dname = assets_dname

        if writer_queue is None:
            # basic queue if nothing fancy is given
            writer_queue = util_parallel.BlockingJobQueue(
                mode='serial', max_workers=0)
        self.writer_queue = writer_queue

        self.suffix_code = (
            self.chan_code if '|' not in self.chan_code else
            ub.hash_data(self.chan_code)[0:16]
        )

        self.stiching_space = stiching_space
        if stiching_space not in {'video', 'image'}:
            raise NotImplementedError(stiching_space)

        # Setup a dictionary that we will use to make a stitcher for each image
        # as needed.  We use the fact that videos are iterated over
        # sequentially so free up memory of a video after it completes.
        self.image_stitchers = {}
        self._image_scales = {}  # TODO: should be a more general transform
        self._seen_gids = set()
        self._last_vidid = None
        self._last_imgid = None
        self._ready_gids = set()
        # The set of image ids that are currently being finalized
        self._finalizing_gids = set()
        # The set of image ids that have been finalized
        self._finalized_gids = set()

        # Keep track of the number of times we've stitched something into an
        # image.
        self._stitched_gid_patch_histograms = ub.ddict(lambda: 0)

        # TODO: writing predictions and probabilities needs robustness work
        self.write_probs = write_probs
        self.write_preds = write_preds

        if self.write_preds:
            ub.schedule_deprecation(
                'geowatch', 'write_preds', 'needs a different abstraction.',
                deprecate='now')
            from kwcoco import channel_spec
            chan_spec = channel_spec.FusedChannelSpec.coerce(chan_code)
            if self.polygon_categories is None:
                self.polygon_categories = chan_spec.parsed
            # Determine the indexes that we will use for polygon extraction
            _idx_lut = {c: idx for idx, c in enumerate(chan_spec.parsed)}
            self.polygon_idxs = [_idx_lut[c] for c in self.polygon_categories]

        if self.write_probs:
            bundle_dpath = ub.Path(self.result_dataset.bundle_dpath)
            prob_subdir = f'{self.assets_dname}/{self.short_code}'
            self.prob_dpath = (bundle_dpath / prob_subdir).ensuredir()

    def accumulate_image(self, gid, space_slice, data, asset_dsize=None,
                         scale_asset_from_stitchspace=None, is_ready='auto',
                         weights=None, downweight_edges=False, **kwargs):
        """
        Stitches a result into the appropriate image stitcher.

        Args:
            gid (int):
                the image id to stitch into

            space_slice (Tuple[slice, slice] | None):
                the slice (in "output-space") the data corresponds to.
                if None, assumes this is for the entire image.

            data (ndarray | Tensor): the feature or probability data

            asset_dsize (Tuple): the w/h of outputspace
                (i.e. the asset we will write)

            scale_asset_from_stitchspace (float | None):
                the scale to the outspace from from the stitching (i.e.
                image/video) space.

            is_ready (bool): todo, fix this to work better

        Note:
            Output space is asset space for the new asset we are building.
            The actual stitcher holds data in outspace / assetspace.
            May want to adjust termonology here.
        """
        if kwargs.get('dsize', None) is not None:
            asset_dsize = kwargs.get('dsize', None)
            if 0:
                ub.schedule_deprecation(
                    'geowatch', 'dsize', 'arg of accumulate_image',
                    'use asset_dsize instad', deprecate='now')

        if kwargs.get('scale', None) is not None:
            scale_asset_from_stitchspace = kwargs.get('scale', None)
            if 0:
                ub.schedule_deprecation(
                    'geowatch', 'scale', 'arg of accumulate_image',
                    'use scale_asset_from_stitchspace instad', deprecate='now')

        self._stitched_gid_patch_histograms[gid] += 1
        data = kwarray.atleast_nd(data, 3)
        dset = self.result_dataset
        img = dset.index.imgs[gid]
        if self.stiching_space == 'video':
            vidid = img.get('video_id', None)
            # Create the stitcher if it does not exist
            if gid not in self.image_stitchers:
                if asset_dsize is None:
                    if vidid is None:
                        # Assume fake video space
                        height, width = img['height'], img['width']
                    else:
                        video = dset.index.videos[vidid]
                        height, width = video['height'], video['width']
                else:
                    width, height = asset_dsize
                if self.num_bands == 'auto':
                    if len(data.shape) == 3:
                        self.num_bands = data.shape[2]
                    else:
                        raise NotImplementedError
                asset_dims = (height, width, self.num_bands)
                self.image_stitchers[gid] = kwarray.Stitcher(
                    asset_dims, device=self.device, dtype=self.dtype)
                self._image_scales[gid] = scale_asset_from_stitchspace

            if is_ready == 'auto':
                is_ready = self._last_vidid is not None and vidid != self._last_vidid

            if is_ready:
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
        elif self.stiching_space == 'image':
            # Create the stitcher if it does not exist
            vidid = img.get('video_id', None)
            if gid not in self.image_stitchers:
                if asset_dsize is None:
                    height, width = img['height'], img['width']
                else:
                    width, height = asset_dsize
                if self.num_bands == 'auto':
                    if len(data.shape) == 3:
                        self.num_bands = data.shape[2]
                    else:
                        raise NotImplementedError
                asset_dims = (height, width, self.num_bands)
                self.image_stitchers[gid] = kwarray.Stitcher(
                    asset_dims, device=self.device)
                self._image_scales[gid] = scale_asset_from_stitchspace

            if is_ready == 'auto':
                is_ready = self._last_imgid is not None and gid != self._last_imgid
            if is_ready:
                # Assuming read if the last image has changed
                # This check needs a rework
                self._ready_gids.add(self._last_imgid)
        else:
            raise NotImplementedError(self.stiching_space)

        self._last_imgid = gid
        self._last_vidid = vidid

        stitcher: kwarray.Stitcher = self.image_stitchers[gid]

        asset_space_slice = space_slice
        self._stitcher_center_weighted_add(stitcher, asset_space_slice, data,
                                           weights,
                                           downweight_edges=downweight_edges)

    @staticmethod
    def _stitcher_center_weighted_add(stitcher, asset_space_slice, data,
                                      weights=None, downweight_edges=False):
        """
        TODO: refactor
        """
        from geowatch.utils import util_kwimage

        if weights is not None:
            weights = kwarray.ArrayAPI.numpy(weights)
            # Hack, the dataloader should always provide weights aligned with
            # the output, but we have offbyone errors, so just force things to
            # work while we figure those out.
            data, weights = _force_shape_agreement_by_cropping2d(data, weights)

        if downweight_edges:
            _center_weights = util_kwimage.upweight_center_mask(data.shape[0:2])

            if weights is None:
                weights = _center_weights
            else:
                weights = weights * _center_weights

        if weights is None:
            # TODO: allow weights to be None for stitching performance in this
            # case.
            weights = np.ones(data.shape[0:2], dtype=np.float32)

        is_2d = len(data.shape) == 2
        is_3d = len(data.shape) == 3

        if asset_space_slice is None:
            # Assume this data is for the entire image.
            h, w = stitcher.shape[0:2]
            asset_space_slice = kwimage.Box.from_dsize((w, h)).to_slice()

        if is_3d:
            weights = weights[..., None]

        shapes_disagree = (
            stitcher.shape[0] < asset_space_slice[0].stop or
            stitcher.shape[1] < asset_space_slice[1].stop
        )
        if shapes_disagree:
            # By embedding the space slice in the stitcher dimensions we can
            # get a slice corresponding to the valid region in the stitcher,
            # and the extra padding encodes the valid region of the data we are
            # trying to stitch into.
            subslice, padding = kwarray.embed_slice(asset_space_slice[0:2],
                                                    stitcher.shape[0:2])

            slice_h = (asset_space_slice[0].stop - asset_space_slice[0].start)
            slice_w = (asset_space_slice[1].stop - asset_space_slice[1].start)
            # data.shape[0]
            # data.shape[1]
            _fixup_slice = (
                slice(padding[0][0], slice_h - padding[0][1]),
                slice(padding[1][0], slice_w - padding[1][1]),
            )
            subdata = data[_fixup_slice]
            subweights = weights[_fixup_slice]

            asset_slice = subslice
            asset_data = subdata
            asset_weights = subweights
        else:
            # Normal case
            asset_slice = asset_space_slice
            asset_data = data
            asset_weights = weights

        # Handle stitching nan values
        invalid_output_mask = np.isnan(asset_data)
        if np.any(invalid_output_mask):
            if is_3d:
                spatial_valid_mask = (
                    1 - invalid_output_mask.any(axis=2, keepdims=True))
            else:
                assert is_2d
                spatial_valid_mask = (1 - invalid_output_mask)
            asset_weights = asset_weights * spatial_valid_mask
            asset_data[invalid_output_mask] = 0

        asset_slice = fix_slice(asset_slice)

        HACK_FIX_SHAPE = 1
        if HACK_FIX_SHAPE:
            # Something is causing an off by one error, not sure what it is
            # this hack just forces the slice to agree.
            dh, dw = asset_data.shape[0:2]
            box = kwimage.Box.from_slice(asset_slice)
            sw, sh = box.dsize
            if sw > dw:
                box = box.resize(width=dw)
            if sh > dh:
                box = box.resize(height=dh)
            if sw < dw:
                asset_data = asset_data[:, 0:sw]
                asset_weights = asset_weights[:, 0:sw]
            if sh < dh:
                asset_data = asset_data[0:sh]
                asset_weights = asset_weights[0:sh]
            asset_slice = box.to_slice()

        try:
            stitcher.add(asset_slice, asset_data, weight=asset_weights)
        except IndexError:
            print(f'asset_slice={asset_slice}')
            print(f'asset_weights.shape={asset_weights.shape}')
            print(f'asset_data.shape={asset_data.shape}')
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

    def submit_finalize_image(self, gid):
        """
        Like finalize image, but submits the job to the manager's writer queue,
        which could be asynchronous.
        """
        self._finalizing_gids.add(gid)
        self.writer_queue.submit(self.finalize_image, gid)

    def flush_images(self):
        """
        Allow the writer queue to finish finalizing any incomplete images
        before allowing the process to procede.
        """
        self.writer_queue.wait_until_finished()

    @property
    def seen_image_ids(self):
        return self._seen_gids

    def finalize_image(self, gid):
        """
        Finalizes the stitcher for this image, deletes it, and adds
        its hard and/or soft predictions to the CocoDataset.

        Args:
            gid (int): the image-id to finalize
        """
        import os
        self._finalizing_gids.add(gid)

        # Remove this image from the managed set.
        img = self.result_dataset.index.imgs[gid]

        self._ready_gids.difference_update({gid})

        try:
            # stitcher = self.image_stitchers.get(gid)
            stitcher = self.image_stitchers.pop(gid)
        except KeyError:
            if gid in self._seen_gids:
                raise KeyError((
                    'Attempted to finalize image gid={}, but we already '
                    'finalized it').format(gid))
            else:
                raise KeyError((
                    'Attempted to finalize image gid={}, but no data '
                    'was ever accumulated for it ').format(gid))

        self._seen_gids.add(gid)

        scale_asset_from_stitchspace = self._image_scales.pop(gid)

        # Get the final stitched feature for this image
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered')
            final_probs = stitcher.finalize()
        final_probs = kwarray.atleast_nd(final_probs, 3)

        # NOTE: could find and record the valid prediction regions.
        # Given a (rectilinear) non-convex multipolygon where we are guarenteed
        # that all of the angles in the polygon are right angles, what is an
        # efficient algorithm to decompose it into a minimal set of disjoint
        # rectangles?
        # https://stackoverflow.com/questions/5919298/
        # Or... just write out a polygon... KISS
        # Mark that we made a prediction on this image.
        if self.write_prediction_attrs:
            final_weights = kwarray.atleast_nd(stitcher.weights, 3)
            is_predicted_pixel = final_weights.any(axis=2).astype('uint8')
            _mask = kwimage.Mask(is_predicted_pixel, 'c_mask')
            _poly = _mask.to_multi_polygon()
            predicted_region = _poly.to_geojson()
            img['prediction_region'] = predicted_region
            img['has_predictions'] = ub.udict.union(
                img.get('has_predictions', {}),
                {self.chan_code: True}
            )

        # Get spatial relationship between the stitch space and image space
        if self.stiching_space == 'video':
            vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', {'type': 'affine'}))
            img_from_stitch = vid_from_img.inv()
        elif self.stiching_space == 'image':
            img_from_stitch = kwimage.Affine.eye()
        else:
            raise AssertionError

        n_anns = 0
        total_prob = 0

        asset_from_stitch = kwimage.Affine.coerce(scale=scale_asset_from_stitchspace)
        stitch_from_asset = asset_from_stitch.inv()
        img_from_asset = img_from_stitch @ stitch_from_asset

        if self.write_probs:
            # This currently exists as an example to demonstrate how a
            # prediction script can write a pre-fusion TA-2 feature to disk and
            # register it with the kwcoco file.

            if self.prob_format == 'cog':
                prob_ext = '.tif'
                imwrite_backend = 'gdal'
            elif self.prob_format == 'png':
                prob_ext = f'.{self.prob_format}'
                imwrite_backend = 'cv2'
            else:
                raise KeyError(f'unknown prob_format={self.prob_format!r}')
            #
            # Save probabilities (or feature maps) as a new auxiliary image
            bundle_dpath = self.result_dataset.bundle_dpath
            new_fname = (  # FIXME
                img.get('name', str(img['id'])) +
                f'_{self.suffix_code}{prob_ext}'
            )
            new_fpath = self.prob_dpath / new_fname

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

            total_prob += np.nansum(final_probs)

            # Save the prediction to disk
            write_kwargs = self.imwrite_kwargs.copy()

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
                if prob_ext != '.tif':
                    warnings.warn(
                        'Cannot save geospatial information unless the '
                        'output format is cog / tif')

            if self.prob_format == 'png':
                write_kwargs.clear()
                quantize_dtype = np.uint8
            else:
                quantize_dtype = np.int16
                write_kwargs['metadata'] = {
                    'channels': self.chan_code,
                }

            if self.quantize:
                # Quantize
                if self.expected_minmax is None:
                    old_min, old_max = None, None
                else:
                    old_min, old_max = self.expected_minmax
                quant_probs, quantization = quantize_image(
                    final_probs, old_min=old_min, old_max=old_max,
                    quantize_dtype=quantize_dtype)
                write_data = quant_probs
                aux['quantization'] = quantization
            else:
                write_data = final_probs
                quantization = None

            if self.prob_format != 'png':
                write_kwargs['metadata'] = {
                    'quantization': quantization,
                }

            try:
                kwimage.imwrite(
                    os.fspath(new_fpath), write_data, space=None, backend=imwrite_backend,
                    **write_kwargs,
                )
            except Exception as ex:
                print()
                print('ERROR ex = {}'.format(ub.urepr(ex, nl=1)))
                print('new_fpath = {}'.format(ub.urepr(new_fpath, nl=1)))
                print(f'imwrite_backend={imwrite_backend}')
                print(f'write_data.shape={write_data.shape}')
                print(f'write_data.dtype={write_data.dtype}')
                print(f'write_kwargs={write_kwargs}')
                raise

        if self.write_preds:
            from geowatch.tasks.tracking.utils import mask_to_polygons
            ub.schedule_deprecation(
                'geowatch', 'write_preds', 'needs a different abstraction.',
                deprecate='now')
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
                for score, asset_poly in scored_polys:
                    # Transform the video polygon into image space
                    img_poly = asset_poly.warp(img_from_asset)
                    bbox = list(img_poly.box().boxes.to_coco())[0]
                    # Add the polygon as an annotation on the image
                    self.result_dataset.add_annotation(
                        image_id=gid, category_id=cid,
                        bbox=bbox, segmentation=img_poly, score=score)

        info = {
            'n_anns': n_anns,
            'total_prob': total_prob,
        }
        self._finalized_gids.add(gid)
        return info


def quantize_image(imdata, old_min=None, old_max=None, quantize_dtype=np.int16):
    """
    New version of quantize_float01

    TODO:
        - [ ] How does this live relative to dequantize in delayed image?
        It seems they should be tied somehow.

    Args:
        imdata (ndarray): image data to quantize

        old_min (float | None):
            a stanard floor for minimum values to make quantization consistent
            across images. If unspecified chooses the minimum value in the
            data.

        old_max (float | None):
            a stanard ceiling for maximum values to make quantization
            consistent across images. If unspecified chooses the maximum value
            in the data.

        quantize_dtype (dtype):
            which type of integer to quantize as

    Returns:
        Tuple[ndarray, Dict] - new data with encoding information

    Note:
        Setting old_min / old_max indicates the possible extend of the input
        data (and it will be clipped to it). It does not mean that the input
        data has to have those min and max values, but it should be between
        them.

    Example:
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
        >>> from delayed_image.helpers import dequantize
        >>> # Test error when input is not nicely between 0 and 1
        >>> imdata = (np.random.randn(32, 32, 3) - 1.) * 2.5
        >>> quant1, quantization1 = quantize_image(imdata)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))
        >>> #
        >>> for i in range(1, 20):
        >>>     print('i = {!r}'.format(i))
        >>>     quant2, quantization2 = quantize_image(imdata, old_min=-i, old_max=i)
        >>>     recon2 = dequantize(quant2, quantization2)
        >>>     error2 = np.abs((recon2 - imdata)).sum()
        >>>     print('error2 = {!r}'.format(error2))

    Example:
        >>> # Test dequantize with uint8
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
        >>> from delayed_image.helpers import dequantize
        >>> imdata = np.random.randn(32, 32, 3)
        >>> quant1, quantization1 = quantize_image(imdata, quantize_dtype=np.uint8)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))

    Example:
        >>> # Test quantization with different signed / unsigned combos
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
        >>> print(quantize_image(None, 0, 1, np.int16))
        >>> print(quantize_image(None, 0, 1, np.int8))
        >>> print(quantize_image(None, 0, 1, np.uint8))
        >>> print(quantize_image(None, 0, 1, np.uint16))
    """
    if imdata is None:
        if old_min is None and old_max is None:
            old_min = 0
            old_max = 1
        elif old_min is None:
            old_min = old_max - 1
        elif old_max is None:
            old_max = old_min + 1
    else:
        invalid_mask = np.isnan(imdata)
        if old_min is None or old_max is None:
            valid_data = imdata[~invalid_mask].ravel()
            if len(valid_data) > 0:
                if old_min is None:
                    old_min = int(np.floor(valid_data.min()))
                if old_max is None:
                    old_max = int(np.ceil(valid_data.max()))
            else:
                old_min = 0
                old_max = 1

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
        new_imdata = (
            (imdata.clip(old_min, old_max) - old_min) * quant_factor +
            quantize_min)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered')
            new_imdata = new_imdata.astype(quantize_dtype)
        new_imdata[invalid_mask] = quantize_nan
    else:
        new_imdata = None

    return new_imdata, quantization


def quantize_float01(imdata, old_min=0, old_max=1, quantize_dtype=np.int16):
    """
    DEPRECATE IN FAVOR OF quantize_image

    Note:
        Setting old_min / old_max indicates the possible extend of the input
        data (and it will be clipped to it). It does not mean that the input
        data has to have those min and max values, but it should be between
        them.

    Example:
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
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
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
        >>> from delayed_image.helpers import dequantize
        >>> imdata = np.random.randn(32, 32, 3)
        >>> quant1, quantization1 = quantize_float01(imdata, old_min=0, old_max=1,
        >>>                                          quantize_dtype=np.uint8)
        >>> recon1 = dequantize(quant1, quantization1)
        >>> error1 = np.abs((recon1 - imdata)).sum()
        >>> print('error1 = {!r}'.format(error1))

    Example:
        >>> # Test quantization with different signed / unsigned combos
        >>> from geowatch.tasks.fusion.coco_stitcher import *  # NOQA
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
        new_imdata = (
            imdata.clip(old_min, old_max) - old_min) * quant_factor + quantize_min
        new_imdata = new_imdata.astype(quantize_dtype)
        new_imdata[invalid_mask] = quantize_nan
    else:
        new_imdata = None

    return new_imdata, quantization


def fix_slice(sl):
    if isinstance(sl, slice):
        return _fix_slice(sl)
    elif isinstance(sl, (tuple, list)) and isinstance(ub.peek(sl), slice):
        return _fix_slice_tup(sl)
    else:
        raise TypeError(repr(sl))


def _fix_int(d):
    return None if d is None else int(d)


def _fix_slice(d):
    return slice(_fix_int(d.start), _fix_int(d.stop), _fix_int(d.step))


def _fix_slice_tup(sl):
    return tuple(map(_fix_slice, sl))


def _force_shape_agreement_by_cropping2d(data1, data2):
    """
    I feel like I've written this before.

    Args:
        data1 (ndarray): data with ndim >= 2, first two dims are height / width
        data2 (ndarray): data with ndim >= 2, first two dims are height / width
    """
    if data1.shape[0:2] != data2.shape[0:2]:
        h1, w1 = data1.shape[0:2]
        h2, w2 = data2.shape[0:2]

        dh = abs(h1 - h2)
        dw = abs(w1 - w2)

        if dh > 10 or dw > 10:

            import kwutil
            IGNORE_OFF_BY_ONE_STITCHING = kwutil.util_environ.envflag('IGNORE_OFF_BY_ONE_STITCHING', False)
            if not IGNORE_OFF_BY_ONE_STITCHING:
                raise AssertionError(
                    'This function is for hacking away off-by-one-errors, '
                    'but the difference in shapes was too large: '
                    f'data1.shape={data1.shape}, data1.shape={data2.shape}')
        h3 = min(h1, h2)
        w3 = min(w1, w2)
        new_data1 = data1[0:h3, 0:w3]
        new_data2 = data2[0:h3, 0:w3]
    else:
        new_data1 = data1
        new_data2 = data2
    return new_data1, new_data2
