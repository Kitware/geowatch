"""
I dont like the name of this file. I want to rename it, but it exists to keep
the size of the datamodule down for now.

TODO:
    - [ ] Break BalancedSampleTree and BalancedSampleForest into their own balanced sampling module.
    - [ ] Make a good augmentation module
    - [ ] Determine where MultiscaleMask should live.
"""
import numpy as np
import ubelt as ub
import kwimage
import kwarray


try:
    from line_profiler import profile
except Exception:
    profile = ub.identity


@ub.memoize
@profile
def resolve_scale_request(request=None, data_gsd=None):
    """
    Helper for handling user and machine specified spatial scale requests

    Args:
        request (None | float | str):
            Indicate a relative or absolute requested scale.  If given as a
            float, this is interpreted as a scale factor relative to the
            underlying data.  If given as a string, it will accept the format
            "{:f} *GSD" and resolve to an absolute GSD.  Defaults to 1.0.

        data_gsd (None | float):
            if specified, this indicates the GSD of the underlying data.
            (Only valid for geospatial data). TODO: is there a better
            generalization?

    Returns:
        Dict[str, Any] : resolved : containing keys
            scale (float): the scale factor to obtain the requested
            gsd (float | None): if data_gsd is given, this is the absolute
                GSD of the request.

    Note:
        The returned scale is relative to the DATA. If you are resizing a
        sampled image, then use it directly, but if you are adjusting a sample
        WINDOW, then it needs to be used inversely.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> resolve_scale_request(1.0)
        >>> resolve_scale_request('native')
        >>> resolve_scale_request('10 GSD', data_gsd=10)
        >>> resolve_scale_request('20 GSD', data_gsd=10)

    Example:
        >>> from geowatch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> import ubelt as ub
        >>> grid = list(ub.named_product({
        >>>     'request': ['10GSD', '30GSD'],
        >>>     'data_gsd': [10, 30],
        >>> }))
        >>> grid += list(ub.named_product({
        >>>     'request': [None, 1.0, 2.0, 0.25, 'native'],
        >>>     'data_gsd': [None, 10, 30],
        >>> }))
        >>> for kwargs in grid:
        >>>     print('kwargs = {}'.format(ub.urepr(kwargs, nl=0)))
        >>>     resolved = resolve_scale_request(**kwargs)
        >>>     print('resolved = {}'.format(ub.urepr(resolved, nl=0)))
        >>>     print('---')

    """
    # FIXME: rectify with util_resolution
    final_gsd = None
    final_scale = None

    if request is None:
        final_scale = 1.0
        final_gsd = data_gsd
    elif isinstance(request, str):
        if request == 'native':
            final_gsd = 'native'
            final_scale = 'native'
        elif request.lower().endswith('gsd'):
            if data_gsd is None:
                raise ValueError(
                    'The request was given in terms of GSD, but '
                    'the underlying data GSD was unspecified')
            final_gsd = float(request[:-3].strip())
            final_scale = data_gsd / final_gsd
        else:
            final_scale = float(request)
    else:
        final_scale = float(request)

    if final_gsd is None:
        if data_gsd is not None:
            final_gsd = np.array(data_gsd) / final_scale

    resolved = {
        'scale': final_scale,
        'gsd': final_gsd,
        'data_gsd': data_gsd,
    }
    return resolved


def abslog_scaling(arr):
    orig_sign = np.nan_to_num(np.sign(arr))
    shifted = np.abs(arr) + 1
    shifted = np.log(shifted)
    shifted[np.isnan(shifted)] = 0.1
    return orig_sign * shifted


def fliprot(img, rot_k=0, flip_axis=None, axes=(0, 1)):
    """
    Args:
        img (ndarray): H, W, C

        rot_k (int): number of ccw rotations

        flip_axis(Tuple[int, ...]):
            either [], [0], [1], or [0, 1].
            0 is the y axis and 1 is the x axis.

        axes (Typle[int, int]): the location of the y and x axes

    Example:
        >>> img = np.arange(16).reshape(4, 4)
        >>> unique_fliprots = [
        >>>     {'rot_k': 0, 'flip_axis': None},
        >>>     {'rot_k': 0, 'flip_axis': (0,)},
        >>>     {'rot_k': 1, 'flip_axis': None},
        >>>     {'rot_k': 1, 'flip_axis': (0,)},
        >>>     {'rot_k': 2, 'flip_axis': None},
        >>>     {'rot_k': 2, 'flip_axis': (0,)},
        >>>     {'rot_k': 3, 'flip_axis': None},
        >>>     {'rot_k': 3, 'flip_axis': (0,)},
        >>> ]
        >>> for params in unique_fliprots:
        >>>     img_fw = fliprot(img, **params)
        >>>     img_inv = inv_fliprot(img_fw, **params)
        >>>     assert np.all(img == img_inv)
    """
    if rot_k != 0:
        img = np.rot90(img, k=rot_k, axes=axes)
    if flip_axis is not None:
        _flip_axis = np.asarray(axes)[flip_axis]
        img = np.flip(img, axis=_flip_axis)
    return img


def fliprot_annot(annot, rot_k, flip_axis=None, axes=(0, 1), canvas_dsize=None):
    """
    Ignore:
        >>> from geowatch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> import kwimage
        >>> H, W = 121, 153
        >>> canvas_dsize = (W, H)
        >>> box1 = kwimage.Boxes.random(1).scale((W, H)).quantize()
        >>> ltrb = box1.data
        >>> rot_k = 4
        >>> annot = box1
        >>> annot = box1.to_polygons()[0]
        >>> annot1 = annot.copy()
        >>> unique_fliprots = [
        >>>     {'rot_k': 0, 'flip_axis': None},
        >>>     {'rot_k': 0, 'flip_axis': (0,)},
        >>>     {'rot_k': 1, 'flip_axis': None},
        >>>     {'rot_k': 1, 'flip_axis': (0,)},
        >>>     {'rot_k': 2, 'flip_axis': None},
        >>>     {'rot_k': 2, 'flip_axis': (0,)},
        >>>     {'rot_k': 3, 'flip_axis': None},
        >>>     {'rot_k': 3, 'flip_axis': (0,)},
        >>> ]
        >>> results = []
        >>> for params in unique_fliprots:
        >>>     annot2 = fliprot_annot(annot, canvas_dsize=canvas_dsize, **params)
        >>>     annot3 = inv_fliprot_annot(annot2, canvas_dsize=canvas_dsize, **params)
        >>>     results.append({
        >>>         'annot2': annot2,
        >>>         'annot3': annot3,
        >>>         'params': params,
        >>>     })

        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> image1 = kwimage.grab_test_image('astro', dsize=(W, H))
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(results))
        >>> for result in results:
        >>>     image2 = fliprot(image1.copy(), **result['params'])
        >>>     image3 = inv_fliprot(image2.copy(), **result['params'])
        >>>     annot2 = result['annot2']
        >>>     annot3 = result['annot3']
        >>>     canvas1 = annot1.draw_on(image1.copy(), edgecolor='kitware_green', fill=False)
        >>>     canvas2 = annot2.draw_on(image2.copy(), edgecolor='kitware_blue', fill=False)
        >>>     canvas3 = annot3.draw_on(image3.copy(), edgecolor='kitware_red', fill=False)
        >>>     canvas = kwimage.stack_images([canvas1, canvas2, canvas3], axis=1)
        >>>     kwplot.imshow(canvas, pnum=pnum_(), title=ub.urepr(result['params'], nl=0, compact=1, nobr=1))
    """
    # TODO: can use the new `Affine.fliprot` when 0.9.22 releases

    import kwimage
    if rot_k != 0:
        x0 = canvas_dsize[0] / 2
        y0 = canvas_dsize[1] / 2
        # generalized way
        # Translate center of old canvas to the origin
        T1 = kwimage.Affine.translate((-x0, -y0))
        # Construct the rotation
        tau = np.pi * 2
        theta = -(rot_k * tau / 4)
        R = kwimage.Affine.rotate(theta=theta)
        # Find the center of the new rotated canvas
        canvas_box = kwimage.Box.from_dsize(canvas_dsize)
        new_canvas_box = canvas_box.warp(R)
        x2 = new_canvas_box.width / 2
        y2 = new_canvas_box.height / 2
        # Translate to the center of the new canvas
        T2 = kwimage.Affine.translate((x2, y2))
        # print(f'T1=\n{ub.urepr(T1)}')
        # print(f'R=\n{ub.urepr(R)}')
        # print(f'T2=\n{ub.urepr(T2)}')
        A = T2 @ R @ T1
        annot = annot.warp(A)
        # TODO: specialized faster way
        # lt_x, lt_y, rb_x, rb_y = boxes.components
    else:
        x2 = y2 = None

    # boxes = kwimage.Boxes(ltrb, 'ltrb')
    if flip_axis is not None:
        if x2 is None:
            x2 = canvas_dsize[0] / 2
            y2 = canvas_dsize[1] / 2
        # Make the flip matrix
        F = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for axis in flip_axis:
            mdim = 1 - axis
            F[mdim, mdim] *= -1
        T1 = kwimage.Affine.translate((-x2, -y2))
        T2 = kwimage.Affine.translate((x2, y2))
        A = T2 @ F @ T1
        annot = annot.warp(A)

    return annot


def inv_fliprot_annot(annot, rot_k, flip_axis=None, axes=(0, 1), canvas_dsize=None):
    if rot_k % 2 == 1:
        canvas_dsize = canvas_dsize[::-1]
    annot = fliprot_annot(annot, -rot_k, flip_axis=None, axes=axes, canvas_dsize=canvas_dsize)
    if rot_k % 2 == 1:
        canvas_dsize = canvas_dsize[::-1]
    annot = fliprot_annot(annot, 0, flip_axis=flip_axis, axes=axes, canvas_dsize=canvas_dsize)
    return annot


def inv_fliprot(img, rot_k=0, flip_axis=None, axes=(0, 1)):
    """
    Undo a fliprot

    Args:
        img (ndarray): H, W, C
    """
    if flip_axis is not None:
        _flip_axis = np.asarray(axes)[flip_axis]
        img = np.flip(img, axis=_flip_axis)
    if rot_k != 0:
        img = np.rot90(img, k=-rot_k, axes=axes)
    return img


@ub.memoize
def _string_to_hashvec(key):
    """
    Transform a string into a 16D float32 uniformly distributed random Tensor
    based on the hash of the string.

    Note there are magic numbers hard-coded in this function, and is the reason
    for the blake3 dependency. Would likely be better to make it configurable
    and use sha256 as the default.
    """
    key_hash = ub.hash_data(key, base=16, hasher='blake3').encode()
    key_tensor = np.frombuffer(memoryview(key_hash), dtype=np.int32).astype(np.float32)
    key_tensor = key_tensor / np.linalg.norm(key_tensor)
    return key_tensor


def _boxes_snap_to_edges(given_box, snap_target, inplace=False):
    """
    Ignore:
        >>> from geowatch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> import kwimage
        >>> from geowatch.tasks.fusion.datamodules.data_utils import _string_to_hashvec, _boxes_snap_to_edges
        >>> from geowatch.tasks.fusion.datamodules.data_utils import _boxes_snap_to_edges
        >>> snap_target = kwimage.Boxes([[0, 0, 10, 10]], 'ltrb')
        >>> given_box = kwimage.Boxes([[-3, 5, 3, 13]], 'ltrb')
        >>> adjusted_box = _boxes_snap_to_edges(given_box, snap_target)
        >>> print('adjusted_box = {!r}'.format(adjusted_box))

        _boxes_snap_to_edges(kwimage.Boxes([[-3, 3, 20, 13]], 'ltrb'), snap_target)
        _boxes_snap_to_edges(kwimage.Boxes([[-3, -3, 3, 3]], 'ltrb'), snap_target)
        _boxes_snap_to_edges(kwimage.Boxes([[7, 7, 15, 15]], 'ltrb'), snap_target)
    """
    s_x1, s_y1, s_x2, s_y2 = snap_target.components
    g_x1, g_y1, g_x2, g_y2 = given_box.components

    xoffset1 = -np.minimum((g_x1 - s_x1), 0)
    yoffset1 = -np.minimum((g_y1 - s_y1), 0)

    xoffset2 = np.minimum((s_x2 - g_x2), 0)
    yoffset2 = np.minimum((s_y2 - g_y2), 0)

    xoffset = (xoffset1 + xoffset2).ravel()[0]
    yoffset = (yoffset1 + yoffset2).ravel()[0]

    adjusted_box = given_box.translate((xoffset, yoffset), inplace=inplace)
    return adjusted_box


def samecolor_nodata_mask(stream, hwc, relevant_bands, use_regions=0,
                          samecolor_values=None):
    """
    Find a 2D mask that indicates what values should be set to nan.
    This is typically done by finding clusters of zeros in specific bands.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> import kwcoco
        >>> import kwarray
        >>> stream = kwcoco.FusedChannelSpec.coerce('foo|red|green|bar')
        >>> stream_oset = ub.oset(stream)
        >>> relevant_bands = ['red', 'green']
        >>> relevant_band_idxs = [stream_oset.index(b) for b in relevant_bands]
        >>> rng = kwarray.ensure_rng(0)
        >>> hwc = (rng.rand(32, 32, stream.numel()) * 3).astype(int)
        >>> use_regions = 0
        >>> samecolor_values = {0}
        >>> samecolor_mask = samecolor_nodata_mask(
        >>>     stream, hwc, relevant_bands, use_regions=use_regions,
        >>>     samecolor_values=samecolor_values)
        >>> assert samecolor_mask.sum() == (hwc[..., relevant_band_idxs] == 0).any(axis=2).sum()
    """
    from geowatch.utils import util_kwimage
    stream_oset = ub.oset(stream)
    relevant_band_idxs = [stream_oset.index(b) for b in relevant_bands]
    relevant_masks = []
    for b_sl in relevant_band_idxs:
        bands = hwc[:, :, b_sl]
        bands = np.ascontiguousarray(bands)
        if use_regions:
            # Speed up the compuation by doing this at a coarser scale
            is_samecolor = util_kwimage.find_samecolor_regions(
                bands, scale=0.4, min_region_size=49,
                values=samecolor_values)
        else:
            # Faster histogram method
            is_samecolor = util_kwimage.find_high_frequency_values(
                bands, values=samecolor_values)
        relevant_masks.append(is_samecolor)

    if len(relevant_masks) == 1:
        samecolor_mask = relevant_masks[0]
    else:
        samecolor_mask = (np.stack(relevant_masks, axis=2) > 0).any(axis=2)
    return samecolor_mask


class MultiscaleMask:
    """
    A helper class to build up a mask indicating what pixels are unobservable
    based on data from different resolution.

    In othe words, if you have multiple masks, and each mask has a different
    resolution, then this will iteravely upscale the masks to the largest
    resolution so far and perform a logical or. This helps keep the memory
    footprint small.

    TODO:
        Does this live in kwimage?

    CommandLine:
        xdoctest -m geowatch.tasks.fusion.datamodules.data_utils MultiscaleMask --show

    Example:
        >>> from geowatch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> image = kwimage.grab_test_image()
        >>> image = kwimage.ensure_float01(image)
        >>> rng = kwarray.ensure_rng(1)
        >>> mask1 = kwimage.Mask.random(shape=(12, 12), rng=rng).data
        >>> mask2 = kwimage.Mask.random(shape=(32, 32), rng=rng).data
        >>> mask3 = kwimage.Mask.random(shape=(16, 16), rng=rng).data
        >>> omask = MultiscaleMask()
        >>> omask.update(mask1)
        >>> omask.update(mask2)
        >>> omask.update(mask3)
        >>> masked_image = omask.apply(image, np.nan)
        >>> # Now we can use our upscaled masks on an image.
        >>> masked_image = kwimage.fill_nans_with_checkers(masked_image, on_value=0.3)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> inputs = kwimage.stack_images(
        >>>     [kwimage.atleast_3channels(m * 255) for m in [mask1, mask2, mask3]],
        >>>     pad=2, bg_value='kw_green', axis=1)
        >>> kwplot.imshow(inputs, pnum=(1, 3, 1), title='input masks')
        >>> kwplot.imshow(omask.mask, pnum=(1, 3, 2), title='final mask')
        >>> kwplot.imshow(masked_image, pnum=(1, 3, 3), title='masked image')
        >>> kwplot.show_if_requested()
    """

    def __init__(self):
        self.mask = None
        self._fraction = None

    def update(self, mask):
        """
        Expand the observable mask to the larger data and take the logical or
        of the resized masks.
        """
        self._fraction = None
        if len(mask.shape) > 2:
            if len(mask.shape) != 3 or mask.shape[2] != 1:
                raise ValueError(f'bad mask shape {mask.shape}')
            mask = mask[..., 0]
        if self.mask is None:
            self.mask = mask
        else:
            mask1 = self.mask
            mask2 = mask
            dsize1 = mask1.shape[0:2][::-1]
            dsize2 = mask2.shape[0:2][::-1]
            if dsize1 != dsize2:
                area1 = np.prod(dsize1)
                area2 = np.prod(dsize2)
                if area2 > area1:
                    mask1, mask2 = mask2, mask1
                    dsize1, dsize2 = dsize2, dsize1
                # Enlarge the smaller mask
                mask2 = mask2.astype(np.uint8)
                mask2 = kwimage.imresize(mask2, dsize=dsize1,
                                         interpolation='nearest')
            self.mask = np.logical_or(mask1, mask2)

    def apply(self, image, value):
        """
        Set the locations in ``image`` that correspond to this mask to
        ``value``.
        """
        mask = self.mask
        if mask is None:
            return image
        dsize1 = image.shape[0:2][::-1]
        dsize2 = mask.shape[0:2][::-1]
        if dsize1 != dsize2:
            # Ensure the mask corresponds to the image size
            mask = mask.astype(np.uint8)
            mask = kwimage.imresize(mask, dsize=dsize1,
                                    interpolation='nearest')
        mask = kwarray.atleast_nd(mask, 3)
        mask = mask.astype(bool)
        assert mask.shape[2] == 1
        mask = np.broadcast_to(mask, image.shape)
        image[mask] = value
        return image

    @property
    def masked_fraction(self):
        if self._fraction is None:
            if self.mask is None:
                self._fraction = 0
            else:
                self._fraction = self.mask.mean()
        return self._fraction
