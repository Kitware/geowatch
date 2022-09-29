"""
I dont like the name of this file. I want to rename it, but it exists to keep
the size of the datamodule down for now.
"""
import numpy as np
import ubelt as ub
import kwarray


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
        >>> from watch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> resolve_scale_request(1.0)
        >>> resolve_scale_request('native')
        >>> resolve_scale_request('10 GSD', data_gsd=10)
        >>> resolve_scale_request('20 GSD', data_gsd=10)

    Example:
        >>> from watch.tasks.fusion.datamodules.data_utils import *  # NOQA
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
        >>>     print('kwargs = {}'.format(ub.repr2(kwargs, nl=0)))
        >>>     resolved = resolve_scale_request(**kwargs)
        >>>     print('resolved = {}'.format(ub.repr2(resolved, nl=0)))
        >>>     print('---')

    """
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
            final_gsd = np.array(data_gsd) * final_scale

    resolved = {
        'scale': final_scale,
        'gsd': final_gsd,
        'data_gsd': data_gsd,
    }
    return resolved


def polygon_distance_transform(poly, shape, dtype):
    """
    Example:
        import cv2
        import kwimage
        poly = kwimage.Polygon.random().scale(32)
        poly_mask = np.zeros((32, 32), dtype=np.uint8)
        poly_mask = poly.fill(poly_mask, value=1)
        dist = cv2.distanceTransform(poly_mask, cv2.DIST_L2, 3)
        ###
        import kwplot
        kwplot.autompl()
        kwplot.imshow(dist, cmap='viridis', doclf=1)
        poly.draw(fill=0, border=1)
    """
    import cv2
    poly_mask = np.zeros_like(shape)
    poly_mask = poly.fill(poly_mask, value=1)
    dist = cv2.distanceTransform(
        src=poly_mask, distanceType=cv2.DIST_L2, maskSize=3)
    return dist


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
    """
    key_hash = ub.hash_data(key, base=16, hasher='blake3').encode()
    key_tensor = np.frombuffer(memoryview(key_hash), dtype=np.int32).astype(np.float32)
    key_tensor = key_tensor / np.linalg.norm(key_tensor)
    return key_tensor


def _boxes_snap_to_edges(given_box, snap_target):
    """
    Ignore:
        >>> from watch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> import kwimage
        >>> from watch.tasks.fusion.datamodules.data_utils import _string_to_hashvec, _boxes_snap_to_edges
        >>> from watch.tasks.fusion.datamodules.data_utils import _boxes_snap_to_edges
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

    adjusted_box = given_box.translate((xoffset, yoffset))
    return adjusted_box


class NestedPool(list):
    """
    Example:
        >>> from watch.tasks.fusion.datamodules.data_utils import *  # NOQA
        >>> nested1 = NestedPool([[[1], [2, 3], [4, 5, 6], [7, 8, 9, 0]], [[11, 12, 13]]])
        >>> print({nested1.sample() for i in range(100)})
        >>> nested2 = NestedPool([[101], [102, 103], [104, 105, 106], [107, 8, 9, 0]])
        >>> print({nested2.sample() for i in range(100)})
        >>> nested3 = NestedPool([nested1, nested2, [4, 59, 9, [], []]])
        >>> print({nested3.sample() for i in range(100)})
        >>> print(ub.repr2(ub.dict_hist(nested3.sample() for i in range(100))))
    """
    def __init__(nested, pools, rng=None):
        super().__init__(pools)
        nested.rng = rng = kwarray.ensure_rng(rng)
        nested.pools = pools

    def sample(nested):
        # Hack for empty lists
        chosen = nested
        i = 0
        while ub.iterable(chosen):
            chosen = nested
            i += 1
            while ub.iterable(chosen):
                i += 1
                num = len(chosen)
                if i > 100000:
                    raise Exception('Too many samples. Bad balance?')
                if not num:
                    break
                idx = nested.rng.randint(0, num)
                chosen = chosen[idx]

        return chosen
