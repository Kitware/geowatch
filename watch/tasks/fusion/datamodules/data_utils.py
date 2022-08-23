import numpy as np


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
