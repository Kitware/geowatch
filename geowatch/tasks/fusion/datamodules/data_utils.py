"""
I dont like the name of this file. I want to rename it, but it exists to keep
the size of the datamodule down for now.
"""
import numpy as np
import ubelt as ub
import kwimage
import kwarray
import networkx as nx


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
    """
    key_hash = ub.hash_data(key, base=16, hasher='blake3').encode()
    key_tensor = np.frombuffer(memoryview(key_hash), dtype=np.int32).astype(np.float32)
    key_tensor = key_tensor / np.linalg.norm(key_tensor)
    return key_tensor


def _boxes_snap_to_edges(given_box, snap_target):
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

    adjusted_box = given_box.translate((xoffset, yoffset))
    return adjusted_box


class BalancedSampleTree(ub.NiceRepr):
    """
    Manages a sampling from a tree of indexes. Helps with balancing
    samples over multiple criteria.

    Example:
        >>> from geowatch.tasks.fusion.datamodules.data_utils import BalancedSampleTree
        >>> # Given a grid of sample locations and attribute information
        >>> # (e.g., region, category).
        >>> sample_grid = [
        >>>     { 'region': 'region1', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "purple" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "red" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "green" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "purple" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region1', 'category': 'rare',       'color': "red" },
        >>>     { 'region': 'region1', 'category': 'rare',       'color': "green" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "red" },
        >>>     { 'region': 'region1', 'category': 'background', 'color': "green" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "blue" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "purple" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "red" },
        >>>     { 'region': 'region2', 'category': 'background', 'color': "green" },
        >>>     { 'region': 'region2', 'category': 'rare',       'color': "purple" },
        >>>     { 'region': 'region2', 'category': 'rare',       'color': "blue" },
        >>> ]
        >>> #
        >>> # First we can just create a flat uniform sampling grid
        >>> # and inspect the imbalance that causes.
        >>> self = BalancedSampleTree(sample_grid)
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist0 = ub.dict_hist([(g['region'], g['category']) for g in sampled])
        >>> print('hist0 = {}'.format(ub.urepr(hist0, nl=1)))
        >>> #
        >>> # We can subdivide the indexes based on region to improve balance.
        >>> self.subdivide('region')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist1 = ub.dict_hist([(g['region'], g['category']) for g in sampled])
        >>> print('hist1 = {}'.format(ub.urepr(hist1, nl=1)))
        >>> #
        >>> # We can further subdivide by category.
        >>> self.subdivide('category')
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist2 = ub.dict_hist([(g['region'], g['category']) for g in sampled])
        >>> print('hist2 = {}'.format(ub.urepr(hist2, nl=1)))
        >>> #
        >>> # We can further subdivide by color, with custom weights.
        >>> weights = { 'red': .25, 'blue': .25, 'green': .4, 'purple': .1 }
        >>> self.subdivide('color', weights=weights)
        >>> print(f'self={self}')
        >>> sampled = list(ub.take(sample_grid, self._sample_many(100)))
        >>> hist3 = ub.dict_hist([
        >>>     (g['region'], g['category'], g['color']) for g in sampled
        >>> ])
        >>> print('hist3 = {}'.format(ub.urepr(hist3, nl=1)))
        >>> hist3_color = ub.dict_hist([(g['color']) for g in sampled])
        >>> print('color weights = {}'.format(ub.urepr(weights, nl=1)))
        >>> print('hist3 (color) = {}'.format(ub.urepr(hist3_color, nl=1)))
    """
    def __init__(self, sample_grid, rng=None):
        super().__init__()
        self.rng = rng = kwarray.ensure_rng(rng)

        # validate input
        if isinstance(sample_grid, list) and sample_grid:
            if isinstance(sample_grid[0], (dict, int, float)):
                self.graph = self._create_graph(sample_grid)
                self._leaf_nodes = [n for n in self.graph.nodes if self.graph.out_degree[n] == 0]
                return
        raise ValueError("""BalancedSampleTree only supports input in the
            form of a flat list or list of dicts.""")

    def _create_graph(self, sample_grid):
        graph = nx.DiGraph()

        # make a special root node
        root_node = '__root__'
        graph.add_node(root_node, weights=None)

        for index, item in enumerate(sample_grid):
            label = f'{index:02d} ' + ub.urepr(item, nl=0, compact=1, nobr=1)
            if isinstance(item, dict):
                graph.add_node(index, label=label, **item)
            else:
                graph.add_node(index, label=label)
            graph.add_edge(root_node, index)
        return graph

    def _get_parent(self, n):
        """ Get the parent of a node (assume a tree). None if it doesnt exist """
        preds = self.graph.pred[n]
        if len(preds):
            assert len(preds) == 1
            return next(iter(preds))
        else:
            return None

    def subdivide(self, key, weights=None):
        remove_edges = []
        add_edges = []
        add_nodes = []

        # Group all leaf nodes by their direct parents
        parent_to_leafs = ub.group_items(self._leaf_nodes, key=lambda n: self._get_parent(n))
        for parent, children in parent_to_leafs.items():
            # Group children by the new attribute
            val_to_subgroup = ub.group_items(children, lambda n:  self.graph.nodes[n][key])
            val_to_subgroup = ub.odict(sorted(val_to_subgroup.items()))
            if len(val_to_subgroup) == 1:
                # Dont need to do anything if no splits were made
                ...
            else:
                # Otherwise, we have to subdivide the children
                for value, subgroup in val_to_subgroup.items():
                    # Use a dotted name to make unambiguous tree splits
                    new_parent = f'{parent}.{key}={value}'
                    # Mark edges to add / remove to implement the split
                    remove_edges.extend([(parent, n) for n in subgroup])
                    add_edges.extend([(parent, new_parent) for n in subgroup])
                    add_edges.extend([(new_parent, n) for n in subgroup])
                    add_nodes.append(new_parent)

                # Add weights to the prior parent
                if weights is not None:
                    weights_group = np.asarray(list(ub.take(weights, val_to_subgroup.keys())))
                    denom = weights_group.sum()
                    if denom == 0:
                        raise NotImplementedError('Zero weighted branches are not handled yet.')
                    weights_group = weights_group / denom
                    self.graph.nodes[parent]['weights'] = weights_group
                else:
                    self.graph.nodes[parent]["weights"] = None

        # Modify the graph
        self.graph.remove_edges_from(remove_edges)
        self.graph.add_nodes_from(add_nodes, weights=None)
        self.graph.add_edges_from(add_edges)

    def _sample_many(self, num):
        for _ in range(num):
            idx = self.sample()
            yield idx

    def sample(self):
        current = '__root__'
        while self.graph.out_degree(current) > 0:
            children = list(self.graph.successors(current))
            num = len(children)

            weights = self.graph.nodes[current]['weights']
            if weights is None:
                idx = self.rng.randint(0, num)
            else:
                idx = self.rng.choice(num, 1, p=weights)[0]

            current = children[idx]
        return current

    def __len__(self):
        return len(list(self._leaf_nodes))

    def __nice__(self):
        n_nodes = self.graph.number_of_nodes()
        n_edges = self.graph.number_of_edges()
        n_leafs = self.__len__()
        n_depth = len(nx.algorithms.dag.dag_longest_path(self.graph))
        return f'nodes={n_nodes}, edges={n_edges}, leafs={n_leafs}, depth={n_depth}'


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
        >>> #kwplot.set_figtitle

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
