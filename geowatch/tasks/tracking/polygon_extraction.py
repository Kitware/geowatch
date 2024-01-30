import portion  # NOQA
import ubelt as ub


class PolygonExtractor:
    """
    Given a timesequence of heatmaps, extract spatially static polygons.

    This class is being developed on ``dev/refine-ac-polys`` in the file:
    geowatch/tasks/tracking/polygon_extraction.py

    Distributed Tweaking on:
        https://colab.research.google.com/drive/1NEJpm36LviesZb45qy59myezi7JHu0bI#scrollTo=G8kHgCXSI3VS

    Example:
        >>> from geowatch.tasks.tracking.polygon_extraction import *  # NOQA
        >>> cls = PolygonExtractor
        >>> self = PolygonExtractor.demo(real_categories=True)
        >>> self.config['algo'] = 'crall'
        >>> print(f'self.heatmap_thwc.shape={self.heatmap_thwc.shape}')

        >>> polys, info = self.predict_polygons(return_info=True)
        >>> label_mask = info['label_mask']

        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> from geowatch.utils import util_kwimage
        >>> kwplot.autompl()
        >>> stacked = self.draw_timesequence()
        >>> canvas = label_mask.colorize()
        >>> for p in polys:
        >>>     canvas = p.draw_on(canvas, fill=0, edgecolor='black')
        >>> pnum_ = kwplot.PlotNums(nSubplots=2)
        >>> kwplot.imshow(stacked, pnum=pnum_(), title='Colorized Time Series')
        >>> kwplot.imshow(canvas, pnum=pnum_(), title='Cluster Labels')
        >>> kwplot.show_if_requested()
    """

    def __init__(self, heatmap_thwc, heatmap_time_intervals=None, bounds=None,
                 classes=None, config=None):
        """
        Args:

            heatmap_thwc (ndarray): An heatmap with dimensions [T, H, W, C]
                where T is number of timesteps and C are feature map reponces
                for each channel.

            heatmap_time_intervals (None | List[Tuple[datetime, datetime]]):
                A list of length T rows. Each row corresponds to a timestep
                in the heatmaps, and contains a tuple of two datetimes:
                corresponding to the start and end of times that
                information corresponds to. If a heatmap only corrseponds to a
                single time, then start and end will be the same.

            bounds (kwimage.MultiPolygon | None):
                if given, this is the polygon detected at BAS time.

            classes (None | List[str]):
                If specified, this corresponds to the class in each position
                of the C dimension.

            config (None | Dict):
                optional configuration

        """
        self.heatmap_thwc = heatmap_thwc
        self.bounds = bounds
        self.classes = classes
        self.heatmap_time_intervals = heatmap_time_intervals

        # For recording intermediate parts of the process.
        self._intermediates = {}

        self.default_config = {
            # 'algo': 'meanshift',
            'algo': 'kmeans',
            'workers': 'avail',
            'robust_normalize': False,
            'positional_encoding': True,
            'positional_encoding_scale': 1.0,
            'scale_factor': 'auto',
            'thresh': 0.3,
            'viz_out_dir': None,
        }
        if config is None:
            config = {}
        self.config = self.default_config | config

    def _predict_leotta(self):
        import numpy as np
        from scipy import ndimage
        heatmap_thwc = self.heatmap_thwc
        assert self.classes is not None

        # import kwarray
        # heatmap_thwc = kwarray.robust_normalize(heatmap_thwc)

        _t, _h, _w, _c = self.heatmap_thwc.shape
        if self.bounds is not None:
            mask = self.bounds.to_mask(dims=(_h, _w)).data

        salient_idx = self.classes.index('ac_salient')
        active_index = self.classes.index('Active Construction')
        # active_index = 2

        # commbine active and saliency and remove NANs
        active_vol = heatmap_thwc[:, :, :, active_index]
        saliency_vol = heatmap_thwc[:, :, :, salient_idx]
        vol = impute_nans2(np.multiply(active_vol, saliency_vol)) * mask[None, :, :]

        # morphological clean-up, a little more in the time direction
        vol = ndimage.grey_closing(vol, (7, 3, 3))
        vol = ndimage.grey_opening(vol, (7, 3, 3))

        # Threshold and connected components
        leotta_thresh = self.config.get('leotta_thresh', 0.3)
        vol_label, label_count = ndimage.label(vol > leotta_thresh)

        # Flatten over time, max label is a heuristic
        max_label = np.max(vol_label, 0)

        # count the number of pixels for each label in 2D

        # remove labels (back in 3D) that were too small
        if 0:
            idx, cnts = np.unique(max_label, return_counts=True)
            to_remove = idx[cnts < 50]
            vol_label[np.isin(vol_label, to_remove)] = 0

        # flatten again without the smaller blobs
        label_img = np.max(vol_label, 0)
        # idx, cnts = np.unique(max_label, return_counts=True)
        return label_img

    def _predict_crall(self):
        import rich
        from geowatch.utils import util_kwimage
        from scipy import ndimage
        import kwarray
        import numpy as np
        raw_heatmap = self.heatmap_thwc

        SHOW = 0
        scale_factor = self.config['scale_factor']
        thresh = self.config['thresh']
        if scale_factor == 'auto':
            scale_factor = 4

        _t, _h, _w, _c = self.heatmap_thwc.shape

        if self.bounds is not None:
            mask = self.bounds.to_mask(dims=(_h, _w)).data
        else:
            mask = None

        if self.config['viz_out_dir']:
            SHOW = 1

        if SHOW:
            import kwplot
            from geowatch.utils import util_kwplot
            dpath = self.config['viz_out_dir']
            finalizer = util_kwplot.FigureFinalizer(dpath, cropwhite=False)

        _n = [1]
        def PRINT_STEP(msg, _n=_n):
            import rich
            n = _n[0]
            rich.print(f'Step {n}: {msg}')
            _n[0] += 1

        rich.print(f'* Given: raw_heatmap.shape={raw_heatmap.shape}')

        # TODO: better downscaling?
        downscaled = raw_heatmap[:, ::scale_factor, ::scale_factor, :]
        if mask is not None:
            small_mask = mask[::scale_factor, ::scale_factor]
        else:
            small_mask = None
        PRINT_STEP(f'Downscale by {scale_factor}x to: {downscaled.shape}')
        downscaled = downscaled.copy()

        PRINT_STEP('Impute NaN')
        imputed = impute_nans2(downscaled)
        rich.print('... Finished Impute')

        if mask is not None:
            PRINT_STEP('Masking Small Heatmap')
            masked = imputed * small_mask[None, :, :, None]
        else:
            masked = imputed
        cube = FeatureCube(masked,
                           TimeIntervalSequence.coerce(self.heatmap_time_intervals),
                           self.classes)
        agg_cube = cube.window_max('12 months')

        if SHOW:
            fig, _ = kwplot.imshow(agg_cube.draw(), title='Input Feature Cube')
            finalizer.finalize(fig, f'step_{_n[0]:03d}_agg_cube.png')

        if 'ac_salient' in self.classes:
            cube1 = agg_cube.take_channels('ac_salient')
        else:
            cube1 = agg_cube.take_channels('salient')

        # cube2 = agg_cube.take_channels('No Activity')
        if 'Site Preparation' in self.classes:
            cube3 = agg_cube.take_channels('Site Preparation')
        else:
            cube3 = None

        if 'Active Construction' in self.classes:
            cube4 = agg_cube.take_channels('Active Construction')
        else:
            cube4 = None

        if 1:
            # hack for old model
            if cube4 is not None:
                mask = (cube1.heatmap_thwc == 0)
                cube1.heatmap_thwc[mask] = cube4.heatmap_thwc[mask]
        # cube5 = agg_cube.take_channels('Post Construction')

        if cube4 is None:
            cube8 = cube1
        else:
            cube8 = cube4 * cube1
        # cube6 = cube2 * cube1
        # cube7 = cube3 * cube1
        # cube9 = cube5 * cube1

        PRINT_STEP('Morphology')
        # cube1.heatmap_thwc = ndimage.grey_closing(cube1.heatmap_thwc, (3, 5, 5, 1))
        # cube1.heatmap_thwc = ndimage.grey_opening(cube1.heatmap_thwc, (3, 5, 5, 1))
        if cube4 is not None:
            cube4.heatmap_thwc = ndimage.grey_closing(cube4.heatmap_thwc, (3, 5, 5, 1))
            cube4.heatmap_thwc = ndimage.grey_opening(cube4.heatmap_thwc, (3, 5, 5, 1))

        # Spatial dilation
        cube1.heatmap_thwc = ndimage.grey_dilation(cube1.heatmap_thwc, (1, 5, 5, 1))

        if cube4 is not None:
            cube4.heatmap_thwc = ndimage.grey_dilation(cube4.heatmap_thwc, (1, 5, 5, 1))
        # cube8.heatmap_thwc = ndimage.grey_closing(cube8.heatmap_thwc, (3, 5, 5, 1))
        # cube8.heatmap_thwc = ndimage.grey_opening(cube8.heatmap_thwc, (3, 5, 5, 1))

        # cube2.heatmap_thwc = (cube2.heatmap_thwc > thresh).astype(np.float32)
        if cube3 is not None:
            cube3.heatmap_thwc = (cube3.heatmap_thwc > thresh).astype(np.float32)
        if cube4 is not None:
            cube4.heatmap_thwc = (cube4.heatmap_thwc > thresh).astype(np.float32)
        # cube5.heatmap_thwc = (cube5.heatmap_thwc > thresh).astype(np.float32)

        # cube6.heatmap_thwc = (cube6.heatmap_thwc > thresh).astype(np.float32)
        # cube7.heatmap_thwc = (cube7.heatmap_thwc > thresh).astype(np.float32)
        cube8.heatmap_thwc = (cube8.heatmap_thwc > thresh).astype(np.float32)
        # cube9.heatmap_thwc = (cube9.heatmap_thwc > thresh).astype(np.float32)

        if SHOW:
            fig2 = kwplot.figure(fnum=2, doclf=1)
            cubes = [cube1, cube4, cube8]
            ncubes = sum([c is not None for c in cubes])
            pnum = kwplot.PlotNums(nSubplots=ncubes)
            if cube1 is not None:
                kwplot.imshow(cube1.draw(), pnum=pnum(), fnum=2, title='Salient')
            if cube4 is not None:
                kwplot.imshow(cube4.draw(), pnum=pnum(), fnum=2, title='Morphed Active')
            if cube8 is not None:
                kwplot.imshow(cube8.draw(), pnum=pnum(), fnum=2, title='Binarized Active * Salient')
            finalizer.finalize(fig2, f'step_{_n[0]:03d}_cubes.png')

        PRINT_STEP('Max Saliency')
        max_saliency = cube1.heatmap_thwc.max(axis=0)[..., 0]
        max_saliency_stats = kwarray.stats_dict(max_saliency)
        print('max_saliency_stats = {}'.format(ub.urepr(max_saliency_stats, nl=1)))

        PRINT_STEP('Volume Labeling')
        vol_label, label_count = ndimage.label(cube8.heatmap_thwc > thresh)
        # Remove small seeds
        idx, cnts = np.unique(vol_label, return_counts=True)
        max_cnts = max(cnts)
        if max_cnts > 200:
            print(f'max_cnts = {ub.urepr(max_cnts, nl=1)}')
            count_thresh = min(50, max_cnts)
            to_remove = idx[cnts < count_thresh]
            n_remove = sum(to_remove)
            print(f'n_remove={n_remove}')
            vol_label[np.isin(vol_label, to_remove)] = 0

        labels1 = vol_label.max(axis=0)[..., 0]
        if SHOW:
            fig3, _ = kwplot.imshow(util_kwimage.colorize_label_image(labels1, label_to_color={0: 'black'}), fnum=3, doclf=1, title='Volume Labels')
            finalizer.finalize(fig3, f'step_{_n[0]:03d}_volume_labels.png')

        unique_labels = np.setdiff1d(np.unique(labels1), [0])

        if self.bounds is not None:
            self._intermediates['small_bounds'] = self.bounds.scale(1 / scale_factor)

            if SHOW:
                fig4, _ = kwplot.imshow(util_kwimage.colorize_label_image(labels1, label_to_color={0: 'black'}), fnum=4, doclf=1, title='Volume Labels With Bounds')
                self._intermediates['small_bounds'].draw(edgecolor='white', fill=0)
                finalizer.finalize(fig4, f'step_{_n[0]:03d}_volume_labels_bounds.png')

        do_watershed = self.bounds is not None
        if do_watershed:
            PRINT_STEP(f'Initialize {len(unique_labels)} Seed Points')
            markers = np.zeros_like(max_saliency, dtype=np.int32)
            marker_points = []
            for labelid in unique_labels:
                mask = labels1 == labelid
                idxmax = (max_saliency * mask).argmax()
                highval = np.unravel_index(idxmax, mask.shape)
                r, c = highval[0], highval[1]
                marker_points.append((c, r))
                markers[r, c] = labelid

            PRINT_STEP('Watershed')
            if SHOW:
                watershed_image = max_saliency
                fig6, _ = kwplot.imshow(watershed_image, fnum=6, doclf=1, title='Watershed Energy')
                import kwimage
                kwimage.Points(xy=np.array(marker_points)).draw()
                finalizer.finalize(fig6, f'step_{_n[0]:03d}_watershed_energy.png')

                ...
            from skimage.segmentation import watershed
            labels = watershed(-max_saliency, markers=markers, mask=small_mask)
        else:
            labels = labels1

        if SHOW:
            fig5, _ = kwplot.imshow(util_kwimage.colorize_label_image(labels, label_to_color={0: 'black'}), fnum=4, doclf=1, title='Final Labels')
            finalizer.finalize(fig5, f'step_{_n[0]:03d}_final_labels.png')
        return labels

    def predict(self):
        """
        Predict the spatial polygons

        Returns:
            ndarray:
                A single [H, W] integer map indicating a cluster label for each
                pixel. I.e. a spatial segmentation of the sites.
        """
        import einops
        import kwarray
        import kwimage
        import kwutil
        import numpy as np
        import rich
        import sklearn
        import sklearn.cluster
        import sklearn.decomposition

        _t, _h, _w, _c = self.heatmap_thwc.shape
        orig_dims = _t * _c

        if 1:
            # defaults
            max_dims = min(32, orig_dims)
            scale_factor = 2

        # Setup Config
        workers = kwutil.util_parallel.coerce_num_workers(self.config['workers'])

        algo = self.config['algo']

        if algo == 'leotta':
            return self._predict_leotta()

        if algo == 'crall':
            return self._predict_crall()

        if algo == 'dbscan':
            dbscan = sklearn.cluster.DBSCAN(
                eps=.009, min_samples=5, metric='cosine', metric_params=None,
                algorithm='auto', leaf_size=30, n_jobs=workers)
            cluster_algo = dbscan
            max_dims = min(8, orig_dims)
            scale_factor = 8
        elif algo == 'meanshift':
            mean_shift = sklearn.cluster.MeanShift(bandwidth=None, seeds=None,
                                                   bin_seeding=False,
                                                   # TODO: set based on resolution
                                                   min_bin_freq=1,
                                                   cluster_all=True,
                                                   n_jobs=workers,
                                                   max_iter=500)
            cluster_algo = mean_shift
            max_dims = min(32, orig_dims)
            scale_factor = 4
        elif algo == 'kmeans':
            from sklearn.cluster import MiniBatchKMeans
            mb_kmeans = MiniBatchKMeans(n_clusters=64, batch_size=4096,
                                        n_init='auto', random_state=1)
            cluster_algo = mb_kmeans
            scale_factor = 2
            max_dims = orig_dims
        else:
            raise KeyError(algo)

        pca = sklearn.decomposition.PCA(n_components=max_dims)
        reduce_dims_algo = pca

        rich.print("[green]--- Start PolygonExtractor Predict ---")
        rich.print
        rich.print('config = {}'.format(ub.urepr(self.config, nl=1)))
        rich.print('* reduce_dims_algo = {}'.format(ub.urepr(reduce_dims_algo, nl=1)))
        rich.print('* cluster_algo = {}'.format(ub.urepr(cluster_algo, nl=1)))

        # Run Algorithm

        def PRINT_STEP(msg, _n=[1]):
            import rich
            n = _n[0]
            rich.print(f'Step {n}: {msg}')
            _n[0] += 1

        raw_heatmap = self.heatmap_thwc

        if self.bounds is not None:
            mask = self.bounds.to_mask(dims=(_h, _w)).data
        else:
            mask = None

        rich.print(f'* Given: raw_heatmap.shape={raw_heatmap.shape}')

        # TODO: better downscaling?
        downscaled = raw_heatmap[:, ::scale_factor, ::scale_factor, :]
        if mask is not None:
            small_mask = mask[::scale_factor, ::scale_factor]
        PRINT_STEP(f'Downscale by {scale_factor}x to: {downscaled.shape}')
        downscaled = downscaled.copy()

        PRINT_STEP('Impute NaN')
        imputed = impute_nans2(downscaled)
        rich.print('... Finished Impute')

        if mask is not None:
            PRINT_STEP('Masking Small Heatmap')
            masked = imputed * small_mask[None, :, :, None]
        else:
            masked = imputed

        SMART_HACK = 0
        if SMART_HACK:
            # from skimage.segmentation import watershed
            # from skimage.feature import peak_local_max
            cx1 = salient_cidx = self.classes.index('ac_salient')  # NOQA
            cx2 = active_cidx = self.classes.index('Active Construction')  # NOQA
            cx3 = siteprep_cidx = self.classes.index('Site Preparation')  # NOQA
            salient_chan = c1 = masked[..., cx1]
            PRINT_STEP('Stacking Modulated Features')
            active_chan = c2 = masked[..., cx2]
            siteprep_chan = c3 = masked[..., cx3]
            c4 = c1 * c2
            c5 = c1 * c3
            to_stack = [c1, c2, c3, c4, c5]

            self._intermediates['salient_chan'] = salient_chan
            self._intermediates['siteprep_chan'] = siteprep_chan
            self._intermediates['active_chan'] = active_chan
            self._intermediates['small_bounds'] = self.bounds.scale(1 / scale_factor)

            small_heatmap = np.stack(to_stack, axis=3)
            rich.print(f'small_heatmap.shape={small_heatmap.shape}')

            PRINT_STEP('Modulate By Saliency')
            max_saliency = salient_chan.max(axis=0, keepdims=1)
            small_heatmap = small_heatmap

            max_saliency_2d = max_saliency[0, :, :]
        else:
            small_heatmap = masked
            max_saliency_2d = None

        if self.config.get('robust_normalize'):
            small_heatmap = kwarray.robust_normalize(small_heatmap)
            PRINT_STEP('Robust Normalize')
        else:
            small_heatmap = small_heatmap
            PRINT_STEP('Skip Robust Normalize')

        t, h, w, c = small_heatmap.shape

        X = einops.rearrange(small_heatmap, 't h w c -> (h w) (t c)')
        PRINT_STEP(f'Rearange (combine time / channels) to: X.shape={X.shape}')

        Xhat = reduce_dims_algo.fit_transform(X)
        PRINT_STEP(f'Reduce dimensionality: Xhat.shape={Xhat.shape}')

        if self.config.get('positional_encoding'):
            scale = self.config.get('positional_encoding_scale', 1)
            rr, cc = np.meshgrid(
                np.linspace(-scale, scale, h),
                np.linspace(-scale, scale, w)
            )
            Xhat2 = np.concatenate([Xhat, rr.T.ravel()[:, None], cc.T.ravel()[:, None]], axis=1)
            PRINT_STEP(f'* Append Positional Encoding. Xhat2.shape={Xhat2.shape}')
        else:
            Xhat2 = Xhat

        PRINT_STEP('Run Clustering Algorithm')
        yhat = cluster_algo.fit_predict(Xhat2) + 1
        rich.print('... Finished Clustering Algorithm')

        small_label_img = einops.rearrange(yhat, '(h w) -> h w', w=w, h=h)
        PRINT_STEP(f'* Convert back to spatial arangement: small_label_img.shape={small_label_img.shape}')
        small_label_img = np.ascontiguousarray(small_label_img).astype(np.uint8)

        if mask is not None:
            PRINT_STEP('Masking label img')
            small_label_img = small_mask * small_label_img

        if max_saliency_2d is not None:
            saliency_mask = max_saliency_2d > 0.2
            small_label_img = saliency_mask * small_label_img
            # coords = peak_local_max(max_saliency_2d, footprint=np.ones((7, 7)), labels=saliency_mask)
            # lbl = watershed(max_saliency_2d, mask=saliency_mask)
            if 0:
                from geowatch.utils import util_kwimage
                import kwplot
                # canvas = util_kwimage.colorize_label_image(lbl)
                kwplot.imshow(max_saliency_2d, fnum=1, doclf=1)
                kwplot.imshow(saliency_mask, fnum=1, doclf=1)
                kwplot.imshow(util_kwimage.colorize_label_image(small_label_img, label_to_color={0: 'black'}))

        # PRINT_STEP('Morphology on Label')
        # small_label_img = kwimage.morphology(
        #     small_label_img, 'close', kernel=5, element='ellipse')

        if 0:
            small_feat = Xhat2.reshape(h, w, -1)
            ohe = kwarray.one_hot_embedding(small_label_img, small_label_img.max() + 1, dim=0)

            superpixel_feats = []
            superpixel_polys = []
            for _ohe in ohe[1:]:
                poly = kwimage.Mask(_ohe.astype(np.uint8), 'c_mask')
                poly = poly.to_multi_polygon(pixels_are='areas')

                poly_feat = (_ohe[:, :, None] * small_feat).sum(axis=(0, 1)) / (_ohe.sum() + 1)
                superpixel_polys.append(poly)
                superpixel_feats.append(poly_feat)

            keep_superpixel_idxs = np.arange(1, len(ohe))
            superpixel_feats = np.array(superpixel_feats)
            flags = superpixel_feats.max(axis=1) >= 0.000
            keep_superpixel_idxs = keep_superpixel_idxs[flags]
            superpixel_feats = superpixel_feats[flags]
            superpixel_polys = list(ub.compress(superpixel_polys, flags))

            dist = ((superpixel_feats[:, None, :] - superpixel_feats[None, :, :]) ** 2).sum(axis=2)
            ii, jj = np.where((dist < 3) & (dist > 0))
            import networkx as nx
            g = nx.Graph()
            g.add_edges_from(zip(ii, jj))

            new_small_label_img = np.zeros_like(small_label_img)

            next_new_id = 1
            for _, cc in enumerate(nx.connected_components(g), start=1):
                cc_polys = list(ub.take(superpixel_polys, cc))
                from shapely.ops import unary_union
                try:
                    cc_poly = unary_union([p.to_shapely().buffer(1) for p in cc_polys])
                except Exception:
                    continue
                for poly in kwimage.MultiPolygon.from_shapely(cc_poly):
                    poly.fill(new_small_label_img, value=next_new_id, pixels_are='areas')
                    next_new_id += 1

            if 0:
                kwplot.imshow(util_kwimage.colorize_label_image(new_small_label_img, label_to_color={0: 'black'}))

            small_label_img = new_small_label_img

        label_img = kwimage.imresize(small_label_img, scale=(scale_factor, scale_factor), interpolation='nearest')
        PRINT_STEP(f'* Step 8. Resize back to full scale: label_img.shape={label_img.shape}')

        rich.print("[green]--- End PolygonExtractor Predict ---")

        self._intermediates = {
            'max_saliency_2d': max_saliency_2d,
            'small_label_img': small_label_img,
            'Xhat2': Xhat2,
        }

        if 0:
            import kwplot
            kwplot.autompl()
            from geowatch.utils import util_kwimage  # NOQA
            canvas = util_kwimage.colorize_label_image(label_img, label_to_color={0: 'black'})
            kwplot.imshow(canvas, fnum=1, doclf=1)

            feature_pca = util_kwimage.ensure_false_color(
                Xhat2.reshape(h, w, -1), method='ortho')
            kwplot.imshow(feature_pca, fnum=2, doclf=1)

        return label_img

    def predict_polygons(self, return_info=False):
        label_img = self.predict()
        label_mask = LabelMask(label_img)
        polys = label_mask.to_multi_polygons()
        if return_info:
            info = {}
            info['label_mask'] = label_mask
            return polys, info
        else:
            return polys

    def show(self):
        import kwplot
        kwplot.autompl()
        stacked = self.draw_timesequence()
        kwplot.imshow(stacked)

    def draw_intermediate(self):
        import kwimage
        import numpy as np
        from geowatch.utils import util_kwimage
        salient = self._intermediates['salient_chan']
        # active = self._intermediates['active_chan']
        # siteprep = self._intermediates['siteprep_chan']
        small_bounds = self._intermediates['small_bounds']

        # heatmaps = np.stack([salient, active, siteprep], axis=3)
        heatmaps = np.stack([salient], axis=3)
        heatmaps = (heatmaps > 0.5).astype(np.float32)

        name_to_color = {cat['name']: kwimage.Color.coerce(cat['color']).as01()
                          for cat in self.classes.cats.values()}
        channel_colors = [name_to_color[n] for n in [
            'ac_salient', 'Site Preparation', 'Active Construction']]
        channel_colors = ['red']

        to_show_frames = [kwimage.nodata_checkerboard(
            util_kwimage.perchannel_colorize(h.astype(np.float32), channel_colors=channel_colors)
        ) for h in heatmaps]

        if self.bounds is not None:
            to_show_frames = [small_bounds.draw_on(frame, edgecolor='white', fill=False) for frame in to_show_frames]

        stacked = kwimage.stack_images_grid(to_show_frames, pad=10, bg_value='kitware_green')
        return stacked

    def draw_timesequence(self):
        import kwimage
        import numpy as np
        import kwarray
        from geowatch.utils import util_kwimage
        heatmaps = self.heatmap_thwc
        norm_heatmaps = kwarray.robust_normalize(heatmaps)

        if self.classes is not None:
            channel_colors = [kwimage.Color.coerce(cat['color']).as01()
                              for cat in self.classes.cats.values()]
            to_show_frames = [kwimage.nodata_checkerboard(
                util_kwimage.perchannel_colorize(h.astype(np.float32), channel_colors=channel_colors)
            ) for h in norm_heatmaps]
        else:
            to_show_frames = [kwimage.nodata_checkerboard(util_kwimage.ensure_false_color(h.astype(np.float32)))
                              for h in norm_heatmaps]
        if self.bounds is not None:
            to_show_frames = [self.bounds.draw_on(frame, edgecolor='white', fill=False) for frame in to_show_frames]
        stacked = kwimage.stack_images_grid(to_show_frames, pad=10, bg_value='kitware_green')
        return stacked

    @classmethod
    def demo(cls, **kwargs):
        """
        Create an instance of the problem on toy data.

        Args:
            **kwargs: passed to :func:`PolygonExtractor.demodata`

        Returns:
            PolygonExtractor
        """
        kwargs = cls.demodata(**kwargs)
        self = cls(**kwargs)
        return self

    @classmethod
    def demodata(cls, real_categories=False, rng=0):
        """
        Create toydata to test and demo the API

        Args:
            real_categories (bool): if False, use fake cateogires

            rng (int | None):
                random seed, or None to use global seed.

        Returns:
            Dict[str, Any]: A dictionary that can be used as kwargs
               to construct and instance of this class.
        """
        import kwimage
        import numpy as np
        import kwcoco
        import kwarray

        rng = kwarray.ensure_rng(rng)

        H, W = 128, 128
        dsize = np.array([W, H])

        # Define a random polygon and create several copies of it.
        poly0 = kwimage.Polygon.random(rng=rng)
        poly0 = poly0.translate((-0.5, -0.5))

        scale_factor = dsize / 4

        poly0 = poly0.scale(scale_factor)
        poly0 = poly0.translate((W / 2, H / 2))

        shift = poly0.box().width * 0.65

        poly1 = poly0.translate((-shift, -shift))
        poly2 = poly0.translate((+shift, +shift))
        poly3 = poly0.translate((-shift, +shift))
        poly4 = poly0.translate((+shift, -shift))

        polys = {
            'poly0': poly0,
            'poly1': poly1,
            'poly2': poly2,
            'poly3': poly3,
            'poly4': poly4,
        }

        # Use bounds of all polys to define a "bounds" object
        bounds = None
        for poly in polys.values():
            if bounds is None:
                bounds = poly
            else:
                bounds = bounds.union(poly)
        bounds = bounds.convex_hull
        bounds = bounds.scale(1.1, about='centroid')

        if real_categories:
            classes = kwcoco.CategoryTree.from_coco([
                {'id': 10, 'name': 'No Activity', 'color': 'tomato'},
                {'id': 11, 'name': 'Site Preparation', 'color': 'gold'},
                {'id': 12, 'name': 'Active Construction', 'color': 'lime'},
                {'id': 13, 'name': 'Post Construction', 'color': 'darkturquoise'},
                {'id': 14, 'name': 'ac_salient', 'color': 'white'},
            ])
        else:
            classes = kwcoco.CategoryTree.from_coco([
                {'id': 20, 'name': 'Red', 'color': 'red'},
                {'id': 21, 'name': 'Green', 'color': 'green'},
                {'id': 22, 'name': 'Blue', 'color': 'blue'},
            ])

        # Define a simple timeseries for each polygon to follow
        # TODO: there might be a nicer way to encode this to allow demodata to
        # capture more aspects of the problem.
        sequences = {
            'poly0': [
                {'frame': 2, 'class': 0},
                {'frame': 3, 'class': 0},
                {'frame': 4, 'class': 0},
                {'frame': 5, 'class': 0},
                {'frame': 6, 'class': 0},
                {'frame': 7, 'class': 0},
                {'frame': 8, 'class': 1},
                {'frame': 9, 'class': 2},
            ],

            'poly1': [
                {'frame': 4, 'class': 0},
                {'frame': 5, 'class': 1},
                {'frame': 6, 'class': 1},
                {'frame': 7, 'class': 1},
                {'frame': 8, 'class': 2},
                {'frame': 9, 'class': 2},
                {'frame': 10, 'class': 2},
            ],

            'poly2':  [
                {'frame': 1, 'class': 0},
                {'frame': 2, 'class': 1},
                {'frame': 3, 'class': 1},
                {'frame': 4, 'class': 1},
                {'frame': 5, 'class': 2},
            ],

            'poly3': [
                {'frame': 12, 'class': 1},
                {'frame': 13, 'class': 2},
                {'frame': 14, 'class': 2},
            ],

            'poly4': [
                {'frame': 4, 'class': 0},
                {'frame': 5, 'class': 1},
                {'frame': 6, 'class': 1},
                {'frame': 7, 'class': 1},
                {'frame': 8, 'class': 2},
                {'frame': 9, 'class': 2},
                {'frame': 10, 'class': 2},
            ],
        }

        # Create a heatmap time sequence and render polygons onto it
        C = len(classes)
        frame_shape = (H, W, C)
        # canvas = np.zeros(frame_shape, dtype=np.float32)
        canvas = rng.rand(*frame_shape).astype(np.float32) * 0.01

        heatmap_frames = [canvas.copy() for _ in range(16)]

        import kwutil
        base = kwutil.util_time.datetime.coerce('2020-01-01')
        delta = kwutil.util_time.timedelta.coerce('2 weeks')
        heatmap_time_intervals = [
            (base + (delta * idx), base + (delta * (idx + 1)))
            for idx in range(len(heatmap_frames))
        ]
        heatmap_time_intervals = [
            TimeInterval.coerce(d) for d in heatmap_time_intervals
        ]

        for key in polys.keys():
            poly = polys[key]
            seq = sequences[key]
            for info in seq:
                idx = info['frame']
                cidx = info['class']
                chan = np.ascontiguousarray(heatmap_frames[idx][..., cidx])
                new_chan = poly.fill(chan, value=1)
                heatmap_frames[idx][..., cidx] = new_chan
                if real_categories:
                    # hack to add in saliency for active classes
                    if classes[cidx] in {'Active Construction', 'Site Preparation'}:
                        cidx = classes.index('ac_salient')
                        chan = np.ascontiguousarray(heatmap_frames[idx][..., cidx])
                        new_chan = poly.fill(chan, value=1)
                        heatmap_frames[idx][..., cidx] = new_chan

        heatmap_thwc = np.array(heatmap_frames)
        kwargs = {
            'bounds': bounds,
            'heatmap_thwc': heatmap_thwc,
            'classes': classes,
            'heatmap_time_intervals': heatmap_time_intervals,
        }
        return kwargs


def impute_nans(data):
    '''
    interpolate to fill nan values

    TODO: tests

    data = np.random.rand(5, 3, 3, 2)
    data[data < 0.1] = np.nan

    data = np.random.rand(54, 295, 296, 5)
    data[data < 0.1] = np.nan

    import timerit
    ti = timerit.Timerit(1, bestof=1, verbose=3)

    for timer in ti.reset('impute_nans2'):
        with timer:
            impute_nans2(data)

    for timer in ti.reset('impute_nans'):
        with timer:
            impute_nans(data)
    '''
    from scipy.interpolate import NearestNDInterpolator
    import numpy as np
    mask = np.isfinite(data)
    valid_points = np.stack(np.where(mask), axis=1)
    invalid_points = np.stack(np.where(~mask), axis=1)
    valid_data = data[mask].ravel()

    if len(data.shape) == 4:
        t, h, w, c = data.shape
        # Hack: scale space to force interpolation over time first space second,
        # and channel last. Ideally we would never interpolate over space or
        # channel, but not sure how to do that atm. Is there a way to only fill in
        # values over the time axis?
        multiplier = np.array([1 / t, 10 * t, 10 * t, t * w * h * 10])[None, :]
    elif len(data.shape) == 3:
        t, h, w = data.shape
        multiplier = np.array([1 / t, 10 * t, 10 * t])[None, :]

    f_nearest = NearestNDInterpolator(valid_points * multiplier, valid_data)
    fill_data = f_nearest(invalid_points * multiplier)

    new_data = data.copy()
    new_data[tuple(invalid_points.T)] = fill_data
    return new_data


def impute_nans2(data):
    import numpy as np
    bad_value_mask = ~np.isfinite(data)
    new_data = data.copy()
    frame_new_prev = None
    for idx in range(data.shape[0]):
        frame_mask = bad_value_mask[idx]
        frame_new = new_data[idx]
        if frame_new_prev is None:
            frame_new[frame_mask] = 0
        else:
            # Impute values from previous frame
            frame_new[frame_mask] = frame_new_prev[frame_mask]
        frame_new_prev = frame_new
    return new_data


class FeatureCube(ub.NiceRepr):
    """
    Container for a [T, H, W, C] heatmap and corresponding T-lengthed time
    intervals.

    Ignore:
        self = FeatureCube.demo()
        print(f'self={self}')
        new = self.window_max('1 month')
        self.heatmap_time_intervals
        new.heatmap_time_intervals

        channels = ['ac_salient']

        import kwplot
        kwplot.autompl()
        kwplot.imshow(self.draw(), pnum=(1, 3, 1))
        kwplot.imshow(new.draw(), pnum=(1, 3, 2))

        new2 = new.take_channels(['ac_salient'])
        kwplot.imshow(new2.draw(), pnum=(1, 3, 3))
    """
    def __init__(self, heatmap_thwc, heatmap_time_intervals=None, classes=None):
        self.heatmap_thwc = heatmap_thwc
        self.heatmap_time_intervals = heatmap_time_intervals
        self.classes = classes

    def __nice__(self):
        return f'{self.heatmap_thwc.shape}'

    def take_channels(self, channels):
        if isinstance(channels, str):
            import kwcoco
            channels = kwcoco.FusedChannelSpec.coerce(channels).as_list()
        chan_idxs = [self.classes.index(c) for c in channels]
        new_heatmaps = self.heatmap_thwc[..., chan_idxs]

        new_graph = self.classes.graph.subgraph(channels)
        new_classes = self.classes.__class__(new_graph)
        new = FeatureCube(new_heatmaps, self.heatmap_time_intervals, classes=new_classes)
        return new

    def window_max(self, window):
        import numpy as np
        groupxs = self.heatmap_time_intervals._compute_time_window(window)
        new_intervals = self.heatmap_time_intervals.apply_grouping(groupxs)
        new_heatmaps = []
        for idxs in groupxs:
            combo = np.nanmax(self.heatmap_thwc[idxs], axis=0)
            new_heatmaps.append(combo)
        new_heatmaps = np.array(new_heatmaps)
        new = FeatureCube(new_heatmaps, new_intervals, classes=self.classes)
        return new

    @classmethod
    def demo(self):
        p = PolygonExtractor.demo(real_categories=1)
        intervals = TimeIntervalSequence.coerce(p.heatmap_time_intervals)
        self = FeatureCube(p.heatmap_thwc, intervals, p.classes)
        return self

    def draw(self):
        import kwimage
        import numpy as np
        import kwarray
        from geowatch.utils import util_kwimage
        heatmaps = self.heatmap_thwc
        norm_heatmaps = kwarray.robust_normalize(heatmaps)

        if self.classes is not None:
            from geowatch import heuristics
            heuristics.ensure_heuristic_category_tree_colors(self.classes)

            channel_colors = [kwimage.Color.coerce(cat['color']).as01()
                              for cat in self.classes.cats.values()]
            to_show_frames = [kwimage.nodata_checkerboard(
                util_kwimage.perchannel_colorize(h.astype(np.float32), channel_colors=channel_colors)
            ) for h in norm_heatmaps]
        else:
            to_show_frames = [kwimage.nodata_checkerboard(util_kwimage.ensure_false_color(h.astype(np.float32)))
                              for h in norm_heatmaps]
        # if self.bounds is not None:
        #     to_show_frames = [self.bounds.draw_on(frame, edgecolor='white', fill=False) for frame in to_show_frames]
        stacked = kwimage.stack_images_grid(to_show_frames, pad=10, bg_value='kitware_green')
        return stacked

    def __mul__(self, other):
        new_heatmap = self.heatmap_thwc * other.heatmap_thwc
        new = FeatureCube(new_heatmap, self.heatmap_time_intervals, self.classes)
        return new


class TimeIntervalSequence(list):
    """
    A list of non-overlapping time intervals

    Ignore:
        from geowatch.tasks.tracking.polygon_extraction import *  # NOQA
        a = TimeInterval.coerce(('2020-01-01', '2020-02-01'))
        b = TimeInterval.coerce(('2020-02-01', '2020-03-01'))
        c = TimeInterval.coerce(('2020-03-01', '2020-03-15'))
        d = TimeInterval.coerce(('2020-03-07', '2020-04-01'))
        e = TimeInterval.coerce(('2020-04-15', '2020-05-01'))
        seq = TimeIntervalSequence([a, b, c, d, e])
        seq._compute_time_window('1 month')
    """

    def apply_grouping(self, groupxs):
        from functools import reduce
        new = TimeIntervalSequence()
        for idxs in groupxs:
            part = list(ub.take(self, idxs))
            new_part = reduce(TimeInterval.union, part[1:], part[0]).enclosure
            new.append(new_part)
        return new

    @classmethod
    def coerce(self, data):
        if isinstance(data, TimeIntervalSequence):
            self = data
        else:
            self = TimeIntervalSequence([TimeInterval.coerce(d) for d in data])
        return self

    def _compute_time_window(self, window):
        """
        Example:
            >>> window = '7days'
            >>> from kwutil import util_time
            >>> self = heatmap_dates = TimeIntervalSequence(map(TimeInterval.coerce, [
            >>>     '2020-01-01', '2020-01-02', '2020-02-01',
            >>>     '2020-02-02', '2020-03-14', '2020-03-23',
            >>>     '2020-04-01', '2020-06-23', '2020-06-26',
            >>>     '2020-06-27', '2020-06-28', ]))
            >>> groupxs = heatmap_dates._compute_time_window(window)
            >>> new = self.apply_grouping(groupxs)
            >>> print(f'groupxs={groupxs}')
            >>> print(f'new={new}')
            >>> groupxs = heatmap_dates._compute_time_window(None)
            >>> new = self.apply_grouping(groupxs)
            >>> print(f'groupxs={groupxs}')
            >>> print(f'new={new}')
            >>> groupxs = heatmap_dates._compute_time_window(3)
            >>> new = self.apply_grouping(groupxs)
            >>> print(f'groupxs={groupxs}')
            >>> print(f'new={new}')
        """
        import kwarray
        from kwutil import util_time
        import numpy as np
        num_frames = len(self)
        if window is None:
            bucket_ids = np.arange(num_frames)
        elif isinstance(window, str):
            delta = util_time.coerce_timedelta(window).total_seconds()
            start_unixtimes = np.array([x.start.timestamp() for x in self])
            start_unixtimes = start_unixtimes - start_unixtimes[0]
            bucket_ids = (start_unixtimes // delta).astype(int)
        elif isinstance(window, int):
            assert num_frames is not None
            frame_indexes = np.arange(num_frames)
            bucket_ids = frame_indexes // window
        else:
            raise NotImplementedError('')
        unique_ids, groupxs = kwarray.group_indices(bucket_ids)
        return groupxs


class TimeInterval(portion.Interval):
    """
    Represents an interval in time.

    Ignore:
        import liberator
        lib = liberator.Liberator()
        lib.add_dynamic(portion.Interval)
        lib.expand(['portion'])
        print(lib.current_sourcecode())

    Example:
        from geowatch.tasks.tracking.polygon_extraction import *  # NOQA
        a = TimeInterval.coerce(('2020-01-01', '2020-02-01'))
        b = TimeInterval.coerce(('2020-02-01', '2020-03-01'))
        c = TimeInterval.coerce(('2020-03-01', '2020-03-15'))
        d = TimeInterval.coerce(('2020-03-07', '2020-04-01'))
        e = TimeInterval.coerce(('2020-04-15', '2020-05-01'))
    """

    @property
    def start(self):
        return self.lower

    @property
    def stop(self):
        return self.upper

    @classmethod
    def coerce(TimeInterval, data):
        import kwutil
        if isinstance(data, TimeInterval):
            self = data
        elif isinstance(data, (str, kwutil.util_time.datetime_cls, float, int)):
            start = kwutil.util_time.datetime.coerce(data)
            stop = start
            self = TimeInterval.closed(start, stop)
        elif ub.iterable(data) and len(data) == 2:
            start, stop = data
            start = kwutil.util_time.datetime.coerce(start)
            stop = kwutil.util_time.datetime.coerce(stop)
            self = TimeInterval.closed(start, stop)
        else:
            raise TypeError
        return self

    @classmethod
    def closed(TimeInterval, start, stop=None):
        """
        """
        if start is None:
            start = -float('inf')
        if stop is None:
            start = -float('inf')
        self = TimeInterval.from_atomic(portion.CLOSED, start, stop, portion.CLOSED)
        return self

    @classmethod
    def random(TimeInterval):
        import kwutil
        a = kwutil.util_time.datetime.random()
        b = kwutil.util_time.datetime.random()
        a, b = sorted([a, b])
        self = TimeInterval.closed(a, b)
        return self


def toydata_demo():
    """
    Example:
        from geowatch.tasks.tracking.polygon_extraction import *  # NOQA
        cls = PolygonExtractor(heatmaps, bounds)
        self = PolygonExtractor.demo()
    """
    from geowatch.utils import util_kwimage
    import kwplot

    def autompl2():
        """
        New autompl with inline logic for notebooks
        """
        import kwplot
        try:
            import IPython
            ipy = IPython.get_ipython()
            ipy.config
            if 'colab' in str(ipy.config['IPKernelApp']['kernel_class']):
                ipy.run_line_magic('matplotlib', 'inline')
        except NameError:
            ...
        kwplot.autompl()

    autompl2()

    self = PolygonExtractor.demo()
    stacked = self.draw_timesequence()
    label_img = self.predict()

    canvas = util_kwimage.colorize_label_image(label_img)
    kwplot.imshow(stacked, pnum=(1, 2, 1))
    kwplot.imshow(canvas, pnum=(1, 2, 2))

    if 0:
        mean_img = self.heatmap_thwc.mean(axis=0)
        import kwplot
        kwplot.plt
        kwplot.imshow(mean_img, fnum=2)


def real_data_demo_case_1():
    from geowatch.utils import util_girder
    from geowatch.utils import util_kwimage
    import pickle
    import kwimage
    import kwarray
    import kwplot
    import numpy as np

    api_url = 'https://data.kitware.com/api/v1'
    heatmap_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654d1423fdc1508d8bec1bc3', hash_prefix='f9eb14a7c0ec99'))

    truth_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654d2624fdc1508d8bec1bca', hash_prefix='c164037856d2ce'))

    bas_bounds_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654d29f6fdc1508d8bec1bcd', hash_prefix='9c555b1adc518b'))

    heatmap_dates_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654d2b11fdc1508d8bec1bd2', hash_prefix='4cdcdf2a40a819'))

    true_info_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654d4f35fdc1508d8bec1c86', hash_prefix='4a68c596804200'))

    truth_labels = pickle.loads(truth_fpath.read_bytes())
    heatmap_thwc = pickle.loads(heatmap_fpath.read_bytes())
    bounds = pickle.loads(bas_bounds_fpath.read_bytes())
    heatmap_time_intervals = pickle.loads(heatmap_dates_fpath.read_bytes())
    truth_info = pickle.loads(true_info_fpath.read_bytes())

    start_time = heatmap_time_intervals[0][0]
    end_time = heatmap_time_intervals[-1][1]
    print('start_time = {}'.format(ub.urepr(start_time, nl=1)))
    print('end_time = {}'.format(ub.urepr(end_time, nl=1)))

    import kwcoco
    classes = kwcoco.CategoryTree.from_coco([
        {'name': 'No Activity', 'color': 'tomato'},
        {'name': 'Site Preparation', 'color': 'gold'},
        {'name': 'Active Construction', 'color': 'lime'},
        {'name': 'Post Construction', 'color': 'darkturquoise'},
        {'name': 'ac_salient', 'color': 'white'},
    ])
    channel_colors = [kwimage.Color.coerce(cat['color']).as01()
                      for cat in classes.cats.values()]

    unique_labels = np.unique(truth_labels)
    unique_colors = kwimage.Color.distinct(len(unique_labels))
    label_to_color = ub.dzip(unique_labels, unique_colors)
    label_to_color[0] = kwimage.Color.coerce('black').as01()

    truth_canvas = util_kwimage.colorize_label_image(truth_labels, label_to_color=label_to_color)

    frame0_canvas = kwarray.robust_normalize(heatmap_thwc[0])
    frame0_canvas = util_kwimage.perchannel_colorize(frame0_canvas, channel_colors=channel_colors)
    frame0_canvas = util_kwimage.ensure_false_color(frame0_canvas)
    frame0_canvas = kwimage.nodata_checkerboard(frame0_canvas)
    frame0_canvas = bounds.draw_on(frame0_canvas, fill=False, edgecolor='white')

    frameN_canvas = kwarray.robust_normalize(heatmap_thwc[-1])
    frameN_canvas = util_kwimage.perchannel_colorize(frameN_canvas, channel_colors=channel_colors)
    frameN_canvas = kwimage.nodata_checkerboard(frameN_canvas)
    frameN_canvas = bounds.draw_on(frameN_canvas, fill=False, edgecolor='white')

    truth_canvas = bounds.draw_on(truth_canvas, fill=False, edgecolor='white')

    # Color truth by status
    import numpy as np
    from geowatch import heuristics
    truth_canvas2 = kwimage.atleast_3channels(np.zeros_like(truth_labels)).astype(np.uint8) + 255
    status_to_color = {}
    for item in ub.ProgIter(list(truth_info.values()), desc='draw truth'):
        poly = kwimage.MultiPolygon.coerce(item['geometry_pxl'])
        color = heuristics.IARPA_STATUS_TO_INFO[item['status']]['color']
        edgecolor = kwimage.Color.random()
        truth_canvas2 = poly.draw_on(truth_canvas2, facecolor=color, edgecolor=edgecolor)
        status_to_color[item['status']] = color
    legend_img = kwplot.make_legend_img(status_to_color)
    truth_canvas2 = kwimage.stack_images([truth_canvas2, legend_img], axis=1)

    kwplot.autompl()
    kwplot.imshow(frame0_canvas, title=f'Frame 0 Colorized: {start_time}', fnum=1)
    kwplot.imshow(frameN_canvas, title=f'Frame N Colorized: {end_time}', fnum=2)
    kwplot.imshow(truth_canvas, title='Truth Labels', fnum=3)
    kwplot.imshow(truth_canvas2, title='Truth Status', fnum=4)

    self = PolygonExtractor(heatmap_thwc, bounds=bounds,
                            heatmap_time_intervals=heatmap_time_intervals,
                            classes=classes)
    stacked = self.draw_timesequence()
    kwplot.imshow(stacked, pnum=(1, 2, 1), fnum=5)

    label_img = self.predict()
    # label_img = self._predict_leotta()

    canvas = util_kwimage.colorize_label_image(label_img)
    kwplot.imshow(canvas, pnum=(1, 2, 2), fnum=5)


def real_data_demo_case_2():
    from geowatch.utils import util_girder
    from geowatch.utils import util_kwimage
    import pickle
    import kwimage
    import kwarray
    import kwplot
    import numpy as np

    api_url = 'https://data.kitware.com/api/v1'
    data_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654d6ce3fdc1508d8bec1c89', hash_prefix='d1425c26580e03'))

    data = pickle.loads(data_fpath.read_bytes())
    heatmap_thwc = data['heatmaps_thwc']
    bounds = kwimage.MultiPolygon.from_shapely(data['bas_gdf'].geometry.unary_union)
    heatmap_time_intervals = data['heatmap_time_intervals']
    truth_gdf = data['truth_gdf']

    t, h, w, c = heatmap_thwc.shape

    truth_colorized = np.ones((h, w, 3))
    truth_labels = np.zeros((h, w), dtype=np.int32)
    idx = 1
    for _, row in truth_gdf.iterrows():
        true_poly = kwimage.MultiPolygon.coerce(row['geometry'])
        truth_colorized = true_poly.draw_on(truth_colorized, color=row['color'])
        if row['status'] not in {'negative', 'ignore'}:
            truth_labels = true_poly.fill(truth_labels, value=idx)
            idx += 1

    truth_labels
    start_time = heatmap_time_intervals[0][0]
    end_time = heatmap_time_intervals[-1][1]
    print('start_time = {}'.format(ub.urepr(start_time, nl=1)))
    print('end_time = {}'.format(ub.urepr(end_time, nl=1)))

    import kwcoco
    classes = kwcoco.CategoryTree.from_coco([
        {'id': 30, 'name': 'No Activity', 'color': 'tomato'},
        {'id': 31, 'name': 'Site Preparation', 'color': 'gold'},
        {'id': 32, 'name': 'Active Construction', 'color': 'lime'},
        {'id': 33, 'name': 'Post Construction', 'color': 'darkturquoise'},
        {'id': 34, 'name': 'ac_salient', 'color': 'white'},
    ])
    print(list(classes))
    list(classes.graph.nodes())
    channel_colors = [kwimage.Color.coerce(cat['color']).as01()
                      for cat in classes.cats.values()]

    unique_labels = np.unique(truth_labels)
    unique_colors = kwimage.Color.distinct(len(unique_labels))
    label_to_color = ub.dzip(unique_labels, unique_colors)
    label_to_color[0] = kwimage.Color.coerce('black').as01()

    truth_canvas = util_kwimage.colorize_label_image(truth_labels, label_to_color=label_to_color)

    frame0_canvas = kwarray.robust_normalize(heatmap_thwc[0])
    frame0_canvas = util_kwimage.perchannel_colorize(frame0_canvas, channel_colors=channel_colors)
    frame0_canvas = util_kwimage.ensure_false_color(frame0_canvas)
    frame0_canvas = kwimage.nodata_checkerboard(frame0_canvas)
    frame0_canvas = bounds.draw_on(frame0_canvas, fill=False, edgecolor='white')

    frameN_canvas = kwarray.robust_normalize(heatmap_thwc[-1])
    frameN_canvas = util_kwimage.perchannel_colorize(frameN_canvas, channel_colors=channel_colors)
    frameN_canvas = kwimage.nodata_checkerboard(frameN_canvas)
    frameN_canvas = bounds.draw_on(frameN_canvas, fill=False, edgecolor='white')

    truth_canvas = bounds.draw_on(truth_canvas, fill=False, edgecolor='white')

    # Color truth by status
    # import numpy as np
    # from geowatch import heuristics
    # truth_canvas2 = kwimage.atleast_3channels(np.zeros_like(truth_labels)).astype(np.uint8) + 255
    # status_to_color = {}
    # for item in ub.ProgIter(list(truth_info.values()), desc='draw truth'):
    #     poly = kwimage.MultiPolygon.coerce(item['geometry_pxl'])
    #     color = heuristics.IARPA_STATUS_TO_INFO[item['status']]['color']
    #     edgecolor = kwimage.Color.random()
    #     truth_canvas2 = poly.draw_on(truth_canvas2, facecolor=color, edgecolor=edgecolor)
    #     status_to_color[item['status']] = color
    # legend_img = kwplot.make_legend_img(status_to_color)
    # truth_canvas2 = kwimage.stack_images([truth_canvas2, legend_img], axis=1)

    kwplot.autompl()
    kwplot.imshow(frame0_canvas, title=f'Frame 0 Colorized: {start_time}', fnum=1)
    kwplot.imshow(frameN_canvas, title=f'Frame N Colorized: {end_time}', fnum=2)
    kwplot.imshow(truth_canvas, title='Truth Labels', fnum=3)
    kwplot.imshow(truth_colorized, title='Truth Status', fnum=4)

    self = PolygonExtractor(heatmap_thwc, bounds=bounds,
                            heatmap_time_intervals=heatmap_time_intervals,
                            classes=classes, config={
                                # 'algo': 'meanshift',
                            })

    # label_img = self.predict()
    # label_img = self._predict_leotta()
    label_img = self._predict_crall()

    canvas = util_kwimage.colorize_label_image(label_img)
    kwplot.imshow(canvas, pnum=(1, 2, 2), fnum=6)

    stacked = self.draw_timesequence()
    kwplot.imshow(stacked, pnum=(1, 2, 1), fnum=6)


def real_data_demo_case_3():
    from geowatch.utils import util_girder
    from geowatch.utils import util_kwimage
    import pickle
    import kwimage
    import kwarray
    import kwplot
    import numpy as np

    api_url = 'https://data.kitware.com/api/v1'
    data_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654e77f4314693d6b1df2655', hash_prefix='1b91956803365b51'))

    data = pickle.loads(data_fpath.read_bytes())
    heatmap_thwc = data['heatmap_thwc']
    bounds = kwimage.MultiPolygon.from_shapely(data['bas_gdf'].geometry.unary_union)
    heatmap_time_intervals = data['heatmap_time_intervals']
    truth_gdf = data['truth_gdf']

    t, h, w, c = heatmap_thwc.shape

    truth_colorized = np.ones((h, w, 3))
    truth_labels = np.zeros((h, w), dtype=np.int32)
    idx = 1
    for _, row in truth_gdf.iterrows():
        true_poly = kwimage.MultiPolygon.coerce(row['geometry'])
        truth_colorized = true_poly.draw_on(truth_colorized, color=row['color'])
        if row['status'] not in {'negative', 'ignore'}:
            truth_labels = true_poly.fill(truth_labels, value=idx)
            idx += 1

    truth_labels
    start_time = heatmap_time_intervals[0][0]
    end_time = heatmap_time_intervals[-1][1]
    print('start_time = {}'.format(ub.urepr(start_time, nl=1)))
    print('end_time = {}'.format(ub.urepr(end_time, nl=1)))

    classes = data['classes']
    channel_colors = [kwimage.Color.coerce(cat['color']).as01()
                      for cat in classes.cats.values()]

    unique_labels = np.unique(truth_labels)
    unique_colors = kwimage.Color.distinct(len(unique_labels))
    label_to_color = ub.dzip(unique_labels, unique_colors)
    label_to_color[0] = kwimage.Color.coerce('black').as01()

    truth_canvas = util_kwimage.colorize_label_image(truth_labels, label_to_color=label_to_color)

    frame0_canvas = kwarray.robust_normalize(heatmap_thwc[0])
    frame0_canvas = util_kwimage.perchannel_colorize(frame0_canvas, channel_colors=channel_colors)
    frame0_canvas = util_kwimage.ensure_false_color(frame0_canvas)
    frame0_canvas = kwimage.nodata_checkerboard(frame0_canvas)
    frame0_canvas = bounds.draw_on(frame0_canvas, fill=False, edgecolor='white')

    frameN_canvas = kwarray.robust_normalize(heatmap_thwc[-1])
    frameN_canvas = util_kwimage.perchannel_colorize(frameN_canvas, channel_colors=channel_colors)
    frameN_canvas = kwimage.nodata_checkerboard(frameN_canvas)
    frameN_canvas = bounds.draw_on(frameN_canvas, fill=False, edgecolor='white')

    truth_canvas = bounds.draw_on(truth_canvas, fill=False, edgecolor='white')

    kwplot.autompl()
    kwplot.imshow(frame0_canvas, title=f'Frame 0 Colorized: {start_time}', fnum=1)
    kwplot.imshow(frameN_canvas, title=f'Frame N Colorized: {end_time}', fnum=2)
    kwplot.imshow(truth_canvas, title='Truth Labels', fnum=3)
    kwplot.imshow(truth_colorized, title='Truth Status', fnum=4)

    self = PolygonExtractor(heatmap_thwc, bounds=bounds,
                            heatmap_time_intervals=heatmap_time_intervals,
                            classes=classes, config={
                                # 'algo': 'meanshift',
                                # 'algo': 'crall',
                                # 'algo': 'leotta',
                            })

    self.config['algo'] = 'crall'
    self.config['scale_factor'] = 1

    polygons, info = self.predict_polygons(return_info=True)
    label_mask = info['label_mask']

    canvas = label_mask.colorize()
    for p in polygons:
        canvas = p.draw_on(canvas, fill=0, edgecolor='black')
    kwplot.imshow(canvas, pnum=(1, 2, 2), fnum=6)

    stacked = self.draw_timesequence()
    kwplot.imshow(stacked, pnum=(1, 2, 1), fnum=6)


def real_data_demo_case_1_fixed():
    from geowatch.utils import util_girder
    from geowatch.utils import util_kwimage
    import pickle
    import kwimage
    import kwarray
    import kwplot
    import numpy as np

    api_url = 'https://data.kitware.com/api/v1'
    data_fpath = ub.Path(util_girder.grabdata_girder(
        api_url, '654eab24314693d6b1df2658', hash_prefix='358bc1e7ec9ab7ba'))

    data = pickle.loads(data_fpath.read_bytes())
    heatmap_thwc = data['heatmap_thwc']
    bounds = kwimage.MultiPolygon.from_shapely(data['bas_gdf'].geometry.unary_union)
    heatmap_time_intervals = data['heatmap_time_intervals']
    truth_gdf = data['truth_gdf']

    t, h, w, c = heatmap_thwc.shape

    truth_colorized = np.ones((h, w, 3))
    truth_labels = np.zeros((h, w), dtype=np.int32)
    idx = 1
    for _, row in truth_gdf.iterrows():
        true_poly = kwimage.MultiPolygon.coerce(row['geometry'])
        truth_colorized = true_poly.draw_on(truth_colorized, color=row['color'])
        if row['status'] not in {'negative', 'ignore'}:
            truth_labels = true_poly.fill(truth_labels, value=idx)
            idx += 1

    truth_labels
    start_time = heatmap_time_intervals[0][0]
    end_time = heatmap_time_intervals[-1][1]
    print('start_time = {}'.format(ub.urepr(start_time, nl=1)))
    print('end_time = {}'.format(ub.urepr(end_time, nl=1)))

    classes = data['classes']
    channel_colors = [kwimage.Color.coerce(cat['color']).as01()
                      for cat in classes.cats.values()]

    unique_labels = np.unique(truth_labels)
    unique_colors = kwimage.Color.distinct(len(unique_labels))
    label_to_color = ub.dzip(unique_labels, unique_colors)
    label_to_color[0] = kwimage.Color.coerce('black').as01()

    truth_canvas = util_kwimage.colorize_label_image(truth_labels, label_to_color=label_to_color)

    frame0_canvas = kwarray.robust_normalize(heatmap_thwc[0])
    frame0_canvas = util_kwimage.perchannel_colorize(frame0_canvas, channel_colors=channel_colors)
    frame0_canvas = util_kwimage.ensure_false_color(frame0_canvas)
    frame0_canvas = kwimage.nodata_checkerboard(frame0_canvas)
    frame0_canvas = bounds.draw_on(frame0_canvas, fill=False, edgecolor='white')

    frameN_canvas = kwarray.robust_normalize(heatmap_thwc[-1])
    frameN_canvas = util_kwimage.perchannel_colorize(frameN_canvas, channel_colors=channel_colors)
    frameN_canvas = kwimage.nodata_checkerboard(frameN_canvas)
    frameN_canvas = bounds.draw_on(frameN_canvas, fill=False, edgecolor='white')

    truth_canvas = bounds.draw_on(truth_canvas, fill=False, edgecolor='white')

    kwplot.autompl()
    kwplot.imshow(frame0_canvas, title=f'Frame 0 Colorized: {start_time}', fnum=1)
    kwplot.imshow(frameN_canvas, title=f'Frame N Colorized: {end_time}', fnum=2)
    kwplot.imshow(truth_canvas, title='Truth Labels', fnum=3)
    kwplot.imshow(truth_colorized, title='Truth Status', fnum=4)

    self = PolygonExtractor(heatmap_thwc, bounds=bounds,
                            heatmap_time_intervals=heatmap_time_intervals,
                            classes=classes, config={})

    # self.config['algo'] = 'meanshift'
    # self.config['algo'] = 'leotta'
    # self.config['leotta_threah'] = 0.2
    # self.config['algo'] = 'kmeans'
    self.config['algo'] = 'crall'
    self.config['scale_factor'] = 8

    polygons, info = self.predict_polygons(return_info=True)
    label_mask = info['label_mask']

    canvas = label_mask.resize(scale=4).colorize()
    for p in polygons:
        canvas = p.scale(4).draw_on(canvas, fill=0, edgecolor='black')
    kwplot.imshow(canvas, pnum=(1, 2, 1), fnum=6)

    stacked = self.draw_timesequence()
    kwplot.imshow(stacked, pnum=(1, 2, 1), fnum=6)


def generate_real_example():
    import kwcoco
    import kwutil
    import pickle  # NOQA
    import numpy as np
    import kwimage  # NOQA
    from geowatch.cli import reproject_annotations

    # Read BAS dataset
    dset = kwcoco.CocoDataset('/home/joncrall/data/dvc-repos/smart_expt_dvc/_airflow/preeval17_batch_v130/KR_R002/sc-fusion/sc_fusion_kwcoco_tracked.json')

    # Classes of interest
    classes = kwcoco.CategoryTree.from_coco([
        {'id': 30, 'name': 'No Activity', 'color': 'tomato'},
        {'id': 31, 'name': 'Site Preparation', 'color': 'gold'},
        {'id': 32, 'name': 'Active Construction', 'color': 'lime'},
        {'id': 33, 'name': 'Post Construction', 'color': 'darkturquoise'},
        {'id': 34, 'name': 'ac_salient', 'color': 'white'},
    ])

    # from geowatch.geoannots import geomodels
    # region = geomodels.RegionModel.coerce('/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/annotations/drop7/region_models/KR_R002.geojson')

    # Create a copy with the Truth
    true_dset = dset.copy()
    sites = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/annotations/drop7/site_models/KR_R002*.geojson'
    true_dset = reproject_annotations.main(dst='return', src=true_dset, sites=sites, role='truth')

    # Choose a Video
    video_id = dset.videos()[0]
    images = dset.images(video_id=video_id)

    resolution = '8GSD'

    # Construct spatial truth and prediction summary
    bas_gdf = coco_make_track_gdf(dset, video_id, resolution)
    truth_gdf = coco_make_track_gdf(true_dset, video_id, resolution)
    bounds = kwimage.MultiPolygon.from_shapely(bas_gdf.geometry.unary_union)

    # Figure out the shape of the canvas at the desired resolution
    delayed = images.coco_images[0].imdelay(channels='red', space='video', resolution=resolution)
    h, w = delayed.shape[0:2]

    import kwplot
    positive_true_polys = []
    nonpositive_true_polys = []
    status_to_color = {}
    for _, row in truth_gdf.iterrows():
        true_poly = kwimage.MultiPolygon.coerce(row['geometry'])
        status_to_color[row['status']] = row['color']
        if row['status'] != {'ignore', 'negative'}:
            positive_true_polys.append(true_poly)

    status_to_color['BAS output'] = 'kitware_blue'

    if 1:
        import kwarray
        coco_img = images.coco_images[0]
        rgb = coco_img.imdelay(channels='red|green|blue', space='video', resolution=resolution).finalize()
        # dets = coco_img._detections_for_resolution(space='video', resolution=resolution)
        canvas = kwarray.robust_normalize(rgb)
        # true_coco_image = true_dset.coco_image(coco_img['id'])
        # true_dets = true_coco_image._detections_for_resolution(space='video')
        positive_true_dets = kwimage.Detections(segmentations=positive_true_polys)
        nonpositive_true_dets = kwimage.Detections(segmentations=nonpositive_true_polys)
        canvas = positive_true_dets.data['segmentations'].draw_on(canvas, alpha=0.5, color='kitware_green', edgecolor='black')
        canvas = nonpositive_true_dets.data['segmentations'].draw_on(canvas, alpha=0.5, color='kitware_gray', edgecolor='black')

        canvas = bounds.draw_on(canvas, color='kitware_blue', alpha=0.5)
        kwplot.imshow(canvas, fnum=2)

    channels = 'Site Preparation|Active Construction|Post Construction|No Activity|ac_salient'
    print(f'Reading heatmaps with: key={channels} @ {resolution}')
    io_workers = kwutil.util_parallel.coerce_num_workers('avail')
    heatmap_iter = imread_many(dset, images, channels=channels, space='video', resolution=resolution, workers=io_workers)
    _heatmaps = list(heatmap_iter)
    heatmap_thwc = np.array(_heatmaps)

    image_dates = [kwutil.util_time.coerce_datetime(d)
                   for d in images.lookup('date_captured')]
    heatmap_time_intervals = list(zip(image_dates, image_dates))

    data = {}
    data['heatmap_thwc'] = heatmap_thwc
    data['heatmap_time_intervals'] = heatmap_time_intervals
    data['bounds'] = bounds
    data['classes'] = classes
    data['bas_gdf'] = bas_gdf
    data['truth_gdf'] = truth_gdf

    if 1:
        # Splot check
        kwargs = ub.compatible(data, PolygonExtractor.__init__)
        self = PolygonExtractor(**kwargs)
        import kwplot
        kwplot.autompl()
        kwplot.imshow(self.draw_timesequence())

    if 0:
        # Upload data
        pickle_fpath = ub.Path.appdir('geowatch/polyextract').ensuredir() / 'demo0-fixed.pkl'
        pickle_fpath.write_bytes(pickle.dumps(data))

        hash_prefix = ub.hash_file(pickle_fpath)[0:16]
        print(hash_prefix)

        # source $HOME/internal/secrets
        ub.cmd(ub.paragraph(
            f'''
            . ~/internal/safe/secrets && \
            girder-client \
                --api-url https://data.kitware.com/api/v1 \
                upload \
                654d10a9fdc1508d8bec1bb6 \
                {pickle_fpath}
            '''), verbose=3, shell=True)


class LabelMask:
    """
    Ignore:
        from geowatch.utils import util_kwimage
        import kwplot
        kwplot.autompl()
        self = LabelMask.demo()
        canvas = self.colorize()
        kwplot.imshow(canvas, doclf=1)
        polys = self.to_multi_polygons()
        for p in polys:
            p.draw(fill=0, edgecolor='black')
    """
    def __init__(self, data):
        self.data = data

    def colorize(self):
        from geowatch.utils import util_kwimage
        canvas = util_kwimage.colorize_label_image(self.data, legend_dpi=300)
        return canvas

    def resize(self, scale=None, dsize=None):
        import kwimage
        new_data = kwimage.imresize(self.data, scale=scale, interpolation='nearest')
        new = self.__class__(new_data)
        return new

    @classmethod
    def demo(LabelMask):
        import numpy as np
        import kwimage
        H, W = 512, 512
        label_img = np.zeros((H, W), dtype=np.int32)
        poly1 = kwimage.Polygon.random().scale((H, W))
        poly2 = kwimage.Polygon.random().scale((H, W))
        label_img = poly1.fill(label_img, value=1)
        label_img = poly2.fill(label_img, value=2)
        self = LabelMask(label_img)
        return self

    def to_multi_polygons(self):
        import kwimage
        label_img = self.data
        poly_datas = _rasterio_labelimg_to_polygons(label_img)
        value_to_polys = ub.group_items(poly_datas, key=lambda p: p['value'])
        multi_polygons = []
        for value, subpolys in value_to_polys.items():
            kwpolys = [
                kwimage.Polygon(**p, metakeys=['value'])
                for p in subpolys
            ]
            mpoly = kwimage.MultiPolygon(kwpolys, meta={'value': value})
            multi_polygons.append(mpoly)
        return multi_polygons


def _rasterio_labelimg_to_polygons(label_img):
    """
    Note:
        The :func:`rasterio.features.shapes` is capable of multi-label polygon
        extraction.

    TODO:
        - [ ] Kwimage needs a good multi-label image polygon extraction tool

    Ignore:
    """
    import numpy as np
    from rasterio import features
    poly_datas = []
    if label_img.size > 0:
        shapes = list(features.shapes(label_img, connectivity=8))
        translate = np.array([-0.5, -0.5]).ravel()[None, :]
        for shape, value in shapes:
            if value > 0:
                coords = shape['coordinates']
                exterior = np.array(coords[0]) + translate
                interiors = [np.array(p) + translate for p in coords[1:]]
                poly_datas.append({
                    'exterior': exterior,
                    'interiors': interiors,
                    'value': value,
                })
    return poly_datas


def coco_make_track_gdf(coco_dset, video_id, resolution=None):
    import geopandas as gpd
    from shapely.ops import unary_union
    from geowatch import heuristics
    coco_images = coco_dset.images(video_id=video_id).coco_images
    tid_to_objs = ub.ddict(list)
    for coco_img in coco_images:
        annots = coco_img.annots()
        track_ids = annots.lookup('track_id')
        polys = coco_img._annot_segmentations(annots.objs, space='video', resolution=resolution)
        ann0 = annots.objs[0]
        status = ann0.get('status', None)
        for tid, poly, ann in zip(track_ids, polys, annots.objs):
            ann['poly'] = poly.to_shapely()
            tid_to_objs[tid].append(ann)

    summary_info = []
    for tid, objs in tid_to_objs.items():
        obj0 = objs[0]
        status = obj0.get('status', None)
        track_poly = unary_union([obj['poly'] for obj in objs])
        if status is None:
            color = 'kitware_blue'
        else:
            color = heuristics.IARPA_STATUS_TO_INFO[status]['color']
        summary_info.append({
            'track_id': tid,
            'status': status,
            'color': color,
            'geometry': track_poly,
        })
    gdf = gpd.GeoDataFrame(summary_info)
    return gdf


def imread_many(dset, gids, channels=None, space='video', resolution=None, workers=0):
    import kwutil
    load_jobs = ub.JobPool(mode='process', max_workers=workers)
    pman = kwutil.util_progress.ProgressManager()
    with load_jobs, pman:
        for gid in pman.progiter(gids, desc=f'submit load images jobs: {channels}'):
            coco_img = dset.coco_image(gid)
            delayed = coco_img.imdelay(channels=channels, space='video', resolution=resolution)
            load_jobs.submit(delayed.finalize)

        for job in pman.progiter(load_jobs.jobs, desc=f'collect load images jobs: {channels}'):
            imdata = job.result()
            yield imdata
