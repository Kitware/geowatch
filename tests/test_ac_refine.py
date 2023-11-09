class PolygonExtractor:
    """

    Example:
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/tests'))
        from test_ac_refine import *  # NOQA
        cls = PolygonExtractor
        self = PolygonExtractor.demo()

    """

    def __init__(self, heatmap_thwc, bounds):
        self.heatmap_thwc = heatmap_thwc
        self.bounds = bounds

    def show(self):
        import scipy
        import scipy.special
        import kwimage
        heatmaps = self.heatmap_thwc
        heatmap_frames = [scipy.special.softmax(frame, axis=2) for frame in heatmaps]
        to_show_frames = [self.bounds.draw_on(frame, edgecolor='white', fill=False) for frame in heatmap_frames]
        import kwplot
        kwplot.autompl()
        stacked = kwimage.stack_images_grid(to_show_frames, pad=10, bg_value='kitware_green')
        kwplot.imshow(stacked)

    @classmethod
    def demo(cls):
        """
        cls = PolygonExtractor
        """
        kwargs = cls.demodata()
        self = cls(**kwargs)
        return self

    @classmethod
    def demodata(cls):
        import kwimage
        import numpy as np

        H, W = 512, 512

        poly0 = kwimage.Polygon.random(rng=0)
        poly0 = poly0.translate((-0.5, -0.5))
        poly0 = poly0.scale((128, 128))
        poly0 = poly0.translate((256, 256))

        poly1 = poly0.translate((-94, -94))
        poly2 = poly0.translate((+94, +94))
        poly3 = poly0.translate((-94, +94))
        poly4 = poly0.translate((+94, -94))

        canvas = np.zeros((H, W, 3), dtype=np.float32)

        polys = {
            'poly0': poly0,
            'poly1': poly1,
            'poly2': poly2,
            'poly3': poly3,
            'poly4': poly4,
        }

        bounds = poly3.union(poly2).union(poly0).union(poly1).union(poly4).convex_hull
        bounds = bounds.scale(1.1, about='centroid')

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
                {'frame': 12, 'class': 0},
                {'frame': 13, 'class': 1},
                {'frame': 14, 'class': 1},
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

        heatmap_frames = [canvas.copy() for _ in range(16)]
        for key in polys.keys():
            poly = polys[key]
            seq = sequences[key]
            for info in seq:
                idx = info['frame']
                cidx = info['class']

                chan = np.ascontiguousarray(heatmap_frames[idx][..., cidx])
                new_chan = poly.fill(chan, value=1)
                heatmap_frames[idx][..., cidx] = new_chan

        if 0:
            import scipy
            import scipy.special
            heatmap_frames = [scipy.special.softmax(frame, axis=2) for frame in heatmap_frames]
            to_show_frames = [bounds.draw_on(frame, edgecolor='white', fill=False) for frame in heatmap_frames]
            import kwplot
            kwplot.autompl()
            stacked = kwimage.stack_images_grid(to_show_frames, pad=10, bg_value='kitware_green')
            kwplot.imshow(stacked)

        heatmap_thwc = np.array(heatmap_frames)
        if 0:
            import xarray as xr
            heatmap_thwc = xr.DataArray(heatmap_thwc, dims=list('THWC'))

        kwargs = {
            'bounds': bounds,
            'heatmap_thwc': heatmap_thwc,
        }
        return kwargs

    def predict(self):
        """
        Example:
            >>> import sys, ubelt
            >>> sys.path.append(ubelt.expandpath('~/code/watch/tests'))
            >>> from test_ac_refine import *  # NOQA
            >>> cls = PolygonExtractor
            >>> self = PolygonExtractor.demo()

        """
        import einops
        import sklearn
        import sklearn.cluster
        import sklearn.decomposition
        import kwutil
        import ubelt as ub

        workers = kwutil.util_parallel.coerce_num_workers('avail')
        print(f'workers={workers}')

        pca = sklearn.decomposition.PCA(n_components=8)

        # dbscan = sklearn.cluster.DBSCAN(
        #     eps=0.5, min_samples=5, metric='cosine', metric_params=None,
        #     algorithm='auto', leaf_size=30, n_jobs=workers)

        mean_shift = sklearn.cluster.MeanShift(bandwidth=None, seeds=None,
                                               bin_seeding=False,
                                               min_bin_freq=1,
                                               cluster_all=True, n_jobs=workers,
                                               max_iter=300)

        small_heatmap = self.heatmap_thwc[:, ::16, ::16, :]
        t, h, w, c = small_heatmap.shape
        X = einops.rearrange(small_heatmap, 't h w c -> (h w) (t c)')
        print(f'X.shape={X.shape}')
        Xhat = pca.fit_transform(X)

        print(f'Xhat.shape={Xhat.shape}')

        print('start mean shift')
        print('mean_shift = {}'.format(ub.urepr(mean_shift, nl=1)))
        yhat = mean_shift.fit_predict(Xhat)

        import numpy as np
        small_label_img = einops.rearrange(yhat, '(h w) -> h w', w=w, h=w)
        small_label_img = np.ascontiguousarray(small_label_img).astype(np.uint8)

        import kwimage
        label_img = kwimage.imresize(small_label_img, scale=(16, 16), interpolation='nearest')

        import scipy
        import scipy.special
        import kwimage
        heatmaps = self.heatmap_thwc
        heatmap_frames = [scipy.special.softmax(frame, axis=2) for frame in heatmaps]
        to_show_frames = [self.bounds.draw_on(frame, edgecolor='white', fill=False) for frame in heatmap_frames]
        import kwplot
        kwplot.autompl()
        stacked = kwimage.stack_images_grid(to_show_frames, pad=10, bg_value='kitware_green')

        from watch.utils import util_kwimage
        canvas = util_kwimage.colorize_label_image(label_img)
        import kwplot
        kwplot.plt
        kwplot.imshow(stacked, pnum=(1, 2, 1))
        kwplot.imshow(canvas, pnum=(1, 2, 2))

        # X = einops.rearrange(self.heatmap_thwc, 't h w c -> (h w) (t c)')

        if 0:
            mean_img = self.heatmap_thwc.mean(axis=0)
            import kwplot
            kwplot.plt
            kwplot.imshow(mean_img, fnum=2)
