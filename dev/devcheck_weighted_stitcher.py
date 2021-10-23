from watch.utils import util_kwimage


def _demo_weighted_stitcher():
    import kwimage
    import kwarray
    import kwplot

    stitch_dims = (512, 512)
    window_dims = (64, 64)

    # Seed a local random number generator
    rng = kwarray.ensure_rng(8022)

    # Create a random heatmap we will use as the dummy "truth" we would like
    # to predict
    heatmap = kwimage.Heatmap.random(dims=stitch_dims, rng=rng)
    truth = heatmap.data['class_probs'][0]

    overlap_to_unweighted = {}
    overlap_to_weighted = {}

    center_weights = util_kwimage.upweight_center_mask(window_dims)

    for overlap in [0, 0.1, 0.3, 0.5, 0.8, 0.9]:
        slider = kwarray.SlidingWindow(stitch_dims, window_dims, overlap=overlap,
                                       keepbound=True, allow_overshoot=True)

        unweighted_sticher = kwarray.Stitcher(stitch_dims, device='numpy')
        weighted_sticher = kwarray.Stitcher(stitch_dims, device='numpy')

        # Seed a local random number generator
        rng = kwarray.ensure_rng(8022)
        for space_slice in slider:

            # Make a (dummy) prediction at this slice
            # Our predition will be a perterbed version of the truth
            real_data = truth[space_slice]
            aff = kwimage.Affine.random(rng=rng, theta=0, shear=0)

            # Perterb spatial location
            pred_data = kwimage.warp_affine(real_data, aff)
            pred_data += (rng.randn(*window_dims) * 0.5)
            pred_data = pred_data.clip(0, 1)

            # Add annoying boundary artifacts
            pred_data[0:3, :] = rng.rand()
            pred_data[-3:None, :] = rng.rand()
            pred_data[:, -3:None] = rng.rand()
            pred_data[:, 0:3] = rng.rand()

            pred_data = kwimage.gaussian_blur(pred_data, kernel=9)

            unweighted_sticher.add(space_slice, pred_data)
            weighted_sticher.add(space_slice, pred_data, weight=center_weights)

        unweighted_stiched_pred = unweighted_sticher.finalize()
        weighted_stiched_pred = weighted_sticher.finalize()
        overlap_to_weighted[overlap] = weighted_stiched_pred
        overlap_to_unweighted[overlap] = unweighted_stiched_pred

    kwplot.autompl()
    pnum_ = kwplot.PlotNums(nCols=2, nSubplots=len(overlap_to_unweighted) * 2 + 2)

    kwplot.imshow(truth, fnum=1, pnum=pnum_(), title='(Dummy) Truth')
    kwplot.imshow(center_weights, fnum=1, pnum=pnum_(), title='Window Weights')

    for overlap in overlap_to_unweighted.keys():
        weighted_stiched_pred = overlap_to_weighted[overlap]
        unweighted_stiched_pred = overlap_to_unweighted[overlap]

        kwplot.imshow(unweighted_stiched_pred, fnum=1, pnum=pnum_(), title=f'Unweighted stitched preds: overlap={overlap}')
        kwplot.imshow(weighted_stiched_pred, fnum=1, pnum=pnum_(), title=f'Weighted stitched preds: overlap={overlap}')
