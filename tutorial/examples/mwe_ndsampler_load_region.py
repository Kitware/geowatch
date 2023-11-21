

def mwe_load_spacetime_region():
    import kwimage
    # import kwcoco
    import numpy as np
    from watch.demo import coerce_kwcoco
    from typing import List
    coco_dset = coerce_kwcoco('watch-msi', dates=True, geodata=True)

    # Keep consistent bands
    keep_ids = [img.img['id'] for img in coco_dset.images().coco_images if 'B11' in img.channels]
    coco_dset = coco_dset.subset(keep_ids)

    import ndsampler
    # from watch.utils import util_kwimage
    sampler = ndsampler.CocoSampler(coco_dset)

    gid1, gid2, gid3 = coco_dset.videos().images[0][0:3]

    # Choose channels likely to be part of the demo set
    chans = 'B11'

    target = {
        'space_slice': kwimage.Boxes([[20, 30, 301, 307]], format='xywh').to_slices()[0],
        'gids': [gid1, gid2, gid3],
        'channels': chans,
    }

    sample = sampler.load_sample(target, with_annots=True)

    detections: List[kwimage.Detections] = sample['annots']['frame_dets']
    imdata = sample['im']

    frame_canvases = []

    for frame, dets in zip(imdata, detections):
        canvas = kwimage.normalize_intensity(frame)
        # fill_nans_with_checkers seems to not work with multi channels
        # this is a hack to fix it.
        canvas = np.concatenate([kwimage.fill_nans_with_checkers(c) for c in canvas.transpose(2, 0, 1)], axis=2)
        assert not np.isnan(canvas).sum()
        canvas = dets.draw_on(canvas)
        frame_canvases.append(canvas)

    final_canvas = kwimage.stack_images(frame_canvases)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(final_canvas)
