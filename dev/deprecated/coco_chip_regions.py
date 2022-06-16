import scriptconfig as scfg
import ubelt as ub
import kwcoco
import kwarray
import netharn as nh
import numpy as np
import kwimage
from skimage.transform import AffineTransform
from os.path import join, exists, abspath, relpath


class ChipRegions(scfg.Config):
    """
    Create a chipped dataset around objects of interest
    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory for the output'),

        'window_overlap': scfg.Value(0),

        'window_dims': scfg.Value((1024, 1024), help='height width of the window'),

        'negative_classes': scfg.Value(
            ['ignore', 'background'],
            help=ub.paragraph(
                '''
                category names that are considered as negatives
                '''
            )),

        'classes_of_interest': scfg.Value(
            None,
            help=ub.paragraph(
                '''
                Only consider specified categories as positives
                '''
            )),
        'verbose': scfg.Value(1, help='verbosity level'),

        'crop_mode': scfg.Value('gdal', help='only gdal for now'),

        'absolute': scfg.Value(False, help=(
            'If False, the output file uses relative paths, otherwise '
            'uses absolute paths')),

        'ratio': scfg.Value(0.25, help='Choose number of negatives as a ratio of positives'),

        'seed': scfg.Value(1019551220, help='random seed for reproducibility')
    }


def main(**kw):
    """
    Ignore:
        # Developer IPython namespace setup
        cd ~/data/dvc-repos/smart_watch_dvc/drop0
        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        from coco_chip_regions import *  # NOQA
        kw = {
            'src': 'AE-Dubai-0001/data.kwcoco.json',
            'dst': 'AE-Dubai-0001-chipped',
        }
    """
    config = ChipRegions(default=kw, cmdline=True)

    window_overlap = config['window_overlap']
    window_dims = config['window_dims']
    verbose = config['verbose']
    negative_classes = config['negative_classes']
    classes_of_interest = config['classes_of_interest']
    crop_mode = config['crop_mode']
    absolute = config['absolute']
    ratio = config['ratio']
    seed = config['seed']

    rng = kwarray.ensure_rng(rng=seed)

    src_dset = kwcoco.CocoDataset(config['src'])
    positives, negatives = select_regions(
        src_dset, window_overlap, window_dims,
        classes_of_interest=classes_of_interest,
        negative_classes=negative_classes, verbose=verbose)

    neg_idxs = np.arange(len(negatives))
    rng.shuffle(neg_idxs)
    num_negatives = int(len(positives) * ratio)
    chosen_neg_idxs = sorted(neg_idxs[:num_negatives])

    chosen_negatives = list(ub.take(negatives, chosen_neg_idxs))
    chosen_positives = positives

    chosen_regions = chosen_positives + chosen_negatives

    dst_bundle_dpath = config['dst']
    ub.ensuredir(dst_bundle_dpath)

    dst_img_dpath = ub.ensuredir((dst_bundle_dpath, '_assets', 'images'))
    dst_dset = kwcoco.CocoDataset()
    dst_dset.fpath = join(dst_bundle_dpath, 'data.kwcoco.json')

    # Copy over all categories (persist category ids)
    for cat in src_dset.cats.values():
        dst_dset.add_category(**cat)

    for gid, sl, aids in ub.ProgIter(chosen_regions, verbose=verbose * 3,
                                     desc='cropping regions'):

        y_min, y_max = sl[0].start, sl[0].stop
        x_min, x_max = sl[1].start, sl[1].stop
        xsize = x_max - x_min
        ysize = y_max - y_min

        src_gpath = src_dset.get_image_fpath(gid)

        # Note: we may have been able to do this in fewer lines with ndsampler
        suffix = '_gid{}_chip{}_{}_{}_{}'.format(gid, x_min, y_min, x_max, y_max)
        dst_gpath = ub.augpath(src_gpath, dpath=dst_img_dpath, suffix=suffix)

        # Sample the chipped image region and write it to disk
        if crop_mode == 'gdal':
            # Gdal crop mode helps copy over relevant metadata
            params = dict(
                xoff=x_min, yoff=y_min, xsize=xsize, ysize=ysize,
                IN=src_gpath, OUT=dst_gpath,
            )
            command = (
                'gdal_translate '
                '-srcwin {xoff} {yoff} {xsize} {ysize} '
                '{IN} {OUT}'
            ).format(**params)
            if not exists(dst_gpath):
                ub.cmd(command, verbose=verbose, check=True)
        else:
            raise NotImplementedError(crop_mode)

        if absolute:
            file_name = abspath(dst_gpath)
        else:
            file_name = relpath(dst_gpath, dst_bundle_dpath)

        # Add the cropped image to the chipped dataset
        src_img = src_dset.imgs[gid]
        dst_img = {
            'file_name': file_name,
            'width': xsize,
            'height': ysize,
        }
        foreign = ub.dict_diff(src_img, set(dst_img) | {'id'})
        dst_img.update(foreign)

        new_gid = dst_dset.add_image(**dst_img)

        transform = AffineTransform(translation=(-x_min, -y_min))
        src_annots = src_dset.annots(aids)
        src_anns = src_annots.objs
        src_dets = src_annots.detections
        dst_dets = src_dets.warp(transform)
        dst_anns = list(dst_dets.to_coco(dset=dst_dset, style='new'))

        # Copy foreign fields from source annotations
        for src_ann, dst_ann in zip(src_anns, dst_anns):
            foreign = ub.dict_diff(src_ann, set(dst_ann) | {'id', 'image_id'})
            dst_ann.update(foreign)
            dst_ann['image_id'] = new_gid
            dst_dset.add_annotation(**dst_ann)

    print('Writing to dst_dset.fpath = {!r}'.format(dst_dset.fpath))
    dst_dset.dump(dst_dset.fpath, newlines=True)


def select_regions(dset, window_overlap, window_dims,
                   classes_of_interest=None,
                   negative_classes={'ignore', 'background'},
                   verbose=1):
    """
    TODO: A Generalized version of this logic should be core in ndsampler.

    Args:
        negative_classes : classes to consider as negative
    """
    from ndsampler import isect_indexer

    keepbound = True

    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    gid_height_width_list = [
        (img['id'], img['height'], img['width'])
        for img in ub.ProgIter(dset.imgs.values(), total=len(dset.imgs),
                               desc='load image sizes', verbose=verbose)]

    if any(h is None or w is None for gid, h, w in gid_height_width_list):
        raise ValueError('All images must contain width and height attrs.')

    @ub.memoize
    def _memo_slider(full_dims, window_dims):
        window_dims_ = full_dims if window_dims == 'full' else window_dims
        slider = nh.util.SlidingWindow(
            full_dims, window_dims_, overlap=window_overlap,
            keepbound=keepbound, allow_overshoot=True)
        slider.regions = list(slider)
        return slider

    gid_to_slider = {
        gid: _memo_slider(full_dims=(height, width), window_dims=window_dims)
        for gid, height, width in ub.ProgIter(
            gid_height_width_list, desc='build sliders', verbose=verbose)
    }

    _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(
        dset, verbose=verbose)

    num_annots_per_image = [
        len(qtree.aid_to_tlbr)
        for gid, qtree in _isect_index.qtrees.items()
    ]
    sum(num_annots_per_image)

    verbose = 1

    gid_to_slider2 = {}
    gid_to_slider3 = {}
    for gid, slider in ub.ProgIter(gid_to_slider.items(),
                                   total=len(gid_to_slider),
                                   desc='finding regions with annots',
                                   verbose=verbose):

        qtree = _isect_index.qtrees[gid]
        annot_aids = np.array(list(qtree.aid_to_tlbr.keys()))
        if len(annot_aids):
            gid_to_slider2[gid] = slider
        else:
            gid_to_slider3[gid] = slider

    positives = []
    negatives = []
    verbose = 0
    all_seen_aids = set()
    for gid, slider in ub.ProgIter(gid_to_slider2.items(),
                                   total=len(gid_to_slider2),
                                   desc='finding regions with annots',
                                   verbose=verbose):

        # image_aids = set(_isect_index.qtrees[gid].aid_to_tlbr.keys())
        qtree = _isect_index.qtrees[gid]
        annot_aids = np.array(list(qtree.aid_to_tlbr.keys()))
        annot_ltrb = np.array(list(qtree.aid_to_tlbr.values())).reshape(-1, 4).astype(np.float32)
        annot_boxes = kwimage.Boxes(annot_ltrb, 'ltrb').to_cxywh()

        # For each image, create a box for each spatial region in the slider
        window_ltrb_ = []
        window_regions = list(slider)
        for region in window_regions:
            y_sl, x_sl = region
            window_ltrb_.append([x_sl.start,  y_sl.start, x_sl.stop, y_sl.stop])
        window_ltrb = np.array(window_ltrb_, dtype=np.float32)
        window_boxes = kwimage.Boxes(window_ltrb, 'ltrb')

        # Mabe a good API would be window_boxes.overlaps(annot_boxes, normalizer='iou')
        # normalizer='dice'
        # normalizer='overlap_area'
        # normalizer='ioaa'
        ious = window_boxes.isect_area(annot_boxes)
        if (ious > 0).any():
            print(ious.sum())
        else:
            print('window_boxes.shape = {!r}'.format(window_boxes.shape))
            print('annot_boxes.shape = {!r}'.format(annot_boxes.shape))
            print('window_boxes = {!r}'.format(window_boxes))
            print('annot_boxes = {!r}'.format(annot_boxes))
            print('slider = {!r}'.format(slider))

        image_seen_aids = set()
        for region, area in zip(window_regions, ious):
            flags = area > 0
            aids = list(map(int, annot_aids[flags]))
            pos_aids = aids
            image_seen_aids.update(aids)
            # aids = sampler.regions.overlapping_aids(gid, box, visible_thresh=0.001)
            if len(pos_aids):
                positives.append((gid, region, aids))
            else:
                negatives.append((gid, region, aids))
        all_seen_aids.update(image_seen_aids)

    print('Found {} positives'.format(len(positives)))
    print('Found {} negatives'.format(len(negatives)))
    return positives, negatives
    # len([gid for gid, a in sampler.dset.gid_to_aids.items() if len(a) > 0])


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/coco_chip_regions.py
    """
    main()
