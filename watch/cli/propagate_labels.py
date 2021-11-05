# -*- coding: utf-8 -*-
"""
DEPRECATE: Use project_annotations instead
"""
import os
import json
import dateutil
import kwcoco
import kwimage
import ubelt as ub
import pathlib
import numpy as np
import scriptconfig as scfg


class PropagateLabelsConfig(scfg.Config):
    """
    This script reads the labels from the kwcoco file and performs the forward
    propagation of labels.
    The final goal of this script is to create a modified kwcoco file.
    The problem with original labels is that in many cases, annotators labeled
    a site in the first few images with a label (say, Active Construction) and
    then this annotation was missing for the next few frames.

    Note:
        # Given a kwcoco file with original annotations, this script forward propagates those annotations
        # and creates a new kwcoco file.

    TODO:
        - [x] Make sure the original annotations are correct. Currently annotations with very low overlap with images are showing up. Could be fixed by relying on geo-coordinates instead?
        - [x] Crop propagated annotations to bounds or valid data mask of new image
        - [x] Pull in and merge annotations from an external kwcoco file
        - [x] Handle splitting and merging tracks (non-unique frame_index)
        - [x] Stop propagation at category change (could be taken care of by external kwcoco file)
        - [x] Parallelize
    """
    default = {
        'src': scfg.Value(help=ub.paragraph(
            '''
            path to the kwcoco file to propagate labels in
            '''), position=1),

        'dst': scfg.Value('propagated_data.kwcoco.json', help=ub.paragraph(
            '''
            Where the output kwcoco file with propagated labels is saved
            '''), position=2),

        'ext': scfg.Value(None, help=ub.paragraph(
            '''
            Path to an optional external kwcoco file to merge annotations from.
            Must have a tag different from src's.
            ''')),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            if specified, visualizations will be written to this directory
            ''')),

        'verbose': scfg.Value(1, help=ub.paragraph(
            '''
            use this to print details
            ''')),

        'validate': scfg.Value(1, help=ub.paragraph(
            '''
            Validate spatial and temporal AOI of each site after propagating
            ''')),

        'crop': scfg.Value(1, help=ub.paragraph(
            '''
            Crop propagated annotations to the valid data mask of the new image
            ''')),

        'max_workers': scfg.Value(None, help=ub.paragraph(
            '''
            Max. number of workers to parallelize over, up to the number of
            regions/ROIs. None is auto; 0 is serial.
            '''))
    }

    # epilog = """
    # Example Usage:
    #     watch-cli propogate_labels --arg1=foobar
    # """


def annotated_band(img):
    # we hope each band has the same valid mask
    # but if not, this field picks out the (probable; heuristic-based)
    # band that the annotation was actually done on
    if img['file_name'] is not None:
        return img
    aux_ix = img.get('aux_annotated_candidate', 0)
    return img['auxiliary'][aux_ix]


def warped_wgs84_to_img(poly, img, inv=False):
    """
    Return poly warped from WGS84 geocoordinates to img's base space

    img['warp_wld_to_pxl'] does not give the appropriate transform for
    ann['segmentation_geo'] because it is in WGS84, not the wld CRS
    """
    from watch.gis import geotiff
    band = annotated_band(img)
    fpath = band['file_name']
    # cacher = ub.Cacher('geotiff_crs_info', depends=fpath)
    # info = cacher.tryload()
    # if info is None:
    if 1:
        try:
            info = geotiff.geotiff_crs_info(fpath)
        except NotImplementedError as e:
            # xdev.embed()
            raise e

        # can't pickle a function...
        # cacher.save(info)

    if inv:
        warp_wld_to_wgs84 = info['wld_to_wgs84']
        # warp_pxl_to_wld = kwimage.Affine.coerce(info['pxl_to_wld'])
        warp_pxl_to_wld = kwimage.Affine.coerce(img['wld_to_pxl']).inv()

        tfms = (warp_pxl_to_wld, warp_wld_to_wgs84)
    else:
        # this is an osr.CoordinateTransform function, not an affine transformation
        # so we can't combine the two steps
        warp_wgs84_to_wld = info['wgs84_to_wld']
        # warp_wld_to_pxl = kwimage.Affine.coerce(info['wld_to_pxl'])
        warp_wld_to_pxl = kwimage.Affine.coerce(img['wld_to_pxl'])

        tfms = (warp_wgs84_to_wld, warp_wld_to_pxl)

    poly = poly.swap_axes()
    for tfm in tfms:
        poly = poly.warp(tfm)

    return poly


def get_warp(gid1, gid2, dataset, use_geo=False):
    # Given a dataset and IDs of two images, the warp between these images is returned
    img1 = dataset.index.imgs[gid1]
    img2 = dataset.index.imgs[gid2]

    if use_geo:
        raise NotImplementedError('this is for the image CRS, not WGS84')
        # Use the geocoordinates as a base space to avoid propagating
        # badly-aligned annotations
        # geo_to_img1 = _warp_wgs84_to_img(img1)
        # geo_to_img2 = _warp_wgs84_to_img(img2)
        # return geo_to_img2 @ geo_to_img1.inv()

    else:
        # Get the transform from each image to the aligned "video-space"
        video_from_img1 = kwimage.Affine.coerce(img1['warp_img_to_vid'])
        video_from_img2 = kwimage.Affine.coerce(img2['warp_img_to_vid'])

        # Build the transform that warps img1 -> video -> img2
        img2_from_video = video_from_img2.inv()
        img2_from_img1 = img2_from_video @ video_from_img1

        return img2_from_img1


def ann_add_orig_info(ann, dset):
    # information needed to pass along to propagated anns
    if 'orig_info' not in ann:
        ann['orig_info'] = {
            'segmentation_orig': ann.get('segmentation', None),
            'segmentation_geos': ann['segmentation_geos'],
            'source_gid': ann['image_id'],
            'source_name': dset.imgs[ann['image_id']]['name'],
            'source_dset': dset.tag
        }

    return ann


def get_warped_ann(previous_ann, warp, image_entry, crop_to_valid=True):
    """
    Returns new annotation by applying a warp and optional crop

    If the new annotation falls outside the new image, returns None instead.

    This should be vectorized across annotations, because util_raster.to_crop
    supports this, since it assumes finding the valid mask of an image is
    expensive. But this is not a problem for the very small sizes we're
    working with here:

    %timeit watch.utils.util_raster.mask(full_tile)
    1.19 s ± 22.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    %timeit watch.utils.util_raster.mask(aligned_crop)
    4.44 ms ± 87.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    """
    from watch.utils import util_raster
    import shapely
    import shapely.ops

    # note we are using the semgentations stored in orig_info because the
    # most recent one may be cropped
    if warp == 'geo':
        try:
            segmentation_geo = (kwimage.MultiPolygon([
                kwimage.Polygon.from_geojson(
                    previous_ann['orig_info']['segmentation_geos'])
            ]))
            # TODO
            # 'geo' will be used when the orig annotation is from an external
            # (un-aligned, possibly nonexistent) image. This means this transformation
            # will be inaccurate, because it is not being done from *this* image's geocoords.
            # this cannot be fixed until external images are found and aligned.
            warped_seg = kwimage.Segmentation(
                warped_wgs84_to_img(segmentation_geo, image_entry))
        except AssertionError:
            # try to correct for lat/lon ordering
            raise
    else:
        segmentation = kwimage.Segmentation.coerce(
            previous_ann['orig_info']['segmentation_orig'])
        warped_seg = segmentation.warp(warp)

    if crop_to_valid:
        # warp to an aux image's space to crop to its valid mask,
        # then warp back to the base space
        band = annotated_band(image_entry)
        warp_aux_to_img = kwimage.Affine.coerce(band['warp_aux_to_img'])

        warped_seg = warped_seg.warp(warp_aux_to_img.inv())
        assert len(warped_seg) == 1, previous_ann['id']
        cropped = util_raster.crop_to(list(
            warped_seg.to_multi_polygon().to_shapely()),
                                      band['file_name'],
                                      bounds_policy='valid')[0]
        # we started with one polygon, but may have been split into more
        # during cropping. Cast it to a MultiPolygon to be safe.
        if cropped is None:
            return None
        elif isinstance(cropped, shapely.geometry.Polygon):
            cropped = shapely.geometry.MultiPolygon([cropped])
        else:
            assert isinstance(cropped, shapely.geometry.MultiPolygon)

        warped_seg = kwimage.Segmentation(
            kwimage.MultiPolygon.from_shapely(cropped))
        warped_seg = warped_seg.warp(warp_aux_to_img)

    warped_bbox = list(warped_seg.bounding_box().to_coco(style='new'))[0]

    # Create a new annotation object
    warped_ann = {'segmentation': warped_seg.to_coco(style='new')}
    warped_ann['bbox'] = warped_bbox
    warped_ann['category_id'] = previous_ann['category_id']
    warped_ann['track_id'] = previous_ann['track_id']
    warped_ann['image_id'] = image_entry['id']
    warped_ann['properties'] = previous_ann['properties']
    warped_ann['orig_info'] = previous_ann['orig_info']

    return warped_ann


def get_canvas_concat_channels(annotations, dataset, img_id):
    delayed = dataset.delayed_load(img_id)

    have_parts = delayed.channels.spec.split('|')
    if len({'r', 'g', 'b'} & set(have_parts)) == 3:
        channels = 'r|g|b'
    elif len({'red', 'green', 'blue'} & set(have_parts)) == 3:
        channels = 'red|green|blue'
    else:
        channels = '|'.join(have_parts[0:3])

    canvas = delayed.take_channels(channels).finalize()
    canvas = kwimage.normalize_intensity(canvas)
    canvas = kwimage.ensure_float01(canvas)
    canvas = np.nan_to_num(canvas)

    # draw on annotations
    # hack because draw_on(color=list) is not supported
    this_dset_anns = []
    ext_dset_anns = []
    for ann in annotations:
        if ann['orig_info']['source_dset'] == dataset.tag:
            this_dset_anns.append(ann)
        else:
            ext_dset_anns.append(ann)
    this_dets = kwimage.Detections.from_coco_annots(
        this_dset_anns, dset=dataset)
    ext_dets = kwimage.Detections.from_coco_annots(ext_dset_anns, dset=dataset)
    canvas = this_dets.draw_on(canvas, color='blue')
    canvas = ext_dets.draw_on(canvas, color='green')
    # try:
    #     # Requires some new changes to kwimage
    #     canvas = this_dets.draw_on(canvas.copy(), color='classes')
    # except Exception:
    #     canvas = this_dets.draw_on(canvas, color='blue')

    # draw on site boundaries
    image = dataset.imgs[img_id]
    video = dataset.index.videos[image['video_id']]
    for site in json.loads(video['properties']['sites']):
        site_geopoly = kwimage.Polygon.from_geojson(site['geometry'])
        site_poly = warped_wgs84_to_img(site_geopoly, image)
        canvas = site_poly.draw_on(canvas,
                                   color='red',
                                   fill=False,
                                   border=True)

    return canvas


def save_visualizations(canvas_chunk, fpath):
    # save visualizations of original and propagated labels
    # Tried to make this thread-safe by using the Axis object
    import kwplot
    plt = kwplot.autoplt()

    fig = plt.figure(figsize=(30, 8))
    n_images = len(canvas_chunk)
    for i, info in enumerate(canvas_chunk):
        before = info['before_canvas']
        after = info['after_canvas']
        ax1 = plt.subplot(2, n_images, i + 1)
        ax1.imshow(before)
        if i == 3:
            ax1.set_title('Original')
        ax1.axis('off')

        ax2 = plt.subplot(2, n_images, n_images + i + 1)
        ax2.imshow(after)
        if i == 3:
            ax2.set_title('Propagated')
        ax2.axis('off')

    from os.path import basename
    fig.suptitle(basename(fpath))

    fig.tight_layout()
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def validate_inbounds(anns, img):
    import pygeos
    from watch.utils import util_raster

    band = annotated_band(img)
    warp_aux_to_img = kwimage.Affine.coerce(band['warp_aux_to_img'])

    # use pygeos to vectorize this bit
    box = pygeos.box(0, 0, *kwimage.load_image_shape(band['file_name'])[1::-1])
    mask = pygeos.from_shapely(
        util_raster.mask(band['file_name'], as_poly=True))
    segs = pygeos.from_shapely([
        kwimage.Segmentation.coerce(ann['segmentation']).warp(
            warp_aux_to_img.inv()).to_multi_polygon().to_shapely()
        for ann in anns
    ])
    geos = pygeos.from_shapely([
        warped_wgs84_to_img(
            kwimage.Polygon.from_geojson(
                ann['orig_info']['segmentation_geos']),
            img).warp(warp_aux_to_img.inv()).to_multi_polygon().to_shapely()
        for ann in anns
    ])

    # getting an empty mask for 7 total blank images
    assert pygeos.contains(box, mask) or pygeos.is_empty(mask)

    # the transforms for these don't line up perfectly
    # up to 16px-area of difference observed when doing video warp
    # they should be identical when doing geo warp
    # TODO see which warp aligns better visually, video or geo
    # assert all(pygeos.contains(geos, segs))
    # assert all(pygeos.area(pygeos.difference(geos, segs)) < 20)

    # this is only true if crop is used
    # actually, only 13 failed instances anyway
    assert all(pygeos.contains(mask, segs)) or pygeos.is_empty(mask)

    # and 4 instances which are completely outside the valid mask without crop
    assert all(pygeos.intersects(mask, geos)) or pygeos.is_empty(mask)


def validate_timebounds(track_id, full_ds):

    anns = full_ds.annots(full_ds.index.trackid_to_aids[track_id])

    # are all the images in the same video?
    imgs = anns.images
    vidids = set(imgs.lookup('video_id'))
    assert len(vidids) == 1
    vidid = vidids.pop()

    # get the matching track/site from the video
    # we can't directly use track_id as a key in sites because of the missing MGRS tile
    # (could differ between sites and region)
    sites = []
    for s in json.loads(full_ds.index.videos[vidid]['properties']['sites']):
        sid = s['properties']['site_id']
        if sid[sid.index('_'):] == track_id:
            sites.append(s)
    assert len(sites) == 1
    site = sites[0]

    # check its time bounds
    img_datetimes = [
        dateutil.parser.parse(dt) for dt in imgs.lookup('date_captured')
    ]
    min_day = dateutil.parser.parse(site['properties']['start_date']).date()
    assert min_day == min(img_datetimes).date()
    # is the track finished?
    try:
        max_day = dateutil.parser.parse(site['properties']['end_date']).date()
        assert max_day == max(img_datetimes).date()
        assert 'Post Construction' in anns.cnames
    except dateutil.parser.ParserError:
        pass


def main(cmdline=False, **kwargs):
    """
    Main function for propagate_labels.

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.cli.propagate_labels import *  # NOQA
        >>> from watch.utils import util_data
        >>> dvc_dpath = util_data.find_smart_dvc_dpath()
        >>> bundle_dpath = dvc_dpath / 'drop1-S2-L8-aligned'
        >>> cmdline = False
        >>> kwargs = {
        >>>     'src': bundle_dpath / 'pre_prop.kwcoco.json',
        >>>     'dst': bundle_dpath / 'post_prob.kwcoco.json',
        >>>     'ext': dvc_dpath / 'drop1/annots.kwcoco.json',
        >>> }
        >>> #kwargs['src'] = '/home/joncrall/data/dvc-repos/smart_watch_dvc/jons-hacked-drop1-S2-L8-aligned/pre-prop2.kwcoco.json'
        >>> main(**kwargs)
    """
    config = PropagateLabelsConfig(default=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    # Read input file
    full_ds = kwcoco.CocoDataset.coerce(config['src'])
    print(ub.repr2(full_ds.basic_stats(), nl=1))

    # Ensure the output directory exists
    dst_fpath = pathlib.Path(config['dst'])
    dst_fpath.parent.mkdir(exist_ok=True, parents=True)

    # Read or stub external dataset
    try:
        ext_ds = kwcoco.CocoDataset.coerce(config['ext'])
        print(ub.repr2(ext_ds.basic_stats(), nl=1))
    except TypeError:
        ext_ds = kwcoco.CocoDataset()

    viz_dpath = config['viz_dpath']
    if viz_dpath not in {None, False}:
        viz_dpath = pathlib.Path(viz_dpath)
        viz_dpath.mkdir(exist_ok=True, parents=True)
    else:
        viz_dpath = None

    print(full_ds.videos().lookup('name'))
    print(ext_ds.videos().lookup('name'))

    print(set(full_ds.annots().lookup('track_id')))
    print(set(ext_ds.annots().lookup('track_id')))

    full_ds_imgnames = set(full_ds.images().lookup('parent_canonical_name'))
    ext_dst_imgnames = set(ext_ds.images().lookup('canonical_name'))
    full_ds_imgnames & ext_dst_imgnames

    # preprocessing step: add new 'orig_info' to every *annotation*.
    # original annotations: source_gid == gid and source_name == name
    # when we propagate labels, we will copy 'orig_info' from the copied
    # annotation. This includes the full segmentation (in case of cropping) and the
    # original image in the src or ext dataset.

    # why doesn't this get saved on dump()??
    full_ds._build_hashid()
    ext_ds._build_hashid()

    if full_ds.tag == '':
        full_ds.tag  = 'data.kwcoco.json'

    ext_ds.taghash = ext_ds.tag + ext_ds.hashid[0:8]
    full_ds.taghash = full_ds.tag + full_ds.hashid[0:8]

    for ann in full_ds.anns.values():
        ann = ann_add_orig_info(ann, full_ds)
    for ann in ext_ds.anns.values():
        ann = ann_add_orig_info(ann, ext_ds)

    # which categories we want to propagate
    # TODO: parameterize
    catnames_to_propagate = [
        'Site Preparation',
        'Active Construction',
        'Unknown',
        # 'No Activity',
        # 'Post Construction',
    ]
    cat_ids_to_propagate = [
        full_ds.index.name_to_cat[c]['id'] for c in catnames_to_propagate
    ]

    full_ds.annots().objs[1]
    ext_ds.annots().objs[1]

    # number of visualizations of every sequence
    frames_per_image = 7

    # a list of newly generated annotation IDs, for debugging purposes
    new_aids = []

    # output dataset to write to
    propagated_ds = full_ds.copy()
    propagated_ds.fpath = config['dst']

    # TODO multithreading breaks this, don't know why
    PROJ_LIB = os.environ.get('PROJ_LIB', None)

    # run job for each video
    max_workers = config['max_workers']
    if max_workers is None:
        max_workers = min(len(full_ds.index.videos), 8)
    executor = ub.Executor(mode='thread', max_workers=max_workers)

    all_video_ids = list(full_ds.index.videos.keys())
    all_video_ids = list(full_ds.index.videos.keys())[0:1]
    jobs = [(vid_id, executor.submit(_propogate_video_worker, vid_id, full_ds,
                                     ext_ds, cat_ids_to_propagate, viz_dpath,
                                     config))
            for vid_id in all_video_ids]
    prog = ub.ProgIter(jobs, total=len(full_ds.index.videos),
                       desc='process video')
    for vid_id, job in prog:
        warped_annotations, canvas_infos = job.result()
        new_aids.extend(
            [propagated_ds.add_annotation(**a) for a in warped_annotations])

        # save visualization
        if viz_dpath is not None:
            vid_name = full_ds.index.videos[vid_id]['name']
            for canvas_chunk in ub.chunks(canvas_infos, frames_per_image):
                min_frame = min([d['frame_num'] for d in canvas_chunk])
                max_frame = max([d['frame_num'] for d in canvas_chunk])
                fname = f'video_{vid_id:04d}_{vid_name}_frames_{min_frame:03d}_to_{max_frame:03d}.jpg'
                fpath = viz_dpath / fname
                save_visualizations(canvas_chunk, fpath)

    if PROJ_LIB is not None:
        os.environ['PROJ_LIB'] = PROJ_LIB

    # save the propagated dataset
    print('Save: propagated_ds.fpath = {!r}'.format(propagated_ds.fpath))
    propagated_ds.dump(propagated_ds.fpath, newlines=True)

    if viz_dpath:
        os.makedirs(viz_dpath, exist_ok=True)
        print('saved visualizations to: {!r}'.format(viz_dpath))

    # print statistics about propagation
    print('original annotations:', full_ds.n_annots, 'propagated annotations:',
          len(new_aids), 'total annotations:', propagated_ds.n_annots)

    # update index (add new anns to tracks) by reloading from disk
    propagated_ds = kwcoco.CocoDataset(propagated_ds.fpath)
    # validate correctness
    if config['validate']:
        for gid, img in propagated_ds.imgs.items():
            try:
                # no idea why PROJ was failing here
                validate_inbounds(propagated_ds.annots(gid=gid).objs, img)
            except AssertionError:
                print(f'image {gid} has OOB annotations')

        for track_id in sorted(propagated_ds.index.trackid_to_aids):
            try:
                validate_timebounds(track_id, propagated_ds)
            except AssertionError:
                print(f'track {track_id} is incomplete or overpropagated')

    return propagated_ds


def build_external_video(vid_id, full_ds, ext_ds):
    """
    Given two datasets with a corresponding video, match images between them
    assuming the gids and names are unreliable, and return an ordered list
    of all images from both videos, with full_ds taking precedence.

    Args:
        vid_id: of a corresponding video in both dsets

        full_ds: main dset to preferentially grab images from

        ext_ds: empty, or external dataset where images may not be well formed or exist,
            but images are a superset of full_ds's, which do exist

    Returns:
        List[Tuple[Dict, bool]]:
            image entries with a boolean flag. The image dictionary is from the
            external dataset when True and from the full dataset when False.
    """

    if ext_ds.n_images == 0:
        return [(i, False) for i in full_ds.images(vidid=vid_id).objs]

    # are these actually the same video?
    full_video = full_ds.index.videos[vid_id]
    ext_video = ext_ds.index.videos[vid_id]

    if (full_video['name'] != ext_video['name']):
        raise AssertionError(ub.paragraph(
            f'''
            video names {full_video["name"]} and {ext_video["name"]} do not agree
            '''))

    full_images = full_ds.images(vidid=vid_id)
    full_imgs = full_images.objs
    full_names = full_images.lookup('parent_canonical_name')

    ext_images = ext_ds.images(vidid=vid_id)
    ext_imgs = ext_images.objs
    ext_names = ext_images.lookup('canonical_name')

    # print('ext_names = {!r}'.format(ext_names[0:]))
    # print('full_names = {!r}'.format(full_names[0:]))

    # For each ext image, find the corresponding image in our dataset
    name_to_idx = {name: idx for idx, name in enumerate(full_names)}
    matched_idxs = [name_to_idx.get(ext, None) for ext in ext_names]
    matched_imgs = [None if idx is None else full_imgs[idx] for idx in matched_idxs]

    # does ext_ds's video completely contain full_ds's [annotated] video?
    num_full_imgs_with_annots = sum(map(bool, full_images.annots))
    num_matched_full_images = sum(m is not None for m in matched_imgs)
    if num_matched_full_images != num_full_imgs_with_annots:
        raise AssertionError(ub.paragraph(
            f'''
            num_matched_full_images={num_matched_full_images},
            num_full_imgs_with_annots={num_full_imgs_with_annots},
            ext_ds={ext_ds},
            full_ds={full_ds},
            '''))
    all_frames_sequence =  [(m, False) if m is not None else (e, True)
                            for m, e in zip(matched_imgs, ext_imgs)]
    return all_frames_sequence


# parallelize over videos
def _propogate_video_worker(vid_id, full_ds, ext_ds, cat_ids_to_propagate,
                            viz_dpath, config):
    """
    - start with a list of images from ext_ds, with the corresponding image
      from full_ds swapped in if we have it (build_external_video)

    - for each image:

      - for each track:

        - if it's already annotated, store all of these annots to propagate
          forward

        - if not, and this image is from full_ds, grab the most recent annots from this track

          - for each annot:

            - if it was last seen in ext_ds, it is a geo-segmentation. warp it
                to this image space from WGS84 using [mostly] the same thing done
                in coco_align_geotiffs. (get_warped_ann(warp='geo'))

            - else, it is a segmentation. Warp it[s original extent] to this
                image through the video space. (get_warped_ann(warp=get_warp()))

            - crop it to the image's valid mask.

      - if this image is from full_ds, save its annotations in the result.
    """
    # a set of all the track IDs in this video

    full_video_gids = full_ds.index.vidid_to_gids[vid_id]

    # time per loop: best=75.407 µs, mean=76.703 ± 2.0 µs
    full_aids = list(ub.flatten(full_ds.images(full_video_gids).annots))
    full_annots = full_ds.annots(full_aids)
    full_tid_to_aids = ub.group_items(full_annots, full_annots.lookup('track_id'))
    track_ids = set(full_tid_to_aids.keys())

    # # time per loop: best=385.911 µs, mean=393.429 ± 2.8 µs
    # track_ids = set(
    #     full_ds.subset(full_video_gids).index.
    #     trackid_to_aids.keys())

    # a dictionary of latest annotation IDs, indexed by the track IDs
    latest_ann_ids = {}

    # a dictionary of latest image IDs, this will be needed to apply affine transform on images
    latest_img_ids = {}

    # whether the latest image ID is from the external dataset
    latest_is_ext = {}

    # canvases for visualizations
    canvas_infos = []

    # results for this video
    warped_annotations = []

    # for all sorted images in this video in both dsets
    all_frames_sequence = build_external_video(vid_id, full_ds, ext_ds)

    #######

    full_frame_idx = 0
    for img, is_ext in all_frames_sequence:
        img_id = img['id']
        if 1:
            # prog.ensure_newline()
            print('{} img_id = {!r}'.format(('ext' if is_ext else 'int'),
                                            img_id))

        frame_dset = (ext_ds if is_ext else full_ds)
        # Update all current tracks to have their latest annotation state
        this_image_annots = frame_dset.annots(gid=img_id)
        this_image_anns = this_image_annots.objs

        this_track_ids = set()
        for ann in this_image_anns:
            if 0:
                print('ann = {}'.format(
                    ub.repr2(ub.dict_diff(
                        ann, {'segmentation', 'segmentation_geos',
                              'properties', 'orig_info'}), nl=0)))
            track_id = ann['track_id']
            this_track_ids.add(track_id)

            # handle non-unique track ids
            if not (latest_img_ids.get(track_id) == img_id
                    and latest_is_ext.get(track_id) == is_ext):
                latest_ann_ids[track_id] = set()
            latest_img_ids[track_id] = img_id
            latest_is_ext[track_id] = is_ext
            latest_ann_ids[track_id].add(ann['id'])

        # don't need to fix anything if this is an external image
        if is_ext:
            continue
        # from now on we know that this_image, img_id is from full_ds

        # if there is anything missing, we are going to fix now
        this_image_fixed_anns = this_image_anns.copy()

        # was there any seen track ID that was not in this image?
        for missing in track_ids - this_track_ids:
            for aid in latest_ann_ids.get(missing, {}):
                previous_annotation = (ext_ds if latest_is_ext[missing] else full_ds).anns[aid]
                # this should be an original annotation
                assert (previous_annotation['image_id'] == previous_annotation['orig_info']['source_gid'] and
                        (previous_annotation['orig_info']['source_dset'] != full_ds.tag) == latest_is_ext[missing])

                # check if the annotation belongs to the list of cats we want to propagate
                if previous_annotation['category_id'] in cat_ids_to_propagate:

                    # get the warp from previous image to this image
                    previous_image_id = latest_img_ids[missing]

                    if previous_annotation['orig_info']['source_dset'] != full_ds.tag:
                        warp = 'geo'
                    else:
                        warp = get_warp(previous_image_id, img_id, full_ds)

                    warped_annotation = get_warped_ann(
                        previous_annotation, warp,
                        full_ds.imgs[img_id],
                        crop_to_valid=config['crop'])

                    # only possible with crop
                    if warped_annotation is None:
                        if config['verbose']:
                            # prog.ensure_newline()
                            print('skipped OOB annotation for image',
                                  img_id, 'track ID', missing)
                        continue

                    # add the propagated annotation in the new kwcoco dataset
                    warped_annotations.append(warped_annotation)

                    # append in the list for visualizations
                    this_image_fixed_anns.append(warped_annotation)

                    if config['verbose']:
                        # prog.ensure_newline()
                        print('added annotation for image', img_id,
                              'track ID', missing)

        # Get "frames_per_image" number of canvases for visualization with original annotations
        # num_frames = len(full_ds.index.vidid_to_gids[vid_id])

        if viz_dpath is not None:
            # is_starting_frame = (j < frames_per_image)
            # is_ending_frame = (j >= (num_frames - frames_per_image))
            # if is_ending_frame or is_starting_frame:
            if True:
                before_canvas = get_canvas_concat_channels(
                    annotations=this_image_anns,
                    dataset=full_ds,
                    img_id=img_id)
                after_canvas = get_canvas_concat_channels(
                    annotations=this_image_fixed_anns,
                    dataset=full_ds,
                    img_id=img_id)
                canvas_infos.append({
                    'before_canvas': before_canvas,
                    'after_canvas': after_canvas,
                    'frame_num': full_frame_idx,
                })

        full_frame_idx += 1

    return warped_annotations, canvas_infos


def __SIMPLE_GEOSPACE_PROPOGATE(ext_ds, vid_id, full_ds):
    """
    This is broken, but it has useful code I dont want to lose quite yet

    This code can interactively (or automatically depending on what is
    commented in / out) write an animation that displays the process of
    propogation in geo-space.

    TODO: Either this script should be fixed up, or this visaulization
    capabilities of this script should be incorporated into the main
    functionality.
    """
    full_video_gids = full_ds.index.vidid_to_gids[vid_id]
    full_aids = list(ub.flatten(full_ds.images(full_video_gids).annots))
    full_annots = full_ds.annots(full_aids)

    all_frames_sequence = build_external_video(vid_id, full_ds, ext_ds)
    INTERACTIVE = 1
    if INTERACTIVE:
        import kwplot
        import geopandas as gpd
        from shapely import ops
        kwplot.autompl()
        wld_map_gdf = gpd.read_file(
            gpd.datasets.get_path('naturalearth_lowres')
        )
        fig = kwplot.figure(fnum=1, doclf=1)
        ax = fig.gca()
        ax = wld_map_gdf.plot(ax=ax)

        combo = ops.unary_union([kwimage.Polygon.coerce(s).to_shapely() for s in full_annots.lookup('segmentation_geos')])
        aoi = combo.convex_hull
        box = kwimage.Polygon.from_shapely(aoi).bounding_box()
        box = box.scale(1.3, about='center')
        minx, miny, maxx, maxy = box.to_tlbr().data[0]
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        import xdev
        # frame_iter = xdev.InteractiveIter(all_frames_sequence)
        frame_iter = (all_frames_sequence)

        def clear_ax(ax):
            attrs = ['lines', 'patches', 'texts', 'tables', 'artists',
                     'images', 'child_axes', 'collections', 'containers']
            for attr in attrs:
                while len(getattr(ax, attr)):
                    getattr(ax, attr)[0].remove()
        clear_ax(ax)

        dpath = ub.ensure_app_cache_dir('watch/tmpviz432/frames')
    else:
        frame_iter = all_frames_sequence

    latest = []
    for frame_idx, (img, is_ext) in enumerate(frame_iter):
        img_id = img['id']
        print('{} img_id = {!r}'.format(('ext' if is_ext else 'int'), img_id))
        frame_dset = (ext_ds if is_ext else full_ds)
        this_annots = frame_dset.annots(gid=img_id)

        cand_anns = [ann.copy() for ann in this_annots.objs]
        for ann in cand_anns:
            kw_poly = kwimage.Polygon.coerce(ann['segmentation_geos'])
            sh_poly = kw_poly.to_shapely()
            ann['sh_poly'] = sh_poly

        # Mark which of the previous annotations our new annotation will update
        cand_flags = []
        for ann1 in cand_anns:
            ann1_flags = []
            for ann2 in latest:
                flag = False
                if ann1['track_id'] == ann2['track_id']:
                    if ann2['sh_poly'].intersects(ann1['sh_poly']):
                        flag = True
                ann1_flags.append(flag)
            # flags = [ann2['sh_poly'].intersects(ann['sh_poly']) for ann2 in latest]
            cand_flags.append(ann1_flags)
        cand_flags = np.array(cand_flags)

        is_new = ~cand_flags.any(axis=1)

        # Gather new annotations that will update old ones
        update_annots = list(ub.compress(cand_anns, ~is_new))
        update_flags = cand_flags[~is_new]
        is_updated = np.zeros(len(latest)).astype(bool)
        for ann, flags in zip(update_annots, update_flags):
            is_updated[flags] = True
            # assert flags.sum() == 1

        new_anns = list(ub.compress(cand_anns, is_new))
        replacement_anns = list(ub.compress(cand_anns, ~is_new))
        removed_anns = list(ub.compress(latest, is_updated))
        keep_anns = list(ub.compress(latest, ~is_updated))

        latest = keep_anns + replacement_anns + new_anns

        if INTERACTIVE:
            clear_ax(ax)
            wld_map_gdf.plot(ax=ax)

            for ann in keep_anns:
                catname = ext_ds.cats[ann['category_id']]['name']
                text = 'keep - {} - {}'.format(catname, ann['track_id'])
                poly = kwimage.Polygon.from_shapely(ann['sh_poly'])
                poly.to_boxes().draw(labels=[text], alpha=0.2)
                poly.draw(color='yellow', alpha=0.5)

            for ann in removed_anns:
                catname = ext_ds.cats[ann['category_id']]['name']
                text = 'removed - {} - {}'.format(catname, ann['track_id'])
                poly = kwimage.Polygon.from_shapely(ann['sh_poly'])
                poly.to_boxes().draw(labels=[text], alpha=0.2)
                poly.draw(color='red', alpha=0.5)

            for ann in new_anns:
                catname = ext_ds.cats[ann['category_id']]['name']
                text = 'new - {} - {}'.format(catname, ann['track_id'])
                poly = kwimage.Polygon.from_shapely(ann['sh_poly'])
                poly.to_boxes().draw(labels=[text], alpha=0.2)
                poly.draw(color='pink', alpha=0.5)

            for ann in replacement_anns:
                catname = ext_ds.cats[ann['category_id']]['name']
                text = 'replacement - {} - {}'.format(catname, ann['track_id'])
                poly = kwimage.Polygon.from_shapely(ann['sh_poly'])
                poly.to_boxes().draw(labels=[text], alpha=0.2)
                poly.draw(color='green', alpha=0.5)

            from kwplot.mpl_make import render_figure_to_image
            img = render_figure_to_image(fig)
            img = kwimage.convert_colorspace(img, src_space='bgr', dst_space='rgb')
            fpath = os.path.join(dpath, 'frame_{:04d}.png'.format(frame_idx))
            kwimage.imwrite(fpath, img)
            xdev.InteractiveIter.draw()


_SubConfig = PropagateLabelsConfig

if __name__ == '__main__':
    r"""
    CommandLine:
        # Command line used to test this script in development

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

        EXT_FPATH=$DVC_DPATH/drop1/data.kwcoco.json
        BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned
        INPUT_KWCOCO=$BUNDLE_DPATH/pre_prop.kwcoco.json
        OUTPUT_KWCOCO=$BUNDLE_DPATH/post_prop.kwcoco.json

        python -m watch.cli.propagate_labels \
                --src $INPUT_KWCOCO \
                --ext $EXT_FPATH \
                --dst $OUTPUT_KWCOCO \
                --viz_dpath=$BUNDLE_DPATH/_propagate_viz5



        python -m watch.cli.coco_visualize_videos \
            --src $INPUT_KWCOCO --viz_dpath $BUNDLE_DPATH/viz_before \
            --num_workers=6

        python -m watch.cli.coco_visualize_videos \
            --src $OUTPUT_KWCOCO --viz_dpath $BUNDLE_DPATH/viz_after \
            --num_workers=6


        # DEBUG CASE
        kwcoco subset --src $INPUT_KWCOCO --dst $INPUT_KWCOCO.small.json \
            --select_videos '.name | startswith("BH_")'

        python -m watch.cli.propagate_labels \
                --src $INPUT_KWCOCO.small.json --dst $OUTPUT_KWCOCO.small.json \
                --viz_dpath=$BUNDLE_DPATH/_propagate_viz_small


    """
    main(cmdline=True)
