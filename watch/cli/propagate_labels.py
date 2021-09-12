# -*- coding: utf-8 -*-
import os
import json
import dateutil
import shapely
import shapely.ops
import pygeos
import kwcoco
import kwimage
import ubelt as ub
import pathlib
import scriptconfig as scfg
from watch.utils import util_raster
from watch.gis import geotiff

import xdev


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
        'src':
        scfg.Value(position=1,
                   help=ub.paragraph('''
            path to the kwcoco file to propagate labels in
            ''')),
        'dst':
        scfg.Value('propagated_data.kwcoco.json',
                   help=ub.paragraph('''
            Where the output kwcoco file with propagated labels is saved
            '''),
                   position=2),
        'ext':
        scfg.Value(None,
                   help=ub.paragraph('''
            Path to an optional external kwcoco file to merge annotations from.
            Must have a tag different from src's.
            ''')),
        'viz_dpath':
        scfg.Value(None,
                   help=ub.paragraph('''
            if specified, visualizations will be written to this directory
            ''')),
        'verbose':
        scfg.Value(1, help="use this to print details"),
        'validate':
        scfg.Value(
            1,
            help="Validate spatial and temporal AOI of each site after propagating"
        ),
        'crop':
        scfg.Value(
            1,
            help="Crop propagated annotations to the valid data mask of the new image"
        ),
        'max_workers':
        scfg.Value(
            None,
            help="Max. number of workers to parallelize over, up to the number of regions/ROIs. None is auto; 0 is serial."
        )
    }

    epilog = """
    Example Usage:
        watch-cli scriptconfig_cli_template --arg1=foobar
    """


def annotated_band(img):
    # we hope each band has the same valid mask
    # but if not, this field picks out the (probable; heuristic-based)
    # band that the annotation was actually done on
    if img['file_name'] is not None:
        return img
    aux_ix = img.get('aux_annotated_candidate', 0)
    return img['auxiliary'][aux_ix]


def warped_wgs84_to_img(poly, img, inv=False):
    '''
    Return poly warped from WGS84 geocoordinates to img's base space

    img['warp_wld_to_pxl'] does not give the appropriate transform for
    ann['segmentation_geo'] because it is in WGS84, not the wld CRS
    '''
    band = annotated_band(img)
    fpath = band['file_name']
    # cacher = ub.Cacher('geotiff_crs_info', depends=fpath)
    # info = cacher.tryload()
    # if info is None:
    if 1:
        try:
            info = geotiff.geotiff_crs_info(fpath)
        except NotImplementedError as e:
            xdev.embed()
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
    '''
    Returns new annotation by applying a warp and optional crop

    If the new annotation falls outside the new image, returns None instead.

    This should be vectorized across annotations, because util_raster.to_crop
    supports this, since it assumes finding the valid mask of an image is
    expensive. But this is not a problem for the very small sizes we're
    working with here:

    >>> %timeit watch.utils.util_raster.mask(full_tile)
    1.19 s ± 22.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    >>> %timeit watch.utils.util_raster.mask(aligned_crop)
    4.44 ms ± 87.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    '''

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
            xdev.embed()
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


def build_external_video(vid_id, full_ds, ext_ds):
    '''
    Given two datasets with a corresponding video, match images between them
    assuming the gids and names are unreliable, and return an ordered list
    of all images from both videos, with full_ds taking precedence.

    Args:
        vid_id: of a corresponding video in both dsets

        full_ds: main dset to preferentially grab images from

        ext_ds: empty, or external dataset where images may not be well formed or exist,
            but images are a superset of full_ds's, which do exist

    Returns:
        List[Tuple[Dict, bool]]: image entries with a boolean flag "is from ext_ds"
    '''

    if ext_ds.n_images == 0:
        return [(i, False) for i in full_ds.images(vidid=vid_id).objs]

    # are these actually the same video?
    assert (full_ds.index.videos[vid_id]['name'] == ext_ds.index.videos[vid_id]
            ['name'])

    full_imgs = full_ds.images(vidid=vid_id).objs
    full_names = full_ds.images(vidid=vid_id).lookup('parent_name')

    ext_imgs = ext_ds.images(vidid=vid_id).objs
    ext_names = ext_ds.images(vidid=vid_id).lookup('canonical_name')

    matched_imgs = []
    for ext in ext_names:
        try:
            matched_imgs.append(full_imgs[full_names.index(ext)])
        except ValueError:
            matched_imgs.append(None)

    # does ext_ds's video completely contain full_ds's [annotated] video?
    full_imgs_with_annots = [
        i for i in full_imgs if len(full_ds.gid_to_aids[i['id']]) > 0
    ]
    assert len([m for m in matched_imgs
                if m is not None]) == len(full_imgs_with_annots), ext_ds

    return [(m, False) if m is not None else (e, True)
            for m, e in zip(matched_imgs, ext_imgs)]


def get_canvas_concat_channels(annotations, dataset, img_id):
    canvas = dataset.delayed_load(img_id, channels='red|green|blue').finalize()
    canvas = kwimage.normalize_intensity(canvas)
    canvas = kwimage.ensure_float01(canvas)

    # draw on annotations
    # hack because draw_on(color=list) is not supported
    this_dset_anns = []
    ext_dset_anns = []
    for ann in annotations:
        if ann['orig_info']['source_dset'] == dataset.tag:
            this_dset_anns.append(ann)
        else:
            ext_dset_anns.append(ann)
    this_dets = kwimage.Detections.from_coco_annots(this_dset_anns,
                                                    dset=dataset)
    ext_dets = kwimage.Detections.from_coco_annots(ext_dset_anns, dset=dataset)
    canvas = this_dets.draw_on(canvas, color='blue')
    canvas = ext_dets.draw_on(canvas, color='green')

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

    fig.tight_layout()
    fig.savefig(fpath, bbox_inches='tight')
    plt.close(fig)


def validate_inbounds(anns, img):

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

    # preprocessing step: add new 'orig_info' to every *annotation*.
    # original annotations: source_gid == gid and source_name == name
    # when we propagate labels, we will copy 'orig_info' from the copied
    # annotation. This includes the full segmentation (in case of cropping) and the
    # original image in the src or ext dataset.

    # why doesn't this get saved on dump()??
    if full_ds.tag == '':
        full_ds.tag  = 'data.kwcoco.json'
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

    # number of visualizations of every sequence
    n_image_viz = 7

    # a list of newly generated annotation IDs, for debugging purposes
    new_aids = []

    # parallelize over videos
    def _job(vid_id, full_ds, ext_ds):
        # a set of all the track IDs in this video
        track_ids = set(
            full_ds.subset(full_ds.index.vidid_to_gids[vid_id]).index.
            trackid_to_aids.keys())

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

        def _is_ext(ann):
            return ann
        # for all sorted images in this video in both dsets
        j = 0
        for img, is_ext in build_external_video(vid_id, full_ds, ext_ds):

            # frame counter for viz
            if not is_ext:
                j += 1

            img_id = img['id']
            if 1:
                # prog.ensure_newline()
                print('{} img_id = {!r}'.format(('ext' if is_ext else 'int'),
                                                img_id))

            this_track_ids = set()

            # Update all current tracks to have their latest annotation state
            this_image_anns = (ext_ds if is_ext else full_ds).annots(
                gid=img_id).objs
            for ann in this_image_anns:
                if 0:
                    print('ann = {}'.format(
                        ub.repr2(ub.dict_diff(
                            ann, {
                                'segmentation', 'segmentation_geos',
                                'properties', 'orig_info'
                            }),
                                 nl=0)))
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

            this_image_fixed_anns = this_image_anns.copy(
            )  # if there is anything missing, we are going to fix now

            # was there any seen track ID that was not in this image?
            for missing in track_ids - this_track_ids:
                for aid in latest_ann_ids.get(missing, {}):
                    previous_annotation = (ext_ds if latest_is_ext[missing] else
                                           full_ds).anns[aid]
                    # this should be an original annotation
                    assert (previous_annotation['image_id'] == previous_annotation['orig_info']['source_gid'] and
                            (previous_annotation['orig_info']['source_dset'] != full_ds.tag) == latest_is_ext[missing])

                    # check if the annotation belongs to the list of cats we want to propagate
                    if previous_annotation[
                            'category_id'] in cat_ids_to_propagate:

                        # get the warp from previous image to this image
                        previous_image_id = latest_img_ids[missing]
                        warped_annotation = get_warped_ann(
                            previous_annotation,
                            ('geo'
                             if previous_annotation['orig_info']['source_dset']
                             != full_ds.tag else get_warp(
                                 previous_image_id, img_id, full_ds)),
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

            # Get "n_image_viz" number of canvases for visualization with original annotations
            num_frames = len(full_ds.index.vidid_to_gids[vid_id])

            if viz_dpath is not None:
                is_starting_frame = (j < n_image_viz)
                is_ending_frame = (j >= (num_frames - n_image_viz))
                if is_ending_frame or is_starting_frame:
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
                        'frame_num': j,
                    })

        return warped_annotations, canvas_infos

    # output dataset to write to
    propagated_ds = full_ds.copy()
    propagated_ds.fpath = config['dst']

    # TODO multithreading breaks this, don't know why
    PROJ_LIB = os.environ['PROJ_LIB']

    # run job for each video
    max_workers = config['max_workers']
    if max_workers is None:
        max_workers = len(full_ds.index.videos)
    executor = ub.Executor(mode='thread', max_workers=max_workers)
    prog = ub.ProgIter(
        [(vid_id, executor.submit(_job, vid_id, full_ds, ext_ds))
         for vid_id in full_ds.index.videos],
        total=len(full_ds.index.videos),
        desc='process video')
    for vid_id, job in prog:
        warped_annotations, canvas_infos = job.result()
        new_aids.extend(
            [propagated_ds.add_annotation(**a) for a in warped_annotations])

        # save visualization
        if viz_dpath is not None:
            vid_name = full_ds.index.videos[vid_id]['name']
            for canvas_chunk in ub.chunks(canvas_infos, n_image_viz):
                min_frame = min([d['frame_num'] for d in canvas_chunk])
                max_frame = max([d['frame_num'] for d in canvas_chunk])
                fname = f'video_{vid_id:04d}_{vid_name}_frames_{min_frame}_to_{max_frame}.jpg'
                fpath = viz_dpath / fname
                save_visualizations(canvas_chunk, fpath)

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


_SubConfig = PropagateLabelsConfig

if __name__ == '__main__':
    r"""
    CommandLine:

        # Command line used to test this script in development

        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        BUNDLE_DPATH=$DVC_DPATH/drop1-S2-L8-aligned

        INPUT_KWCOCO=$BUNDLE_DPATH/data.kwcoco.json
        OUTPUT_KWCOCO=$BUNDLE_DPATH/propagated.kwcoco.json

        python -m watch.cli.coco_visualize_videos \
            --src $INPUT_KWCOCO --viz_dpath $BUNDLE_DPATH/viz_before \
            --num_workers=6

        python -m watch.cli.propagate_labels \
                --src $INPUT_KWCOCO --dst $OUTPUT_KWCOCO \
                --viz_dpath=$BUNDLE_DPATH/_propagate_viz

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
