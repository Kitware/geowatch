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

# import xdev


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
        - [ ] Make sure the original annotations are correct. Currently annotations with very low overlap with images are showing up. Could be fixed by relying on geo-coordinates instead?
        - [x] Crop propagated annotations to bounds or valid data mask of new image
        - [ ] Pull in and merge annotations from an external kwcoco file
        - [x] Handle splitting and merging tracks (non-unique frame_index)
        - [ ] Stop propagation at category change (could be taken care of by external kwcoco file)
        - [ ] Parallelize

    """
    default = {
        'src': scfg.Value(position=1, help=ub.paragraph(
            '''
            path to the kwcoco file to propagate labels in
            ''')),

        'dst': scfg.Value('propagated_data.kwcoco.json', help=ub.paragraph(
            '''
            Where the output kwcoco file with propagated labels is saved
            '''), position=2),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            if specified, visualizations will be written to this directory
            ''')),

        'verbose': scfg.Value(1, help="use this to print details"),

        'validate': scfg.Value(1, help="Validate spatial and temporal AOI of each site after propagating")
    }

    epilog = """
    Example Usage:
        watch-cli scriptconfig_cli_template --arg1=foobar
    """


def get_warp(gid1, gid2, dataset, use_geo=False):
    # Given a dataset and IDs of two images, the warp between these images is returned
    img1 = dataset.index.imgs[gid1]
    img2 = dataset.index.imgs[gid2]
    
    if use_geo:
        # Use the geocoordinates as a base space to avoid propagating
        # badly-aligned annotations
        geo_to_img1 = kwimage.Affine.coerce(img1['wld_to_pxl'])
        geo_to_img2 = kwimage.Affine.coerce(img2['wld_to_pxl'])

        img2_from_img1 = geo_to_img2 @ geo_to_img1.inv()
    
    else:
        # Get the transform from each image to the aligned "video-space"
        video_from_img1 = kwimage.Affine.coerce(img1['warp_img_to_vid'])
        video_from_img2 = kwimage.Affine.coerce(img2['warp_img_to_vid'])

        # Build the transform that warps img1 -> video -> img2
        img2_from_video = video_from_img2.inv()
        img2_from_img1 = img2_from_video @ video_from_img1

    return img2_from_img1

def get_warped_ann(previous_ann,
                   warp,
                   image_entry,
                   previous_image_id,
                   crop_to_valid=True):
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
    segmentation = kwimage.Segmentation.coerce(previous_ann['segmentation'])

    warped_seg = segmentation.warp(warp)

    if crop_to_valid:
        # warp to an aux image's space to crop to its valid mask,
        # then warp back to the base space

        # assume each band has the same valid mask
        # the 'aux_annotated_candidate' attr should be moved from ann to img,
        # then it can be used here
        # but note that it is a heuristic and not guaranteed correct
        band = image_entry['auxiliary'][0]
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
    warped_ann['source_gid'] = previous_image_id
    warped_ann['properties'] = previous_ann['properties']
    warped_ann['segmentation_geos'] = previous_ann['segmentation_geos']

    return warped_ann


def get_canvas_concat_channels(annotations, dataset, img_id):
    canvas = dataset.delayed_load(img_id, channels='red|green|blue').finalize()
    canvas = kwimage.normalize_intensity(canvas)
    canvas = kwimage.ensure_float01(canvas)

    dets = kwimage.Detections.from_coco_annots(annotations, dset=dataset)
    ann_canvas = dets.draw_on(canvas)

    return ann_canvas


def save_visualizations(canvas_chunk, fpath):
    # save visualizations of original and propagated labels
    import kwplot
    plt = kwplot.autoplt()

    plt.figure(figsize=(30, 8))
    n_images = len(canvas_chunk)
    for i, info in enumerate(canvas_chunk):
        before = info['before_canvas']
        after = info['after_canvas']
        plt.subplot(2, n_images, i + 1)
        plt.imshow(before)
        if i == 3:
            plt.title('Original')
        plt.axis('off')

        plt.subplot(2, n_images, n_images + i + 1)
        plt.imshow(after)
        if i == 3:
            plt.title('Propagated')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()


def validate_inbounds(anns, img):
    warp_wld_to_pxl = kwimage.Affine.coerce(img['wld_to_pxl'])

    band = img['auxiliary'][0]
    warp_aux_to_img = kwimage.Affine.coerce(band['warp_aux_to_img'])

    # use pygeos to vectorize this bit
    box = pygeos.box(0, 0, *kwimage.load_image_shape(band['file_name'])[:2])
    mask = pygeos.from_shapely(
        util_raster.mask(band['file_name'], as_poly=True))
    segs = pygeos.from_shapely([
        kwimage.Segmentation.coerce(ann['segmentation']).warp(
            warp_aux_to_img.inv()).to_multi_polygon().to_shapely()
        for ann in anns
    ])
    geos = pygeos.from_shapely([
        kwimage.Polygon.from_geojson(
            ann['segmentation_geos']).warp(warp_wld_to_pxl).warp(
                warp_aux_to_img.inv()).to_multi_polygon().to_shapely()
        for ann in anns
    ])

    assert pygeos.contains(box, mask)
    assert all(pygeos.contains(geos, segs))
    assert all(pygeos.contains(mask, segs))
    assert all(pygeos.intersects(mask, geos))


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
        >>> # xdoctest: +SKIP
        >>> from watch.cli.propagate_labels import *  # NOQA
        >>> dataset_directory = 'drop1-S2-aligned-c1'

        python -m watch.demo.propagate_labels --data_dir=dataset_directory --out_dir='propagation_output'
    """
    config = PropagateLabelsConfig(default=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    # Read input file
    full_ds = kwcoco.CocoDataset.coerce(config['src'])
    print(ub.repr2(full_ds.basic_stats(), nl=1))

    # make a copy of the original kwcoco dataset
    dst_fpath = pathlib.Path(config['dst'])
    # Ensure the output directory exists
    dst_fpath.parent.mkdir(exist_ok=True, parents=True)
    propagated_ds = full_ds.copy()
    propagated_ds.fpath = config['dst']

    viz_dpath = config['viz_dpath']
    if viz_dpath not in {None, False}:
        viz_dpath = pathlib.Path(viz_dpath)
        viz_dpath.mkdir(exist_ok=True, parents=True)
    else:
        viz_dpath = None

    # preprocessing step: in the new dataset, add a new field 'source_gid' to every *annotation*.
    # original annotations: source_gid == gid
    # when we propagate labels, we will add the image id of the original image from which we copied
    # the annotation. In the case of propagated labels, we will have source_gid != gid

    for aid in propagated_ds.index.anns:
        # load annotation
        ann = propagated_ds.index.anns[aid]
        image_id = ann['image_id']
        ann['source_gid'] = image_id

    # which categories we want to propagate
    # TODO: parameterize
    catnames_to_propagate = [
        'Site Preparation',
        'Active Construction',
        'Unknown',
        # 'No Activity',
        # 'Post Construction',
    ]
    cat_ids_to_propagate = [full_ds.index.name_to_cat[c]['id']
                            for c in catnames_to_propagate]

    # number of visualizations of every sequence
    n_image_viz = 7

    # a list of newly generated annotation IDs, for debugging purposes
    new_aids = []

    prog = ub.ProgIter(
        full_ds.index.videos.items(),
        total=len(full_ds.index.videos), desc='process video')
    for vid_id, video in prog:
        image_ids = full_ds.index.vidid_to_gids[vid_id]

        # a set of all the track IDs in this video
        track_ids = set(full_ds.subset(full_ds.index.vidid_to_gids[vid_id]).index.trackid_to_aids.keys())

        # a dictionary of latest annotation IDs, indexed by the track IDs
        latest_ann_ids = {}

        # a dictionary of latest image IDs, this will be needed to apply affine transform on images
        latest_img_ids = {}

        # canvases for visualizations
        canvas_infos = []

        # for all images in this video
        for j, img_id in enumerate(image_ids):
            if 1:
                prog.ensure_newline()
                print('img_id = {!r}'.format(img_id))

            this_track_ids = set()
            this_image_anns = []

            # Update all current tracks to have their latest annotation state
            for aid in list(full_ds.gid_to_aids[img_id]):
                ann = full_ds.anns[aid]
                if 1:
                    print('ann = {}'.format(ub.repr2(ub.dict_diff(ann, {'segmentation', 'segmentation_geos', 'properties'}), nl=0)))
                this_image_anns.append(ann)
                track_id = ann['track_id']
                this_track_ids.add(track_id)
                
                # handle non-unique track ids
                if latest_img_ids.get(track_id) != img_id:
                    latest_ann_ids[track_id] = set()
                latest_img_ids[track_id] = img_id
                latest_ann_ids[track_id].add(aid)

            this_image_fixed_anns = this_image_anns.copy()  # if there is anything missing, we are going to fix now

            # was there any seen track ID that was not in this image?
            for missing in track_ids - this_track_ids:
                for aid in latest_ann_ids.get(missing, {}):
                    previous_annotation = full_ds.anns[aid]
                    # check if the annotation belongs to the list of cats we want to propagate
                    if previous_annotation['category_id'] in cat_ids_to_propagate:

                        # get the warp from previous image to this image
                        # 
                        # NOTE: use_geo must be used if crop_to_valid is used.
                        # this is because segmentation_geo will remain correct,
                        # but segmentation will NOT, when this propagated annot
                        # is used as a source to re-propagate further.
                        previous_image_id = latest_img_ids[missing]
                        warp_previous_to_this_image = get_warp(previous_image_id, img_id, full_ds, use_geo=True)

                        # apply the warp
                        warped_annotation = get_warped_ann(previous_annotation, warp_previous_to_this_image, full_ds.imgs[img_id], previous_image_id, crop_to_valid=True)
                        if warped_annotation is None:
                            if config['verbose']:
                                prog.ensure_newline()
                                print('skipped OOB annotation for image', img_id, 'track ID', missing)
                            continue

                        # add the propagated annotation in the new kwcoco dataset
                        new_aid = propagated_ds.add_annotation(**warped_annotation)
                        new_aids.append(new_aid)

                        # append in the list for visualizations
                        this_image_fixed_anns.append(warped_annotation)

                        if config['verbose']:
                            prog.ensure_newline()
                            print('added annotation for image', img_id, 'track ID', missing)

            # Get "n_image_viz" number of canvases for visualization with original annotations
            num_frames = len(full_ds.index.vidid_to_gids[vid_id])

            if viz_dpath is not None:
                is_starting_frame = (j < n_image_viz)
                is_ending_frame = (j >= (num_frames - n_image_viz))
                if is_ending_frame or is_starting_frame:
                    before_canvas = get_canvas_concat_channels(
                        annotations=this_image_anns, dataset=full_ds,
                        img_id=img_id)
                    after_canvas = get_canvas_concat_channels(
                        annotations=this_image_fixed_anns, dataset=full_ds,
                        img_id=img_id)
                    canvas_infos.append({
                        'before_canvas': before_canvas,
                        'after_canvas': after_canvas,
                        'frame_num': j,
                    })

        # save visualization
        if viz_dpath is not None:
            vid_name = video['name']
            for canvas_chunk in ub.chunks(canvas_infos, n_image_viz):
                min_frame = min([d['frame_num'] for d in canvas_chunk])
                max_frame = max([d['frame_num'] for d in canvas_chunk])
                fname = f'video_{vid_id:04d}_{vid_name}_frames_{min_frame}_to_{max_frame}.jpg'
                fpath = viz_dpath / fname
                save_visualizations(canvas_chunk, fpath)

    # save the propagated dataset
    print('Save: propagated_ds.fpath = {!r}'.format(propagated_ds.fpath))
    propagated_ds.dump(propagated_ds.fpath, newlines=True)

    if viz_dpath:
        os.makedirs(viz_dpath, exist_ok=True)
        print('saved visualizations to: {!r}'.format(viz_dpath))

    # print statistics about propagation
    print('original annotations:', full_ds.n_annots, 'propagated annotations:', len(new_aids), 'total annotations:', propagated_ds.n_annots)
    
    # update index (add new anns to tracks) by reloading from disk
    propagated_ds = kwcoco.CocoDataset(propagated_ds.fpath)
    # validate correctness
    if config['validate']:
        for gid, img in propagated_ds.imgs.items():
            try:
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
