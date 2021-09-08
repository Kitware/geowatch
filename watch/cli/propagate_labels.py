# -*- coding: utf-8 -*-
import kwcoco
import kwimage
import ubelt as ub
import pathlib
import scriptconfig as scfg
from watch.utils import util_raster


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
        # Currently, we are looking at some issues with data and only visualizations are being geenrated.

        python -m watch.cli.propagate_labels.py dataset_fname

    TODO:
        - [ ] Make sure the original annotations are correct. Currently annotations with very low overlap with images are showing up. Could be fixed by relying on geo-coordinates instead?
        - [ ] Crop propagated annotations to bounds or valid data mask of new image
        - [ ] Pull in and merge annotations from an external kwcoco file
        - [ ] Handle splitting and merging tracks (non-unique frame_index)
        - [ ] Stop propagation at category change (could be taken care of by external kwcoco file)

    """
    default = {
        'src': scfg.Value(position=1, help=ub.paragraph(
            '''
            path to the kwcoco file to propagate labels in
            ''')),

        'dst': scfg.Value('propagted_data.kwcoco.json', help=ub.paragraph(
            '''
            Where the output coco file with propagated labels is saved
            '''), position=2),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            if specified, visualizations will be written to this directory
            ''')),

        'verbose': scfg.Value(1, help="use this to print details")
    }

    epilog = """
    Example Usage:
        watch-cli scriptconfig_cli_template --arg1=foobar
    """


def get_warp(gid1, gid2, dataset):
    # Given a dataset and IDs of two images, the warp between these images is returned
    img1 = dataset.index.imgs[gid1]
    img2 = dataset.index.imgs[gid2]

    # Get the transform from each image to the aligned "video-space"
    video_from_img1 = kwimage.Affine.coerce(img1['warp_img_to_vid'])
    video_from_img2 = kwimage.Affine.coerce(img2['warp_img_to_vid'])

    # Build the transform that warps img1 -> video -> img2
    img2_from_video = video_from_img2.inv()
    img2_from_img1 = img2_from_video @ video_from_img1

    return img2_from_img1


def get_warped_ann(previous_ann, warp, image_id, previous_image_id):
    # Returns a new annotation by applying a warp on an existing annotation
    segmentation = kwimage.Segmentation.coerce(previous_ann['segmentation'])

    warped_seg = segmentation.warp(warp)
    warped_bbox = list(warped_seg.bounding_box().to_coco(style='new'))[0]

    # Create a new annotation object
    warped_ann = {'segmentation': warped_seg.to_coco(style='new')}
    warped_ann['bbox'] = warped_bbox
    warped_ann['category_id'] = previous_ann['category_id']
    warped_ann['track_id'] = previous_ann['track_id']
    warped_ann['image_id'] = image_id
    warped_ann['source_gid'] = previous_image_id

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
                    latest_ann_ids[track_id] = {}
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
                        previous_image_id = latest_img_ids[missing]
                        warp_previous_to_this_image = get_warp(previous_image_id, img_id, full_ds)

                        # apply the warp
                        warped_annotation = get_warped_ann(previous_annotation, warp_previous_to_this_image, img_id, previous_image_id)

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
        print('saved visualizations to: {!r}'.format(viz_dpath))

    # print statistics about propagation
    print('original annotations:', full_ds.n_annots, 'propagated annotations:', len(new_aids), 'total annotations:', propagated_ds.n_annots)

    return propagated_ds

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
