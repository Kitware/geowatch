import kwcoco
import kwimage
import os
from os.path import join
import ubelt as ub
import scriptconfig as scfg


class PropagateLabelsConfig(scfg.Config):
    """
    This script reads the labels from the kwcoco file and performs the forward
    propagation of labels.
    The final goal of this script is to create a modified kwcoco file.
    The problem with original labels is that in many cases, annotators labeled
    a site in the first few images with a label (say, Active Construction) and
    then this annotation was missing for the next few frames.


    Notes:

        # Given a kwcoco file with original annotations, this script forward propagates those annotations
        # and creates a new kwcoco file.
        # Currently, we are looking at some issues with data and only visualizations are being geenrated.

        python -m watch.cli.propagate_labels.py dataset_fname


        TODO:
            - [ ] Make sure the original annotations are correct. Currently annotations with very low overlap with images are showing up.
            - [ ] Write the all labels, original and propagated annotations, in another kwcoco file.

    """
    default = {
        'data_dir': scfg.Value('drop1-S2-aligned', help='drop1 aligned directory name', position=1),
        'out_dir': scfg.Value('propagation_output', help="Output directory where visualizations and processed kwcoco files will be saved", position=2),
        'viz_end': scfg.Value(False, help="if True, last few frames will be saved"),
        'verbose': scfg.Value(False, help="use this to print details")
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


def save_visualizations(canvases, canvases_fixed, fname):
    # save visualizations of original and propagated labels
    import kwplot
    plt = kwplot.autoplt()

    plt.figure(figsize=(30, 8))
    n_images = len(canvases)
    for i, c in enumerate(canvases):
        plt.subplot(2, n_images, i + 1)
        plt.imshow(c)
        if i == 3:
            plt.title('Original')
        plt.axis('off')

        plt.subplot(2, n_images, n_images + i + 1)
        plt.imshow(canvases_fixed[i])
        if i == 3:
            plt.title('Propagated')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
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

    args = PropagateLabelsConfig(default=kwargs, cmdline=cmdline)

    # create the output dir
    os.makedirs(args['out_dir'], exist_ok=True)

    # Read input file
    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    coco_fpath = join(dvc_dpath, args['data_dir'], 'data.kwcoco.json')
    full_ds = kwcoco.CocoDataset(coco_fpath)
    print('total video:', full_ds.n_videos)
    print('total images:', full_ds.n_images)
    print('total annotations:', full_ds.n_annots)

    # make a copy of the original kwcoco dataset
    propagated_ds = full_ds.copy()

    # preprocessing step: in the new dataset, add a new filed 'source_gid' to every *annotation*.
    # original annotations: source_gid == gid
    # when we propagate labels, we will add the image id of the original image from which we copied
    # the annotation. In the case of propagated labels, we will have source_gid != gid

    for aid in propagated_ds.index.anns:
        # load annotation
        ann = propagated_ds.index.anns[aid]
        image_id = ann['image_id']
        ann['source_gid'] = image_id

    # which categories we want to propagate
    catnames_to_propagate = [
        'Site Preparation',
        'Active Construction', ]
    categories_to_propagate = [full_ds.index.name_to_cat[c]['id'] for c in catnames_to_propagate]

    # number of visualizations of every sequence
    n_image_viz = 7

    # a list of newly geenrated annotation IDs, for debugging purposes
    new_aids = []

    for vid_id, video in full_ds.index.videos.items():
        image_ids = full_ds.index.vidid_to_gids[vid_id]

        # a set of all the seen track IDs
        seen_track_ids = set()

        # a dictionary of latest annotation IDs, indexed by the track IDs
        latest_ann_ids = {}

        # a dictionary of latest image IDs, this will be needed to apply affine transform on images
        latest_img_ids = {}

        # canvases for visualizations
        canvases = []
        canvases_fixed = []

        # for all images in this video
        for j, img_id in enumerate(image_ids):
            this_track_ids = set()
            this_image_anns = []

            this_image_fixed_anns = []

            aids = full_ds.gid_to_aids[img_id]

            for aid in list(aids):
                anns = full_ds.anns[aid]
                this_image_anns.append(anns)
                track_id = anns['track_id']
                if track_id not in this_track_ids:
                    this_track_ids.update({track_id})

                latest_ann_ids[track_id] = aid
                latest_img_ids[track_id] = img_id

            # add any track IDs to the list of seen track ids
            new_track_ids = this_track_ids - seen_track_ids
            if new_track_ids:
                seen_track_ids.update(new_track_ids)

            this_image_fixed_anns = this_image_anns.copy()  # if there is anything missing, we are going to fix now

            # was there any seen track ID that was not in this image?
            missing_track_ids = seen_track_ids - this_track_ids
            if missing_track_ids:

                for missing in missing_track_ids:
                    if full_ds.anns[latest_ann_ids[missing]]['category_id'] in categories_to_propagate :
                        # check if the annotation belongs to the list of categories that we want to propagate
                        previous_annotation = full_ds.anns[latest_ann_ids[missing]]

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

                        if args['verbose']:
                            print('added annotation for image', img_id, 'track ID', missing)

            # Get "n_image_viz" number of canvases for visualization with original annotations
            num_frames = len(full_ds.index.vidid_to_gids[vid_id])
            store_ending_frame = args['viz_end'] and ((num_frames - j) <=  n_image_viz)
            store_starting_frame = (not args['viz_end']) and (j < n_image_viz)
            if store_starting_frame or store_ending_frame:
                canvases.append(get_canvas_concat_channels(annotations=this_image_anns, dataset=full_ds, img_id=img_id))
                canvases_fixed.append(get_canvas_concat_channels(annotations=this_image_fixed_anns, dataset=full_ds, img_id=img_id))

        # save visualization
        location_string = '_end' if args['viz_end'] else '_start'
        fname = join(args['out_dir'], 'video_' + str(vid_id) + location_string + '.jpg')
        save_visualizations(canvases, canvases_fixed, fname)

    # save the propagated dataset
    propagated_fname = join(args['out_dir'], 'propagted_data.kwcoco.json')
    propagated_ds.dump(propagated_fname)

    # print statistics about propagation
    print('original annotations:', full_ds.n_annots, 'propagated annotations:', len(new_aids), 'total annotations:', propagated_ds.n_annots)

    # Running a debugging check. This can be removed in future
    verify_ds = kwcoco.CocoDataset(propagated_fname)
    original_aids_n = 0
    propagated_aids_n = 0
    for aid in verify_ds.index.anns:
        # load annotation
        ann = verify_ds.index.anns[aid]
        image_id = ann['image_id']
        if image_id == ann['source_gid']:
            original_aids_n += 1
        else:
            propagated_aids_n += 1

    print('verification of annotation types -- no. of original anns:', original_aids_n, ', no. of propagated anns:', propagated_aids_n)

    # ToDo
    # [x] write the code to add annotation objects
    # [x] save the new kwcoco data to disk
    # [ ] use site dates (or any other mechanism) to make sure propagation is limited only to correct frames

    return 0

if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.propagate_labels
    """
    main(cmdline=True)