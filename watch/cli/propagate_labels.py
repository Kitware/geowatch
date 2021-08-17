"""
This script reads the labels from the kwcoco file and performs the forward propagation of labels.
The final goal of this script is to create a modified kwcoco file.
The problem with original labels is that in many cases, annotators labeled a site in the first few images with a label (say, Active Construction) and then this annotation was missing for the next few frames.


Notes:

    # Given a kwcoco file with original annotations, this script forward propagates those annotations
    # and creates a new kwcoco file.
    # Currently, we are looking at some issues with data and only visualizations are being geenrated.

    python -m watch.cli.propagate_labels.py dataset_fname


    TODO:
        - [ ] Make sure the original annotations are correct. Currently annotations with very low overlap with images are showing up.
        - [ ] Write the all labels, original and propagated annotations, in another kwcoco file.
         
"""
import sys
import argparse
import kwcoco, kwimage
import matplotlib.pyplot as plt
import kwplot
import os
from os.path import join
import ubelt as ub
import numpy as np


def get_canvas_concat_channels(annotations, dataset, img_id):
    canvas = dataset.delayed_load(img_id, channels='red|green|blue').finalize()
    canvas = kwimage.normalize_intensity(canvas)
    canvas = kwimage.ensure_float01(canvas)

    dets = kwimage.Detections.from_coco_annots(annotations, dset=dataset)
    ann_canvas = dets.draw_on(canvas)
    
    return ann_canvas

def save_visualizations(canvases, canvases_fixed, fname):
    # save visualizations of original and propagated labels
    
    plt.figure(figsize=(30,8))
    n_images = len(canvases)
    for i, c in enumerate(canvases):
        plt.subplot(2, n_images, i+1)
        plt.imshow(c)
        if i==3:
            plt.title('Original')
        plt.axis('off')

        plt.subplot(2, n_images, n_images + i+1)
        plt.imshow(canvases_fixed[i])
        if i==3:
            plt.title('Propagated')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    

def main(args):
    """
    Main function for propagate_labels.

    Example:
        >>> from watch.cli.propagate_labels import *  # NOQA
        >>> dataset_directory = 'drop1-S2-aligned-c1'
        >>> python -m watch.demo.propagate_labels --data_dir=dataset_directory --out_dir='propagation_output'

    """
    # read arguments
    parser = argparse.ArgumentParser(
        description="Forward propagate labels")
    parser.add_argument("--data_dir", default= 'drop1-S2-aligned-c1', 
    help="drop1 directory name, defualt is drop1-S2-aligned-c1. The kwcoco file from this directory will be read")
    parser.add_argument("--out_dir", default='propagation_output', help="Output directory where visualizations and processed kwcoco files will be saved")
    parser.add_argument("--viz_end", default=False, action="store_true", help="if True, last few frames will be saved")
    parser.add_argument("--verbose", default=False, action="store_true", help="use this to print details")
    args = parser.parse_args(args)
    
    # create the output dir
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Read input file
    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    coco_fpath = join(dvc_dpath, args.data_dir, 'data.kwcoco.json')
    full_ds = kwcoco.CocoDataset(coco_fpath)
    print('total video:', full_ds.n_videos)
    print('total images:', full_ds.n_images)
    print('total annotations:', full_ds.n_annots)
    
    # which categories we want to propagate
    catnames_to_propagate = [
        'Site Preparation',
        'Active Construction',]
    categories_to_propagate = [full_ds.index.name_to_cat[c]['id'] for c in catnames_to_propagate]    
    
    # number of visualizations of every sequence 
    n_image_viz = 7
    
    # we save the ending frames of every video sequence if this is set to True 
    # viz_end = True
    # deprecating from code, this is now passed as an argument
    
    
    for vid_id, video in full_ds.index.videos.items():
        image_ids = full_ds.index.vidid_to_gids[vid_id]

        # a set of all the seen track IDs
        seen_track_ids = set()
        # a dictionary of latest annotation IDs, indexed by the track IDs
        latest_ann_ids = {}   

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
                        this_image_fixed_anns.append( full_ds.anns[latest_ann_ids[missing]] )
                        if args.verbose:
                            print('added annotation for image', img_id, 'track ID', missing)

            # Get "n_image_viz" number of canvases for visualization with original annotations
            store_starting_frame = args.viz_end and ((video['num_frames'] - j) <=  n_image_viz)
            store_ending_frame = (not args.viz_end) and (j < n_image_viz)
            if store_starting_frame or store_ending_frame:
                 canvases.append(get_canvas_concat_channels(annotations=this_image_anns, dataset=full_ds, img_id=img_id))
                 canvases_fixed.append(get_canvas_concat_channels(annotations=this_image_fixed_anns, dataset=full_ds, img_id=img_id))

        # save visualization
        location_string = '_end' if args.viz_end else '_start'
        fname = join(args.out_dir, 'video_'+str(vid_id)+location_string+'.jpg')
        save_visualizations(canvases, canvases_fixed, fname)
    
    # ToDO: write the code to add annotation objects
    # ToDo: save the new kwcoco data to disk
    
    return 0

if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.propagate_labels
    """
    sys.exit(main(sys.argv[1:]))
