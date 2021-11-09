import kwcoco
import kwimage
import matplotlib.pyplot as plt
import numpy as np
import os
from watch.utils import kwcoco_extensions

def get_rgb(dset, gid):
    img = dset.index.imgs[gid]
    coco_img = kwcoco_extensions.CocoImage(img, dset)
    r = coco_img.delay('red', space='video').finalize()
    g = coco_img.delay('green', space='video').finalize()
    b = coco_img.delay('blue', space='video').finalize()
    rgb = np.concatenate((r, g, b), axis=2)
    rgb = kwimage.normalize_intensity(rgb)
    return rgb


def get_pred_seg(dset, gid, render_track_id=False):
    img = dset.index.imgs[gid]
    shape = (img['height'], img['width'])
    pred_canvas = np.zeros(shape, dtype=np.uint8)

    pred_anns = dset.annots(gid=gid)
    pred_dets = dset.annots(gid=gid).detections

    aids = pred_anns.aids

    # filter annotations and only keep ones with 'track_id'
    track_ids = []
    valid_aids = []
    valid_segmentations = []
    for i, aid in enumerate(aids):
        if 'track_id' in dset.anns[aid]:
            # print('track id', dset.anns[aid]['track_id'])
            track_ids.append(dset.anns[aid]['track_id'])
            valid_aids.append(aid)
            valid_segmentations.append(pred_dets.data['segmentations'][i])
        else:
            print('skipping this ann')

    for i, pred_sseg in enumerate(valid_segmentations):
        track_now = track_ids[i]
        # print('track id', track_now)
        
        render_value = track_now if render_track_id else 1
        pred_canvas = pred_sseg.fill(pred_canvas, value=render_value)
    
    return pred_canvas


def get_gt_seg(dset, gid, render_track_id=False):
    img = dset.index.imgs[gid]
    shape = (img['height'], img['width'])
    
    # Create a truth "panoptic segmentation" style mask
    true_canvas = np.zeros(shape, dtype=np.uint8)
    true_dets = dset.annots(gid=gid).detections

    true_anns = dset.annots(gid=gid)
    aids = true_anns.aids
    track_ids = [dset.anns[aid]['track_id'] for aid in aids]
    for i, true_sseg in enumerate(true_dets.data['segmentations']):
        track_now = track_ids[i]
        render_value = track_now if render_track_id else 1
        true_canvas = true_sseg.fill(true_canvas, value=render_value)
        
    return true_canvas


def render_pred_gt(pred_canvas, gt_canvas):
    # assumes both canvases to be binary
    # output color coding:
    # TN=white(R=1,G=1,B=1), TP=Green(R=0,G=1,B=0), FN=Yellow(R=1,G=1,B=0), FP=Red(R=1,G=0,B=0)
    shape = pred_canvas.shape
    out_canvas = np.zeros((pred_canvas.shape[0], pred_canvas.shape[1], 3))
    
    tn = (gt_canvas == 0) & (pred_canvas == 0)
    tp = (gt_canvas == 1) & (pred_canvas == 1)
    fn = (gt_canvas == 1) & (pred_canvas == 0)
    fp = (gt_canvas == 0) & (pred_canvas == 1)
    
    # R
    out_canvas[:,:,0] = np.clip(1-tp, a_min=0, a_max=1)
    
    # G
    out_canvas[:,:,1] = np.clip(1-fp, a_min=0, a_max=1)

    # B
    out_canvas[:,:,2] = np.clip(tn, a_min=0, a_max=1)
    
    return out_canvas


def visualize_videos(pred_dset, true_dset, out_dir='./_assets'):
    os.makedirs(out_dir, exist_ok=True)
    val_vid_ids = [6, 7]

    for vidid, _ in pred_dset.index.videos.items():
        gids = pred_dset.index.vidid_to_gids[vidid]

        n_images_to_visualize = 8
        sample_spacing = len(gids)//n_images_to_visualize
        gid_list = np.arange(start=0, stop=len(gids), step=sample_spacing)

        plt.figure(figsize=(20,5))
        for j in range(n_images_to_visualize):
            plt.subplot(2, n_images_to_visualize, j+1)
            pred_canvas = get_pred_seg(pred_dset, gids[gid_list[j]])
            gt_canvas = get_gt_seg(true_dset, gids[gid_list[j]])
            coded_canvas = render_pred_gt(pred_canvas, gt_canvas)
            plt.imshow(coded_canvas, interpolation='nearest')
            plt.title('image:'+str(gid_list[j]))
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

            # RGB
            plt.subplot(2, n_images_to_visualize, j+1+n_images_to_visualize)
            rgb = get_rgb(pred_dset, gids[gid_list[j]])
            plt.imshow(rgb)
            ax = plt.gca()
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

        fname = out_dir+'/video_'+str(vidid)+'_tracks.jpg'
        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight')
        plt.close()