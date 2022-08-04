"""
Maybe a CV2 tracker could work for us?



References:
    https://livecodestream.dev/post/object-tracking-with-opencv/
    https://pyimagesearch.com/2018/08/06/tracking-multiple-objects-with-opencv/

Requires:
    pip install opencv-contrib-python-headless
"""

import kwcoco
import kwplot
import numpy as np
import cv2
import kwimage
kwplot.autompl()

dset = kwcoco.CocoDataset.demo('vidshapes1-frames30')


video = dset.videos().objs[0]

vid_width = video['width']
vid_height = video['height']
vid_dims = (vid_height, vid_width)

video_images = dset.images(vidid=video['id'])


video_frames = []
video_boxes = []

for gid in video_images:
    coco_img = dset.coco_image(gid)
    imgspace_dets = dset.annots(gid=gid).detections
    vidspace_dets = imgspace_dets.warp(coco_img.warp_vid_from_img)
    vidspace_dets.data.pop('keypoints', None)

    frame_heatmap = np.zeros((vid_dims), dtype=np.float32)

    # object_slices = blip_boxes.quantize().to_slices()
    # for sl in object_slices:
    #     frame_heatmap[sl] = 1

    # import kwarray
    # sticher = kwarray.Stitcher(vid_dims)
    object_boxes = vidspace_dets.boxes
    blip_boxes = object_boxes.scale(0.5, about='centroid')
    for cx, cy, w, h in blip_boxes.to_cxywh().data:
        poly_mask = np.zeros((vid_dims), dtype=np.uint8)
        # Could just fill the extent and translate here
        poly = kwimage.Polygon.circle((cx, cy), r=(w, h))
        rel_mask, offset = poly.to_relative_mask(return_offset=True)
        dist_msk = cv2.distanceTransform(
            src=rel_mask.data, distanceType=cv2.DIST_L2, maskSize=3)
        dist_msk = dist_msk / dist_msk.max()
        rx, ry = offset
        rh, rw = rel_mask.shape
        big_sl = kwimage.Boxes([[rx, ry, rw, rh]], 'xywh').to_slices()[0]
        exiting = frame_heatmap[big_sl]
        frame_heatmap[big_sl] = np.maximum(exiting, dist_msk)

    frame_heatmap = kwimage.gaussian_blur(frame_heatmap, sigma=6)
    video_frames.append(frame_heatmap)
    video_boxes.append(object_boxes)

    # heatmap = vidspace_dets.rasterize((1, 1), vid_dims)


first_boxes = video_boxes[0]
first_frame = video_frames[0]
bbox = first_boxes.to_xywh().quantize().data[0]

multitracker = cv2.legacy.MultiTracker_create()

for first_box in first_boxes.to_xywh().quantize().data:
    # tracker = cv2.TrackerMIL_create()
    multitracker.add(cv2.legacy.TrackerMIL_create(), first_frame, first_box)

# multitracker.init(first_frame, bbox)


import xdev  # NOQA
for frame in xdev.InteractiveIter(video_frames):
    # status_flag, tracked_bbox = tracker.update(frame)
    (status_flag, tracked_bboxes) = multitracker.update(frame)
    print(f'status_flag={status_flag}')
    new_bbox = kwimage.Boxes(tracked_bboxes, 'xywh')
    canvas = new_bbox.draw_on(frame.copy())
    kwplot.imshow(canvas)
    # new_bbox.draw()
    xdev.InteractiveIter.draw()
