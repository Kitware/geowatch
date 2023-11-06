def demo():
    """
    Show how to get the predicted heatmaps in video space at a resolution for
    each frame in a video. Also show how to get the corresponding rasterized
    annotations.
    """
    import watch
    import kwimage
    import numpy as np

    if 1:
        coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True)

    chosen_video = coco_dset.videos().objs[0]
    video_id = chosen_video['id']
    images_for_video = coco_dset.images(video_id=video_id)

    resolution = '10GSD'

    def get_annots_in_requested_space(coco_img, space='video', resolution=None):
        """
        This is slightly less than ideal in terms of API, but it will work for
        now.
        """
        # Build transform from image to requested space
        warp_vid_from_img = coco_img._warp_vid_from_img
        scale = coco_img._scalefactor_for_resolution(space='video',
                                                     resolution=resolution,
                                                     RESOLUTION_KEY='target_gsd')
        warp_req_from_vid = kwimage.Affine.scale(scale)
        warp_req_from_img = warp_req_from_vid @ warp_vid_from_img

        # Get annotations in "Image Space"
        annots = coco_img.dset.annots(image_id=coco_img.img['id'])
        imgspace_dets = annots.detections
        reqspace_dets = imgspace_dets.warp(warp_req_from_img)
        return reqspace_dets

    frame_data_list = []
    frame_truth_list = []

    for coco_img in images_for_video.coco_images:
        # Get the image data in "requested space"
        delayed = coco_img.imdelay('salient', space='video', resolution=resolution, RESOLUTION_KEY='target_gsd')
        detections = get_annots_in_requested_space(coco_img, space='video', resolution=resolution)

        frame_data = delayed.finalize()

        h, w = frame_data.shape[0:2]

        # TODO: filter by class or whatever you want here
        truth_canvas = np.zeros((h, w))
        detections.data['segmentations'].fill(truth_canvas, value=1)
        frame_truth_list.append(truth_canvas)
        frame_data_list.append(frame_data)
