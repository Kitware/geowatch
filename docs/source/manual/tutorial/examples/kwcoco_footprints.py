def kwcoco_footprint_example():
    import watch
    import kwimage
    import kwcoco

    USE_TOYDATA = 0
    if USE_TOYDATA:
        coco_dset = watch.coerce_kwcoco()
    else:
        dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        kwcoco_fpath = dvc_dpath / 'Drop4-BAS/data_vali.kwcoco.json'
        coco_dset = kwcoco.CocoDataset(kwcoco_fpath)

    # Loop through all videos / images like:
    for video_id in coco_dset.videos():
        for image_id in coco_dset.images(video_id=video_id):

            # Construct a CocoImage object, which is a helper to make it eaiser
            # to query pixel and warp / transform information (note annotation
            # access could be easier).
            coco_img : kwcoco.CocoImage = coco_dset.coco_image(image_id)

            # The coco image doesn't give access to annotations yet, but the
            # regular coco API does, so we can lookup an "annotations" object
            # representing all annotations in the image.
            annots = coco_dset.annots(gid=coco_img.img['id'])

            # Accessing the "detections" attribute returns a kwimage.Detections
            # object in IMAGE_SPACE (i.e. aligned to the pixels of the highest
            # resolution asset / auxiliary band registered in the image).
            dets : kwimage.Detections = annots.detections
            # The detections object contains boxes, segmentation, and category
            # information.
            dets.data['boxes']
            dets.data['segmentations']
            dets.data['class_idxs']
            # The class indexes are are indexes into these object "classes"
            dets.meta['classes']

            # However because the detections are returned in IMAGE space.  we
            # may need to warp them into VIDEO space. Fortunately this is easy.
            # The CocoImage object provides this transform matrix.
            warp_vid_from_img : kwimage.Affine = coco_img.warp_vid_from_img

            # The kwimage "warp" method makes it easy jointly transform all
            # annotation objects.
            vidspace_dets = dets.warp(warp_vid_from_img)

            # The underlying warped videospace boxes can be accessed as such:
            boxes : kwimage.Boxes = vidspace_dets.boxes

            # But we can also access polygons
            polys : kwimage.SegmentationList = vidspace_dets.data['segmentations']

            # Bounding boxes of polygons can be accessed by taking the bounds
            # of the polygons. (Which may be more accurate because transforming
            # polygons is lossless whereas transforming boxes is lossy)
            box_accum = []
            for poly in polys:
                box_accum.append(poly.to_boxes())
            boxes_v2 = kwimage.Boxes.concatenate(box_accum)
