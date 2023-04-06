def find_wv_images_for_pan_sharpen():
    import watch

    # This respects the environ DVC_DPATH
    # or assumes the DVC repo is in $HOME/data/dvc-repos/smart_watch_dvc
    dvc_dpath = watch.utils.util_data.find_smart_dvc_dpath()
    kwcoco_fpath = dvc_dpath / 'Drop1-Aligned-L1/data.kwcoco.json'

    import kwcoco
    coco_dset = kwcoco.CocoDataset(kwcoco_fpath)

    # Define what channels we want
    desired_channels = {'red', 'green', 'blue', 'panchromatic'}
    desired_sensors = {'WV'}
    valid_gids = []

    # For each image in the dataset
    for gid in coco_dset.index.imgs.keys():
        coco_img = coco_dset.coco_image(gid)
        sensor = coco_img.img['sensor_coarse']
        # If the sensor is one we are interested in,
        if sensor in desired_sensors:
            have_channels = coco_img.channels.fuse().as_set()
            # And, if we have all of the channels we were looking for
            if have_channels.issuperset(desired_channels):
                # Then add the image to the list of "valid" image ids
                valid_gids.append(gid)

    import ubelt as ub
    for obj in coco_dset.images(valid_gids).objs:
        name = obj['name']
        print(' * name = {}'.format(ub.urepr(name, nl=1)))
        for aux in obj['auxiliary']:
            aux_fname = aux['file_name']
            print('    * aux_fname = {}'.format(ub.urepr(aux_fname, nl=1)))
            aux_chan = aux['channels']
            print('    * aux_chan = {}'.format(ub.urepr(aux_chan, nl=1)))
