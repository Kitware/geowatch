
def main():
    """
    Walk over all files in a kwcoco bundle, find the ones that are
    unregistered, and optionally remove them.
    """
    import kwcoco
    import kwimage
    import ubelt as ub
    dset = kwcoco.CocoDataset('data.kwcoco.json')

    registered_paths = []
    for gid in dset.images():
        coco_img = dset.coco_image(gid)
        registered_paths.extend(list(coco_img.iter_image_filepaths()))

    existing_image_paths = []
    for r, ds, fs in ub.Path('.').walk():
        for f in fs:
            if f.lower().endswith(kwimage.im_io.IMAGE_EXTENSIONS):
                existing_image_paths.append(r / f)


    existing_image_paths = [p.absolute() for p in existing_image_paths]
    registered_paths = [ub.Path(p).absolute() for p in registered_paths]

    assert not ub.find_duplicates(registered_paths)
    assert not ub.find_duplicates(existing_image_paths)

    registered_paths = set(registered_paths)
    existing_image_paths = set(existing_image_paths)

    missing_fpaths = registered_paths - existing_image_paths
    unregistered_fpaths = existing_image_paths - registered_paths
