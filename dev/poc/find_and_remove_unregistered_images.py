#!/usr/bin/env python3
"""
TODO: add to kwcoco
"""
import scriptconfig as scfg
import ubelt as ub


class RemoveUnregisteredImagesConfig(scfg.DataConfig):
    """
    Walk over all files in a kwcoco bundle, find the ones that are
    unregistered, and optionally remove them.
    """
    src = scfg.Value(None, nargs='+', help='all kwcoco paths that register data in this bundle', position=1)
    io_workers = scfg.Value('avail', help='number of io workers')
    yes = scfg.Value(False, isflag=True, help='if True say yes to everything')
    image_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        if specified, only the specified path(s) will have unregistered images
        removed, otherwise this argument will be inferred as the bundle
        directories belonging to the specified kwcoco files.
        '''))


def _check_registered(dset):
    registered_paths = []
    for gid in dset.images():
        coco_img = dset.coco_image(gid)
        registered_paths.extend(list(coco_img.iter_image_filepaths()))
    registered_paths = [ub.Path(p).absolute() for p in registered_paths]
    registered_dups = ub.find_duplicates(registered_paths)
    if registered_dups:
        print('ERROR: Duplicates')
        for fpath, idxs in registered_dups.items():

            found_dup_gids = []
            # No fast index for this.
            for gid in dset.images():
                coco_img = dset.coco_image(gid)
                paths = {ub.Path(p).absolute() for p in coco_img.iter_image_filepaths()}
                if fpath in paths:
                    found_dup_gids.append(gid)

            for gid in found_dup_gids:
                coco_img = dset.coco_image(gid)
                print('coco_img.video = {}'.format(ub.urepr(coco_img.video, nl=1)))
                print('coco_img.img = {}'.format(ub.urepr(coco_img.img, nl=1)))
                print(f'coco_img={coco_img}')
        raise AssertionError('Registered files have duplicates')
    return registered_paths


def _find_existing_images(image_dpath):
    import kwimage
    existing_image_paths = []
    for r, ds, fs in image_dpath.walk():
        for f in fs:
            if f.lower().endswith(kwimage.im_io.IMAGE_EXTENSIONS):
                existing_image_paths.append(r / f)
    existing_image_paths = [p.absolute() for p in existing_image_paths]
    assert not ub.find_duplicates(existing_image_paths)
    return existing_image_paths


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/watch/dev/poc'))
        >>> from find_and_remove_unregistered_images import *  # NOQA
        >>> cmdline = 0
        >>> kwargs = dict(src='*.kwcoco.*')
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = RemoveUnregisteredImagesConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    import rich
    rich.print('config = ' + ub.urepr(config, nl=1))
    import os
    import kwcoco
    from kwutil import util_path
    fpaths = util_path.coerce_patterned_paths(config.src)

    rich.print('Will read fpaths = {}'.format(ub.urepr(fpaths, nl=1)))
    datasets = list(kwcoco.CocoDataset.coerce_multiple(fpaths, workers=config.io_workers))

    if config.image_dpath is None:
        image_dpaths = [ub.Path(d.bundle_dpath).absolute() for d in datasets]
        if not ub.allsame(image_dpaths):
            raise ValueError('All input datasets must share the same bundle')
    else:
        image_dpaths = util_path.coerce_patterned_paths(config.image_dpath)
        image_dpaths = [d.absolute() for d in image_dpaths]
        assert len(image_dpaths) == 1, 'only one image dpath handled for now'

    all_registered_paths = []
    for dset in datasets:
        registered_paths = _check_registered(dset)
        all_registered_paths.extend(registered_paths)

    image_dpath = image_dpaths[0]
    existing_image_paths = _find_existing_images(image_dpath)

    registered_paths = set(registered_paths)
    existing_image_paths = set(existing_image_paths)

    missing_fpaths = registered_paths - existing_image_paths
    unregistered_fpaths = existing_image_paths - registered_paths

    rich.print(f'len(existing_image_paths) = {len(existing_image_paths)}')
    rich.print(f'len(registered_paths)     = {len(registered_paths)}')
    rich.print(f'len(missing_fpaths)       = {len(missing_fpaths)}')
    rich.print(f'len(unregistered_fpaths)  = {len(unregistered_fpaths)}')

    if len(unregistered_fpaths) > 0:
        import rich.prompt
        rich.print('unregistered_fpaths = {}'.format(ub.urepr([os.fspath(f) for f in unregistered_fpaths], nl=1)))
        rich.print(f'len(existing_image_paths) = {len(existing_image_paths)}')
        rich.print(f'len(registered_paths)     = {len(registered_paths)}')
        rich.print(f'len(missing_fpaths)       = {len(missing_fpaths)}')
        rich.print(f'len(unregistered_fpaths)  = {len(unregistered_fpaths)}')
        ans = config.yes or rich.prompt.Confirm.ask(f'Delete these {len(unregistered_fpaths)} unregistered files?')

        if ans:
            ## ACTUALLY DELETE
            for p in ub.ProgIter(unregistered_fpaths, desc='deleting'):
                p.delete()

            # Find and remove empty directories
            _remove_empty_dirs(image_dpath)
        else:
            raise RuntimeError('User abort')
    else:
        print('No unregistered files')


def _remove_empty_dirs(dpath):
    empty_dpaths = True
    while empty_dpaths:
        empty_dpaths = []
        for r, ds, fs in dpath.walk():
            if not ds and not fs:
                empty_dpaths.append(r)
        for d in empty_dpaths:
            d.rmdir()

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/find_and_remove_unregistered_images.py
        python -m find_and_remove_unregistered_images
    """
    main()
