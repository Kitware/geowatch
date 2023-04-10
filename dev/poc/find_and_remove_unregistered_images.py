#!/usr/bin/env python3
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


def _find_existing_images(bundle_dpath):
    import kwimage
    existing_image_paths = []
    for r, ds, fs in bundle_dpath.walk():
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
    from watch.utils import util_path
    fpaths = util_path.coerce_patterned_paths(config.src)

    rich.print('Will read fpaths = {}'.format(ub.urepr(fpaths, nl=1)))
    datasets = list(kwcoco.CocoDataset.coerce_multiple(fpaths, workers=config.io_workers))

    all_bundles = [ub.Path(d.bundle_dpath).resolve() for d in datasets]
    if not ub.allsame(all_bundles):
        raise ValueError('All input datasets must share the same bundle')

    all_registered_paths = []
    for dset in datasets:
        registered_paths = _check_registered(dset)
        all_registered_paths += registered_paths

    bundle_dpath = all_bundles[0]
    existing_image_paths = _find_existing_images(bundle_dpath)

    registered_paths = set(registered_paths)
    existing_image_paths = set(existing_image_paths)

    missing_fpaths = registered_paths - existing_image_paths
    unregistered_fpaths = existing_image_paths - registered_paths

    rich.print(f'{len(missing_fpaths)=}')
    rich.print(f'{len(unregistered_fpaths)=}')

    if len(unregistered_fpaths) > 0:
        import rich.prompt
        rich.print('unregistered_fpaths = {}'.format(ub.urepr([os.fspath(f) for f in unregistered_fpaths], nl=1)))
        ans = config.yes or rich.prompt.Confirm.ask(f'Delete these {len(unregistered_fpaths)} unregistered files?')

        if ans:
            ## ACTUALLY DELETE
            for p in ub.ProgIter(unregistered_fpaths, desc='deleting'):
                p.delete()

            # Find and remove empty directories
            _remove_empty_dirs(bundle_dpath)
        else:
            raise RuntimeError('User abort')
    else:
        print('No unregistered files')


def _remove_empty_dirs(bundle_dpath):
    empty_dpaths = True
    while empty_dpaths:
        empty_dpaths = []
        for r, ds, fs in bundle_dpath.walk():
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
