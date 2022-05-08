import os
import kwcoco
import ubelt as ub
from watch.utils import util_path


def main(*src):
    fpaths = util_path.coerce_patterned_paths(src)
    if len(src) == 1:
        max_workers = 0
    else:
        max_workers = min(len(src), 8)
    verbose = (max_workers == 0)
    print(f'max_workers={max_workers}')
    print(f'verbose={verbose}')
    jobs = ub.JobPool('process', max_workers=max_workers)
    for coco_fpath in ub.ProgIter(fpaths, desc='special reroot coco', verbose=3):
        jobs.submit(special_reroot_worker, coco_fpath, verbose=verbose)

    for job in jobs.as_completed(desc='collect jobs'):
        fpath, num_modified = job.result()
        print(f'wrote {fpath=}, {num_modified=}')


def special_reroot_worker(coco_fpath, verbose=0):
    with ub.Timer('read coco_fpath = {!r}'.format(coco_fpath), verbose=verbose):
        dset = kwcoco.CocoDataset(coco_fpath)
    any_modified = special_reroot_single(dset, verbose=verbose)
    if verbose:
        print(f'{len(any_modified)=}')
    if any_modified:
        dset.validate()
        if verbose:
            print('writing dset.fpath = {!r}'.format(dset.fpath))
        dset.dump(dset.fpath, newlines=True)
        if verbose:
            print('wrote dset.fpath = {!r}'.format(dset.fpath))
    num_modified = len(any_modified)
    return dset.fpath, num_modified


def special_reroot_single(dset, verbose=0):
    bundle_dpath = ub.Path(dset.bundle_dpath).absolute()
    resolved_bundle_dpath = ub.Path(dset.bundle_dpath).resolve()

    any_modified = []
    for gid in ub.ProgIter(dset.images(), desc='special reroot', verbose=verbose):
        coco_img: kwcoco.CocoImage = dset.coco_image(gid)
        for obj in coco_img.iter_asset_objs():
            old_fname = ub.Path(obj['file_name'])
            fpath = (bundle_dpath / old_fname)
            if fpath.exists():
                new_fname = resolve_relative_to(fpath, resolved_bundle_dpath)
                if new_fname != old_fname:
                    assert (bundle_dpath / new_fname).exists()
                    any_modified.append(f'{old_fname} -> {new_fname}')
                    obj['file_name'] = os.fspath(new_fname)
    return any_modified


def resolve_relative_to(path, dpath, strict=False):
    """
    Given a path, try to resolve its symlinks such that it is relative to the
    given dpath.

    Ignore:
        def _symlink(self, target, verbose=0):
            return ub.Path(ub.symlink(target, self, verbose=verbose))
        ub.Path._symlink = _symlink

        # TODO: try to enumerate all basic cases

        base = ub.Path.appdir('kwcoco/tests/reroot')
        base.delete().ensuredir()

        drive1 = (base / 'drive1').ensuredir()
        drive2 = (base / 'drive2').ensuredir()

        data_repo1 = (drive1 / 'data_repo1').ensuredir()
        cache = (data_repo1 / '.cache').ensuredir()
        real_file1 = (cache / 'real_file1').touch()

        real_bundle = (data_repo1 / 'real_bundle').ensuredir()
        real_assets = (real_bundle / 'assets').ensuredir()

        # Symlink file outside of the bundle
        link_file1 = (real_assets / 'link_file1')._symlink(real_file1)
        real_file2 = (real_assets / 'real_file2').touch()
        link_file2 = (real_assets / 'link_file2')._symlink(real_file2)


        # A symlink to the data repo
        data_repo2 = (drive1 / 'data_repo2')._symlink(data_repo1)
        data_repo3 = (drive2 / 'data_repo3')._symlink(data_repo1)
        data_repo4 = (drive2 / 'data_repo4')._symlink(data_repo2)

        # A prediction repo TODO
        pred_repo5 = (drive2 / 'pred_repo5').ensuredir()

        _ = ub.cmd(f'tree -a {base}', verbose=3)

        fpaths = []
        for r, ds, fs in os.walk(base, followlinks=True):
            for f in fs:
                if 'file' in f:
                    fpath = ub.Path(r) / f
                    fpaths.append(fpath)


        dpath = real_bundle.resolve()

        for path in fpaths:
            # print(f'{path}')
            # print(f'{path.resolve()=}')
            resolved_rel = resolve_relative_to(path, dpath)
            print('resolved_rel = {!r}'.format(resolved_rel))
    """
    try:
        resolved_abs = resolve_directory_symlinks(path)
        resolved_rel = resolved_abs.relative_to(dpath)
    except ValueError:
        if strict:
            raise
        else:
            return path
    return resolved_rel


def resolve_directory_symlinks(path):
    """
    Only resolve symlinks of directories
    """
    return path.parent.resolve() / path.name
    # prev = path
    # curr = prev.parent
    # while prev != curr:
    #     if curr.is_symlink():
    #         rhs = path.relative_to(curr)
    #         resolved_lhs = curr.resolve()
    #         new_path = resolved_lhs / rhs
    #         return new_path
    #     prev = curr
    #     curr = prev.parent
    # return path


def _cleanup_lightning_logs():
    from watch.utils import util_path
    training_dpaths = util_path.coerce_patterned_paths('./**/lightning_logs/version_*')

    to_remove = []
    for train_dpath in training_dpaths:
        checkpoint_dpath = train_dpath / 'checkpoints'
        num_checkpoints = len(list(checkpoint_dpath.glob('*')))
        if num_checkpoints == 0:
            to_remove.append(train_dpath)
    for dpath in to_remove:
        dpath.delete()

    # Reduce number of diagnostic images
    actions = []
    for train_dpath in training_dpaths:
        train_viz_dpath = train_dpath / 'monitor/train/batch'
        vali_viz_dpath = train_dpath / 'monitor/vali/batch'
        actions += clean_viz_dpath(train_viz_dpath, max_keep=100)
        actions += clean_viz_dpath(vali_viz_dpath, max_keep=300)

    dry = 1
    all_actions = actions
    _execute_actions(all_actions, dry=dry)


def clean_viz_dpath(viz_dpath, max_keep=300):
    def _choose_action(file_infos):
        import kwarray
        file_infos = kwarray.shuffle(file_infos, rng=0)
        n_keep = max_keep
        for info in file_infos[:n_keep]:
            info['action'] = 'keep'
        for info in file_infos[n_keep:]:
            info['action'] = 'delete'
    exts = ['*.png', '*.jpg' ]

    all_actions = []
    for ext in exts:
        fpaths = list(viz_dpath.glob(ext))
        file_infos = [{'size': p.stat().st_size, 'fpath': p}
                      for p in fpaths]
        _choose_action(file_infos)
        all_actions.extend(file_infos)
    return all_actions


def _execute_actions(all_actions, dry=True):
    print(f'{len(all_actions)=}')
    grouped_actions = ub.group_items(all_actions, lambda x: x['action'])

    import xdev
    for key, group in grouped_actions.items():
        size = xdev.byte_str(sum([s['size'] for s in group]))
        print('{:>4} images:  {:>4}, size={}'.format(key.capitalize(), len(group), size))

    if dry:
        print('Dry run')
    else:
        delete = grouped_actions.get('delete', [])
        delete_fpaths = [item['fpath'] for item in delete]
        if delete_fpaths:
            for p in ub.ProgIter(delete_fpaths, desc='deleting'):
                ub.delete(p)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/special_reroot.py combo*.kwcoco.json
    """
    import fire
    fire.Fire(main)
