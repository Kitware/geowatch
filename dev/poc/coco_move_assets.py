
def move_assets():
    import kwcoco
    from kwutil import util_path
    import ubelt as ub

    bundle_dpath = ub.Path('.')

    cache_dpath = (bundle_dpath / '_cache').ensuredir()
    dirty_fpath = cache_dpath / 'dirty_flag'
    if dirty_fpath.exists():
        raise Exception('The repo is in a dirty state and hasnt been cleaned up')
    dirty_fpath.touch()

    old_prefix = '_assets'
    new_prefix = 'raw_bands'

    old_path = bundle_dpath / '_assets'
    new_path = bundle_dpath / 'raw_bands'

    paths = util_path.coerce_patterned_paths('*.kwcoco.zip')
    orig_dsets = list(kwcoco.CocoDataset.coerce_multiple(paths, workers='avail'))

    DEBUGGING = 1
    if DEBUGGING:
        dsets = [d.copy() for d in ub.ProgIter(orig_dsets, desc='copy datasets')]
    else:
        dsets = orig_dsets

    for dset in ub.ProgIter(dsets, desc='checking before reroot'):
        assert not dset.missing_images()

    print('Moving assets')
    old_path.move(new_path)

    for dset in ub.ProgIter(dsets, desc='modify kwcoco files'):
        ...
        dset.reroot(old_prefix=old_prefix, new_prefix=new_prefix, check=True)

    for dset in ub.ProgIter(dsets, desc='checking reroot'):
        assert not dset.missing_images()

    for dset in ub.ProgIter(dsets, desc='write kwcoco files'):
        dset.dump(indent=' ' * 4)

    dirty_fpath.delete()
