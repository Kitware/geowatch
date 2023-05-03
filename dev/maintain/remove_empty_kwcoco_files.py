import ubelt as ub
kwcoco_fpaths = list(ub.Path('.').glob('imgonly*.kwcoco.*'))
import kwcoco
dset_iter = kwcoco.CocoDataset.coerce_multiple(kwcoco_fpaths, workers='avail')


good_fpaths = []
bad_fpaths = []

for dset in ub.ProgIter(dset_iter, total=len(kwcoco_fpaths)):
    if dset.n_images == 0:
        bad_fpaths.append(dset.fpath)
    else:
        good_fpaths.append(dset.fpath)

for bad in bad_fpaths:
    fpath = ub.Path(bad)
    region_id = fpath.stem.split('-')[1].split('.')[0]
    fpath.delete()
