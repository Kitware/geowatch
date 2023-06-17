#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class RemoveEmptyKWCocoFiles(scfg.DataConfig):
    src = scfg.Value('*.kwcoco.*', nargs='*', help='path to one more more kwcoco files')


def main(cmdline=1, **kwargs):
    """
    Example:
        >>> # xdoctest: +SKIP
        >>> cmdline = 0
        >>> kwargs = dict()
        >>> main(cmdline=cmdline, **kwargs)
    """
    config = RemoveEmptyKWCocoFiles.cli(cmdline=cmdline, data=kwargs, strict=True)

    import rich
    import kwcoco
    from kwutil.util_path import coerce_patterned_paths
    from kwutil import util_progress
    from rich.prompt import Confirm
    rich.print('config = ' + ub.urepr(config, nl=1))

    kwcoco_fpaths = coerce_patterned_paths(config.src)
    print('kwcoco_fpaths = {}'.format(ub.urepr(kwcoco_fpaths, nl=1)))
    dset_iter = kwcoco.CocoDataset.coerce_multiple(kwcoco_fpaths, workers='avail')

    good_fpaths = []
    bad_fpaths = []

    pman = util_progress.ProgressManager()
    with pman:
        for dset in pman.progiter(dset_iter, total=len(kwcoco_fpaths)):
            if dset.n_images == 0:
                bad_fpaths.append(dset.fpath)
            else:
                good_fpaths.append(dset.fpath)

    print('bad_fpaths = {}'.format(ub.urepr(bad_fpaths, nl=1)))

    num_bad = len(bad_fpaths)
    num_good = len(good_fpaths)
    num_total = num_good + num_bad

    ans = Confirm.ask(f'delete {num_bad} / {num_total} kwcoco files?')
    if ans:
        for bad in ub.ProgIter(bad_fpaths):
            fpath = ub.Path(bad)
            # region_id = fpath.stem.split('-')[1].split('.')[0]
            fpath.delete()


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/watch/dev/poc/remove_empty_kwcoco_files.py
    """
    main()
