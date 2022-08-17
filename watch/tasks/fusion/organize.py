"""
Helper script for organizing experimental directory structures

python -m watch.tasks.fusion.organize

"""
import ubelt as ub
import json
import os


def suggest_paths(test_dataset=None, package_fpath=None, workdir=None,
                  sidecar2=False, as_json=True, pred_cfg=None, pred_cfgstr=None):
    """
    Suggest an organized set of paths for where data should be written.

    Attempt to reduce parameterization by suggesting paths for where
    predictions and evaluations should be written depending on the parameters
    you are using.

    Args:
        test_dataset (str):
            path to the testing kwcoco filepath

        package_fpath (str):
            the path to the model checkpoint / package.

        workdir (str | None):
            if specified forces use of this root directory for predictions and
            evaluations, otherwise the predictions are written next to the
            package that is doing the predictions.

    Example:
        >>> from watch.tasks.fusion.organize import *  # NOQA
        >>> test_dataset = 'vali.kwcoco.json'
        >>> package_fpath = '/models/fusion/eval1_cand/packages/expt1/package_abc.pt'
        >>> suggestions = suggest_paths(test_dataset, package_fpath, sidecar2=1, as_json=False)
        >>> print('suggestions = {}'.format(ub.repr2(suggestions, nl=1, align=':', sort=0)))

        >>> suggestions = suggest_paths(test_dataset, package_fpath, sidecar2=1, as_json=False, workdir='/my_tmp_eval')
        >>> print('suggestions = {}'.format(ub.repr2(suggestions, nl=1, align=':', sort=0)))
    """

    suggestions = {}

    if test_dataset is not None:
        # TODO: better way to choose the test-dataset-identifier - needs a
        # hashid
        test_dataset = ub.Path(test_dataset)
        test_dset_name = '_'.join((list(test_dataset.parts[-2:-1]) + [test_dataset.stem]))
        # test_dset_name = test_dataset.stem
    else:
        test_dset_name = 'unknown_test_dset'

    if package_fpath is not None:

        if pred_cfgstr is None:
            if pred_cfg is None:
                pred_cfgstr = 'unknown'
            else:
                pred_cfgstr = ub.hash_data(pred_cfg)[0:8]

        _register_hashid_reverse_lookup(pred_cfgstr, pred_cfg, type='pred_cfg')

        pred_cfg_dname = 'predcfg_' + pred_cfgstr

        # TODO: This should be handled by watch.dvc.expt_manager

        package_fpath = ub.Path(package_fpath)
        # pred_dname = 'pred_' + package_fpath.stem
        pred_dname = package_fpath.name

        if sidecar2:
            # Make assumptions about directory structure
            expt_name = package_fpath.parent.name
            pkg_dpath = package_fpath.parent.parent
            candidate_dpath = pkg_dpath.parent
            # package_name = package_fpath.stem

            if pkg_dpath.name != 'packages':
                print('Warning: might not have the right dir structure')

            if workdir is None:
                workdir = candidate_dpath
            else:
                workdir = ub.Path(workdir)

            pred_root = workdir / 'pred' / expt_name
            pred_dpath = pred_root / pred_dname / test_dset_name / pred_cfg_dname

            eval_root = workdir / 'eval' / expt_name
            eval_dpath = eval_root / pred_dname / test_dset_name / pred_cfg_dname / 'eval'
        else:
            if workdir is None:
                workdir = package_fpath.parent
            else:
                workdir = ub.Path(workdir)

            pred_dpath = workdir / pred_dname / test_dset_name / pred_cfg_dname
            eval_dpath = pred_dpath / 'eval'

        pred_dataset = pred_dpath / 'pred.kwcoco.json'

        suggestions['package_fpath'] = os.fspath(package_fpath)

        suggestions['pred_dpath'] = os.fspath(pred_dpath)

        suggestions['pred_dataset'] = os.fspath(pred_dataset)

        suggestions['eval_dpath'] = os.fspath(eval_dpath)

        suggestions['package_cfgstr'] = package_fpath.stem
        suggestions['pred_cfgstr'] = pred_cfgstr

    # TODO: make this return a dict, and handle jsonification
    # in the CLI main
    if as_json:
        suggestion_text = json.dumps(suggestions)
        return suggestion_text
    else:
        return suggestions


def _register_hashid_reverse_lookup(key, data, type='global'):
    """
    Make a lookup table of hashes we've made, so we can refer to what the heck
    those directory names mean!

    Example:
        from watch.tasks.fusion.organize import *  # NOQA
        from watch.tasks.fusion.organize import _register_hashid_reverse_lookup
        data = {'test': 'data'}
        key = ub.hash_data(data)[0:8]
        _register_hashid_reverse_lookup(key, data)
        _register_hashid_reverse_lookup('conflict', 'conflict1')
        _register_hashid_reverse_lookup('conflict', 'conflict2')

    """
    import shelve
    import os
    from watch.utils.util_locks import Superlock

    FULL_TEXT = 1
    DPATH_TEXT = 1

    rlut_dpath = ub.Path.appdir('watch/hash_rlut', type).ensuredir()
    shelf_fpath = rlut_dpath / 'hash_rlut.shelf'
    text_fpath = rlut_dpath / 'hash_rlut.txt'
    file_dpath = (rlut_dpath / 'hash_rlut').ensuredir()
    lock_fpath = rlut_dpath / 'flock.lock'
    lock = Superlock(thread_key='hash_rlut', lock_fpath=lock_fpath)
    blake3 = ub.hash_data(data, hasher='blake3')
    row = {'data': data, 'blake3': blake3}
    info = {}
    with lock:
        shelf = shelve.open(os.fspath(shelf_fpath))
        with shelf:
            # full_shelf = dict(shelf)
            if key not in shelf:
                shelf[key] = [row]
                info['status'] = 'new'
            else:
                datas = shelf[key]
                found = False
                for other in datas:
                    if other['blake3'] == row['blake3']:
                        found = True
                        break
                if not found:
                    info['status'] = 'conflict'
                    datas.append(row)
                    shelf[key] = datas
                else:
                    info['status'] = 'exists'

            if FULL_TEXT:
                full_shelf = dict(shelf)
            else:
                full_shelf = None

        if info['status'] != 'exists':
            # Convinience
            if FULL_TEXT:
                full_shelf = dict(shelf)
                full_text = ub.repr2(full_shelf, nl=3)
                text_fpath.write_text(full_text)

            if DPATH_TEXT:
                fpath = file_dpath / key
                datas_text = ub.repr2(datas, nl=3)
                fpath.write_text(datas_text)

    return info


if __name__ == '__main__':
    import fire
    fire.Fire()
