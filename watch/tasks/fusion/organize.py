"""
Helper script for organizing experimental directory structures

python -m watch.tasks.fusion.organize

"""
import ubelt as ub
import json
import os


def suggest_paths(test_dataset=None, package_fpath=None, workdir=None,
                  sidecar2=False, as_json=True, pred_cfg=None):
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

        if pred_cfg is None:
            pred_cfg_dname = 'predcfg_unknown'
        else:
            pred_cfg_dname = 'predcfg_' + ub.hash_data(pred_cfg)[0:8]

        package_fpath = ub.Path(package_fpath)
        pred_dname = 'pred_' + package_fpath.stem

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

            pred_dpath = workdir / pred_dname / test_dset_name
            eval_dpath = pred_dpath / 'eval'

        pred_dataset = pred_dpath / 'pred.kwcoco.json'

        suggestions['package_fpath'] = os.fspath(package_fpath)

        suggestions['pred_dpath'] = os.fspath(pred_dpath)

        suggestions['pred_dataset'] = os.fspath(pred_dataset)

        suggestions['eval_dpath'] = os.fspath(eval_dpath)

    # TODO: make this return a dict, and handle jsonification
    # in the CLI main
    if as_json:
        suggestion_text = json.dumps(suggestions)
        return suggestion_text
    else:
        return suggestions


def make_nice_dirs():
    # DEPRECATE
    from watch.utils import util_data
    dvc_dpath = util_data.find_smart_dvc_dpath()
    train_base = dvc_dpath / 'training'
    dataset_names = [
        'Drop1_October2021',
        'Drop1_November2021',
        'Drop1_2020-11-17',
    ]
    user_machine_dpaths = list(train_base.glob('*/*'))
    # all_checkpoint_paths = []
    for um_dpath in user_machine_dpaths:
        for dset_name in dataset_names:
            dset_dpath = um_dpath / dset_name
            runs_dpath = (dset_dpath / 'runs')
            nice_root_dpath = (dset_dpath / 'nice')
            nice_root_dpath.mkdir(exist_ok=True,)

            for nice_link in nice_root_dpath.glob('*'):
                if ub.util_links.islink(nice_link):
                    if not nice_link.exists():
                        nice_link.unlink()

            lightning_log_dpaths = list(runs_dpath.glob('*/lightning_logs'))
            for ll_dpath in lightning_log_dpaths:
                dname = ll_dpath.parent.name
                nice_dpath = nice_root_dpath / dname
                version_dpaths = sorted(ll_dpath.glob('*'))
                version_dpath = version_dpaths[-1]
                ub.symlink(version_dpath, nice_dpath, verbose=1)


def make_eval_symlinks():
    """
    DEPRECATE
    """
    from watch.utils import util_data
    dvc_dpath = util_data.find_smart_dvc_dpath()

    # HACK: HARD CODED
    # model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    model_dpath = dvc_dpath / 'models/fusion/SC-20201117'
    eval_link_base = model_dpath / 'eval_links'
    eval_link_base.mkdir(exist_ok=True)

    eval_dpaths = list(model_dpath.glob('*/*/*/eval'))
    for eval_dpath in eval_dpaths:
        # Hack: find a better way to get info needed to make a nice folder name
        eval_link_name = 'eval_' + eval_dpath.parent.name + '_' + eval_dpath.parent.parent.name
        eval_nice_dpath = eval_link_base / eval_link_name
        ub.symlink(eval_dpath, eval_nice_dpath)


def make_pred_symlinks():
    """
    DEPRECATE
    """
    from watch.utils import util_data
    dvc_dpath = util_data.find_smart_dvc_dpath()

    # HACK: HARD CODED
    model_dpath = dvc_dpath / 'models/fusion/unevaluated-activity-2021-11-12'
    link_base = model_dpath / 'pred_links'
    link_base.mkdir(exist_ok=True)

    dpaths = list(model_dpath.glob('*/pred_*/*'))
    for dpath in dpaths:
        # Hack: find a better way to get info needed to make a nice folder name
        link_name = 'pred_' + dpath.name + '_' + dpath.parent.name
        nice_dpath = link_base / link_name
        ub.symlink(dpath, nice_dpath)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/organize.py make_nice_dirs
        python ~/code/watch/watch/tasks/fusion/organize.py make_eval_symlinks
        python ~/code/watch/watch/tasks/fusion/organize.py make_pred_symlinks
    """
    import fire
    fire.Fire()
