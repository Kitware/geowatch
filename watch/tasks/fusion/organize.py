"""
Helper script for organizing experimental directory structures

python -m watch.tasks.fusion.organize

"""


def suggest_paths(test_dataset=None, package_fpath=None, pred_root=None):
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

        pred_root (str | None):
            if specified forces use of this root directory for predictions,
            otherwise the predictions are written next to the package that is
            doing the predictions.

    Example:
        >>> from watch.tasks.fusion.organize import *  # NOQA
        >>> test_dataset = 'vali.kwcoco.json'
        >>> package_fpath = '/foo/package_abc.pt'
        >>> suggest_paths(test_dataset, package_fpath)
        >>> suggest_paths(test_dataset, package_fpath)
    """
    import pathlib
    import json

    suggestions = {}

    if test_dataset is not None:
        test_dataset = pathlib.Path(test_dataset)
        test_dset_name = test_dataset.stem
    else:
        test_dset_name = 'unknown_test_dset'

    if package_fpath is not None:
        package_fpath = pathlib.Path(package_fpath)
        pred_dname = 'pred_' + package_fpath.stem

        if pred_root is None:
            pred_root = package_fpath.parent
        else:
            pred_root = pathlib.Path(pred_root)

        pred_dpath = pred_root / pred_dname / test_dset_name
        pred_dataset = pred_dpath / 'pred.kwcoco.json'

        suggestions['pred_dpath'] = str(pred_dpath)

        suggestions['pred_dataset'] = str(pred_dataset)

        suggestions['eval_dpath'] = str(pred_dpath / 'eval')

    # TODO: make this return a dict, and handle jsonification
    # in the CLI main
    return json.dumps(suggestions)


def make_nice_dirs():
    from watch.utils import util_data
    import ubelt as ub
    import pathlib  # NOQA
    dvc_dpath = util_data.find_smart_dvc_dpath()
    train_base = dvc_dpath / 'training'
    dataset_names = [
        'Drop1_October2021',
        'Drop1_November2021',
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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/organize.py make_nice_dirs
    """
    import fire
    fire.Fire()
