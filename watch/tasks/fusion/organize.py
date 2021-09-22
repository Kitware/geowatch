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

        suggestions['eval_dpath'] = str(pred_dataset / 'eval')

    import json
    return json.dumps(suggestions)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/organize.py
    """
    import fire
    fire.Fire()
