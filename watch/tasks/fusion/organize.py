"""
Helper script for organizing experimental directory structures

python -m watch.tasks.fusion.organize

"""


def suggest_paths(test_dataset=None, package_fpath=None):
    """
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
        pred_dpath = package_fpath.parent / pred_dname / test_dset_name
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
