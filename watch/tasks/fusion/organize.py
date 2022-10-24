"""
Helper script for organizing experimental directory structures

THIS IS SLOWY BEING DEPRECATED

python -m watch.tasks.fusion.organize
"""
import ubelt as ub
import json
import os


def suggest_paths(test_dataset=None, package_fpath=None, workdir=None,
                  sidecar2=True, as_json=True, pred_cfg=None, pred_cfgstr=None):
    """
    DEPRECATED

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
        >>> package_fpath = 'work/models/fusion/eval1_cand/packages/expt1/package_abc.pt'
        >>> suggestions = suggest_paths(test_dataset, package_fpath, sidecar2=1, as_json=False)
        >>> print('suggestions = {}'.format(ub.repr2(suggestions, nl=1, align=':', sort=0)))

        >>> suggestions = suggest_paths(test_dataset, package_fpath, sidecar2=1, as_json=False, workdir='/my_tmp_eval')
        >>> print('suggestions = {}'.format(ub.repr2(suggestions, nl=1, align=':', sort=0)))

        >>> package_fpath = 'foobar/eval1_cand/packages/expt1/package_abc.pt'
        >>> suggestions = suggest_paths(test_dataset, package_fpath, sidecar2=1, as_json=False)
        >>> print('suggestions = {}'.format(ub.repr2(suggestions, nl=1, align=':', sort=0)))
    """

    suggestions = {}

    # New logic that will supercede this file
    from watch.mlops.expt_manager import ExperimentState
    state = ExperimentState('*', '*')
    test_dset_name = state._condense_test_dset(test_dataset)

    if package_fpath is not None:
        if pred_cfgstr is None:
            pred_cfg_dname = state._condense_pred_cfg(pred_cfg)
        else:
            pred_cfg_dname = 'predcfg_' + pred_cfgstr

        package_fpath = ub.Path(package_fpath)
        if sidecar2:
            condensed = {}
            condensed.update({
                'trk_model': package_fpath.name,
                'test_trk_dset': test_dset_name,
                'trk_pxl_cfg': pred_cfg_dname,
            })
            try:
                condensed.update(state._parse_pattern_attrs(state.templates['pkg_trk_pxl_fpath'], package_fpath))
            except Exception:
                if package_fpath.parent.parent != 'packages':
                    print('Assumptions are broken')
                condensed['expt'] = package_fpath.parent
                condensed['expt_dvc_dpath'] = package_fpath.parent.parent.parent.parent
                condensed['dataset_code'] = package_fpath.parent.parent.parent.name

            if workdir is None:
                workdir = state.storage_template_prefix.format(**condensed)

            workdir = ub.Path(workdir)
            pred_fpath = workdir / state.volitile_templates['pred_trk_pxl_fpath'].format(**condensed)
            pred_dpath = pred_fpath.parent

            eval_fpath = workdir / state.versioned_templates['eval_trk_pxl_fpath'].format(**condensed)
            eval_dpath = eval_fpath.parent.parent
        else:
            if workdir is None:
                workdir = package_fpath.parent
            else:
                workdir = ub.Path(workdir)
            pred_dname = package_fpath.name
            pred_dpath = workdir / pred_dname / test_dset_name / pred_cfg_dname
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


if __name__ == '__main__':
    import fire
    fire.Fire()
