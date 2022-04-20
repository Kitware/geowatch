import ubelt as ub
import pandas as pd
import os
import glob


def eval3_report():
    import watch
    dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    gsd10_dpath = dvc_dpath / 'models/fusion/eval3_candidates'
    gsd1_dpath = dvc_dpath / 'models/fusion/eval3_sc_candidates'

    summary_stats = []
    evaluations = []

    gsd_dpaths = {
        'gsd10': gsd10_dpath,
        'gsd1': gsd1_dpath,
    }
    for gds_key, dpath in gsd_dpaths.items():
        experiments = list((dpath / 'packages').glob('*'))
        models = list((dpath / 'packages').glob('*/*'))

        pixel_evals = dvc_globbed_info('pxl', dpath / 'eval/*/*/*/*/eval/curves/measures2.*')
        iarpa_bas_evals = dvc_globbed_info('bas', dpath / 'eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.*')
        iarpa_sc_evals = dvc_globbed_info('sc', dpath / 'eval/*/*/*/*/eval/actclf/*/iarpa_*/scores/merged/summary3.*')

        evaluations += pixel_evals
        evaluations += iarpa_bas_evals
        evaluations += iarpa_sc_evals

        row = {
            'gsd': gds_key,
            'num_experiments': len(experiments),
            'num_models': len(models),
            'num_pxl_evals': len(pixel_evals),
            'num_bas_evals': len(iarpa_bas_evals),
            'num_sc_evals': len(iarpa_sc_evals),
        }
        summary_stats.append(row)

    if True:
        # Ensure dvc is synced
        missing_dvc = []
        missing_real = []
        for row in evaluations:
            if row['dvc_fpath'] is None:
                missing_dvc.append(row['fpath'])
            if row['fpath'] is None:
                missing_real.append(row['dvc_fpath'])

        from watch.utils import simple_dvc
        if missing_real:
            dvc = simple_dvc.SimpleDVC.coerce(missing_real[0])
            dvc.pull(missing_real, remote='aws')

    df = pd.DataFrame(summary_stats)
    print(df)


def dvc_globbed_info(type, pat):
    eval_fpaths = list(glob.glob(os.fspath(pat)))
    rows = []
    for k, group in ub.group_items(eval_fpaths, lambda x: (ub.Path(x).parent, ub.Path(x).name.split('.')[0])).items():
        dvc_fpath = None
        fpath = None
        for g in group:
            if g.endswith('.dvc'):
                dvc_fpath = g
            else:
                fpath = g

        row = {
            'type': type,
            'fpath': fpath,
            'dvc_fpath': dvc_fpath,
        }
        rows.append(row)
    return rows
