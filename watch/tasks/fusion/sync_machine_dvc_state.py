"""
Synchronize DVC states across the machine.
"""
import glob
import ubelt as ub
import platform
from watch.utils import simple_dvc


EVAL_GLOB_PATTERNS = {
    'pxl_10': 'models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json',
    'trk_10': 'models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json',
    'pxl_1': 'models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json',
    'act_1': 'models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/actclf/*/*_eval/scores/merged/summary3.json',
}


def main():
    """
    from watch.tasks.fusion.sync_machine_dvc_state import *  # NOQA
    """
    import watch
    from watch.tasks.fusion import repackage
    dvc_hdd_dpath = watch.find_smart_dvc_dpath(hardware='hdd')

    try:
        dvc_ssd_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
    except Exception:
        dvc_ssd_dpath = None

    mode = 'commit'
    if dvc_ssd_dpath is not None:
        # If the SSD has stuff, add it, but the HDD is primary
        dvc_dpath = dvc_ssd_dpath
        dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath)
        dvc.git_pull()
        sync_checkpoints(dvc_dpath, mode='list')
        sync_checkpoints(dvc_dpath, mode=mode)

    # HDD part
    dvc_dpath = dvc_hdd_dpath
    dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath)
    dvc.git_pull()

    sync_checkpoints(dvc_dpath, mode='list')
    sync_checkpoints(dvc_dpath, mode=mode)

    eval_df = evaluation_state(dvc_dpath)


def sync_checkpoints(dvc_dpath, mode='list'):
    from watch.tasks.fusion import repackage

    dataset_codes = [
        'Cropped-Drop3-TA1-2022-03-10',
        'Aligned-Drop3-TA1-2022-03-10',
        'Aligned-Drop3-L1',
    ]
    # hack for old scheme
    storage_repl = {
        'Aligned-Drop3-TA1-2022-03-10': 'eval3_candidates',
        'Cropped-Drop3-TA1-2022-03-10': 'eval3_sc_candidates',
    }

    train_coded_paths = list((dvc_dpath / "training").glob('*/*/*'))
    code_to_path = {p.name: p for p in train_coded_paths}

    for dataset_code in dataset_codes:
        print(f'dataset_code={dataset_code}')
        path = code_to_path.get(dataset_code, None)
        if path is not None:
            storage_code = storage_repl.get(dataset_code, dataset_code)
            train_dpath = str(path / 'runs/*')
            storage_dpath = dvc_dpath / 'models/fusion' / storage_code

            repackage.gather_checkpoints(
                dvc_dpath=dvc_dpath, storage_dpath=storage_dpath,
                train_dpath=train_dpath,
                mode=mode)


def evaluation_state(dvc_dpath):
    """
    Get a list of dictionaries with information for each known evaluation.

    Information includes its real path if it exists, its dvc path if it exists
    and what sort of actions need to be done to synchronize it.
    """
    eval_rows = []
    for type, suffix in EVAL_GLOB_PATTERNS.items():
        raw_pat = str(dvc_dpath / suffix)
        dvc_pat = raw_pat + '.dvc'
        found_raw = list(glob.glob(raw_pat))
        found_dvc = list(glob.glob(dvc_pat))
        lut = {k: {'raw': k, 'dvc': None} for k in found_raw}
        for found_dvc in found_dvc:
            k = found_dvc[:-4]
            row = lut.setdefault(k, {})
            row.setdefault('raw', None)
            row['dvc'] = found_dvc
        rows = list(lut.values())
        for row in rows:
            row['type'] = type
        eval_rows.extend(rows)

    for row in eval_rows:
        row['has_dvc'] = (row['dvc'] is not None)
        row['has_raw'] = (row['raw'] is not None)
        row['has_both'] = row['has_dvc'] and row['has_raw']

        row['needs_pull'] = row['has_dvc'] and not row['has_raw']

        row['is_link'] = None
        row['unprotected'] = None
        row['needs_push'] = None

        if row['has_raw']:
            p = ub.Path(row['raw'])
            row['is_link'] = p.is_symlink()
            row['needs_push'] = not row['has_dvc']
            if row['has_dvc']:
                row['unprotected'] = not row['is_link']

    import pandas as pd
    eval_df = pd.DataFrame(eval_rows)
    print(eval_df.drop(['type', 'raw', 'dvc'], axis=1).sum().to_frame().T)
    print(eval_df.groupby('type').sum())
    return eval_df


def pull_all_evals(dvc_dpath):
    dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath)
    dvc.git_pull()
    eval_df = evaluation_state(dvc_dpath)
    pull_fpaths = eval_df[eval_df.needs_pull]['dvc'].tolist()
    dvc.pull(pull_fpaths, remote='aws')


def commit_unstaged_evals(dvc_dpath):
    dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath)

    eval_df = evaluation_state(dvc_dpath)

    is_weird = (eval_df.is_link & (~eval_df.has_dvc))
    weird_df = eval_df[is_weird]
    if len(weird_df):
        print(f'weird_df=\n{weird_df}')

    to_push = eval_df[eval_df.needs_push == True]  # NOQA
    assert not to_push['has_dvc'].any()
    to_push_fpaths = to_push['raw'].tolist()
    print(f'to_push=\n{to_push}')

    dvc.add(to_push_fpaths)
    dvc.git_commitpush(f'Sync models from {platform.node()}')
    dvc.push(to_push_fpaths, remote='aws')
