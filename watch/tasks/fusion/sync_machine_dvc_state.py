"""
Synchronize DVC states across the machine.
"""
import glob
import ubelt as ub
import platform
from watch.utils import simple_dvc


EVAL_GLOB_PATTERNS = [
    'models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json',
    'models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json',
    'models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/actclf/*/*_eval/scores/merged/summary3.json',
    'models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json',
]


def main():
    import watch
    dvc_hdd_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    dvc_ssd_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
    dvc_dpaths = [
        dvc_ssd_dpath,
        dvc_hdd_dpath
    ]
    dvc_dpath = dvc_hdd_dpath
    dvc_dpath = dvc_ssd_dpath


def evaluation_state(dvc_dpath):
    found = []
    for s in EVAL_GLOB_PATTERNS:
        raw_pat = str(dvc_dpath / s)
        dvc_pat = raw_pat + '.dvc'
        found_raw = list(glob.glob(raw_pat))
        found_dvc = list(glob.glob(dvc_pat))
        lut = {k: {'raw': k, 'dvc': None} for k in found_raw}
        for found_dvc in found_dvc:
            k = found_dvc[:-4]
            row = lut.setdefault(k, {})
            row.setdefault('raw', None)
            row['dvc'] = found_dvc
        found.extend(list(lut.values()))

    for row in found:
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
    df = pd.DataFrame(found)
    print(df.sum())
    return found


def pull_all_evals(dvc_dpath):
    dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath)
    dvc.git_pull()
    found = evaluation_state(dvc_dpath)

    pull_rows = [row for row in found if row['needs_pull']]
    pull_fpaths = [row['dvc'] for row in pull_rows]

    dvc.pull(pull_fpaths, remote='aws')


def commit_unstaged_evals(dvc_dpath):
    dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath)

    found = evaluation_state(dvc_dpath)

    for row in found:
        if not row['is_link']:
            assert not row['has_dvc'], 'probably not tracked'

    unstaged = []
    for p in found:
        path = ub.Path(p)
        if not path.is_symlink():
            unstaged.append(path)

    for path in unstaged:
        dvc_path = path.augment(tail='.dvc')
        assert not dvc_path.exists()

    dvc.add(unstaged)
    dvc.git_commitpush(f'Sync models from {platform.node()}')
    dvc.push(unstaged, remote='aws')
