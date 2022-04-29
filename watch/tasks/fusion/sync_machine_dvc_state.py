"""
Synchronize DVC states across the machine.
"""
import glob
import ubelt as ub
import platform
import scriptconfig as scfg
from watch.utils import simple_dvc

DATASET_CODES = [
    'Cropped-Drop3-TA1-2022-03-10',
    'Aligned-Drop3-TA1-2022-03-10',
    'Aligned-Drop3-L1',
]

# hack for old scheme
STORAGE_REPL = {
    'Aligned-Drop3-TA1-2022-03-10': 'eval3_candidates',
    'Cropped-Drop3-TA1-2022-03-10': 'eval3_sc_candidates',
}


EVAL_GLOB_PATTERNS = {
    'pxl_ta1_10': 'models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json',
    'trk_ta1_10': 'models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json',

    'pxl_l1_10': 'models/fusion/Aligned-Drop3-L1/eval/*/*/*/*/eval/curves/measures2.json',
    'trk_l1_10': 'models/fusion/Aligned-Drop3-L1/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json',

    'pxl_ta1_1': 'models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/curves/measures2.json',
    'act_ta1_1': 'models/fusion/eval3_sc_candidates/eval/*/*/*/*/eval/actclf/*/*_eval/scores/merged/summary3.json',
}


class SyncMachineConfig(scfg.Config):
    """
    Certain parts of these names have special nomenclature to make them easier
    to work with in Python and Bash.

    The "watch" module comes with a few nice command line programs. Given a
    machine with the "watch" environment, the watch DVC repo is accessed as
    follows:

        DVC_DPATH=$(smartwatch_dvc)

    The workdir is where a user on a machine puts all of their experiments.

        WORKDIR=$DVC_DPATH/training/$HOSTNAME/$USER

    Before we start an experment, we must choose a dataset. Lets use an
    example:

        DATASET_CODE=Aligned-Drop3-L1

    Along with the DVC directory, this should uniquely specify a kwcoco dataset
    bundle (although it might not specify the specific view of that dataset,
    -- views can have different GSD or be bundle subsets). The directory
    of this bundle should be:

        KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE

    and it should be the case that, there is a kwcoco manifest that describes
    the entire bundle called:

        MAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data.kwcoco.json

    But we may have precomputed splits also e.g

        TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_train.kwcoco.json
        VALI_FPATH=$KWCOCO_BUNDLE_DPATH/data_nowv_vali.kwcoco.json

    Each experiment should also be given a name:

        EXPERIMENT_NAME=Drop3_L1_BASELINE_BAS_V001

    For each experiment you choose an experiment name on a dataset.

        DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

    The packaging part of this script works with

        $DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/$EXPT_MODEL_GLOBNAME/*.pt
    """
    default = {
        'push': scfg.Value(True, help='if True, will push results to the dvc_remote'),
        'pull': scfg.Value(True, help='if True, will pull results to the dvc_remote'),
        'dvc_remote': scfg.Value('aws', help='dvc remote to sync to/from'),

        'packages': scfg.Value(True, help='sync packages'),
        'evals': scfg.Value(True, help='sync evaluations'),

        'dataset_codes': scfg.Value(None, help=ub.paragraph(
            '''
            if unset, will use the defaults, otherwise this should be a list of
            the DVC dataset bundle names that we want to consider.  Note: we do
            make assumptions that the about where these names go in paths.
            We may make this more general in the future.

            Namely:

                # Training runs go here go here.
                <dvc_dpath>/training/*/*/<dataset_code>/runs/<expt_name>/lightning_logs

                # Packages go here.
                <dvc_dpath>/models/fusion/<dataset_code>/packages

                # Evaluations go here.
                <dvc_dpath>/models/fusion/<dataset_code>/eval
            ''')),
    }


class DVCSyncManager(ub.NiceRepr):
    """
    Implements an API around our DVC structure, which can be described as
    follows.

    <dvc_dpath>
        * [<dataset_code>, ...]

        * training
            * <hostname>/<user>/<dataset_code>/runs/<expt_name>/lightning_logs/...

        * models
            * <task>
            * fusion/<storage_code>/packages/<expt_name>/<model_name.pt>
            * fusion/<storage_code>/pred/<expt_name>/pred_<model_name.pt>/***
            * fusion/<storage_code>/eval/<expt_name>/pred_<model_name.pt>/***

    Note:
        moving forward dataset_code and storage_code should always be the same.
        so storage_code=dataset_code. But we have a weird case that we still
        support ATM.

    A breakdown of the packages dir is:
        packages/<expt_name>/<model_name.pt>
    """

    def __nice__(self):
        return str(self.dvc)

    def __init__(self, dvc_dpath, dvc_remote, dataset_codes=None):
        self.dvc_dpath = dvc_dpath
        self.dvc_remote = dvc_remote
        self.dataset_codes = dataset_codes
        self.dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath, remote=dvc_remote)

    def push_evals(self):
        dvc = self.dvc
        eval_df = evaluation_state(self.dvc_dpath)

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
        dvc.push(to_push_fpaths)

    def pull_evals(self):
        dvc = self.dvc
        dvc.git_pull()
        eval_df = evaluation_state(self.dvc_dpath)
        pull_fpaths = eval_df[eval_df.needs_pull]['dvc'].tolist()
        dvc.pull(pull_fpaths)

    def pull_packages(self):
        raise NotImplementedError

    def push_packages(self):
        from watch.tasks.fusion import repackage
        dvc_dpath = self.dvc_dpath
        train_coded_paths = list((dvc_dpath / "training").glob('*/*/*'))
        code_to_path = {p.name: p for p in train_coded_paths}
        mode = 'commit'
        for dataset_code in self.dataset_codes:
            print(f'dataset_code={dataset_code}')
            path = code_to_path.get(dataset_code, None)
            if path is not None:
                storage_code = STORAGE_REPL.get(dataset_code, dataset_code)
                train_dpath = str(path / 'runs/*')
                storage_dpath = dvc_dpath / 'models/fusion' / storage_code / 'packages'
                repackage.gather_checkpoints(
                    dvc_dpath=dvc_dpath, storage_dpath=storage_dpath,
                    train_dpath=train_dpath, dvc_remote=self.dvc_remote, mode=mode)

    def sync(self, push=True, pull=True, evals=True, packages=True):
        if push:
            if packages:
                self.push_packages()
            if evals:
                self.push_evals()


def main(cmdline=True, **kwargs):
    """
    from watch.tasks.fusion.sync_machine_dvc_state import *  # NOQA
    """
    import watch

    config = SyncMachineConfig(cmdline=cmdline, data=kwargs)
    dvc_remote = config['dvc_remote']

    if config['dataset_codes'] is None:
        dataset_codes = DATASET_CODES
    else:
        raise Exception('must be defualt for now')

    dvc_hdd_dpath = watch.find_smart_dvc_dpath(hardware='hdd')

    try:
        dvc_ssd_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
    except Exception:
        ssd_manager = None
    else:
        ssd_manager = DVCSyncManager(
            dvc_ssd_dpath, dvc_remote=dvc_remote, dataset_codes=dataset_codes)

    # If the SSD has stuff, push it, but don't pull to SSD
    if ssd_manager is not None:
        synckw = ub.compatible(config, ssd_manager.sync)
        synckw['pull'] = False
        ssd_manager.sync(**synckw)

    # Do everything to the HDD.
    hdd_manager = DVCSyncManager(
        dvc_hdd_dpath, dvc_remote=dvc_remote, dataset_codes=dataset_codes)
    synckw = ub.compatible(config, hdd_manager.sync)
    hdd_manager.sync(**synckw)


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

    # import numpy as np

    for row in eval_rows:
        row['has_dvc'] = (row['dvc'] is not None)
        row['has_raw'] = (row['raw'] is not None)
        row['has_both'] = row['has_dvc'] and row['has_raw']

        row['needs_pull'] = row['has_dvc'] and not row['has_raw']

        row['is_link'] = False
        row['unprotected'] = False
        row['needs_push'] = False

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


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/sync_machine_dvc_state.py
        python -m watch.tasks.fusion.sync_machine_dvc_state --push=True --pull=False
        python -m watch.tasks.fusion.sync_machine_dvc_state --push=True --pull=False --help
    """
    main(cmdline=True)
