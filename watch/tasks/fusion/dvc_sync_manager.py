"""
Synchronize DVC states across the machine.

Example:
    python -m watch.tasks.fusion.dvc_sync_manager "pull evals"
"""
import pandas as pd
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


class WatchDVCState:
    def __init__(self, dvc_dpath):
        pass


class ExperimentState:
    """
    Ignore:
        >>> from watch.tasks.fusion.dvc_sync_manager import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> dataset_code = 'Aligned-Drop3-TA1-2022-03-10'
        >>> self = ExperimentState(dvc_dpath, dataset_code)
        >>> gen = self.measure_rows()
        >>> row = ub.peek(gen)
        >>> table = self.measure_table()
        >>> print(table)
    """
    def __init__(self, dvc_dpath, dataset_code, storage_code=None):
        self.dvc_dpath = dvc_dpath
        self.dataset_code = dataset_code
        if storage_code is None:
            storage_code = STORAGE_REPL.get(dataset_code, dataset_code)
        self.storage_code = storage_code
        self.storage_dpath = self.dvc_dpath / 'models/fusion' / storage_code

        self.patterns = {
            'expt': '*',
            'dset': '*',
            'model': '*',
            'pred_cfg': '*',
            'trk_cfg': '*',
            'act_cfg': '*',
        }
        self.measure_templates = {
            'pxl': 'eval/{expt}/{model}/{dset}/{pred_cfg}/eval/curves/measures2.json',
            'trk': 'eval/{expt}/{model}/{dset}/{pred_cfg}/eval/tracking/{trk_cfg}/iarpa_eval/scores/merged/summary2.json',
            'act': 'eval/{expt}/{model}/{dset}/{pred_cfg}/eval/actclf/{act_cfg}/iarpa_sc_eval/scores/merged/summary3.json',
        }
        self.measure_patterns = {}
        self._build_path_patterns()

    def _build_path_patterns(self):
        self.measure_patterns = {
            k: self.storage_dpath / v.format(**self.patterns)
            for k, v in self.measure_templates.items()}

    @classmethod
    def _dvcglob(cls, pat):
        """
        Ignore:
            >>> import watch
            >>> dvc_dpath = watch.find_smart_dvc_dpath()
            >>> bundle_dpath = dvc_dpath / 'deprecated/drop1-S2-L8-aligned'
            >>> list(ExperimentState._dvcglob(bundle_dpath / '*'))
        """
        from watch.utils import util_pattern
        import os
        pat = os.fspath(pat)
        mpat = util_pattern.Pattern.coerce(pat)
        default = {'raw': None, 'dvc': None}
        id_to_row = ub.ddict(default.copy)
        paths = list(map(ub.Path, mpat.paths(recursive=0)))
        dvc_ext = '.dvc'
        len_ext = len(dvc_ext)
        for path in paths:
            parent = path.parent
            name = path.name
            if name.endswith(dvc_ext):
                type = 'dvc'
                raw_path = parent / name[:-len_ext]
                # dvc_path = path
            else:
                type = 'raw'
                raw_path = path
                # dvc_path = parent / (name + '.dvc')
            row = id_to_row[raw_path]
            row[type] = path
            yield row

    def measure_rows(self, attrs=1):
        keys = ['pxl', 'act', 'trk']
        for key in keys:
            pat = self.measure_patterns['pxl']
            for row in self._dvcglob(pat):
                row['type'] = key
                row['has_dvc'] = (row['dvc'] is not None)
                row['has_raw'] = (row['raw'] is not None)
                row['needs_pull'] = row['has_dvc'] and not row['has_raw']
                row['is_link'] = False
                row['unprotected'] = False
                row['needs_push'] = False
                if attrs:
                    row['dataset_code'] = self.dataset_code
                    row['dataset_code'] = self.dataset_code

                if row['has_raw']:
                    p = ub.Path(row['raw'])
                    row['is_link'] = p.is_symlink()
                    row['needs_push'] = not row['has_dvc']
                    if row['has_dvc']:
                        row['unprotected'] = not row['is_link']
                yield row

    def measure_table(self):
        """
        Get a list of dictionaries with information for each known evaluation.

        Information includes its real path if it exists, its dvc path if it exists
        and what sort of actions need to be done to synchronize it.
        """
        eval_rows = list(self.measure_rows())
        # import numpy as np
        eval_df = pd.DataFrame(eval_rows)
        print(eval_df.drop(['type', 'raw', 'dvc'], axis=1).sum().to_frame().T)
        print(eval_df.groupby('type').sum())
        return eval_df


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
        'command': scfg.Value(None, help='if specified, will overload other options', position=1),
        'push': scfg.Value(True, help='if True, will push results to the dvc_remote'),
        'pull': scfg.Value(True, help='if True, will pull results to the dvc_remote'),

        'packages': scfg.Value(True, help='sync packages'),
        'evals': scfg.Value(True, help='sync evaluations'),

        'dvc_remote': scfg.Value('aws', help='dvc remote to sync to/from'),

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

    Example:
        >>> from watch.tasks.fusion.dvc_sync_manager import *  # NOQA
        >>> # Default config is used if not provided
        >>> self = DVCSyncManager.coerce()
    """

    def __nice__(self):
        return str(self.dvc)

    def __init__(self, dvc_dpath, dvc_remote='aws', dataset_codes=None):
        self.dvc_dpath = dvc_dpath
        self.dvc_remote = dvc_remote
        self.dataset_codes = dataset_codes
        self.dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath, remote=dvc_remote)
        self._evaluation_state()

    @classmethod
    def coerce(cls):
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
        dvc_remote = 'aws'
        dataset_codes = DATASET_CODES
        self = cls(dvc_dpath=dvc_dpath, dvc_remote=dvc_remote,
                   dataset_codes=dataset_codes)
        return self

    def _evaluation_state(self):
        states = []
        for dataset_code in self.dataset_codes:
            state = ExperimentState(self.dvc_dpath, dataset_code)
            states.append(state)
        self.states = states

    def evaluation_state(self):
        rows = list(ub.flatten(state.measure_rows() for state in self.states))
        df = pd.DataFrame(rows)
        return df

    def push_evals(self):
        dvc = self.dvc
        eval_df = self.evaluation_state()

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
        eval_df = self.evaluation_state()
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
        if pull:
            # if packages:
            #     self.pull_packages()
            if evals:
                self.pull_evals()


def main(cmdline=True, **kwargs):
    """
    from watch.tasks.fusion.dvc_sync_manager import *  # NOQA
    """
    import watch

    config = SyncMachineConfig(cmdline=cmdline, data=kwargs)
    command = config['command']
    if command is not None:
        config['push'] = False
        config['pull'] = False
        config['evals'] = False
        config['packages'] = False
        if 'pull' in command:
            config['pull'] = True
        if 'push' in command:
            config['push'] = True
        if 'evals' in command:
            config['evals'] = True
        if 'packages' in command:
            config['packages'] = True

    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    dvc_remote = config['dvc_remote']

    if config['dataset_codes'] is None:
        dataset_codes = DATASET_CODES
    else:
        raise Exception('must be defualt for now')

    # If we have an SSD, and it has stuff, push it, but don't pull to SSD
    try:
        dvc_ssd_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
    except Exception:
        ssd_manager = None
    else:
        ssd_manager = DVCSyncManager(
            dvc_ssd_dpath, dvc_remote=dvc_remote, dataset_codes=dataset_codes)
    if ssd_manager is not None:
        synckw = ub.compatible(config, ssd_manager.sync)
        synckw['pull'] = False
        ssd_manager.sync(**synckw)

    # Do everything to the HDD.
    try:
        dvc_hdd_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    except Exception:
        dvc_hdd_dpath = watch.find_smart_dvc_dpath()
    hdd_manager = DVCSyncManager(
        dvc_hdd_dpath, dvc_remote=dvc_remote, dataset_codes=dataset_codes)
    synckw = ub.compatible(config, hdd_manager.sync)
    hdd_manager.sync(**synckw)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/dvc_sync_manager.py
        python -m watch.tasks.fusion.dvc_sync_manager --push=True --pull=False
        python -m watch.tasks.fusion.dvc_sync_manager --push=True --pull=False --help
    """
    main(cmdline=True)
