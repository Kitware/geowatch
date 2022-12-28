r"""
This is the CLI for expt_state

Synchronize DVC states across the machine.

This is a new Phase2 Variant of this script.

Example:

    export DVC_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
    export DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
    cd $DVC_EXPT_DPATH

    python -m watch mlops "status" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch mlops "status" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch mlops "pull packages evals" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch mlops "push packages evals"

    python -m watch mlops "status" --dataset_codes Drop4-SC

    python -m watch mlops "list"

    # On training machine
    python -m watch mlops "push packages"
    python -m watch mlops "push packages" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"

    # On testing machine
    python -m watch mlops "pull packages"
    python -m watch mlops "status"

    # Run evals on testing machine
    python -m watch mlops "evaluate" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"

    # On testing machine
    python -m watch mlops "push evals"

    # On analysis machine
    python -m watch mlops "pull evals"


TODO:
    ### Make the Experiment Evaluation Reporter more robust and generalize to
    ### more problems.

    It should quickly show the best models for various metric and it should be
    easy for the user to inspect them further.  For example say the best model
    of interest was:

    MODEL_OF_INTEREST="Drop4_BAS_Retrain_V002_epoch=45-step=23552"
    MODEL_OF_INTEREST="Drop4_BAS_Continue_15GSD_BGR_V004_epoch=78-step=323584"

    # TODO:
    # There is a problem with multiple .pt suffixes, just dont use any

    # You should be able to pull things wrt to that model

    python -m watch mlops "pull packages" --model_pattern="${MODEL_OF_INTEREST}*"
    python -m watch mlops "pull evals" --model_pattern="${MODEL_OF_INTEREST}*"
    python -m watch mlops "status" --model_pattern="${MODEL_OF_INTEREST}*"

Ignore:
    python -m watch mlops "evaluate" \
        --enable_pred=1 \
        --enable_eval=1 \
        --enable_actclf=1 \
        --enable_actclf_eval=1 \
        --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC" \
        --devices="0,1,2,3" --run=1

    python -m watch mlops "evaluate" \
        --bas_thresh=0.0,0.01,0.1 \
        --set_cover_algo=approx,exact \
        --enable_pred=1 \
        --enable_eval=1 \
        --enable_track=1 \
        --enable_iarpa_eval=1 \
        --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC" \
        --devices="0,1" --run=0 \
        --skip_existing=True

        # --enable_track=1 \
        # --enable_iarpa_eval=1 \
"""
import pandas as pd
import ubelt as ub
import platform
import scriptconfig as scfg
from watch.utils import simple_dvc
from watch import heuristics
from watch.mlops.expt_state import ExperimentState, summarize_tables


class ExptManagerConfig(scfg.DataConfig):
    """
    Certain parts of these names have special nomenclature to make them easier
    to work with in Python and Bash.

    The "watch" module comes with a few nice command line programs. Given a
    machine with the "watch" environment, the watch DVC repo is accessed as
    follows:

        DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
        DVC_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")

    The workdir is where a user on a machine puts all of their experiments.

        WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

    Before we start an experment, we must choose a dataset. Lets use an
    example:

        DATASET_CODE=Aligned-Drop3-L1

    Along with the DVC directory, this should uniquely specify a kwcoco dataset
    bundle (although it might not specify the specific view of that dataset,
    -- views can have different GSD or be bundle subsets). The directory
    of this bundle should be:

        KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH/$DATASET_CODE

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

        $DVC_EXPT_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/$EXPT_MODEL_GLOBNAME/*.pt
    """
    command = scfg.Value(None, nargs='*', help='if specified, will overload other options', position=1)

    dvc_remote = scfg.Value('aws', help='dvc remote to sync to/from')

    expt_dvc_dpath = scfg.Value('auto', help='path to the experiment dvc dpath')
    data_dvc_dpath = scfg.Value('auto', help='path to the data dvc dpath')

    model_pattern = scfg.Value('*', help='if specified restrict to models matching this name pattern')

    dataset_codes = scfg.Value(None, nargs='+', help=ub.paragraph(
        '''
        if unset, will use the defaults, otherwise this should be a list of
        the DVC dataset bundle names that we want to consider.  Note: we do
        make assumptions that the about where these names go in paths.
        We may make this more general in the future.

        Namely:

            # Training runs go here go here.
            <expt_dvc_dpath>/training/*/*/<dataset_code>/runs/<expt_name>/lightning_logs

            # Packages go here.
            <expt_dvc_dpath>/models/fusion/<dataset_code>/packages

            # Evaluations go here.
            <expt_dvc_dpath>/models/fusion/<dataset_code>/eval

        NOTE: THIS SPECIFIC FORMAT IS IN HIGH FLUX. DOCS MAY BE OUTDATED
        '''))


def main(cmdline=True, **kwargs):
    """
    from watch.mlops.expt_manager import *  # NOQA
    """
    config = ExptManagerConfig(cmdline=cmdline, data=kwargs)
    print('ExptManagerConfig config = {}'.format(ub.repr2(dict(config), nl=1)))
    command = config['command']

    available_actions = [
        'status', 'evaluate', 'push', 'pull', 'list', 'report',
    ]
    available_targets = [
        'packages', 'evals'
    ]

    actions = []
    targets = []

    if command is not None and command:
        for c in command:
            for a in available_actions:
                if a in c:
                    actions.append(a)
            for t in available_targets:
                if t in c:
                    targets.append(t)

    # print(f'actions={actions}')
    # print(f'targets={targets}')
    # print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    dvc_remote = config['dvc_remote']

    if config['dataset_codes'] is None:
        dataset_codes = heuristics.DATASET_CODES
    else:
        dataset_codes = config['dataset_codes']

    if config['expt_dvc_dpath'] == 'auto':
        config['expt_dvc_dpath'] = heuristics.auto_expt_dvc()
    if config['data_dvc_dpath'] == 'auto':
        config['data_dvc_dpath'] = heuristics.auto_data_dvc()

    expt_dvc_dpath = config['expt_dvc_dpath']
    data_dvc_dpath = config['data_dvc_dpath']

    manager = DVCExptManager(
        expt_dvc_dpath, dvc_remote=dvc_remote, dataset_codes=dataset_codes,
        data_dvc_dpath=data_dvc_dpath, model_pattern=config['model_pattern'])

    if 'pull' in actions:
        manager.pull(targets)

    if 'push' in actions:
        manager.push(targets)

    if 'status' in actions:
        manager.summarize()

    if 'list' in actions:
        manager.list()

    if 'evaluate' in actions:
        self = manager
        for state in self.states:
            state.schedule_evaluation()

    if 'report' in actions:
        self = manager
        from watch.mlops import expt_report
        dvc_manager = manager
        reporter = expt_report.EvaluationReporter(dvc_manager)
        reporter.load()
        reporter.status()
        reporter.plot()


class DVCExptManager(ub.NiceRepr):
    """
    Implements an API around our DVC structure, which can be described as
    follows.

    TODO:
        - [ ] If we can somehow generate the output paths based on the
        pipeline, then we will be in a very good position.

    <data_dvc_dpath>
        * [<dataset_code>, ...]

    <expt_dvc_dpath>
        * training
            * <hostname>/<user>/<dataset_code>/runs/<expt_name>/lightning_logs/...

        * models
            * <task>
            * fusion/<dataset_code>/packages/<expt_name>/<model_name.pt>
            * fusion/<dataset_code>/pred/<expt_name>/<model_name.pt>/***
            * fusion/<dataset_code>/eval/<expt_name>/<model_name.pt>/***

    A breakdown of the packages dir is:
        packages/<expt_name>/<model_name.pt>

    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_EXPT_DPATH)
        >>> from watch.mlops.expt_manager import *  # NOQA
        >>> import watch
        >>> manager = DVCExptManager.coerce(watch.find_dvc_dpath(tags='phase2_expt'))
        >>> manager.summarize()

    Ignore:
        broke = df[df['is_broken']]
    """

    def __nice__(manager):
        return str(manager.dvc)

    def __init__(manager, expt_dvc_dpath, dvc_remote='aws', dataset_codes=None,
                 data_dvc_dpath=None, model_pattern='*'):
        manager.model_pattern = model_pattern
        manager.expt_dvc_dpath = expt_dvc_dpath
        manager.data_dvc_dpath = data_dvc_dpath
        manager.dvc_remote = dvc_remote
        manager.dataset_codes = dataset_codes
        manager.dvc = simple_dvc.SimpleDVC.coerce(expt_dvc_dpath, remote=dvc_remote)
        manager._build_states()

    def summarize(manager):
        # for state in manager.states:
        #     state.summarize()
        tables = manager.cross_referenced_tables()
        summarize_tables(tables)

    def list(manager):
        tables = manager.cross_referenced_tables()

        if 'staging' in tables:
            todrop = ['expt_dvc_dpath', 'raw', 'ckpt_path', 'spkg_fpath', 'pkg_fpath', 'lightning_version', 'ckpt_exists']
            df = tables['staging']
            print(df.drop(ub.oset(todrop) & df.columns, axis=1).to_string())

        if 'volitile' in tables:
            volitile_drop = ['expt_dvc_dpath', 'raw', '']
            todrop = volitile_drop
            df = tables['volitile']
            print(df.drop(ub.oset(todrop) & df.columns, axis=1).to_string())

        if 'versioned' in tables:
            volitile_drop = ['raw', 'dvc', 'expt_dvc_dpath', 'expt',
                             'is_broken', 'is_link', 'has_raw', 'has_dvc',
                             'unprotected', 'needs_pull', 'needs_push',
                             'has_orig']
            todrop = volitile_drop
            df = tables['versioned']
            type_to_versioned = dict(list(df.groupby('type')))
            for type, subdf in type_to_versioned.items():
                print(f'type={type}')
                print(subdf.drop(ub.oset(todrop) & df.columns, axis=1).to_string())

    @classmethod
    def coerce(cls, expt_dvc_dpath=None):
        import watch
        if expt_dvc_dpath is None:
            expt_dvc_dpath = watch.find_smart_dvc_dpath()
        dvc_remote = 'aws'
        dataset_codes = heuristics.DATASET_CODES
        manager = cls(expt_dvc_dpath=expt_dvc_dpath, dvc_remote=dvc_remote,
                      dataset_codes=dataset_codes)
        return manager

    def _build_states(manager):
        states = []
        for dataset_code in manager.dataset_codes:
            state = ExperimentState(
                manager.expt_dvc_dpath, dataset_code, dvc_remote=manager.dvc_remote,
                data_dvc_dpath=manager.data_dvc_dpath,
                model_pattern=manager.model_pattern)
            states.append(state)
        manager.states = states

    def versioned_table(manager, **kw):
        rows = list(ub.flatten(state.versioned_rows(**kw) for state in manager.states))
        df = pd.DataFrame(rows)
        missing = ub.oset(ExperimentState.VERSIONED_COLUMNS) - df.columns
        if len(missing):
            df.loc[:, missing] = None
        return df

    def volitile_table(manager, **kw):
        rows = list(ub.flatten(state.volitile_table(**kw) for state in manager.states))
        df = pd.DataFrame(rows)
        missing = ub.oset(ExperimentState.VOLITILE_COLUMNS) - df.columns
        if len(missing):
            df.loc[:, missing] = None
        return df

    def evaluation_table(manager):
        rows = list(ub.flatten(state.evaluation_rows() for state in manager.states))
        df = pd.DataFrame(rows)
        return df

    def cross_referenced_tables(manager):
        import pandas as pd
        table_accum = ub.ddict(list)
        for state in manager.states:
            tables = state.cross_referenced_tables()
            for k, v in tables.items():
                if not v.empty:
                    # col 'n_pred_act_poly_sites_fpath' can be duplicated
                    table_accum[k].append(
                        v.loc[:, ~v.columns.duplicated()]
                    )
        combo_tables = ub.udict(table_accum).map_values(lambda vs: pd.concat(vs))
        return combo_tables

    def push_evals(manager):
        dvc = manager.dvc
        eval_df = manager.evaluation_table()
        summarize_tables({'versioned': eval_df})

        is_weird = (eval_df.is_link & (~eval_df.has_dvc))
        weird_df = eval_df[is_weird]
        if len(weird_df):
            print(f'weird_df=\n{weird_df}')

        # Determine what evaluations need to be added to DVC
        to_add = eval_df[~eval_df['has_dvc']]
        if len(to_add):
            to_add_paths = to_add['raw']
            # paths = to_add_paths
            dvc.add(to_add_paths)
            ...

        to_push = eval_df[eval_df.needs_push == True]  # NOQA
        assert not to_push['has_dvc'].any()

        # TODO: if we want to allow modifications we need to find
        # unprotected files (or changed files on non-symlink dvc repos)
        # to_push = eval_df[(eval_df.needs_push == True) | (eval_df.unprotected == True)]  # NOQA

        to_push_fpaths = to_push['raw'].tolist()
        print(f'to_push=\n{to_push}')
        if len(to_push_fpaths):
            # dvc.add(to_push_fpaths)
            dvc.git_commitpush(f'Sync evals from {platform.node()}')
            dvc.push(to_push_fpaths)

    def pull_evals(manager):
        manager.dvc.git_pull()

        eval_df = manager.evaluation_table()
        summarize_tables({'versioned': eval_df})

        # manager.summarize()
        print(f'manager.expt_dvc_dpath={manager.expt_dvc_dpath}')
        print(len(eval_df))
        if len(eval_df) > 0:
            eval_df = eval_df[~eval_df['is_broken']]
            pull_rows = eval_df[eval_df.needs_pull]
            pull_fpaths = pull_rows['dvc'].tolist()
        else:
            pull_fpaths = []
        print(f'{len(pull_fpaths)=}')
        for p in pull_fpaths:
            assert p.exists()
        manager.dvc.pull(pull_fpaths)

    def pull_packages(manager):
        # Assume just one git repo and manually pull
        manager.dvc.git_pull()

        pkg_df = manager.versioned_table(types=['pkg_fpath'])
        pull_df = pkg_df[pkg_df['needs_pull'].astype(bool)]
        pull_fpaths = pull_df['dvc'].tolist()
        manager.dvc.pull(pull_fpaths)

    def push_packages(manager):
        """
        TODO: break this up into smaller components.
        """
        # from watch.tasks.fusion import repackage
        # mode = 'commit'
        for state in manager.states:
            state.push_packages()

    def push(manager, targets):
        if 'packages' in targets:
            manager.push_packages()

        if 'evals' in targets:
            manager.push_evals()

    def pull(manager, targets):
        if 'packages' in targets:
            manager.pull_packages()
        if 'evals' in targets:
            manager.pull_evals()

    def reverse_hash_lookup(manager, key):
        # This probably doesn't belong here
        from watch.utils.reverse_hashid import ReverseHashTable
        ReverseHashTable.query(key, verbose=1)


# TODO: can we hook into DVC more efficiently to query this?
# def _check_ignore_tables(paths, dvc):
#     import os
#     dvc_root = dvc.dvc_root
#     @ub.memoize
#     def make_gitignore_pattern(root):
#         from watch.utils import util_pattern
#         # ignore_fpath = (root / '.gitignore')
#         ignore_fpath = (root / '.dvcignore')
#         if ignore_fpath.exists():
#             ignore_lines = [p.strip() for p in ignore_fpath.read_text().split('\n')
#                             if not p.startswith('#')]
#             ignore_pats = [str(p) for p in ignore_lines if p]
#             pat = util_pattern.MultiPattern.coerce(ignore_pats, hint='glob')
#             return pat
#     for path in ub.ProgIter(paths):
#         rel_path = path.relative_to(dvc_root).parent
#         for i in reversed(range(len(rel_path.parts))):
#             rel_root = dvc_root / ub.Path(*rel_path.parts[0:i])
#             rel_pat = make_gitignore_pattern(rel_root)
#             if rel_pat is not None:
#                 print(f'rel_root={rel_root}')
#                 suffix_path = os.fspath(path.relative_to(rel_root))
#                 if rel_pat.match(suffix_path):
#                     raise Exception


__config__ = ExptManagerConfig


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/mlops/expt_manager.py "pull all"
        python -m watch.mlops.expt_manager "status"
        python -m watch.mlops.expt_manager "push all"
        python -m watch.mlops.expt_manager "pull evals"

        python -m watch.mlops.expt_manager "evaluate"

        python -m watch.mlops.expt_manager "pull all"
        python -m watch.mlops.expt_manager "pull packages"
    """
    main(cmdline=True)
