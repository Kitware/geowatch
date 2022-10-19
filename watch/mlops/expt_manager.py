r"""
Synchronize DVC states across the machine.

This is a new Phase2 Variant of this script.

Example:

    export DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data")
    export DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
    cd $DVC_EXPT_DPATH

    python -m watch mlops "status" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch mlops "status" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch mlops "pull packages evals" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch mlops "push packages evals"

    python -m watch mlops "status"

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
import warnings
import parse
import pandas as pd
import ubelt as ub
import platform
import scriptconfig as scfg
from watch.utils import simple_dvc
from watch.utils import util_pattern
from watch.utils import util_path
from watch import heuristics


class ExptManagerConfig(scfg.Config):
    """
    Certain parts of these names have special nomenclature to make them easier
    to work with in Python and Bash.

    The "watch" module comes with a few nice command line programs. Given a
    machine with the "watch" environment, the watch DVC repo is accessed as
    follows:

        EXPT_DVC_DPATH=$(smartwatch_dvc --tags="phase2_expt")
        DATA_DVC_DPATH=$(smartwatch_dvc --tags="phase2_data")

    The workdir is where a user on a machine puts all of their experiments.

        WORKDIR=$EXPT_DVC_DPATH/training/$HOSTNAME/$USER

    Before we start an experment, we must choose a dataset. Lets use an
    example:

        DATASET_CODE=Aligned-Drop3-L1

    Along with the DVC directory, this should uniquely specify a kwcoco dataset
    bundle (although it might not specify the specific view of that dataset,
    -- views can have different GSD or be bundle subsets). The directory
    of this bundle should be:

        KWCOCO_BUNDLE_DPATH=$DATA_DVC_DPATH/$DATASET_CODE

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

        $EXPT_DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/$EXPT_MODEL_GLOBNAME/*.pt
    """
    default = {
        'command': scfg.Value(None, nargs='*', help='if specified, will overload other options', position=1),

        'dvc_remote': scfg.Value('aws', help='dvc remote to sync to/from'),

        'expt_dvc_dpath': scfg.Value('auto', help='path to the experiment dvc dpath'),
        'data_dvc_dpath': scfg.Value('auto', help='path to the data dvc dpath'),

        'model_pattern': scfg.Value('*', help='if specified restrict to models matching this name pattern'),

        'dataset_codes': scfg.Value(None, nargs='+', help=ub.paragraph(
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
            ''')),
    }


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


# TODO: rename to DVCMultiStateManager and DVCStateManager
class DVCExptManager(ub.NiceRepr):
    """
    Implements an API around our DVC structure, which can be described as
    follows.

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
        >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
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
                table_accum[k].append(v)
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
        dvc = manager.dvc
        dvc.git_pull()
        eval_df = manager.evaluation_table()
        summarize_tables({'versioned': eval_df})

        # manager.summarize()
        print(f'manager.expt_dvc_dpath={manager.expt_dvc_dpath}')
        print(len(eval_df))
        eval_df = eval_df[~eval_df['is_broken']]
        pull_rows = eval_df[eval_df.needs_pull]
        pull_fpaths = pull_rows['dvc'].tolist()
        print(f'{len(pull_fpaths)=}')
        for p in pull_fpaths:
            assert p.exists()
        dvc.pull(pull_fpaths)

    def pull_packages(manager):
        # TODO: git pull
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


class ExperimentState(ub.NiceRepr):
    """

    Ignore:
        >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
        >>> from watch.mlops.expt_manager import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
        >>> dvc_remote = 'aws'
        >>> self = ExperimentState(expt_dvc_dpath, dataset_code, dvc_remote, data_dvc_dpath)
        >>> self.summarize()

    Ignore:
        >>> # Just show patterns:
        >>> from watch.mlops.expt_manager import *  # NOQA
        >>> self = ExperimentState('<expt_dpath>', '<dset_code>')
        >>> print('self.templates = {}'.format(ub.repr2(self.templates, nl=1, sort=0)))

    Ignore:
        table[table.type == 'pkg_fpath']['model'].unique()
    """

    def __init__(self, expt_dvc_dpath, dataset_code, dvc_remote=None,
                 data_dvc_dpath=None, model_pattern='*'):

        if isinstance(model_pattern, str) and model_pattern.endswith('.txt') and ub.Path(model_pattern).exists():
            model_pattern = [
                p.strip()
                for p in ub.Path(model_pattern).read_text().split('\n')
                if p.strip()]

        # from watch.mlops.schedule_evaluation import schedule_evaluation
        # from watch.utils import util_pattern
        # model_pattern = util_pattern.MultiPattern.coerce(model_pattern, hint='glob')

        self.expt_dvc_dpath = ub.Path(expt_dvc_dpath)

        if data_dvc_dpath is None:
            import watch
            try:
                data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', envvar='DATA_DVC_DPATH')
            except Exception:
                pass
        self.data_dvc_dpath = data_dvc_dpath
        self.dataset_code = dataset_code
        self.dvc_remote = dvc_remote
        self.training_dpath = self.expt_dvc_dpath / 'training'

        # TODO: the name "fusion" should be a high level task or group not be hard coded.
        # TODO: the name "models" should be configurable. It's the versioning place.
        # We could move the pred out of the models subdir

        # Denote which of the keys represent hashed information that could be
        # looked up via the rlut.
        self.hashed_cfgkeys = [
            'trk_pxl_cfg',
            'trk_poly_cfg',
            'act_pxl_cfg',
            'act_poly_cfg',
            'crop_cfg',
        ]
        self.condensed_keys = self.hashed_cfgkeys + [
            'test_trk_dset',
            'test_act_dset',
            'trk_model',
            'act_model',
            'crop_src_dset',
            'crop_id',
        ]

        ### Experimental, add in SC dependencies
        self.staging_template_prefix = '{expt_dvc_dpath}/training/{host}/{user}/{dataset_code}/'
        self.storage_template_prefix = '{expt_dvc_dpath}/models/fusion/{dataset_code}/'

        self.patterns = {
            # General
            'trk_expt': '*',
            'act_expt': '*',
            'expt_dvc_dpath': expt_dvc_dpath,
            'dataset_code': dataset_code,
            ### Versioned
            'test_trk_dset': '*',
            'test_act_dset': '*',
            'trk_model': model_pattern,  # hack, should have ext
            'act_model': model_pattern,  # hack, should have ext
            'trk_pxl_cfg': '*',
            'trk_poly_cfg': '*',
            'act_pxl_cfg': '*',
            'act_poly_cfg': '*',
            'crop_src_dset': '*',
            'crop_cfg': '*',
            'crop_id': '*',
            'trk_poly_id': '*',
            'regions_id': '*',
            #### Staging
            'host': '*',
            'user': '*',
            'lightning_version': '*',
            'checkpoint': '*',  # hack, should have ext
            'stage_model': '*',  # hack, should have ext
            ### Deprecated
            'model': model_pattern,  # hack, should have ext
            'expt': '*',
        }

        self.staging_templates = {
            'ckpt': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{checkpoint}.ckpt',
            'spkg': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{model}.pt',
        }

        # directory suffixes after the pred/eval type
        task_dpath_suffix = {
            'trk_pxl_dpath'  : 'trk/{trk_model}/{test_trk_dset}/{trk_pxl_cfg}',
            'trk_poly_dpath' : 'trk/{trk_model}/{test_trk_dset}/{trk_pxl_cfg}/{trk_poly_cfg}',

            'act_pxl_dpath'  : 'act/{act_model}/{test_act_dset}/{act_pxl_cfg}',
            'act_poly_dpath' : 'act/{act_model}/{test_act_dset}/{act_pxl_cfg}/{act_poly_cfg}',

            'crop_dpath': 'crop/{crop_src_dset}/{regions_id}/{crop_cfg}/{crop_id}',
        }

        task_dpaths = {
            'pred_trk_pxl_dpath'   : 'pred/' + task_dpath_suffix['trk_pxl_dpath'],
            'pred_trk_poly_dpath'  : 'pred/' + task_dpath_suffix['trk_poly_dpath'],
            'pred_act_pxl_dpath'   : 'pred/' + task_dpath_suffix['act_pxl_dpath'],
            'pred_act_poly_dpath'  : 'pred/' + task_dpath_suffix['act_poly_dpath'],

            'crop_dpath'           : task_dpath_suffix['crop_dpath'],

            'eval_trk_pxl_dpath'   : 'eval/' + task_dpath_suffix['trk_pxl_dpath'],
            'eval_trk_poly_dpath'  : 'eval/' + task_dpath_suffix['trk_poly_dpath'],
            'eval_act_pxl_dpath'   : 'eval/' + task_dpath_suffix['act_pxl_dpath'],
            'eval_act_poly_dpath'  : 'eval/' + task_dpath_suffix['act_poly_dpath'],
        }

        self.volitile_templates = {
            'pred_trk_pxl_fpath'        : task_dpaths['pred_trk_pxl_dpath'] + '/pred.kwcoco.json',
            'pred_trk_poly_kwcoco'      : task_dpaths['pred_trk_poly_dpath'] + '/tracks.kwcoco.json',
            'pred_trk_poly_sites_fpath'          : task_dpaths['pred_trk_poly_dpath'] + '/site_tracks_manifest.json',
            'pred_trk_poly_site_summaries_fpath' : task_dpaths['pred_trk_poly_dpath'] + '/site_summary_tracks_manifest.json',
            'pred_trk_poly_sites_dpath'          : task_dpaths['pred_trk_poly_dpath'] + '/sites',
            'pred_trk_poly_site_summaries_dpath' : task_dpaths['pred_trk_poly_dpath'] + '/site-summaries',
            'pred_trk_poly_viz_stamp' : task_dpaths['pred_trk_poly_dpath'] + '/_viz.stamp',

            'crop_fpath'              : task_dpaths['crop_dpath'] + '/crop.kwcoco.json',

            'pred_act_pxl_fpath'   : task_dpaths['pred_act_pxl_dpath'] + '/pred.kwcoco.json',
            'pred_act_poly_kwcoco' : task_dpaths['pred_act_poly_dpath'] + '/activity_tracks.kwcoco.json',
            'pred_act_poly_sites_fpath'  : task_dpaths['pred_act_poly_dpath'] + '/site_activity_manifest.json',
            # 'pred_act_poly_site_summaries_fpath' : task_dpaths['pred_act_poly_dpath'] + '/site_summary_activity_manifest.json',
            'pred_act_poly_sites_dpath'  : task_dpaths['pred_act_poly_dpath'] + '/sites',
            # 'pred_act_poly_site_summaries_dpath' : task_dpaths['pred_act_poly_dpath'] + '/site-summaries',
            'pred_act_poly_viz_stamp' : task_dpaths['pred_act_poly_dpath'] + '/_viz.stamp',
        }

        self.versioned_templates = {
            # TODO: rename curves to pixel
            'pkg_fpath'            : 'packages/{expt}/{model}.pt',  # by default packages dont know what task they have (because they may have multiple)
            'pkg_trk_pxl_fpath'    : 'packages/{trk_expt}/{trk_model}.pt',
            'pkg_act_pxl_fpath'    : 'packages/{act_expt}/{act_model}.pt',
            'eval_trk_pxl_fpath'   : task_dpaths['eval_trk_pxl_dpath'] + '/curves/measures2.json',
            'eval_trk_poly_fpath'  : task_dpaths['eval_trk_poly_dpath'] + '/merged/summary2.json',
            'eval_act_pxl_fpath'   : task_dpaths['eval_act_pxl_dpath'] + '/curves/measures2.json',
            'eval_act_poly_fpath'  : task_dpaths['eval_act_poly_dpath'] + '/merged/summary3.json',
        }

        # User specified config mapping a formatstr variable to a set of items
        # that will cause a row to be ignored if it has one of those values
        # when a table is being built.
        self.blocklists = {
            k: set() for k in self.patterns.keys()
        }

        self.templates = {}
        for k, v in self.staging_templates.items():
            self.templates[k] = self.staging_template_prefix + v

        for k, v in self.volitile_templates.items():
            self.templates[k] = self.storage_template_prefix + v

        for k, v in self.versioned_templates.items():
            self.templates[k] = self.storage_template_prefix + v

        for k, v in task_dpaths.items():
            self.templates[k] = self.storage_template_prefix + v

        self.path_patterns_matrix = []
        self._build_path_patterns()

        # These are some locations that I used to know
        self.legacy_versioned_templates = {
            (self.storage_template_prefix + 'eval/{trk_expt}/{model}/{test_dset}/{pred_cfg}/eval/curves',
             self.storage_template_prefix + 'eval/{trk_expt}/{model}/{test_dset}/{pred_cfg}/eval/eval_pxl/curves'),
            (self.storage_template_prefix + 'eval/{trk_expt}/{model}/{test_dset}/{pred_cfg}/eval/heatmaps',
             self.storage_template_prefix + 'eval/{trk_expt}/{model}/{test_dset}/{pred_cfg}/eval/eval_pxl/heatmaps'),
            ##
            # Move activity metrics to depend on pred_pxl_cfg, trk_cfg and
            ##
            # (self.storage_template_prefix + 'eval/{expt}/{model}/{test_dset}/{pred_cfg}/eval/actclf/',
            #  self.storage_template_prefix + 'eval/{expt}/{model}/{test_dset}/{pred_cfg}/eval/tracking/truth/actclf/'),
        }

    def _make_cross_links(self):
        # Link between evals and predictions
        eval_rows = list(self.evaluation_rows())
        num_links = 0
        for row in ub.ProgIter(eval_rows, desc='linking evals and preds'):
            if row['has_raw']:
                eval_type = row['type']
                pred_type = eval_type.replace('eval', 'pred')
                eval_dpath = ub.Path(self.templates[eval_type + '_dpath'].format(**row))
                pred_dpath = ub.Path(self.templates[pred_type + '_dpath'].format(**row))
                if eval_dpath.exists() and pred_dpath.exists():
                    pred_lpath = eval_dpath / '_pred_link'
                    eval_lpath = pred_dpath / '_eval_link'
                    ub.symlink(pred_dpath, pred_lpath, verbose=1, overwrite=True)
                    ub.symlink(eval_dpath, eval_lpath, verbose=1, overwrite=True)
                    num_links += 1
        print(f'made {num_links} links')

    VERSIONED_COLUMNS = [
        'type', 'has_dvc', 'has_raw', 'needs_pull', 'is_link', 'is_broken',
        'unprotected', 'needs_push', 'raw', 'dvc', 'dataset_code']

    VOLITILE_COLUMNS = [
        'type', 'raw', 'model', 'dataset_code'
    ]

    STAGING_COLUMNS = [
        'ckpt_exists', 'is_packaged', 'is_copied', 'needs_package', 'needs_copy']

    def _build_path_patterns(self):
        _patterns = ub.udict(self.patterns).map_values(
            lambda x: x if ub.iterable(x) else [x])
        self._pattern_matrix = list(ub.named_product(_patterns))

        self.path_patterns_matrix = [
            ub.udict({
                k: v.format(**patterns)
                for k, v in self.templates.items()
            })
            for patterns in self._pattern_matrix
        ]
        # print('self.path_patterns_matrix = {}'.format(ub.repr2(self.path_patterns_matrix, nl=1)))

    def __nice__(self):
        return self.dataset_code

    def _parse_pattern_attrs(self, template, path):
        row = {}
        parser = parse.Parser(str(template))
        results = parser.parse(str(path))
        if results is None:
            raise AssertionError(f'Failed to match path={path} to template={template}')
            parser = parse.Parser(str(template)[:-4])
            results = parser.parse(str(path))
        if results is not None:
            row.update(results.named)
        else:
            warnings.warn('warning: bad attrs')
        return row

    def relevant_reverse_hashes(self):
        raise NotImplementedError

    def _block_non_existing_rhashes(self):
        # TODO: helper, could be refactored
        state = self
        state._build_path_patterns()
        orig_eval_table = state.evaluation_table()
        for cfgkey in state.hashed_cfgkeys:
            if cfgkey in orig_eval_table:
                unique_keys = orig_eval_table[cfgkey].dropna().unique()
                for key in unique_keys:
                    from watch.utils.reverse_hashid import ReverseHashTable
                    candidates = ReverseHashTable.query(key, verbose=0)
                    if not candidates:
                        state.blocklists[cfgkey].add(key)

    def staging_rows(self):
        """
        A staging item are items that are the result of non-deterministic
        processes like training. These are not versioned or recomputable.
        These are things in the training directory that need to be repackaged
        or copied into the versioned folder.
        """
        # Gather checkpoints and packages from the training directory.
        # Some checkpoints may not have been repackaged yet.
        # Some packages may have had their checkpoints deleted.
        # None of these files are in DVC, this is entirely volitile state.
        default = {'ckpt_path': None, 'spkg_fpath': None}
        _id_to_row = ub.ddict(default.copy)

        rows = []
        key = 'ckpt'
        for pat in [p[key] for p in self.path_patterns_matrix]:
            mpat = util_pattern.Pattern.coerce(pat)
            # Find all checkpoints
            for ckpt_path in list(mpat.paths()):
                if ckpt_path.suffix != '.ckpt':
                    continue
                row = default.copy()
                row['ckpt_path'] = ckpt_path
                row['type'] = 'ckpt'
                row['is_packaged'] = False
                row['ckpt_exists'] = True

                _attrs = self._parse_pattern_attrs(self.templates[key], ckpt_path)
                row.update(_attrs)
                rows.append(row)
                _id_to_row[ckpt_path] = row

        # Find repackaged checkpoints
        key = 'spkg'  # stands for staged package
        for pat in [p[key] for p in self.path_patterns_matrix]:
            mpat = util_pattern.Pattern.coerce(pat)
            for spkg_fpath in list(mpat.paths()):
                # Does this correspond to an existing checkpoint?
                _attrs = self._parse_pattern_attrs(self.templates[key], spkg_fpath)

                # Hack: making assumption about naming pattern
                spkg_stem = spkg_fpath.stem
                ckpt_stem = ''.join(spkg_stem.partition('_epoch')[-2:])[1:]
                ckpt_path = spkg_fpath.parent / (ckpt_stem + '.ckpt')

                if ckpt_path.exists():
                    # Modify existing row
                    row = _id_to_row[ckpt_path]
                else:
                    # Add new row
                    row = default.copy()
                    row['checkpoint'] = ckpt_stem
                    row['ckpt_exists'] = False
                    row['type'] = 'ckpt'
                    rows.append(row)
                row['spkg_fpath'] = spkg_fpath
                row['is_packaged'] = True
                row.update(_attrs)

        for row in rows:
            fname = row['checkpoint']

            if row.get('spkg_fpath', None) is None:
                # HACK!!!
                row['model'] = None
            else:
                row['model'] = ub.Path(row['spkg_fpath']).name

            # Hack: making name assumptions
            info = checkpoint_filepath_info(fname)
            row.update(info)

            # Where would we expect to put this file?
            kw = ub.udict(row).subdict({'expt', 'model'})
            kw['expt_dvc_dpath'] = self.expt_dvc_dpath
            kw['dataset_code'] = self.dataset_code
            row['pkg_fpath'] = ub.Path(self.templates['pkg_fpath'].format(**kw))
            row['is_copied'] = row['pkg_fpath'].exists()

        return rows

    def volitile_rows(self):
        """
        A volitile item is something that is derived from something versioned
        (so it is recomputable), but it is not versioned itself. These are
        raw prediction, tracking, and classification results.
        """
        keys = [
            'pred_trk_pxl_fpath',
            'pred_trk_poly_sites_fpath',
            'pred_trk_poly_site_summaries_fpath',
            'eval_act_poly_fpath'
        ]
        for key in keys:
            for pat in [p[key] for p in self.path_patterns_matrix]:
                found = util_path.coerce_patterned_paths(pat)
                for path in found:
                    row = {
                        'type': key,
                        'raw': path,
                    }
                    _attrs = self._parse_pattern_attrs(self.templates[key], path)
                    row.update(_attrs)
                    yield row

    def evaluation_rows(self, with_attrs=1, types=None, notypes=None):
        keys = [
            'eval_trk_pxl_fpath',
            'eval_trk_poly_fpath',
            'eval_act_pxl_fpath',
            'eval_act_poly_fpath'
        ]
        yield from self.versioned_rows(with_attrs=with_attrs, types=keys)

    def versioned_rows(self, with_attrs=1, types=None, notypes=None):
        """
        Versioned items are things that are tracked with DVC. These are
        packages and evaluation measures.

        Ignore:
            types = None
            notypes = None
            with_attrs = 1
        """
        keys = [
            'eval_trk_pxl_fpath',
            'eval_act_poly_fpath',
            'eval_trk_poly_fpath',
            'pkg_fpath'
        ]
        if types is not None:
            keys = types
        if notypes is not None:
            keys = list(ub.oset(keys) - set(notypes))
        for key in keys:
            for pat in [p[key] for p in self.path_patterns_matrix]:
                found = list(util_path.sidecar_glob(
                    pat, sidecar_ext='.dvc', sidecar_key='dvc', main_key='raw'))
                for row in found:
                    row['type'] = key
                    row['has_dvc'] = (row['dvc'] is not None)
                    row['has_raw'] = (row['raw'] is not None)
                    row['needs_pull'] = row['has_dvc'] and not row['has_raw']
                    row['is_link'] = False
                    row['is_broken'] = False
                    row['unprotected'] = False
                    row['needs_push'] = False
                    if with_attrs:
                        if row['raw']:
                            path = row['raw']
                        else:
                            path = row['dvc'].augment(ext='')
                        row['dataset_code'] = self.dataset_code
                        _attrs = self._parse_pattern_attrs(self.templates[key], path)

                        if self.blocklists is not None:
                            blocked = False
                            for k, v in _attrs.items():
                                if k in self.blocklists:
                                    if v in self.blocklists[k]:
                                        blocked = True
                            if blocked:
                                continue

                        row.update(_attrs)

                    if row['has_raw']:
                        p = ub.Path(row['raw'])
                        row['is_link'] = p.is_symlink()
                        row['is_broken'] = row['is_link'] and not p.exists()
                        row['unprotected'] = row['has_dvc'] and not row['is_link']
                        row['needs_push'] = not row['has_dvc']
                    yield row

    def volitile_table(self):
        volitile_rows = list(self.volitile_rows())
        volitile_df = pd.DataFrame(volitile_rows)
        if len(volitile_df) == 0:
            volitile_df[self.VOLITILE_COLUMNS] = 0
        return volitile_df

    def staging_table(self):
        # import numpy as np
        staging_rows = list(self.staging_rows())
        staging_df = pd.DataFrame(staging_rows)

        if len(staging_df) == 0:
            staging_df[self.STAGING_COLUMNS] = 0
        return staging_df

    def versioned_table(self, **kw):
        """
        Get a list of dictionaries with information for each known evaluation.

        Information includes its real path if it exists, its dvc path if it exists
        and what sort of actions need to be done to synchronize it.
        """
        eval_rows = list(self.versioned_rows(**kw))
        eval_df = pd.DataFrame(eval_rows)
        if len(eval_df) == 0:
            eval_df[self.VERSIONED_COLUMNS] = 0
            # ['type', 'has_dvc', 'has_raw', 'needs_pull', 'is_link', 'is_broken', 'is_unprotected', 'needs_push', 'dataset_code']] = 0
        return eval_df

    def evaluation_table(self):
        rows = list(self.evaluation_rows())
        df = pd.DataFrame(rows)
        return df

    def cross_referenced_tables(self):
        import kwarray
        # Cross reference the versioned table with the staging table to
        # populate items in the staging table. Namely, if we have already
        # completed the staging process or not.
        staging_df = self.staging_table()
        versioned_df = self.versioned_table()
        volitile_df = self.volitile_table()

        if len(volitile_df) and len(versioned_df):
            # Determine how many volitile items (i.e. predictions) we
            # have on disk that correspond with our versioned data
            # volitile_keys = ['pred_pxl', 'pred_trk', 'pred_act']

            _grouper_keys = ['trk_model', 'act_model', 'model']
            vol_grouper_keys = ub.oset(_grouper_keys) & volitile_df.columns
            ver_grouper_keys = ub.oset(_grouper_keys) & versioned_df.columns
            grouper_keys = list(vol_grouper_keys & ver_grouper_keys)

            if 0:
                versioned_df.drop(['raw', 'dvc', 'dataset_code', 'expt_dvc_dpath'], axis=1)
            group_to_volitile = dict(list(volitile_df.groupby(grouper_keys)))
            group_to_versioned = dict(list(versioned_df.groupby(grouper_keys)))

            pred_keys = [
                'pred_trk_pxl_fpath',
                'pred_act_pxl_fpath',

                'pred_trk_poly_sites_fpath',
                'pred_trk_poly_site_summaries_fpath',

                'pred_act_poly_sites_fpath',
                # 'pred_act_poly_site_summaries_fpath',

                'pred_act_poly_sites_fpath'
            ]
            npred_keys = ['n_' + k for k in pred_keys]

            versioned_df.loc[:, npred_keys] = 0
            for groupvals, subdf in group_to_versioned.items():
                associated = group_to_volitile.get(groupvals, None)
                if associated is not None:
                    counts = associated.value_counts('type').rename(lambda x: 'n_' + x, axis=0)
                    versioned_df.loc[subdf.index, counts.index] += counts

        if len(staging_df) and len(versioned_df):
            # import xdev
            # with xdev.embed_on_exception_context:
            spkg_was_copied = kwarray.isect_flags(staging_df['model'], versioned_df['model'])
            staging_df['is_copied'] = spkg_was_copied
            # num_need_repackage = (~staging_df['is_packaged']).sum()
            # print(f'num_need_repackage={num_need_repackage}')

            # Lightning might produce the same checkpoint multiple times.  I'm not
            # sure if these checkpoints are actually different. Either way if they
            # are different, the difference should only be slight.  Given that we
            # now know which versions were stages, filter duplicates
            #
            # Given duplicates, prioritize:
            # staged, packaged, higher lightning version, lower checkpoint version.
            priority = [
                {'name': 'is_copied', 'ascending': 1},
                {'name': 'is_packaged', 'ascending': 1},
                {'name': 'lightning_version', 'ascending': 1},
                {'name': 'ckpt_ver', 'ascending': 1},
            ]
            by = [t['name'] for t in priority]
            ascending = [t['ascending'] for t in priority]
            deduped = []
            for k, g in staging_df.groupby(['expt', 'lightning_version', 'epoch', 'step']):
                if len(g) == 1:
                    deduped.append(g)
                else:
                    # Choose one from the group with highest priority
                    prioritized = g.sort_values(by=by, ascending=ascending)
                    choice = prioritized.iloc[0:1]
                    deduped.append(choice)
            staging_df = pd.concat(deduped)

            # Add info from staging into the versioned table
            versioned_has_orig = kwarray.isect_flags(versioned_df['model'], staging_df['model'])
            versioned_df['has_orig'] = versioned_has_orig
        else:
            staging_df['is_copied'] = False
            versioned_df['has_orig'] = False

        # TODO: cross reference the volitile table

        tables = ub.udict({
            'staging': staging_df,
            'versioned': versioned_df,
            'volitile': volitile_df,
        })
        return tables

    def summarize(self):
        """
        Ignore:
            >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
            >>> from watch.mlops.expt_manager import *  # NOQA
            >>> import watch
            >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
            >>> #expt_dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
            >>> dataset_code = 'Cropped-Drop3-TA1-2022-03-10'
            >>> self = ExperimentState(expt_dvc_dpath, dataset_code)
            >>> self.summarize()
        """
        tables = self.cross_referenced_tables()
        summarize_tables(tables)

    def push_packages(self):
        """
        This does what repackage used to do.
        Repackages checkpoints as torch packages, copies them to the DVC repo,
        and then adds them to DVC.

        >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
        >>> from watch.mlops.expt_manager import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
        >>> self = ExperimentState(expt_dvc_dpath, dataset_code, data_dvc_dpath)
        >>> self.summarize()
        """
        from rich.prompt import Confirm

        mode = 'all'

        staging_df = self.staging_table()
        needs_package = staging_df[~staging_df['is_packaged']]

        print(f'There are {len(needs_package)} checkpoints that need to be repackaged')
        if mode == 'interact':
            flag = Confirm.ask('Do you want to repackage?')
            if not flag:
                raise UserAbort

        if 'ckpt_path' in needs_package:
            to_repackage = needs_package['ckpt_path'].values.tolist()
        else:
            to_repackage = []
        print(to_repackage)
        if to_repackage:
            # NOTE: THIS RELIES ON KNOWING ABOUT THE SPECIFIC MODEL CODE.
            # IT WOULD BE NICE IF WE DIDN'T NEED THAT HERE.
            repackage(to_repackage)

        # Rebuild the tables to ensure we are up to date
        tables = self.cross_referenced_tables()
        staging_df, versioned_df, volitile_df = ub.take(tables, ['staging', 'versioned', 'volitile'])
        needs_copy = staging_df[~staging_df['is_copied']]
        print(needs_copy)
        print(f'There are {len(needs_copy)} packages that need to be copied')
        if mode == 'interact':
            flag = Confirm.ask('Do you want to copy?')
            if not flag:
                raise UserAbort

        import shutil
        for row in ub.ProgIter(needs_copy.to_dict('records'), desc='Copy packages to DVC dir'):
            kw = ub.udict(row).subdict({'expt', 'model'})
            kw['expt_dvc_dpath'] = self.expt_dvc_dpath
            kw['dataset_code'] = self.dataset_code
            pkg_fpath = ub.Path(self.templates['pkg_fpath'].format(**kw))
            src, dst = (row['spkg_fpath'], pkg_fpath)
            dst.parent.ensuredir()
            shutil.copy(src, dst)

        # Rebuild the tables to ensure we are up to date
        tables = self.cross_referenced_tables()
        staging_df, versioned_df, volitile_df = ub.take(tables, ['staging', 'versioned', 'volitile'])
        needs_add_flags = (~versioned_df['has_dvc'] | versioned_df['unprotected'])
        needs_dvc_add = versioned_df[needs_add_flags]
        print(needs_dvc_add)
        print(f'There are {len(needs_dvc_add)} packages that need DVC add/push')
        if mode == 'interact':
            flag = Confirm.ask('Do you want to run DVC add/push?')
            if not flag:
                raise UserAbort

        if len(needs_dvc_add):
            from watch.utils.simple_dvc import SimpleDVC
            dvc_api = SimpleDVC(self.expt_dvc_dpath)
            toadd_pkg_fpaths = needs_dvc_add['raw'].to_list()
            dvc_api.add(toadd_pkg_fpaths)
            push_jobs = 8
            dvc_api.push(toadd_pkg_fpaths, remote=self.dvc_remote,
                         jobs=push_jobs, recursive=True)

            import platform
            hostname = platform.node()
            dvc_api.git_commitpush(f'new packaged models from {hostname}')

        print(ub.codeblock(
            f"""
            # On the evaluation remote you need to run something like:
            DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
            cd $DVC_EXPT_DPATH
            git pull
            dvc pull -r aws --recursive models/fusion/{self.dataset_code}

            python -m watch.mlops.expt_manager "pull packages" --dvc_dpath=$DVC_EXPT_DPATH
            python -m watch.mlops.expt_manager "status packages" --dvc_dpath=$DVC_EXPT_DPATH
            python -m watch.mlops.expt_manager "evaluate" --dvc_dpath=$DVC_EXPT_DPATH

            # setup right params
            # python -m tasks.fusion.schedule_inference schedule_evaluation --gpus=auto --run=True

            """))

    def schedule_evaluation(state):
        # python -m watch.tasks.fusion.schedule_evaluation schedule_evaluation \
        #         --devices="0,1" \
        #         --model_globstr="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages/$EXPT_MODEL_GLOBNAME/*.pt" \
        #         --test_dataset="$VALI_FPATH" \
        #         --enable_pred=1 \
        #         --enable_eval=1 \
        #         --enable_actclf=1 \
        #         --enable_actclf_eval=1 \
        #         --draw_heatmaps=0 \
        #         --without_alternatives \
        #         --skip_existing=True --backend=slurm --run=0
        # from watch.tasks.fusion.schedule_evaluation import schedule_evaluation
        from watch.mlops.schedule_evaluation import schedule_evaluation
        model_globstr = [p['pkg_fpath'] for p in state.path_patterns_matrix]

        # NOTE: this should more often be specified as a cmdline arg maybe
        # jsonargparse can help with getting this nested correctly.
        test_kwcoco_fpath = state.data_dvc_dpath / state.dataset_code / 'data_vali.kwcoco.json'
        annotations_dpath = state.data_dvc_dpath / 'annotations'
        # TODO: how do we make scriptconfig do modal CLIs easilly?
        # need to configure
        eval_kw = {
            'test_dataset': test_kwcoco_fpath,
            'model_globstr': model_globstr,
            # 'run': None,
            # 'run': 1,
            'annotations_dpath': annotations_dpath,
            'devices': [0, 1],
        }
        # table = manager.versioned_table()
        # schedule_evaluation(cmdline=False, **eval_kw)
        schedule_evaluation(cmdline=1, **eval_kw)

    def _condense_test_dset(self, test_dataset):
        """
        This does what "organize" used to do.
        """
        if test_dataset is None:
            test_dset_name = 'unknown_test_dset'
        else:
            test_dataset = ub.Path(test_dataset)
            test_dset_name = '_'.join((list(test_dataset.parts[-2:-1]) + [test_dataset.stem]))

        # Register our condensed named.
        from watch.utils.reverse_hashid import ReverseHashTable
        rhash = ReverseHashTable(type='test_dset')
        rhash.register(test_dset_name, test_dataset)
        return test_dset_name

    def _condense_cfg(self, params, type):
        human_opts = ub.dict_isect(params, {})
        other_opts = ub.dict_diff(params, human_opts)
        if len(human_opts):
            human_part = ub.repr2(human_opts, compact=1) + '_'
        else:
            human_part = ''
        cfgstr_suffix = human_part + ub.hash_data(other_opts)[0:8]
        cfgstr = f'{type}_{cfgstr_suffix}'
        from watch.utils.reverse_hashid import ReverseHashTable
        rhash = ReverseHashTable(type=type)
        rhash.register(cfgstr, params)
        return cfgstr

    def _condense_pred_cfg(self, pred_cfg):
        """
        This does what "organize" used to do.
        """
        # Register our condensed named.
        if pred_cfg is None:
            pred_cfgstr = 'unknown'
        else:
            pred_cfgstr = ub.hash_data(pred_cfg)[0:8]
        pred_cfg_dname  = 'predcfg_' + pred_cfgstr
        from watch.utils.reverse_hashid import ReverseHashTable
        rhash = ReverseHashTable(type='pred_cfg')
        rhash.register(pred_cfg_dname, pred_cfg)
        return pred_cfg_dname

    def _condense_trk_cfg(self, bas_track_cfg):
        """
        This does what "organize" used to do.
        """
        human_opts = ub.dict_isect(bas_track_cfg, {})
        other_opts = ub.dict_diff(bas_track_cfg, human_opts)
        if len(human_opts):
            human_part = ub.repr2(human_opts, compact=1) + '_'
        else:
            human_part = ''
        cfgstr = human_part + ub.hash_data(other_opts)[0:8]
        # pred_bundle_dpath = pred_fpath.parent
        trk_cfg_dname = f'trackcfg_{cfgstr}'

        from watch.utils.reverse_hashid import ReverseHashTable
        rhash = ReverseHashTable(type='pred_cfg')
        rhash.register(trk_cfg_dname, bas_track_cfg)
        return trk_cfg_dname

    def _condense_model(self, model):
        return ub.Path(model).name

    def _condense_act_cfg(self, act_cfg):
        """
        This does what "organize" used to do.
        """
        human_opts = ub.dict_isect(act_cfg, {})
        other_opts = ub.dict_diff(act_cfg, human_opts)
        if len(human_opts):
            human_part = ub.repr2(human_opts, compact=1) + '_'
        else:
            human_part = ''
        cfgstr = human_part + ub.hash_data(other_opts)[0:8]
        acf_cfg_dname = f'actcfg_{cfgstr}'
        from watch.utils.reverse_hashid import ReverseHashTable
        rhash = ReverseHashTable(type='pred_cfg')
        rhash.register(acf_cfg_dname, act_cfg)
        return acf_cfg_dname


def summarize_tables(tables):
    """
    pip install rich-dataframe
    """
    from rich import print
    from rich.panel import Panel
    import rich
    console = rich.get_console()
    staging_df = tables.get('staging', None)
    volitile_df = tables.get('volitile', None)
    versioned_df = tables.get('versioned', None)

    if staging_df is not None:
        title = '[yellow] Staging Summary (Training)'

        if len(staging_df):
            staging_df['needs_copy'] = (~staging_df['is_copied'])
            staging_df['needs_package'] = (~staging_df['is_packaged'])
            body_df = staging_df[['ckpt_exists', 'is_packaged', 'is_copied', 'needs_package', 'needs_copy']].sum().to_frame().T
            body = console.highlighter(str(body_df))
        else:
            body = console.highlighter('There are no unversioned staging items')
        print(Panel(body, title=title))

    _grouper_keys = ub.oset(['dataset_code', 'test_trk_dset', 'test_act_dset', 'type'])

    if volitile_df is not None:
        title = ('[bright_blue] Volitile Summary (Predictions)')
        if len(volitile_df):
            grouper_keys = list(_grouper_keys & volitile_df.columns)
            num_pred_types = volitile_df.groupby(grouper_keys).nunique()
            body_df = num_pred_types
            body = console.highlighter(str(body_df))
        else:
            body = console.highlighter('There are no volitile items')

        print(Panel(body, title=title))

    if versioned_df is not None:
        title = ('[bright_green] Versioned Summary (Models and Evaluations)')
        # if 'has_orig' not in versioned_df.columns:
        #     versioned_df['has_orig'] = np.nan
        # version_bitcols = ['has_raw', 'has_dvc', 'is_link', 'is_broken', 'needs_pull', 'needs_push', 'has_orig']
        version_bitcols = ['has_raw', 'has_dvc', 'is_link', 'is_broken', 'needs_pull', 'needs_push']
        if len(versioned_df):
            grouper_keys = list(_grouper_keys & versioned_df.columns)
            body_df = versioned_df.groupby(grouper_keys)[version_bitcols].sum()
            body = console.highlighter(str(body_df))
        else:
            body = console.highlighter('There are no versioned items')
        print(Panel(body, title=title))


def checkpoint_filepath_info(fname):
    """
    Finds information encoded in the checkpoint/model file path.

    TODO:
        We need to ensure this info is encoded inside the file header as well!

    Ignore
        parse.parse('{prefix}foo={bar}', 'foo=3')
        parse.parse('{prefix}foo={bar}', 'afoao=3')

    Example:
        >>> from watch.mlops.expt_manager import *  # NOQA
        >>> fnames = [
        >>>     'epoch=1-step=10.foo',
        >>>     'epoch=1-step=10-v2.foo',
        >>>     'epoch=1-step=10',
        >>>     'epoch=1-step=10-v2',
        >>>     'junkepoch=1-step=10.foo',
        >>>     'junk/epoch=1-step=10-v2.foo',
        >>>     'junk-epoch=1-step=10',
        >>>     'junk_epoch=1-step=10-v2',
        >>> ]
        >>> for fname in fnames:
        >>>     info = checkpoint_filepath_info(fname)
        >>>     print(f'info={info}')
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v0'}
        info={'epoch': 1, 'step': 10, 'ckpt_ver': 'v2'}
    """
    # We assume it must have this
    suffix = ''.join(fname.partition('epoch=')[1:])
    # Hack: making name assumptions
    parsers = [
        parse.Parser('epoch={epoch:d}-step={step:d}-{ckpt_ver}.{ext}'),
        parse.Parser('epoch={epoch:d}-step={step:d}.{ext}'),
        parse.Parser('epoch={epoch:d}-step={step:d}-{ckpt_ver}'),
        parse.Parser('epoch={epoch:d}-step={step:d}'),
    ]
    # results = parser.parse(str(path))
    info = None
    for parsers in parsers:
        result = parsers.parse(suffix)
        if result:
            break
    if result:
        info = result.named
        if 'ckpt_ver' not in info:
            info['ckpt_ver'] = 'v0'
        info = ub.dict_diff(info, {'ext', 'prefix'})
    return info


@ub.memoize
def load_meta(fpath):
    import yaml
    with open(fpath, 'r') as file:
        data = yaml.safe_load(file)
    return data


def repackage(checkpoint_fpath, force=False, dry=False):
    """
    TODO:
        generalize this beyond the fusion model, also refactor.

    checkpoint_fpath

    Ignore:
        >>> import ubelt as ub
        >>> checkpoint_fpath = ub.expandpath(
        ...     '/home/joncrall/data/dvc-repos/smart_watch_dvc/models/fusion/checkpoint_DirectCD_smt_it_joint_p8_raw9common_v5_tune_from_onera_epoch=2-step=2147.ckpt')

    checkpoint_fpath = '/home/joncrall/remote/namek/smart_watch_dvc/training/namek/joncrall/Drop1_October2021/runs/Saliency_smt_it_joint_p8_rgb_uconn_ukyshared_v001/lightning_logs/version_1/checkpoints/epoch=53-step=28457.ckpt'
    """
    import os
    # For now there is only one model, but in the future we will need
    # some sort of modal switch to package the correct metadata
    from watch.tasks.fusion import methods
    from watch.utils import util_path
    checkpoint_fpaths = util_path.coerce_patterned_paths(checkpoint_fpath)
    package_fpaths = []
    for checkpoint_fpath in checkpoint_fpaths:
        # If we have a checkpoint path we can load it if we make assumptions
        # init method from checkpoint.
        checkpoint_fpath = os.fspath(checkpoint_fpath)

        x = ub.Path(ub.augpath(checkpoint_fpath, ext='.pt'))
        package_name = x.name

        # Can we precompute the package name of this checkpoint?
        train_dpath_hint = None
        if checkpoint_fpath.endswith('.ckpt'):
            path_ = ub.Path(checkpoint_fpath)
            if path_.parent.stem == 'checkpoints':
                train_dpath_hint = path_.parent.parent

        meta_fpath = None
        if train_dpath_hint is not None:
            # Look at the training config file to get info about this
            # experiment
            candidates = list(train_dpath_hint.glob('fit_config.yaml'))
            if len(candidates) == 1:
                meta_fpath = candidates[0]
                data = load_meta(meta_fpath)
                if 'name' in data:
                    # Use the metadata package name if it exists
                    expt_name = data['name']
                else:
                    # otherwise, hack to put experiment name in package name
                    # based on an assumed directory structure
                    expt_name = ub.Path(data['default_root_dir']).name
                if expt_name not in package_name:
                    package_name = expt_name + '_' + package_name

        package_fpath = x.parent / package_name

        if force or not package_fpath.exists():
            if not dry:
                import netharn as nh
                xpu = nh.XPU.coerce('cpu')
                checkpoint = xpu.load(checkpoint_fpath)

                # checkpoint = torch.load(checkpoint_fpath)
                print(list(checkpoint.keys()))
                hparams = checkpoint['hyper_parameters']

                if 'input_sensorchan' not in hparams:
                    # HACK: we had old models that did not save their hparams
                    # correctly. Try to fix them up here. The best we can do
                    # is try to start a small training run with the exact same
                    # settings and capture fixed model state from that.
                    if meta_fpath is None:
                        raise Exception('we cant do a fix without the meta fpath')

                    hackfix_hparams_fpath = meta_fpath.augment(prefix='hackfix_')
                    if not hackfix_hparams_fpath.exists():
                        # Do this once per experiment group to save time.
                        import tempfile
                        tmp_dpath = ub.Path(tempfile.mkdtemp())
                        tmp_root = (tmp_dpath / package_name)
                        ub.cmd(f'python -m watch.tasks.fusion.fit '
                               f'--config "{meta_fpath}" --default_root_dir "{tmp_root}" '
                               f'--max_epochs=0 --max_epoch_length=1', system=1, verbose=3)
                        tmp_llogs_dpaths = sorted((tmp_root / 'lightning_logs').glob('*'))
                        assert tmp_llogs_dpaths, 'cannot fix this model'
                        tmp_hparams_fpath = tmp_llogs_dpaths[-1] / 'hparams.yaml'
                        import shutil
                        shutil.copy(tmp_hparams_fpath, hackfix_hparams_fpath)

                    import yaml
                    with open(hackfix_hparams_fpath, 'r') as file:
                        hacked_hparams = yaml.load(file, yaml.Loader)
                    hacked_hparams = ub.udict(hacked_hparams)
                    # Select the known problematic variables
                    problem_hparams = hacked_hparams.subdict([
                        'classes', 'dataset_stats', 'input_sensorchan',
                        'input_channels'])
                    hparams.update(problem_hparams)
                    # hacked_hparams - hparams

                if 'input_channels' in hparams:
                    import kwcoco
                    # Hack for strange pickle issue
                    chan = hparams['input_channels']
                    if chan is not None:
                        if not hasattr(chan, '_spec') and hasattr(chan, '_info'):
                            chan = kwcoco.ChannelSpec.coerce(chan._info['spec'])
                            hparams['input_channels'] = chan
                        else:
                            hparams['input_channels'] = kwcoco.ChannelSpec.coerce(chan.spec)

                method = methods.MultimodalTransformer(**hparams)
                state_dict = checkpoint['state_dict']
                method.load_state_dict(state_dict)

                if train_dpath_hint is not None:
                    method.train_dpath_hint = train_dpath_hint

                method.save_package(os.fspath(package_fpath))
        package_fpaths.append(os.fspath(package_fpath))
    return package_fpaths


class UserAbort(Exception):
    pass


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

