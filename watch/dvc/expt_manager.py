"""
Synchronize DVC states across the machine.

This is a new Phase2 Variant of this script.
The original proof of concept is in
~/code/watch/watch/tasks/fusion/dvc_sync_manager.py

Example:
    python -m watch.dvc.expt_manager "list"
    python -m watch.dvc.expt_manager "pull evals"
    python -m watch.dvc.expt_manager "push evals"
    python -m watch.dvc.expt_manager "push packages evals"
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


# TODO: replace globals with a global config if necessary
DATASET_CODES = [
    'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC',
]


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
                <expt_dvc_dpath>/training/*/*/<dataset_code>/runs/<expt_name>/lightning_logs

                # Packages go here.
                <expt_dvc_dpath>/models/fusion/<dataset_code>/packages

                # Evaluations go here.
                <expt_dvc_dpath>/models/fusion/<dataset_code>/eval
            ''')),
    }


def main(cmdline=True, **kwargs):
    """
    from watch.dvc.expt_manager import *  # NOQA
    """
    import watch

    config = ExptManagerConfig(cmdline=cmdline, data=kwargs)
    command = config['command']
    dolist = 0
    if command is not None:
        config['push'] = False
        config['pull'] = False
        config['evals'] = False
        config['packages'] = False
        if 'list' in command:
            dolist = True
        if 'all' in command:
            config['packages'] = True
            config['evals'] = True
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

    expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
    hdd_manager = DVCExptManager(
        expt_dvc_dpath, dvc_remote=dvc_remote, dataset_codes=dataset_codes)
    synckw = ub.compatible(config, hdd_manager.sync)
    hdd_manager.sync(**synckw)

    if dolist:
        hdd_manager.summarize()


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
        >>> from watch.dvc.expt_manager import *  # NOQA
        >>> import watch
        >>> self = DVCExptManager.coerce(watch.find_dvc_dpath(tags='phase2_expt'))
        >>> self.summarize()

    Ignore:
        broke = df[df['is_broken']]
    """

    def __nice__(self):
        return str(self.dvc)

    def __init__(self, expt_dvc_dpath, dvc_remote='aws', dataset_codes=None):
        self.expt_dvc_dpath = expt_dvc_dpath
        self.dvc_remote = dvc_remote
        self.dataset_codes = dataset_codes
        self.dvc = simple_dvc.SimpleDVC.coerce(expt_dvc_dpath, remote=dvc_remote)
        self._build_states()

    def summarize(self):
        for state in self.states:
            state.summarize()
        versioned_df = self.versioned_table()
        summarize_versioned_df(versioned_df)

    @classmethod
    def coerce(cls, expt_dvc_dpath=None):
        import watch
        if expt_dvc_dpath is None:
            expt_dvc_dpath = watch.find_smart_dvc_dpath()
        dvc_remote = 'aws'
        dataset_codes = DATASET_CODES
        self = cls(expt_dvc_dpath=expt_dvc_dpath, dvc_remote=dvc_remote,
                   dataset_codes=dataset_codes)
        return self

    def _build_states(self):
        states = []
        for dataset_code in self.dataset_codes:
            state = ExperimentState(
                self.expt_dvc_dpath, dataset_code, dvc_remote=self.dvc_remote)
            states.append(state)
        self.states = states

    def versioned_table(self, **kw):
        rows = list(ub.flatten(state.versioned_rows(**kw) for state in self.states))
        df = pd.DataFrame(rows)
        return df

    def evaluation_table(self):
        rows = list(ub.flatten(state.evaluation_rows() for state in self.states))
        df = pd.DataFrame(rows)
        return df

    def push_evals(self):
        dvc = self.dvc
        eval_df = self.evaluation_table()
        summarize_versioned_df(eval_df)

        is_weird = (eval_df.is_link & (~eval_df.has_dvc))
        weird_df = eval_df[is_weird]
        if len(weird_df):
            print(f'weird_df=\n{weird_df}')

        to_push = eval_df[eval_df.needs_push == True]  # NOQA
        assert not to_push['has_dvc'].any()

        # TODO: if we want to allow modifications we need to find
        # unprotected files (or changed files on non-symlink dvc repos)
        # to_push = eval_df[(eval_df.needs_push == True) | (eval_df.unprotected == True)]  # NOQA

        to_push_fpaths = to_push['raw'].tolist()
        print(f'to_push=\n{to_push}')
        if len(to_push_fpaths):
            dvc.add(to_push_fpaths)
            dvc.git_commitpush(f'Sync evals from {platform.node()}')
            dvc.push(to_push_fpaths)

    def pull_evals(self):
        dvc = self.dvc
        dvc.git_pull()
        eval_df = self.evaluation_table()
        summarize_versioned_df(eval_df)

        # self.summarize()
        print(f'self.expt_dvc_dpath={self.expt_dvc_dpath}')
        print(len(eval_df))
        # import xdev
        # xdev.embed()
        eval_df = eval_df[~eval_df['is_broken']]
        pull_rows = eval_df[eval_df.needs_pull]
        pull_fpaths = pull_rows['dvc'].tolist()
        print(f'{len(pull_fpaths)=}')
        dvc.pull(pull_fpaths)

    def pull_packages(self):
        pkg_df = self.versioned_table(types=['pkg'])
        pull_df = pkg_df[pkg_df['needs_pull']]

        pull_fpaths = pull_df['dvc'].tolist()
        self.dvc.pull(pull_fpaths)

    def push_packages(self):
        # from watch.tasks.fusion import repackage
        # mode = 'commit'
        for state in self.states:
            state.push_packages()
            # # TODO: use the "state" staging table instead
            # if 0:
            #     import kwarray
            #     state_df = state.versioned_table()
            #     stage_df = state.staging_table()
            #     spkg_was_copied = kwarray.isect_flags(stage_df['model'], state_df['model'])
            #     stage_df['spkg_was_copied'] = spkg_was_copied
            #     num_need_repackage = (~stage_df['is_packaged']).sum()
            #     print(f'num_need_repackage={num_need_repackage}')
            # else:
            #     dataset_code = state.dataset_code
            #     print(f'dataset_code={dataset_code}')
            #     train_dpath = state.training_dpath / '*/*' / state.dataset_code / 'runs'
            #     storage_dpath = state.storage_dpath / 'packages'
            #     repackage.gather_checkpoints(
            #         expt_dvc_dpath=state.expt_dvc_dpath, storage_dpath=storage_dpath,
            #         train_dpath=train_dpath, dvc_remote=self.dvc_remote, mode=mode)

    def sync(self, push=True, pull=True, evals=True, packages=True):
        if push:
            if packages:
                self.push_packages()
            if evals:
                self.push_evals()
        if pull:
            if packages:
                self.pull_packages()
            if evals:
                self.pull_evals()


class ExperimentState(ub.NiceRepr):
    """

    Ignore:
        >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
        >>> from watch.dvc.expt_manager import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
        >>> self = ExperimentState(expt_dvc_dpath, dataset_code, data_dvc_dpath)
        >>> self.summarize()

    Ignore:
        table[table.type == 'pkg']['model'].unique()
    """

    def __init__(self, expt_dvc_dpath, dataset_code, dvc_remote=None):

        self.expt_dvc_dpath = expt_dvc_dpath
        self.dataset_code = dataset_code
        self.dvc_remote = dvc_remote
        self.storage_dpath = self.expt_dvc_dpath / 'models/fusion' / dataset_code
        self.training_dpath = self.expt_dvc_dpath / 'training'
        self.patterns = {
            # General
            'expt': '*',
            'expt_dvc_dpath': expt_dvc_dpath,
            'dataset_code': dataset_code,
            ### Versioned
            'test_dset': '*',
            'model': '*',  # hack, should have ext
            'pred_cfg': '*',
            'trk_cfg': '*',
            'act_cfg': '*',
            #### Staging
            'host': '*',
            'user': '*',
            'lightning_version': '*',
            'checkpoint': '*',  # hack, should have ext
            'stage_model': '*',  # hack, should have ext
        }

        self.staging_template_prefix = '{expt_dvc_dpath}/training/{host}/{user}/{dataset_code}/'
        self.storage_template_prefix = '{expt_dvc_dpath}/models/fusion/{dataset_code}/'

        self.staging_templates = {
            'ckpt': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{checkpoint}.ckpt',
            'spkg': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{model}.pt',
        }

        # Volitile (unused: todo incorporate)
        self.volitile_templates = {
            'pred_pxl': 'pred/{expt}/{model}/{test_dset}/{pred_cfg}/pred.kwcoco.json',
            'pred_trk': 'pred/{expt}/{model}/{test_dset}/{pred_cfg}/tracking/{trk_cfg}/tracks.json',
            'pred_act': 'pred/{expt}/{model}/{test_dset}/{pred_cfg}/actclf/{act_cfg}/activity_tracks.json',
        }

        self.versioned_templates = {
            'pkg': 'packages/{expt}/{model}.pt',
            'eval_pxl': 'eval/{expt}/{model}/{test_dset}/{pred_cfg}/eval/curves/measures2.json',
            'eval_trk': 'eval/{expt}/{model}/{test_dset}/{pred_cfg}/eval/tracking/{trk_cfg}/iarpa_eval/scores/merged/summary2.json',
            'eval_act': 'eval/{expt}/{model}/{test_dset}/{pred_cfg}/eval/actclf/{act_cfg}/iarpa_sc_eval/scores/merged/summary3.json',
        }

        self.templates = {}
        for k, v in self.staging_templates.items():
            self.templates[k] = self.staging_template_prefix + v
        for k, v in self.volitile_templates.items():
            self.templates[k] = self.storage_template_prefix + v
        for k, v in self.versioned_templates.items():
            self.templates[k] = self.storage_template_prefix + v

        ub.dict_union(
            self.staging_templates,
            self.volitile_templates,
            self.versioned_templates,
        )

        self.path_patterns = {}
        self._build_path_patterns()

    # def _check(self):
    #     from watch.utils import util_pattern
    #     pat = self.path_patterns['pkg']
    #     # pat = self.path_patterns['eval_act']
    #     p1 = [p for p in list(util_pattern.Pattern.coerce(pat).paths()) if not p.name.endswith('.dvc')]
    #     p2 = list(util_pattern.Pattern.coerce(pat + '.dvc').paths())
    #     for p in p2:
    #         if not p.augment(ext='').exists():
    #             break
    #     print(len(p1))
    #     print(len(p2))
    # self.
    # pass

    def _build_path_patterns(self):
        self.path_patterns = {
            k: v.format(**self.patterns)
            for k, v in self.templates.items()}

    def __nice__(self):
        return self.dataset_code

    def _parse_pattern_attrs(self, key, path):
        row = {}
        template = self.templates[key]
        parser = parse.Parser(str(template))
        results = parser.parse(str(path))
        if results is None:
            raise AssertionError(path)
            parser = parse.Parser(str(template)[:-4])
            results = parser.parse(str(path))
        if results is not None:
            row.update(results.named)
        else:
            warnings.warn('warning: bad attrs')
        return row

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
        default = {'ckpt_path': None, 'spkg_path': None}
        _id_to_row = ub.ddict(default.copy)

        key = 'ckpt'
        pat = self.path_patterns[key]
        mpat = util_pattern.Pattern.coerce(pat)
        # Find all checkpoints
        rows = []
        for ckpt_path in list(mpat.paths()):
            if ckpt_path.suffix != '.ckpt':
                continue
            row = default.copy()
            row['ckpt_path'] = ckpt_path
            row['type'] = 'ckpt'
            row['is_packaged'] = False
            row['ckpt_exists'] = True

            _attrs = self._parse_pattern_attrs(key, ckpt_path)
            row.update(_attrs)
            rows.append(row)
            _id_to_row[ckpt_path] = row

        # Find repackaged checkpoints
        key = 'spkg'  # stands for staged package
        pat = self.path_patterns[key]
        mpat = util_pattern.Pattern.coerce(pat)
        for spkg_path in list(mpat.paths()):
            # Does this correspond to an existing checkpoint?
            _attrs = self._parse_pattern_attrs(key, spkg_path)

            # Hack: making assumption about naming pattern
            spkg_stem = spkg_path.stem
            ckpt_stem = ''.join(spkg_stem.partition('_epoch')[-2:])[1:]
            ckpt_path = spkg_path.parent / (ckpt_stem + '.ckpt')

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
            row['spkg_path'] = spkg_path
            row['is_packaged'] = True
            row.update(_attrs)

        # Hack: making name assumptions
        for row in rows:
            fname = row['checkpoint']
            info = checkpoint_filepath_info(fname)
            row.update(info)
        return rows

    # TODO: add another variant for non-versioned prediction files

    def volitile_rows(self):
        """
        A volitile item is something that is derived from something versioned
        (so it is recomputable), but it is not versioned itself. These are
        raw prediction, tracking, and classification results.
        """

    def evaluation_rows(self, with_attrs=1, types=None, notypes=None):
        keys = ['eval_pxl', 'eval_act', 'eval_trk']
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
        keys = ['eval_pxl', 'eval_act', 'eval_trk', 'pkg']
        if types is not None:
            keys = types
        if notypes is not None:
            keys = list(ub.oset(keys) - set(notypes))
        for key in keys:
            pat = self.path_patterns[key]
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
                    _attrs = self._parse_pattern_attrs(key, path)
                    row.update(_attrs)

                if row['has_raw']:
                    p = ub.Path(row['raw'])
                    row['is_link'] = p.is_symlink()
                    row['is_broken'] = row['is_link'] and not p.exists()
                    row['unprotected'] = row['has_dvc'] and not row['is_link']
                    row['needs_push'] = not row['has_dvc']
                yield row

    def staging_table(self):
        # import numpy as np
        staging_rows = list(self.staging_rows())
        staging_df = pd.DataFrame(staging_rows)

        if len(staging_df) == 0:
            staging_df[['ckpt_exists', 'is_packaged', 'is_staged', 'needs_package', 'needs_stage']] = 0
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
            eval_df[['type', 'has_dvc', 'has_raw', 'needs_pull', 'is_link', 'is_broken', 'is_unprotected', 'needs_push', 'dataset_code']] = 0
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

        if len(staging_df) and len(versioned_df):
            spkg_was_copied = kwarray.isect_flags(staging_df['model'], versioned_df['model'])
            staging_df['is_copied'] = spkg_was_copied
            num_need_repackage = (~staging_df['is_packaged']).sum()
            print(f'num_need_repackage={num_need_repackage}')

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
            versioned_has_staged = kwarray.isect_flags(versioned_df['model'], staging_df['model'])
            versioned_df['has_staged'] = versioned_has_staged
        else:
            staging_df['is_copied'] = False
            versioned_df['has_staged'] = False
        return staging_df, versioned_df

    def summarize(self):
        """
        Ignore:
            >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
            >>> from watch.dvc.expt_manager import *  # NOQA
            >>> import watch
            >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
            >>> #expt_dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
            >>> dataset_code = 'Cropped-Drop3-TA1-2022-03-10'
            >>> self = ExperimentState(expt_dvc_dpath, dataset_code)
            >>> self.summarize()
        """
        staging_df, versioned_df = self.cross_referenced_tables()
        summarize_staging_df(staging_df)
        summarize_versioned_df(versioned_df)

    def push_packages(self):
        """
        This does what repackage used to do.
        Repackages checkpoints as torch packages, copies them to the DVC repo,
        and then adds them to DVC.

        >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
        >>> from watch.dvc.expt_manager import *  # NOQA
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

        from watch.tasks.fusion.repackage import repackage
        to_repackage = needs_package['ckpt_path'].values.tolist()
        print(to_repackage)
        if to_repackage:
            # NOTE: THIS RELIES ON KNOWING ABOUT THE SPECIFIC MODEL CODE.
            # IT WOULD BE NICE IF WE DIDN'T NEED THAT HERE.
            repackage(to_repackage)

        # Rebuild the tables to ensure we are up to date
        staging_df, versioned_df = self.cross_referenced_tables()
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
            pkg_fpath = ub.Path(self.templates['pkg'].format(**kw))
            src, dst = (row['spkg_path'], pkg_fpath)
            dst.parent.ensuredir()
            shutil.copy(src, dst)

        # Rebuild the tables to ensure we are up to date
        staging_df, versioned_df = self.cross_referenced_tables()
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

            # setup right params
            # python -m tasks.fusion.schedule_inference schedule_evaluation --gpus=auto --run=True
            """))


def summarize_versioned_df(versioned_df):
    import numpy as np
    print('Versioned summary')
    if 'has_staged' not in versioned_df.columns:
        versioned_df['has_staged'] = np.nan

    version_bitcols = ['has_raw', 'has_dvc', 'is_link', 'is_broken', 'needs_pull', 'needs_push', 'has_staged']
    needy = versioned_df.groupby(['dataset_code', 'type'])[version_bitcols].sum()
    print(needy)


def summarize_staging_df(staging_df):
    print('Staging summary')
    staging_df['needs_stage'] = (~staging_df['is_staged'])
    staging_df['needs_package'] = (~staging_df['is_packaged'])
    print(staging_df[['ckpt_exists', 'is_packaged', 'is_staged', 'needs_package', 'needs_stage']].sum().to_frame().T)


def checkpoint_filepath_info(fname):
    """
    Finds information encoded in the checkpoint/model file path.

    TODO:
        We need to ensure this info is encoded inside the file header as well!

    Ignore
        parse.parse('{prefix}foo={bar}', 'foo=3')
        parse.parse('{prefix}foo={bar}', 'afoao=3')

    Example:
        >>> from watch.dvc.expt_manager import *  # NOQA
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


class UserAbort(Exception):
    pass


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/dvc/expt_manager.py "pull all"
        python -m watch.dvc.expt_manager "push all"
        python -m watch.dvc.expt_manager "pull evals"
        python -m watch.dvc.expt_manager "pull all"
        python -m watch.dvc.expt_manager "pull packages"
    """
    main(cmdline=True)
