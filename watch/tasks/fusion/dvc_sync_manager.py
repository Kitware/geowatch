"""
Synchronize DVC states across the machine.

Example:
    python -m watch.tasks.fusion.dvc_sync_manager "pull evals"
    python -m watch.tasks.fusion.dvc_sync_manager "push evals"
    python -m watch.tasks.fusion.dvc_sync_manager "push packages evals"
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
    'Cropped-Drop3-TA1-2022-03-10',
    'Aligned-Drop3-TA1-2022-03-10',
    'Aligned-Drop3-L1',
]

# hack for old scheme
STORAGE_REPL = {
    'Aligned-Drop3-TA1-2022-03-10': 'eval3_candidates',
    'Cropped-Drop3-TA1-2022-03-10': 'eval3_sc_candidates',
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


def main(cmdline=True, **kwargs):
    """
    from watch.tasks.fusion.dvc_sync_manager import *  # NOQA
    """
    import watch

    config = SyncMachineConfig(cmdline=cmdline, data=kwargs)
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

    if dolist:
        self.summarize()


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
        >>> self = DVCSyncManager.coerce(watch.find_smart_dvc_dpath(hardware='hdd'))
        >>> #df = self.versioned_table()
        >>> #print(df)
        >>> self.summarize()

    Ignore:
        broke = df[df['is_broken']]
    """

    def __nice__(self):
        return str(self.dvc)

    def __init__(self, dvc_dpath, dvc_remote='aws', dataset_codes=None):
        self.dvc_dpath = dvc_dpath
        self.dvc_remote = dvc_remote
        self.dataset_codes = dataset_codes
        self.dvc = simple_dvc.SimpleDVC.coerce(dvc_dpath, remote=dvc_remote)
        self._build_states()

    def summarize(self):
        versioned_df = self.versioned_table()
        summarize_versioned_df(versioned_df)

    @classmethod
    def coerce(cls, dvc_dpath=None):
        import watch
        if dvc_dpath is None:
            dvc_dpath = watch.find_smart_dvc_dpath()
        dvc_remote = 'aws'
        dataset_codes = DATASET_CODES
        self = cls(dvc_dpath=dvc_dpath, dvc_remote=dvc_remote,
                   dataset_codes=dataset_codes)
        return self

    def _build_states(self):
        states = []
        for dataset_code in self.dataset_codes:
            state = ExperimentState(self.dvc_dpath, dataset_code)
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
        print(f'self.dvc_dpath={self.dvc_dpath}')
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
        from watch.tasks.fusion import repackage
        mode = 'commit'
        for state in self.states:
            # TODO: use the "state" staging table instead
            if 0:
                import kwarray
                state_df = state.versioned_table()
                stage_df = state.staging_table()
                spkg_was_copied = kwarray.isect_flags(stage_df['model'], state_df['model'])
                stage_df['spkg_was_copied'] = spkg_was_copied
                num_need_repackage = (~stage_df['is_packaged']).sum()
                print(f'num_need_repackage={num_need_repackage}')
            else:
                dataset_code = state.dataset_code
                print(f'dataset_code={dataset_code}')
                train_dpath = state.training_dpath / '*/*' / state.dataset_code / 'runs'
                storage_dpath = state.storage_dpath / 'packages'
                repackage.gather_checkpoints(
                    dvc_dpath=state.dvc_dpath, storage_dpath=storage_dpath,
                    train_dpath=train_dpath, dvc_remote=self.dvc_remote, mode=mode)

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
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.tasks.fusion.dvc_sync_manager import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_smart_dvc_dpath()
        >>> dataset_code = 'Aligned-Drop3-TA1-2022-03-10'
        >>> self = ExperimentState(dvc_dpath, dataset_code)
        >>> gen = self.versioned_rows(['eval_trk'])
        >>> row = ub.peek(gen)
        >>> table = self.versioned_table()
        >>> print(table[['type', 'raw']])

    Ignore:
        table[table.type == 'pkg']['model'].unique()
    """
    def __init__(self, dvc_dpath, dataset_code, storage_code=None):
        self.dvc_dpath = dvc_dpath
        self.dataset_code = dataset_code
        if storage_code is None:
            storage_code = STORAGE_REPL.get(dataset_code, dataset_code)
        self.storage_code = storage_code
        self.storage_dpath = self.dvc_dpath / 'models/fusion' / storage_code
        self.training_dpath = self.dvc_dpath / 'training'
        self.patterns = {
            # General
            'expt': '*',
            'dvc_dpath': dvc_dpath,
            'dataset_code': dataset_code,
            'storage_code': storage_code,
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

        self.staging_template_prefix = '{dvc_dpath}/training/{host}/{user}/{dataset_code}/'
        self.storage_template_prefix = '{dvc_dpath}/models/fusion/{storage_code}/'

        self.staging_templates = {
            'ckpt': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{checkpoint}.ckpt',
            'spkg': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{model}.pt',
        }

        # Volitile (unused: todo incorporate)
        self.volitile_templates = {
            'pred_pxl': 'pred/{expt}/pred_{model}/{test_dset}/{pred_cfg}/pred.kwcoco.json',
            'pred_trk': 'pred/{expt}/pred_{model}/{test_dset}/{pred_cfg}/tracking/{trk_cfg}/tracks.json',
            'pred_act': 'pred/{expt}/pred_{model}/{test_dset}/{pred_cfg}/actclf/{act_cfg}/activity_tracks.json',
        }

        self.versioned_templates = {
            'pkg': 'packages/{expt}/{model}.pt',
            'eval_pxl': 'eval/{expt}/pred_{model}/{test_dset}/{pred_cfg}/eval/curves/measures2.json',
            'eval_trk': 'eval/{expt}/pred_{model}/{test_dset}/{pred_cfg}/eval/tracking/{trk_cfg}/iarpa_eval/scores/merged/summary2.json',
            'eval_act': 'eval/{expt}/pred_{model}/{test_dset}/{pred_cfg}/eval/actclf/{act_cfg}/iarpa_sc_eval/scores/merged/summary3.json',
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
        # import numpy as np
        eval_rows = list(self.versioned_rows(**kw))
        eval_df = pd.DataFrame(eval_rows)
        # print(eval_df.drop(['type', 'raw', 'dvc'], axis=1).sum().to_frame().T)
        # print(eval_df.groupby('type').sum())
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
            staging_df['is_staged'] = spkg_was_copied
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
                {'name': 'is_staged', 'ascending': 1},
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
            staging_df['is_staged'] = False
            staging_df['is_packaged'] = False
            versioned_df['has_staged'] = False
        return staging_df, versioned_df

    def summarize(self):
        """
        Ignore:
            >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
            >>> from watch.tasks.fusion.dvc_sync_manager import *  # NOQA
            >>> import watch
            >>> dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
            >>> #dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
            >>> dataset_code = 'Cropped-Drop3-TA1-2022-03-10'
            >>> self = ExperimentState(dvc_dpath, dataset_code)
            >>> self.summarize()

            versioned_df = self.versioned_table()
            flags = (versioned_df['model'] == 'CropDrop3_SC_s2wv_rgb_xver6_V019_epoch=119-step=30719')
            versioned_df[flags]

            flags = versioned_df['model'].apply(lambda x: 'V019_epoch=119-step=30719' in x)
            flags.sum()

            y = self.versioned_table(types=['pkg'])
            y[y['expt'].apply(lambda x: 'V019' in x)]
            flags = y['model'].apply(lambda x: '30719' in x)
            flags.sum()
            y[flags]
        """
        staging_df, versioned_df = self.cross_referenced_tables()
        summarize_staging_df(staging_df)
        summarize_versioned_df(versioned_df)


def summarize_versioned_df(versioned_df):
    import numpy as np
    print('Versioned summary')
    if 'has_staged' not in versioned_df.columns:
        versioned_df['has_staged'] = np.nan

    version_bitcols = ['has_raw', 'has_dvc', 'is_link', 'is_broken', 'needs_pull', 'needs_push', 'has_staged']
    needy = versioned_df.groupby(['dataset_code', 'type'])[version_bitcols].sum()
    print(needy)

    # want_cols = [
    #     'type', 'dataset_code', 'expt', 'model',
    #     # 'test_dset',
    #     'pred_cfg', 'act_cfg', 'trk_cfg']

    # have_cols = versioned_df.columns.intersection(want_cols)
    # config_rows = versioned_df[have_cols]
    # if 0:
    #     # Uniqueness Breakdown
    #     print(config_rows.sort_values('expt').value_counts(dropna=False).to_string())
    # # description = config_rows.describe()
    # # print(description.to_string())
    # # Description with more stuff (not sure if there is a way to get pandas
    # # describe to do this.
    # for dataset_code, group in versioned_df.groupby('dataset_code'):
    #     desc2_parts = []
    #     for col in have_cols:
    #         col_vals = config_rows[col]
    #         col_freq = col_vals.value_counts()
    #         col_description = {'name': col}
    #         col_description['count'] = (~col_vals.isnull()).sum()
    #         col_description['unique'] = col_freq.size
    #         # col_description['max_freq'] = col_freq.max()
    #         # col_description['min_freq'] = col_freq.min()
    #         # col_description['med_freq'] = int(col_freq.median())
    #         desc2_parts.append(col_description)
    #     description2 = pd.DataFrame(desc2_parts).set_index('name').T
    #     description2.columns.name = None
    #     print('')
    #     print(f'dataset_code={dataset_code}')
    #     print('Number of Unique Entries')
    #     print(description2.to_string())
    #     print('Number of Models & Evaluations')
    #     print(group.groupby('type')[version_bitcols].sum())
    # model_to_types = {}
    # for model, group in versioned_df.groupby('model'):
    #     model_to_types[model] = tuple(sorted(group['type'].unique()))
    # types_to_models = ub.invert_dict(model_to_types, 0)
    # model_eval_counts = pd.DataFrame([ub.map_vals(len, types_to_models)])
    # print('Number of evals / package type for each model')
    # # print('Ideally each model has a package, pixel eval, and either act or trk eval')
    # # print('Models with evals, but no packages are a problem, probably a bug, maybe needs a git pull?')
    # print(model_eval_counts)
    # # print(types_to_models[('eval_pxl',)])


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
        >>> from watch.tasks.fusion.dvc_sync_manager import *  # NOQA
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


def pull_from_s3_notes():
    """

    aws s3 --profile iarpa ls s3://kitware-smart-watch-data/sync_root/
    aws s3 --profile iarpa sync s3://kitware-smart-watch-data/sync_root/ $HOME/data/aws-sync/sync_root/

    ls $HOME/data/aws-sync/sync_root/ta2-train*/*/Aligned-Drop3-L1/runs
    ls $HOME/data/aws-sync/sync_root/ta2-train*/*/Aligned-Drop3-L1/runs

    ls $HOME/data/aws-sync/sync_root/ta2-train*/*/Aligned-Drop3-L1/runs/*/lightning_logs/*/checkpoints

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    DATASET_CODE=Aligned-Drop3-L1
    EXPT_GROUP_CODE=Aligned-Drop3-L1
    KWCOCO_BUNDLE_DPATH=$DVC_DPATH/$DATASET_CODE
    TRAIN_DPATH=$HOME/data/aws-sync/sync_root/*/*/$DATASET_CODE/runs/*/lightning_logs
    python -m watch.tasks.fusion.repackage gather_checkpoints \
        --dvc_dpath="$DVC_DPATH" \
        --storage_dpath="$DVC_DPATH/models/fusion/$EXPT_GROUP_CODE/packages" \
        --train_dpath="$TRAIN_DPATH" \
        --push_jobs=8 --dvc_remote=aws \
        --mode=commit
    """


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/dvc_sync_manager.py "pull all"
        python -m watch.tasks.fusion.dvc_sync_manager "push all"
        python -m watch.tasks.fusion.dvc_sync_manager "pull evals"
        python -m watch.tasks.fusion.dvc_sync_manager "pull all"

        python -m watch.tasks.fusion.dvc_sync_manager "pull packages"
    """
    main(cmdline=True)
