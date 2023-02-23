r"""
This is the CLI for expt_state

Synchronize DVC states across the machine.

This is a new Phase2 Variant of this script.

Example:

    export DVC_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data")
    export DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
    cd $DVC_EXPT_DPATH

    python -m watch.mlops.manager "status" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch.mlops.manager "status" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch.mlops.manager "pull packages evals" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"
    python -m watch.mlops.manager "push packages evals"

    python -m watch.mlops.manager "status" --dataset_codes Drop4-SC

    python -m watch.mlops.manager "list" --dataset_codes Drop4-BAS
    python -m watch.mlops.manager "list" --dataset_codes Drop6

    # On training machine
    python -m watch.mlops.manager "push packages"
    python -m watch.mlops.manager "push packages" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"

    # On testing machine
    python -m watch.mlops.manager "pull packages"
    python -m watch.mlops.manager "status"

    # Run evals on testing machine
    python -m watch.mlops.manager "evaluate" --dataset_codes "Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC"

    # On testing machine
    python -m watch.mlops.manager "push evals"

    # On analysis machine
    python -m watch.mlops.manager "pull evals"


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

    python -m watch.mlops.manager "pull packages" --model_pattern="${MODEL_OF_INTEREST}*"
    python -m watch.mlops.manager "pull evals" --model_pattern="${MODEL_OF_INTEREST}*"
    python -m watch.mlops.manager "status" --model_pattern="${MODEL_OF_INTEREST}*"

    python -m watch.mlops.manager "status" --dataset_codes=Drop6
    python -m watch.mlops.manager "add packages" --dataset_codes=Drop6
"""
import pandas as pd
import ubelt as ub
# import platform
import scriptconfig as scfg
from watch.utils import simple_dvc
from watch import heuristics
import warnings
import parse
from watch.utils import util_pattern
from watch.utils import util_path
from watch.mlops import repackager


class ManagerConfig(scfg.DataConfig):
    """
    Certain parts of these names have special nomenclature to make them easier
    to work with in Python and Bash.
    """
    command = scfg.Value(None, nargs='*', help='if specified, will overload other options', position=1)

    dvc_remote = scfg.Value('aws', help='dvc remote to sync to/from')

    expt_dvc_dpath = scfg.Value('auto', help='path to the experiment dvc dpath')

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

        NOTE: THIS SPECIFIC FORMAT IS IN HIGH FLUX. DOCS MAY BE OUTDATED
        '''))


def main(cmdline=True, **kwargs):
    """
    from watch.mlops.manager import *  # NOQA
    """
    config = ManagerConfig.cli(cmdline=cmdline, data=kwargs)
    print('ManagerConfig config = {}'.format(ub.repr2(dict(config), nl=1)))
    command = config['command']

    available_actions = [
        'status', 'evaluate', 'push', 'pull', 'list',
        'report', 'add',
    ]
    available_targets = [
        'packages',
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

    print(f'actions={actions}')
    print(f'targets={targets}')
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    dvc_remote = config['dvc_remote']

    if config['dataset_codes'] is None:
        dataset_codes = heuristics.DATASET_CODES
    else:
        dataset_codes = config['dataset_codes']

    if config['expt_dvc_dpath'] == 'auto':
        config['expt_dvc_dpath'] = heuristics.auto_expt_dvc()

    expt_dvc_dpath = config['expt_dvc_dpath']

    manager = DVCExptManager(
        expt_dvc_dpath, dvc_remote=dvc_remote, dataset_codes=dataset_codes,
        model_pattern=config['model_pattern'])

    if 'pull' in actions:
        manager.pull(targets)

    if 'add' in actions and 'packages' in targets:
        manager.add_packages()

    if 'push' in actions and 'packages' in targets:
        manager.push_packages()

    # if 'push' in actions:
    #     raise NotImplementedError
    #     manager.push(targets)

    if 'status' in actions:
        manager.summarize()

    if 'list' in actions:
        manager.list()


class DVCExptManager(ub.NiceRepr):
    """
    Implements an API around our DVC structure, which can be described as
    follows.

    TODO:
        - [ ] If we can somehow generate the output paths based on the
        pipeline, then we will be in a very good position.

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
        >>> from watch.mlops.manager import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> dataset_codes = ['Drop4-BAS']
        >>> manager = DVCExptManager(expt_dvc_dpath=expt_dvc_dpath, dataset_codes=dataset_codes)
        >>> manager.list()
        >>> manager.summarize()

        self = manager.stats[0]
        self.list()
        util_pandas.pandas_truncate_items(self.staging_table(), paths=0, max_length=32)[0]


    Ignore:
        broke = df[df['is_broken']]
    """

    def __nice__(manager):
        return str(manager.dvc)

    def __init__(manager, expt_dvc_dpath, dvc_remote='aws', dataset_codes=None,
                 model_pattern='*'):
        manager.model_pattern = model_pattern
        manager.expt_dvc_dpath = expt_dvc_dpath
        manager.dvc_remote = dvc_remote
        manager.dataset_codes = dataset_codes
        manager.dvc = simple_dvc.SimpleDVC.coerce(expt_dvc_dpath, remote=dvc_remote)
        manager._build_states()

    def summarize(manager):
        # for state in manager.states:
        #     state.summarize()
        for state in manager.states:
            state.summarize()

    def list(manager):
        for state in manager.states:
            state.list()

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
                model_pattern=manager.model_pattern)
            states.append(state)
        manager.states = states

    def pull_packages(manager):
        # Assume just one git repo and manually pull
        manager.dvc.git_pull()
        pull_fpaths = []
        for state in manager.states:
            pkg_df = state.versioned_table(types=['pkg_fpath'])
            pull_df = pkg_df[pkg_df['needs_pull'].astype(bool)]
            pull_fpaths += pull_df['dvc'].tolist()
        manager.dvc.pull(pull_fpaths)

    def add_packages(manager):
        """
        TODO: break this up into smaller components.
        """
        # from watch.tasks.fusion import repackage
        # mode = 'commit'
        for state in manager.states:
            state.add_packages()

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

    def pull(manager, targets):
        if 'packages' in targets:
            manager.pull_packages()

    def reverse_hash_lookup(manager, key):
        # This probably doesn't belong here
        from watch.utils.reverse_hashid import ReverseHashTable
        ReverseHashTable.query(key, verbose=1)


class ExperimentState(ub.NiceRepr):
    """

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_EXPT_DPATH)
        >>> from watch.mlops.manager import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
        >>> dataset_code = '*'
        >>> dataset_code = 'Drop4-BAS'
        >>> #dataset_code = 'Drop4-SC'
        >>> dvc_remote = 'aws'
        >>> self = ExperimentState(expt_dvc_dpath, dataset_code, dvc_remote, data_dvc_dpath)
        >>> self.list()
        >>> self.summarize()

    Ignore:
        >>> # Just show patterns:
        >>> from watch.mlops.manager import *  # NOQA
        >>> self = ExperimentState('<expt_dpath>', '<dset_code>')
        >>> print('self.templates = {}'.format(ub.repr2(self.templates, nl=1, sort=0)))

    Ignore:
        table[table.type == 'pkg_fpath']['model'].unique()
    """

    VERSIONED_COLUMNS = [
        'type', 'has_dvc', 'has_raw', 'needs_pull', 'is_link', 'is_broken',
        'unprotected', 'needs_push', 'raw', 'dvc', 'dataset_code']

    STAGING_COLUMNS = [
        'ckpt_exists', 'is_packaged', 'is_copied', 'needs_package', 'needs_copy']

    def __init__(self, expt_dvc_dpath, dataset_code='*', dvc_remote=None,
                 data_dvc_dpath=None, model_pattern='*', storage_dpath=None):

        if isinstance(model_pattern, str) and model_pattern.endswith('.txt') and ub.Path(model_pattern).exists():
            model_pattern = [
                p.strip()
                for p in ub.Path(model_pattern).read_text().split('\n')
                if p.strip()]

        self.expt_dvc_dpath = expt_dvc_dpath = ub.Path(expt_dvc_dpath)

        if data_dvc_dpath is None:
            import watch
            try:
                data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', envvar='DVC_DATA_DPATH')
            except Exception:
                pass
        self.data_dvc_dpath = data_dvc_dpath
        self.dataset_code = dataset_code
        self.dvc_remote = dvc_remote
        self.training_dpath = self.expt_dvc_dpath / 'training'

        if storage_dpath is None or storage_dpath == 'auto':
            storage_dpath = expt_dvc_dpath / 'models/fusion'

        ### Experimental, add in SC dependencies
        self.staging_template_prefix = '{expt_dvc_dpath}/training/{host}/{user}/{dataset_code}/'
        self.storage_template_prefix = '{storage_dpath}/{dataset_code}/'

        self.storage_dpath = storage_dpath

        self.patterns = {
            # General
            'expt_dvc_dpath': expt_dvc_dpath,
            'dataset_code': dataset_code,
            'storage_dpath': storage_dpath,
            #### Staging
            'host': '*',
            'user': '*',
            'ckpt_ver': '*',
            'epoch': '*',
            'step': '*',
            'lightning_version': '*',
            'checkpoint': '*',  # hack, should have ext
            'stage_model': '*',  # hack, should have ext
            ### Deprecated
            'model': model_pattern,  # hack, should have ext
            'imodel': model_pattern,
            'smodel': model_pattern,
            'expt': '*',
        }

        self.staging_templates = {
            # Staged checkpoint
            'ckpt': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{checkpoint}.ckpt',

            # Staged package
            'spkg': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{smodel}.pt',

            # Interrupted models
            'ipkg': 'runs/{expt}/lightning_logs/{lightning_version}/package-interupt/{imodel}.pt',
        }

        self.versioned_templates = {
            'pkg_fpath'            : 'packages/{expt}/{expt}_{ckpt_ver}_epoch{epoch}_step{step}.pt',  # by default packages dont know what task they have (because they may have multiple)
        }

        # that will cause a row to be ignored if it has one of those values
        # when a table is being built.
        self.blocklists = {
            k: set() for k in self.patterns.keys()
        }

        self.templates = {}
        for k, v in self.staging_templates.items():
            self.templates[k] = self.staging_template_prefix + v

        for k, v in self.versioned_templates.items():
            self.templates[k] = self.storage_template_prefix + v

        self.path_patterns_matrix = []
        self._build_path_patterns()

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
            raise RuntimeError(f'Failed to match path={path} to template={template}')
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
        default = {'ckpt_path': None, 'spkg_fpath': None}
        _id_to_row = ub.ddict(default.copy)

        rows = []
        key = 'ckpt'  # a raw checkpoint
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
                    # No corresponding raw checkpoint exists, add new row
                    row = default.copy()
                    row['checkpoint'] = ckpt_stem
                    row['ckpt_exists'] = False
                    row['type'] = 'ckpt'
                    rows.append(row)
                row['spkg_fpath'] = spkg_fpath
                row['is_packaged'] = True
                row.update(_attrs)

        # Find interrupted checkpoints
        key = 'ipkg'  # stands for interrupted package
        for pat in [p[key] for p in self.path_patterns_matrix]:
            mpat = util_pattern.Pattern.coerce(pat)
            for spkg_fpath in list(mpat.paths()):
                # Does this correspond to an existing checkpoint?
                _attrs = self._parse_pattern_attrs(self.templates[key], spkg_fpath)

                # Hack: making assumption about naming pattern
                spkg_stem = spkg_fpath.stem
                ckpt_stem = ''.join(spkg_stem.partition('_epoch')[-2:])[1:]
                ckpt_path = spkg_fpath.parent / (ckpt_stem + '.ckpt')

                # The checkpoint itself wont exist in this case
                # Always add a new row
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

            # Hack: making name assumptions
            info = checkpoint_filepath_info(fname)
            row.update(info)

            row.pop('imodel', None)
            row.pop('smodel', None)

            # Where would we expect to put this file?
            kw = ub.udict(row).subdict({'expt', 'ckpt_ver', 'epoch', 'step'})
            kw['expt_dvc_dpath'] = self.expt_dvc_dpath
            kw['dataset_code'] = self.dataset_code
            kw['storage_dpath'] = self.storage_dpath
            # This is the name we would version this with.
            row['pkg_fpath'] = ub.Path(self.templates['pkg_fpath'].format(**kw))
            row['is_copied'] = row['pkg_fpath'].exists()

        return rows

    def versioned_rows(self, with_attrs=1, types=None, notypes=None):
        """
        Versioned items are things that are tracked with DVC. These are
        packages and evaluation measures.

        Ignore:
            types = None
            notypes = None
            with_attrs = 1
        """
        keys = ['pkg_fpath']
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

    def staging_table(self):
        # import numpy as np
        staging_rows = list(self.staging_rows())
        staging_df = pd.DataFrame(staging_rows)

        DEDUP = 1  # get rid of duplicate or near duplicate checkpoints
        if DEDUP:
            if len(staging_df) > 0:
                chosen = []
                for _, group in staging_df.groupby(['expt', 'epoch', 'step']):
                    if len(group) > 1:
                        group = group.sort_values('ckpt_ver').iloc[0:1]
                    chosen.append(group)
                staging_df = pd.concat(chosen, axis=0).sort_index().reset_index(drop=True)

        if len(staging_df) == 0:
            staging_df[self.STAGING_COLUMNS] = 0
        return staging_df

    def versioned_table(self, **kw):
        """
        Get a list of dictionaries with information for each known evaluation.

        Information includes its real path if it exists, its dvc path if it exists
        and what sort of actions need to be done to synchronize it.
        """
        versioned_rows = list(self.versioned_rows(**kw))
        versioned_df = pd.DataFrame(versioned_rows)
        if len(versioned_df) == 0:
            versioned_df[self.VERSIONED_COLUMNS] = 0
            # ['type', 'has_dvc', 'has_raw', 'needs_pull', 'is_link', 'is_broken', 'is_unprotected', 'needs_push', 'dataset_code']] = 0
        return versioned_df

    def cross_referenced_tables(self):
        import kwarray
        # Cross reference the versioned table with the staging table to
        # populate items in the staging table. Namely, if we have already
        # completed the staging process or not.
        staging_df = self.staging_table()
        versioned_df = self.versioned_table()

        if len(staging_df) and len(versioned_df):
            # import xdev
            # with xdev.embed_on_exception_context:
            spkg_was_copied = kwarray.isect_flags(staging_df['pkg_fpath'], versioned_df['raw'])
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
            staging_df = pd.concat(deduped).sort_index().reset_index(drop=True)
        else:
            staging_df['is_copied'] = False

        tables = ub.udict({
            'staging': staging_df,
            'versioned': versioned_df,
        })
        return tables

    def list(self):
        tables = self.cross_referenced_tables()
        ready_packages = None
        if 'staging' in tables:
            todrop = ['expt_dvc_dpath', 'raw', 'ckpt_path', 'spkg_fpath', 'pkg_fpath', 'lightning_version', 'ckpt_exists']
            df = tables['staging']

            print(df.drop(ub.oset(todrop) & df.columns, axis=1).to_string())

        if 'versioned' in tables:
            todrop = ['raw', 'dvc', 'expt_dvc_dpath', 'expt', 'is_broken',
                      'is_link', 'has_raw', 'has_dvc', 'unprotected',
                      'needs_pull', 'needs_push']
            df = tables['versioned']
            type_to_versioned = dict(list(df.groupby('type')))

            sub = df[df['type'] == 'pkg_fpath']
            ready_packages = list(map(str, sub['raw'][sub['has_raw']].tolist()))

            for type, subdf in type_to_versioned.items():
                print(f'type={type}')
                print(subdf.drop(ub.oset(todrop) & df.columns, axis=1).to_string())

        if ready_packages is not None:
            print('ready_packages = {}'.format(ub.urepr(ready_packages, nl=1)))

    def summarize(self):
        """
        from mlops.aggregate import Aggregate
        agg = Aggregate(table)
        agg.build()

        Ignore:
            >>> # xdoctest: +REQUIRES(env:DVC_EXPT_DPATH)
            >>> from watch.mlops.manager import *  # NOQA
            >>> import watch
            >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
            >>> #expt_dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
            >>> dataset_code = 'Cropped-Drop3-TA1-2022-03-10'
            >>> self = ExperimentState(expt_dvc_dpath, dataset_code)
            >>> self.summarize()
        """
        tables = self.cross_referenced_tables()
        summarize_tables(tables)

    def package_checkpoints(self, mode='interact'):
        from rich.prompt import Confirm
        staging_df = self.staging_table()
        needs_package = staging_df[~staging_df['is_packaged']]

        print(f'There are {len(needs_package)} checkpoints that need to be repackaged')
        if mode == 'interact' and len(needs_package):
            flag = Confirm.ask('Do you want to repackage?')
            if not flag:
                raise UserAbort

        if 'ckpt_path' in needs_package:
            to_repackage = needs_package['ckpt_path'].values.tolist()
        else:
            to_repackage = []
        print('to_repackage = {}'.format(ub.repr2(to_repackage, nl=1)))
        if to_repackage:
            # NOTE: THIS RELIES ON KNOWING ABOUT THE SPECIFIC MODEL CODE.
            # IT WOULD BE NICE IF WE DIDN'T NEED THAT HERE.
            repackager.repackage(to_repackage)

    def copy_packages_to_dvc(self, mode='interact'):
        from rich.prompt import Confirm
        # Rebuild the tables to ensure we are up to date
        tables = self.cross_referenced_tables()
        staging_df, versioned_df = ub.take(tables, ['staging', 'versioned'])
        needs_copy = staging_df[~staging_df['is_copied']]
        print(needs_copy)
        print(f'There are {len(needs_copy)} packages that need to be copied')
        if mode == 'interact':
            flag = Confirm.ask('Do you want to copy?')
            if not flag:
                raise UserAbort

        for row in ub.ProgIter(needs_copy.to_dict('records'), desc='Copy packages to DVC dir'):
            src, dst = (row['spkg_fpath'], row['pkg_fpath'])
            if src is not None:
                dst.parent.ensuredir()
                ub.Path(src).copy(dst)

    def add_packages_to_dvc(self, mode='interact'):
        from rich.prompt import Confirm
        perf_config = {
            'push_workers': 8,
        }
        # Rebuild the tables to ensure we are up to date
        tables = self.cross_referenced_tables()
        staging_df, versioned_df = ub.take(tables, ['staging', 'versioned'])
        needs_add_flags = (~versioned_df['has_dvc'] | versioned_df['unprotected'])
        needs_dvc_add = versioned_df[needs_add_flags]
        print(needs_dvc_add)
        print(f'There are {len(needs_dvc_add)} packages that need DVC add/push')
        if len(needs_dvc_add):
            if mode == 'interact':
                flag = Confirm.ask('Do you want to run DVC add/push?')
                if not flag:
                    raise UserAbort

            import platform
            from watch.utils.simple_dvc import SimpleDVC
            hostname = platform.node()
            dvc_api = SimpleDVC(self.expt_dvc_dpath)

            toadd_pkg_fpaths = needs_dvc_add['raw'].to_list()

            dvc_api.git_commitpush(f'new packaged models from {hostname}')

            dvc_api.add(toadd_pkg_fpaths, verbose=1)
            dvc_api.push(
                toadd_pkg_fpaths, remote=self.dvc_remote,
                jobs=perf_config['push_workers'],
                recursive=True, verbose=1)

        print(ub.codeblock(
            f"""
            # On the evaluation remote you need to run something like:
            DVC_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
            cd $DVC_EXPT_DPATH
            git pull
            dvc pull -r aws --recursive models/fusion/{self.dataset_code}

            python -m watch.mlops.manager "pull packages" --dvc_dpath=$DVC_EXPT_DPATH
            python -m watch.mlops.manager "status packages" --dvc_dpath=$DVC_EXPT_DPATH
            """))

    def add_packages(self):
        """
        This does what repackage used to do.
        Repackages checkpoints as torch packages, copies them to the DVC repo,
        and then adds them to DVC.
        """
        mode = 'all'
        self.package_checkpoints(mode=mode)
        self.copy_packages_to_dvc(mode=mode)

    def push_packages(self):
        """
        This does what repackage used to do.
        Repackages checkpoints as torch packages, copies them to the DVC repo,
        and then adds them to DVC.

        >>> # xdoctest: +REQUIRES(env:DVC_EXPT_DPATH)
        >>> from watch.mlops.manager import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
        >>> self = ExperimentState(expt_dvc_dpath, dataset_code, data_dvc_dpath)
        >>> self.summarize()
        """
        mode = 'all'
        self.package_checkpoints(mode=mode)
        self.copy_packages_to_dvc(mode=mode)
        self.add_packages_to_dvc(mode=mode)


def checkpoint_filepath_info(fname):
    """
    Finds information encoded in the checkpoint/model file path.

    TODO:
        We need to ensure this info is encoded inside the file header as well!

    Ignore
        parse.parse('{prefix}foo={bar}', 'foo=3')
        parse.parse('{prefix}foo={bar}', 'afoao=3')

    Example:
        >>> from watch.mlops.old.expt_state import *  # NOQA
        >>> fnames = [
        >>>     'epoch1_step10.foo',
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
    suffix = ''.join(fname.partition('epoch')[1:])
    # Hack: making name assumptions
    parsers = [
        parse.Parser('epoch={epoch:d}-step={step:d}-{ckpt_ver}.{ext}'),
        parse.Parser('epoch={epoch:d}-step={step:d}.{ext}'),
        parse.Parser('epoch={epoch:d}-step={step:d}-{ckpt_ver}'),
        parse.Parser('epoch={epoch:d}-step={step:d}'),
        parse.Parser('epoch{epoch:d}_step{step:d}.{ext}'),
        parse.Parser('epoch{epoch:d}_step{step:d}'),
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


def summarize_tables(tables):
    """
    pip install rich-dataframe
    """
    from rich import print
    from rich.panel import Panel
    import rich
    console = rich.get_console()
    staging_df = tables.get('staging', None)
    versioned_df = tables.get('versioned', None)

    table_shapes = ub.udict(tables).map_values(lambda x: x.shape)
    title = '[blue] Table Summary'
    print(title)
    print('table_shapes = {}'.format(ub.repr2(table_shapes, nl=1, align=':', sort=0)))

    if staging_df is not None:
        title = '[yellow] Staging Summary (Training Checkpoints)'

        if len(staging_df):
            staging_df['needs_copy'] = (~staging_df['is_copied'])
            staging_df['needs_package'] = (~staging_df['is_packaged'])
            body_df = staging_df[['ckpt_exists', 'is_packaged', 'is_copied', 'needs_package', 'needs_copy']].sum().to_frame().T
            body = console.highlighter(str(body_df))
        else:
            body = console.highlighter('There are no unversioned staging items')
        print(Panel(body, title=title))

    _grouper_keys = ub.oset(['dataset_code', 'type'])

    if versioned_df is not None:
        title = ('[bright_green] Versioned Summary (Packaged Models)')
        # version_bitcols = ['has_raw', 'has_dvc', 'is_link', 'is_broken', 'needs_pull', 'needs_push']
        version_bitcols = ['has_raw', 'has_dvc', 'is_link', 'is_broken', 'needs_pull', 'needs_push']
        if len(versioned_df):
            grouper_keys = list(_grouper_keys & versioned_df.columns)
            body_df = versioned_df.groupby(grouper_keys)[version_bitcols].sum()
            body = console.highlighter(str(body_df))
        else:
            body = console.highlighter('There are no versioned items')
        print(Panel(body, title=title))


class UserAbort(Exception):
    pass


__config__ = ManagerConfig


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/mlops/mlops.manager.py "pull all"
        python -m watch.mlops.manager "status"
        python -m watch.mlops.manager "list"
        python -m watch.mlops.manager "pull packages"

        # python -m watch.mlops.manager "push packages"
        # python -m watch.mlops.manager "pull evals"

        # python -m watch.mlops.manager "evaluate"
        # python -m watch.mlops.manager "pull packages"
    """
    main(cmdline=True)
