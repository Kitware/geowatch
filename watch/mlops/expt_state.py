"""
TODO:
    each unique run needs a pipeline-UUID so we can run the same experiment
    multiple times and not break the data model. We should be able to tell it
    to use any of the pre-existing variants with the same configuration though.
"""
import warnings
import parse
import pandas as pd
import ubelt as ub
from watch.utils import util_pattern
from watch.utils import util_path
from watch.mlops import repackager


class ExperimentState(ub.NiceRepr):
    """

    Ignore:
        >>> # xdoctest: +REQUIRES(env:EXPT_DVC_DPATH)
        >>> from watch.mlops.expt_state import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
        >>> dataset_code = '*'
        >>> dataset_code = 'Drop4-SC'
        >>> dvc_remote = 'aws'
        >>> self = ExperimentState(expt_dvc_dpath, dataset_code, dvc_remote, data_dvc_dpath)
        >>> self.summarize()

    Ignore:
        >>> # Just show patterns:
        >>> from watch.mlops.expt_state import *  # NOQA
        >>> self = ExperimentState('<expt_dpath>', '<dset_code>')
        >>> print('self.templates = {}'.format(ub.repr2(self.templates, nl=1, sort=0)))

    Ignore:
        table[table.type == 'pkg_fpath']['model'].unique()
    """

    def __init__(self, expt_dvc_dpath, dataset_code='*', dvc_remote=None,
                 data_dvc_dpath=None, model_pattern='*', storage_dpath=None):

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

        # Some of the "ids" are build from hashes of other configurations.
        # This denotes what deps should be hashed and in what order.
        self.hashid_dependencies = {
            'trk_poly_id': ['trk_model', 'test_trk_dset', 'trk_pxl_cfg', 'trk_poly_cfg'],
            'crop_id': ['regions_id', 'crop_cfg', 'crop_src_dset']
        }

        if storage_dpath is None or storage_dpath == 'auto':
            storage_dpath = expt_dvc_dpath / 'models/fusion'

        ### Experimental, add in SC dependencies
        self.staging_template_prefix = '{expt_dvc_dpath}/training/{host}/{user}/{dataset_code}/'
        self.storage_template_prefix = '{storage_dpath}/{dataset_code}/'

        self.storage_dpath = storage_dpath

        self.patterns = {
            # General
            'trk_expt': '*',
            'act_expt': '*',
            'expt_dvc_dpath': expt_dvc_dpath,
            'dataset_code': dataset_code,
            'storage_dpath': storage_dpath,
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
            # Staged checkpoint
            'ckpt': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{checkpoint}.ckpt',

            # Staged package
            'spkg': 'runs/{expt}/lightning_logs/{lightning_version}/checkpoints/{model}.pt',

            # Interrupted models
            'ipkg': 'runs/{expt}/lightning_logs/{lightning_version}/package-interupt/{model}.pt',
        }

        # directory suffixes after the pred/eval type
        task_dpath_suffix = {
            'trk_pxl_dpath'  : 'trk/{trk_model}/{test_trk_dset}/{trk_pxl_cfg}',
            'trk_poly_dpath' : 'trk/{trk_model}/{test_trk_dset}/{trk_pxl_cfg}/{trk_poly_cfg}',

            'act_pxl_dpath'  : 'act/{act_model}/{test_act_dset}/{act_pxl_cfg}',
            'act_poly_dpath' : 'act/{act_model}/{test_act_dset}/{act_pxl_cfg}/{act_poly_cfg}',

            'crop_dpath': 'crop/{crop_src_dset}/{regions_id}/{crop_cfg}/{crop_id}',
        }

        # TODO:
        # Can we abstract this so the only piece of user input is a definition
        # of a set of steps? The steps can define the path templates that they
        # want and the variables they use.

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
            raise RuntimeError(f'Failed to match path={path} to template={template}')
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

        # Find interrupted checkpoints
        key = 'ipkg'  # stands for staged package
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
            'pred_act_poly_sites_fpath'
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

                    ADD_CROPID_HACK = 0
                    # We will not do this for now and handle it in the result
                    # parser, but neither solution is great. Need a better way
                    # to find the "join" of these tables
                    if ADD_CROPID_HACK:
                        # special handling for adding tracking / cropping
                        # params to the activity row. We should figure out a
                        # way of making this more general in the future.
                        if row['type'] == 'pred_act_poly_sites_fpath':
                            if row['test_act_dset'].startswith('crop'):
                                # Fixme dataset name ids need a rework
                                crop_id = row['test_act_dset'].split('_crop.kwcoco')[0]

                                # There needs to be a search step for the crop
                                # dataset, which is not ideal.
                                pats = self.patterns.copy()
                                pats['crop_id'] = crop_id
                                pats = ub.udict(pats).map_values(str)
                                pat = self.templates['crop_fpath'].format(**pats)
                                _found = util_path.coerce_patterned_paths(pat)
                                if _found:
                                    assert len(_found) == 1, 'should not have dups here'
                                    found = _found[0]
                                    _crop_attrs = ub.udict(self._parse_pattern_attrs(self.templates['crop_fpath'], found))
                                    _crop_attrs = _crop_attrs - row
                                    row.update(_crop_attrs)
                                    # Can we find the tracking params too?
                                    # It looks like we'd need to parse out the
                                    # file, so no, not here. We need to change the
                                    # path scheme to fix that.
                                    # import json
                                    # dataset = json.loads(found.read_text())
                                    # self._parse_pattern_attrs(self.templates['crop_fpath']

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

            x = versioned_df.copy()
            x.loc[:, npred_keys] = 0
            for groupvals, subdf in group_to_versioned.items():
                associated = group_to_volitile.get(groupvals, None)
                if associated is not None:
                    counts = associated.value_counts('type').rename(lambda x: 'n_' + x, axis=0)
                    counts
                    # FIXME? Not sure what broke.
                    # versioned_df.loc[subdf.index, counts.index] += counts

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
            >>> from watch.mlops.expt_state import *  # NOQA
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
        >>> from watch.mlops.expt_state import *  # NOQA
        >>> import watch
        >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt')
        >>> data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
        >>> dataset_code = 'Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC'
        >>> self = ExperimentState(expt_dvc_dpath, dataset_code, data_dvc_dpath)
        >>> self.summarize()
        """
        from rich.prompt import Confirm

        perf_config = {
            'push_workers': 8,
        }

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
        print('to_repackage = {}'.format(ub.repr2(to_repackage, nl=1)))
        if to_repackage:
            # NOTE: THIS RELIES ON KNOWING ABOUT THE SPECIFIC MODEL CODE.
            # IT WOULD BE NICE IF WE DIDN'T NEED THAT HERE.
            repackager.repackage(to_repackage)

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
            dvc_api.push(
                toadd_pkg_fpaths, remote=self.dvc_remote,
                jobs=perf_config['push_workers'],
                recursive=True)

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

            python -m watch.mlops.expt_state "pull packages" --dvc_dpath=$DVC_EXPT_DPATH
            python -m watch.mlops.expt_state "status packages" --dvc_dpath=$DVC_EXPT_DPATH
            python -m watch.mlops.expt_state "evaluate" --dvc_dpath=$DVC_EXPT_DPATH

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
        from watch.utils.reverse_hashid import condense_config
        cfgstr = condense_config(params, type)
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
        if model is None:
            return None
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

    table_shapes = ub.udict(tables).map_values(lambda x: x.shape)
    title = '[blue] Table Summary'
    print(title)
    print('table_shapes = {}'.format(ub.repr2(table_shapes, nl=1, align=':', sort=0)))

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

    _grouper_keys = ub.oset([
        'dataset_code',
        # 'test_trk_dset',
        # 'test_act_dset',
        'type'
    ])

    if volitile_df is not None:
        title = ('[bright_blue] Volitile Summary (Predictions)')
        if len(volitile_df):
            grouper_keys = list(_grouper_keys & volitile_df.columns)
            num_pred_types = volitile_df.groupby(grouper_keys, dropna=False).nunique()
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
        >>> from watch.mlops.expt_state import *  # NOQA
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


class UserAbort(Exception):
    pass
