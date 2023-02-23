import kwarray
import ubelt as ub


class SmartGlobalHelper:
    """
    A class for SMART-specific hacks and defaults for mlops

    Should be stateless
    """

    def shared_palletes(self, macro_table):
        # import kwplot
        import numpy as np
        import seaborn as sns
        from watch.utils import util_pandas

        def filterkeys(keys):
            return [k for k in keys if 'specified' not in k]

        _macro_table = util_pandas.DotDictDataFrame(macro_table)
        thresh_keys = filterkeys(_macro_table.query_column('thresh'))
        bas_thresh_keys = [k for k in thresh_keys if 'bas_poly' in k]

        # Pre determine some palettes
        shared_palletes_groups = []
        shared_palletes_groups.append(bas_thresh_keys)

        shared_palletes_groups.append(filterkeys(_macro_table.query_column('learning_rate')))

        shared_palletes_groups.append(filterkeys(_macro_table.query_column('chip_dims')))
        shared_palletes_groups.append(
            filterkeys(_macro_table.query_column('resolution')) +
            filterkeys(_macro_table.query_column('output_space_scale')) +
            filterkeys(_macro_table.query_column('output_resolution')) +
            filterkeys(_macro_table.query_column('input_space_scale')) +
            filterkeys(_macro_table.query_column('input_resolution')) +
            filterkeys(_macro_table.query_column('window_space_scale')) +
            filterkeys(_macro_table.query_column('window_resolution'))
        )

        param_to_palette = {}
        for group_params in shared_palletes_groups:
            try:
                unique_vals = np.unique(macro_table[group_params].values)
            except TypeError:
                unique_vals = set(macro_table[group_params].values.ravel())

            # 'Spectral'
            if len(unique_vals) > 5:
                unique_colors = sns.color_palette('Spectral', n_colors=len(unique_vals))
                # kwplot.imshow(_draw_color_swatch(unique_colors), fnum=32)
            else:
                unique_colors = sns.color_palette(n_colors=len(unique_vals))
            palette = ub.dzip(unique_vals, unique_colors)
            param_to_palette.update({p: palette for p in group_params})
        return param_to_palette

    def label_modifier(self):
        """
        Build the label modifier for the SMART task.
        """
        from watch.utils import util_kwplot
        modifier = util_kwplot.LabelModifier()

        modifier.add_mapping({
            'blue|green|red|nir': 'BGRN',
            'blue|green|red|nir,invariants.0:17': 'invar',
            'blue|green|red|nir|swir16|swir22': 'BGNRSH'
        })

        @modifier.add_mapping
        def humanize_label(text):
            text = text.replace('package_epoch0_step41', 'EVAL7')
            text = text.replace('params.', '')
            text = text.replace('metrics.', '')
            text = text.replace('fit.', 'fit.')
            return text
        return modifier

    def mark_star_models(self, macro_table):
        #### Hack for models of interest.
        star_params = []
        p1 = macro_table[(
            # (macro_table['bas_poly.moving_window_size'] == 200) &
            (macro_table['params.bas_pxl.package_fpath'] == 'package_epoch0_step41') &
            (macro_table['params.bas_pxl.chip_dims'] == '[128, 128]') &
            (macro_table['params.bas_poly.thresh'] == 0.12)  &
            (macro_table['params.bas_poly.max_area_sqkm'] == 'None') &
            (macro_table['params.bas_poly.moving_window_size'] == 'None')
        )]['param_hashid'].iloc[0]
        star_params = [p1]
        p2 = macro_table[(
            # (macro_table['bas_poly.moving_window_size'] == 200) &
            (macro_table['params.bas_pxl.package_fpath'] == 'Drop4_BAS_2022_12_15GSD_BGRN_V10_epoch=0-step=4305') &
            (macro_table['params.bas_pxl.chip_dims'] == '[224, 224]') &
            (macro_table['params.bas_poly.thresh'] == 0.13)  &
            (macro_table['params.bas_poly.max_area_sqkm'] == 'None') &
            (macro_table['params.bas_poly.moving_window_size'] == 200)
        )]['param_hashid'].iloc[0]
        star_params += [p2]
        p3 = macro_table[(
            # (macro_table['bas_poly.moving_window_size'] == 200) &
            (macro_table['params.bas_pxl.package_fpath'] == 'Drop4_BAS_15GSD_BGRNSH_invar_V8_epoch=16-step=8704') &
            (macro_table['params.bas_pxl.chip_dims'] == '[256, 256]') &
            (macro_table['params.bas_poly.thresh'] == 0.17)  &
            (macro_table['params.bas_poly.max_area_sqkm'] == 'None') &
            (macro_table['params.bas_poly.moving_window_size'] == 'None')
        )]['param_hashid'].iloc[0]
        star_params += [p3]
        macro_table['is_star'] = kwarray.isect_flags(macro_table['param_hashid'], star_params)

    def old_hacked_model_case(self, macro_table):
        from watch.utils.util_pandas import DotDictDataFrame
        fit_params = DotDictDataFrame(macro_table)['fit']
        unique_packages = macro_table['bas_pxl.package_fpath'].drop_duplicates()
        # unique_fit_params = fit_params.loc[unique_packages.index]
        pkgmap = {}
        pkgver = {}
        for id, pkg in unique_packages.items():
            pkgver[pkg] = 'M{:02d}'.format(len(pkgver))
            pid = pkgver[pkg]
            out_gsd = fit_params.loc[id, 'fit.output_space_scale']
            in_gsd = fit_params.loc[id, 'fit.input_space_scale']
            assert in_gsd == out_gsd
            new_name = f'{pid}'
            if pkg == 'package_epoch0_step41':
                new_name = f'{pid}_NOV'
            pkgmap[pkg] = new_name
        macro_table['bas_pxl.package_fpath'] = macro_table['bas_pxl.package_fpath'].apply(lambda x: pkgmap.get(x, x))

    def default_vantage_points(self, eval_type):
        if eval_type == 'bas_poly_eval':
            vantage_points = [
                {
                    'metric1': 'metrics.bas_poly_eval.bas_faa_f1',
                    'metric2': 'metrics.bas_poly_eval.bas_tpr',

                    'scale1': 'linear',
                    'scale2': 'linear',

                    'objective1': 'maximize',
                },

                {
                    'metric1': 'metrics.bas_poly_eval.bas_tpr',
                    'metric2': 'metrics.bas_poly_eval.bas_f1',

                    'objective1': 'maximize',

                    'scale1': 'linear',
                    'scale2': 'linear',
                },

                {
                    'metric1': 'metrics.bas_poly_eval.bas_ppv',
                    'metric2': 'metrics.bas_poly_eval.bas_tpr',

                    'scale1': 'linear',
                    'scale2': 'linear',

                    'objective1': 'maximize',
                },

                {
                    'metric1': 'metrics.bas_poly_eval.bas_f1',
                    'metric2': 'metrics.bas_poly_eval.bas_ffpa',

                    'scale1': 'linear',
                    'scale2': 'linear',

                    'objective1': 'maximize',
                },

                {
                    'metric1': 'metrics.bas_poly_eval.bas_space_FAR',
                    'metric2': 'metrics.bas_poly_eval.bas_tpr',

                    'scale1': 'linear',
                    'scale2': 'linear',

                    'objective1': 'minimize',
                },
            ]
        elif eval_type == 'bas_pxl_eval':
            vantage_points = [
                {
                    'metric1': 'metrics.bas_pxl_eval.salient_AP',
                    'metric2': 'metrics.bas_pxl_eval.salient_AUC',

                    'scale1': 'linear',
                    'scale2': 'linear',

                    'objective1': 'maximize',
                },
            ]

        return vantage_points

    def _default_metrics(self, agg):
        _display_metrics_suffixes = []
        if agg.type == 'bas_poly_eval':
            _display_metrics_suffixes = [
                'bas_tp',
                'bas_fp',
                'bas_fn',
                'bas_f1',
                'bas_ffpa',
                'bas_faa_f1',
                'bas_tpr',
                'bas_ppv',
            ]
            _primary_metrics_suffixes = [
                # 'bas_faa_f1'
                'bas_faa_f1',
            ]
        elif agg.type == 'sc_poly_eval':
            _primary_metrics_suffixes = [
                'sc_macro_f1', 'bas_faa_f1'
            ]
        elif agg.type == 'bas_pxl_eval':
            _primary_metrics_suffixes = [
                'salient_AP',
                'salient_APUC',
                'salient_AUC',
            ]
        elif agg.type == 'sc_pxl_eval':
            _primary_metrics_suffixes = [
                'coi_mAP',
                'coi_mAPUC',
                'coi_mAUC',
            ]
        else:
            raise NotImplementedError(agg.type)
        return _primary_metrics_suffixes, _display_metrics_suffixes


SMART_HELPER = SmartGlobalHelper()
