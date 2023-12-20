
def best_separating_threshold(x, y):
    from kwcoco.metrics.confusion_vectors import BinaryConfusionVectors
    import kwarray
    data = kwarray.DataFrameArray({
        'is_true': y.values,
        'pred_score': x.values,
    })
    bincfsn_vec = BinaryConfusionVectors(data)
    dct = bincfsn_vec.measures().asdict()
    f1_score, f1_thresh = dct['_max_f1']
    return f1_score, f1_thresh


def find_relevant_stuff():
    # import watch
    from watch.mlops.aggregate import AggregateLoader
    import ubelt as ub

    target = '/home/joncrall/remote/namek/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_namek_v1'
    # expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')

    load_kwargs = {
        'target': [
            target
        ],
        'pipeline': 'sc',
        'io_workers': 'avail',
    }
    with ub.Timer('load'):
        loader = AggregateLoader(**load_kwargs)
        eval_type_to_agg = loader.coerce_aggregators()
    _agg = eval_type_to_agg['sc_poly_eval']
    agg = _agg.filterto(models=['Drop7-Cropped2GSD_SC_bgrn_gnt_split6_V84_epoch17_step1548'])
    agg.build_macro_tables()
    best = agg.report_best()
    top_param_hashid = list(best.top_param_lut.keys())[-1]
    _agg = _agg.filterto(param_hashids=[top_param_hashid])

    poly_fpaths = _agg.index['fpath'].to_list()
    # Manually run confusion analysis

    confusion_paths = []
    for p in poly_fpaths:
        p = ub.Path(p)
        print(p.parent)
        pred_dpath = p.parent / 'confusion_analysis/confusion_groups/pred/'
        if pred_dpath.exists():
            confusion_paths.append(pred_dpath)

    sites = []
    from watch.geoannots import geomodels
    for dpath in confusion_paths:
        sites.extend(list(geomodels.SiteModel.coerce_multiple(dpath, workers=8)))
    site_score_analysis(sites)


def relevant_stuff_case1():
    dpath = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_ac_static_small_baseline_v1/eval/flat/sc_poly_eval/sc_poly_eval_id_372c0c95/confusion_analysis/confusion_groups/pred/'
    from watch.geoannots import geomodels
    sites = list(geomodels.SiteModel.coerce_multiple(dpath, workers=8))
    site_score_analysis(sites)


def site_score_analysis(sites):
    """
    """
    import pandas as pd
    import ubelt as ub
    from watch.cli.run_tracker import smooth_observation_scores
    # import geowatch_tpl
    # from watch.utils import util_gis
    # shapestats = geowatch_tpl.import_submodule('shapestats')

    simple_types = {
        'unhandled_cfsn_system_rejected': 'negative',
        'sm_pos_match': 'positive',
        'sm_completely_wrong': 'negative',
        'sm_ignore': 'ignore',
    }
    import numpy as np

    grid = list(ub.named_product({
        # 'smooth_mode': ['conv3', 'ewma'],
        'smooth_mode': ['ewma'],
        # 'smoothing': [0.6, 0.63, 0.65, 0.66, 0.7],
        'smoothing': [0.66],
    }))

    meta_rows = []
    for grid_item in grid:
        obs_rows = []

        site_rows = []
        for site in ub.ProgIter(sites):

            confusion = site.header['properties']['cache']['confusion']
            cfsn_type = confusion['type']
            ideal_label = simple_types[cfsn_type]

            if ideal_label == 'ignore':
                continue

            positive = (ideal_label == 'positive')

            cfsn_status = confusion.get('te_association_status', 'tn')
            if cfsn_status == 'tn':
                assert cfsn_type == 'unhandled_cfsn_system_rejected'

            if cfsn_status == '0':
                if cfsn_type == 'sm_pos_match':
                    cfsn_status = 'tp'
                else:
                    cfsn_status = 'tn'
                    raise Exception

            observations = list(site.observations())

            # current_phase = observations['current_phase']
            obs_multi_scores = [obs['properties']['cache']
                                ['raw_multi_scores'] for obs in observations]
            assert all(len(ms) == 1 for ms in obs_multi_scores)
            obs_scores = [ms[0] for ms in obs_multi_scores]
            raw_scores = pd.DataFrame(obs_scores)

            smooth_observation_scores(observations, smoothing=grid_item['smoothing'], smooth_mode=grid_item['smooth_mode'])
            smooth_scores = pd.DataFrame([
                obs['properties']['cache']['smooth_scores'] for obs in observations])

            from kwutil.util_time import coerce_datetime
            start_date = coerce_datetime(observations[0]['properties']['observation_date'])

            for idx, obs in enumerate(observations):
                obs_time = coerce_datetime(obs['properties']['observation_date'])
                s1 = obs['properties']['cache']['raw_multi_scores'][0]
                s2 = obs['properties']['cache']['smooth_scores']
                s1 = {'raw_' + k: v for k, v in s1.items()}
                s2 = {'smooth_' + k: v for k, v in s2.items()}
                row = s1 | s2
                row['site_id'] = site.site_id
                row['index'] = idx
                duration = obs_time - start_date
                row.update({
                    'positive': positive,
                    'region_id': site.region_id,
                    'cfsn_status': cfsn_status,
                    'duration': duration,
                })
                obs_rows.append(row)

            curr_labels = [o['properties']['current_phase'] for o in observations]
            active_labels = {'Site Preparation', 'Active Construction'}
            flags = [lbl in active_labels for lbl in curr_labels]

            # site_crs84_gdf = site.pandas_site()
            # site_utm_gdf = util_gis.project_gdf_to_local_utm(site_crs84_gdf, mode=1)
            # df = site_utm_gdf
            # import kwimage
            # obox_whs = [kwimage.MultiPolygon.from_shapely(s).oriented_bounding_box().extent
            #             for s in df.geometry]
            # df['obox_major'] = [max(e) for e in obox_whs]
            # df['obox_minor'] = [min(e) for e in obox_whs]
            # df['major_obox_ratio'] = df['obox_major'] / df['obox_minor']
            # df['area'] = df.geometry.area
            # df['isoperimetric_quotient'] = df.geometry.apply(shapestats.ipq)
            # df['boundary_amplitude'] = df.geometry.apply(shapestats.compactness.boundary_amplitude)
            # df['eig_seitzinger'] = df.geometry.apply(shapestats.compactness.eig_seitzinger)

            # subscores = raw_scores[flags]
            # [flags]
            # if subscores.isna().any():
            #     raise Exception

            site_rows.append({
                # 'rt_area': np.sqrt(df.iloc[0]['area']),
                # 'isoperimetric_quotient': df['isoperimetric_quotient'].iloc[0],
                # 'boundary_amplitude': df['boundary_amplitude'].iloc[0],
                # 'eig_seitzinger': df['eig_seitzinger'].iloc[0],
                # 'major_obox_ratio': df['major_obox_ratio'].iloc[0],
                'frac_active': np.mean(flags),

                'smooth_salient_mean': smooth_scores['ac_salient'].mean(),
                'smooth_salient_max': smooth_scores['ac_salient'].max(),

                'raw_salient_mean': raw_scores['ac_salient'].mean(),
                'raw_salient_max': raw_scores['ac_salient'].max(),
                'raw_salient_min': raw_scores['ac_salient'].min(),

                'active_max': raw_scores['Active Construction'].max(),
                'active_mean': raw_scores['Active Construction'].mean(),

                'siteprep_max': raw_scores['Site Preparation'].max(),
                'siteprep_mean': raw_scores['Site Preparation'].mean(),

                # 'cfsn_type': cfsn_type,
                'positive': positive,
                'region_id': site.region_id,
                'cfsn_status': cfsn_status,
            })

        import kwimage
        palette = {
            'tp': kwimage.Color.coerce('kitware_green').as01(),
            'fp': kwimage.Color.coerce('kitware_red').as01(),
            'tn': kwimage.Color.coerce('kitware_blue').as01(),
        }

        if 0:
            import kwplot
            sns = kwplot.autosns()
            obs_df = pd.DataFrame(obs_rows)
            fig = kwplot.figure(fnum=1, doclf=True)
            ax = fig.gca()
            ax.cla()
            for _, grp in ub.ProgIter(list(obs_df.groupby('site_id'))):
                sns.lineplot(
                    data=grp, x='duration', y='smooth_ac_salient', hue='cfsn_status',
                    legend=False, ax=ax, palette=palette)

        scalars = pd.DataFrame(site_rows)
        if 0:
            scalars = scalars[scalars['region_id'] == 'KR_R002']
        scalars = scalars.fillna(0)

        y = scalars['positive']
        X = scalars.drop(['positive', 'region_id', 'cfsn_status'], axis=1)
        # Find the most informative feature
        rows = []
        for k in X.columns:
            x = X[k]
            best_f1, best_thresh = best_separating_threshold(x, y)
            rows.append({
                'key': k,
                'best_f1': best_f1,
                'best_thresh': best_thresh,
            })
        separators = pd.DataFrame(rows).sort_values('best_f1')
        separators = separators.set_index('key')
        print(separators)

        meta_row = separators.loc['smooth_salient_max'].to_dict()
        meta_row.update(grid_item)
        meta_rows.append(meta_row)

    print(pd.DataFrame(meta_rows).sort_values('best_f1'))

    if 0:
        # Do it per region
        rows = []
        for group_key, group in scalars.groupby('region_id'):
            y = group['positive']
            X = group.drop(['positive', 'region_id', 'cfsn_status'], axis=1)
            for k in X.columns:
                x = X[k]
                best_f1, best_thresh = best_separating_threshold(x, y)
                rows.append({
                    'key': k,
                    'best_f1': best_f1,
                    'best_thresh': best_thresh,
                    'group_key': group_key,
                })
        separators = pd.DataFrame(rows).sort_values('best_f1')
        separators = separators.set_index('key')
        print(separators)
        piv = separators.reset_index().pivot(index=['group_key'], columns=['key'], values=['best_thresh'])
        print(piv.T)
        piv.T.loc[piv.std(axis=0).T.sort_values().index]

    import kwplot
    sns = kwplot.autosns()

    sep1 = separators.iloc[-1]
    # sep2 = separators.iloc[-2]
    # key1 = sep1.name
    # key2 = sep2.name
    key1 = 'smooth_salient_max'
    key2 = 'frac_active'
    sep1 = separators.loc[key1]
    # sep2 = separators.loc[key2]

    fig = kwplot.figure(fnum=1, doclf=True)
    ax = fig.gca()
    ax.cla()
    sns.scatterplot(
        data=scalars,
        # x='siteprep_max',
        # y='siteprep_mean',
        x=key1,
        y=key2,
        hue='cfsn_status',
        ax=ax,
        palette=palette,
        style='region_id',
    )
    thresh = sep1['best_thresh']
    ymin, ymax = ax.get_ylim()
    ax.plot([thresh, thresh], [ymin, ymax], '-')


def foo(scalars):
    import sklearn.ensemble
    sklearn.ensemble.RandomForestClassifier

    y = scalars['positive']
    X = scalars.drop(['positive'], axis=1)
    rf = sklearn.ensemble.RandomForestClassifier().fit(X, y)

    y_hat = rf.predict(X)
    y.values & y_hat

    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.feature_selection import SelectFromModel
    from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import LeaveOneOut
    # from sklearn.model_selection import cross_val_score
    # from sklearn.pipeline import make_pipeline

    # feature_selector = SelectFromModel(
    #     RandomForestRegressor(n_jobs=-1), threshold="mean"
    # )
    # pipe = make_pipeline(
    #     feature_selector, RandomForestRegressor(n_jobs=-1)
    # )

    # param_grid = {
    #     # define the grid of the random-forest for the feature selection
    #     "selectfrommodel__estimator__n_estimators": [10, 20],
    #     "selectfrommodel__estimator__max_depth": [3, 5],
    #     # define the grid of the random-forest for the prediction
    #     "randomforestregressor__n_estimators": [10, 20],
    #     "randomforestregressor__max_depth": [5, 8],
    # }
    # grid_search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=3)
    # # You can use the LOO in this way. Be aware that this not a good practise,
    # # it leads to large variance when evaluating your model.
    # # scores = cross_val_score(pipe, X, y, cv=LeaveOneOut(), error_score='raise')
    # scores = cross_val_score(grid_search, X, y, cv=2, error_score='raise')
    # scores.mean()

    # https://stats.stackexchange.com/questions/568695/using-k-fold-cross-validation-of-random-forest-how-many-samples-are-used-to-cre
    param_grid = {
        'max_depth': [2, 3, 4, 5],
        # 'criterion': ['absolute_error', 'squared_error'],
        'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200]
    }
    # Create a base model
    rfCV = sklearn.ensemble.RandomForestClassifier(random_state=1099)

    # Instantiate the grid search model
    regCV = GridSearchCV(
        estimator=rfCV,
        cv=5,
        param_grid=param_grid,
        n_jobs=-1,
        verbose=2,
        return_train_score=True)
    # Fit the grid search to the data
    out = regCV.fit(X, y)
    out.best_score_
