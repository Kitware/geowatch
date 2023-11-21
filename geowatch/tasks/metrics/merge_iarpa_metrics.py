"""
Code to consolidate and merge IARPA results across regions.
"""
import json
import pandas as pd
import numpy as np
import shapely.geometry
import shapely.ops
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import ubelt as ub
from geowatch.heuristics import PHASES as phases


def _read(fpath):
    if fpath.is_file():
        _df = pd.read_csv(fpath, index_col=0)
        if not _df.empty:
            return _df


@dataclass(frozen=True)
class RegionResult:
    region_id: str  # 'KR_R001'
    region_model: Dict
    site_models: List[Dict]
    bas_dpath: Optional[ub.Path] = None  # 'path/to/scores/latest/KR_R001/bas/'
    sc_dpath: Optional[ub.Path] = None   # 'path/to/scores/latest/KR_R001/phase_activity/'
    unbounded_site_status: Optional[Literal['completed', 'partial', 'overall']] = None

    @classmethod
    def from_dpath_and_anns_root(cls, region_dpath,
                                 true_site_dpath, true_region_dpath,
                                 unbounded_site_status='overall'):
        region_dpath = ub.Path(region_dpath)
        region_id = region_dpath.name

        # TODO use overall instead of completed?
        bas_dpath = region_dpath / unbounded_site_status / 'bas'
        bas_dpath = bas_dpath if bas_dpath.is_dir() else None

        sc_dpath = region_dpath / unbounded_site_status / 'phase_activity'
        sc_dpath = sc_dpath if sc_dpath.is_dir() else None

        region_fpath = true_region_dpath / (region_id + '.geojson')
        with open(region_fpath, 'r') as f:
            region_model = json.load(f)

        site_fpaths = sorted(true_site_dpath.glob(f'{region_id}_*.geojson'))
        site_models = [
            json.loads(open(pth).read())
            for pth in site_fpaths
        ]
        return cls(region_id, region_model, site_models,
                   bas_dpath, sc_dpath, unbounded_site_status)

    @property
    def bas_df(self):
        '''
        index:
            region_id, rho, tau
        columns:
            same as merge_bas_metrics_results

        '''
        bas_dpath, region_id = self.bas_dpath, self.region_id
        scoreboard_fpath = bas_dpath / 'scoreboard.csv'
        if scoreboard_fpath.exists():
            scoreboard = pd.read_csv(scoreboard_fpath)
            scoreboard = scoreboard.iloc[:, 1:].copy()
        else:
            import parse
            # ugg we have to parse rho out of the filename.
            fpaths = list(bas_dpath.glob('scoreboard_rho=*.csv'))
            if len(fpaths) == 0:
                raise ValueError('no scoreboards')
            parser = parse.Parser('scoreboard_rho={rho:f}.csv')
            rows = []
            for fpath in fpaths:
                # yo dawg
                rho = parser.parse(fpath.name)['rho']
                row = pd.read_csv(fpath)
                row['rho'] = rho
                rows.append(row)
            scoreboard = pd.concat(rows)
        scoreboard['region_id'] = region_id
        scoreboard = scoreboard.set_index(['region_id', 'rho', 'tau'])
        return scoreboard

    @property
    def site_ids(self) -> List[str]:
        '''
        There are a few possible sets of sites it would make sense to return here.
        - all gt sites
        - "eligible" gt sites that could be matched against, ie with status ==
        "predicted*". This depends on temporal_unbounded handling choice of
        completed, partial, or overall.
        - "matched" gt sites with at least 1 observation matched to at least 1
        observation in a proposed site.

        Currently we are returning "matched" for consistency with the metrics
        framework, but we should consider trying "eligible" to decouple BAS and
        SC metrics; i.e. it would no longer be possible to do worse on SC by
        doing better on BAS.
        '''
        # TODO check out root/(gt|sm)_sites.csv for this
        if 0:    # read all/eligible sites from region json
            site_ids = []
            for s in self.region_model['features']:
                if s['properties']['type'] == 'site_summary':
                    # TODO enumerate good statuses
                    # this should really be done in the metrics framework itself
                    if 'positive' in s['properties']['status']:
                        # TODO adjust sequestered site ids
                        # this will fail for demodata without it
                        site_ids.append(s['properties']['site_id'])
        elif 0:  # read all/eligible sites from site jsons
            site_ids = []
            for site in self.site_models:
                s = site['features'][0]
                if s['properties']['type'] == 'site':
                    if 'positive' in s['properties']['status']:
                        site_ids.append(s['properties']['site_id'])
        elif 0:  # read matched sites from phase table csv
            ph = self.sc_phasetable
            site_ids = ph.index.get_level_values(1).unique()  # assert sites == csv names
            # site_candidates = ph.index.get_level_values(2).unique()
            # if len(site_candidates) != len(sites):
            #     raise NotImplementedError
        else:   # read matched sites from other csvs
            site_ids = ub.oset([])
            fnames = ['ac_confusion_matrix', 'ac_f1', 'ac_temporal_error', 'ap_temporal_error']
            # could parallelize, scales with no. of detected sites
            for fname in fnames:
                for p in self.sc_dpath.glob(f'{fname}_*.csv'):
                    site_id = p.with_suffix('').name.replace(fname + '_', '')
                    if site_id != 'all_sites':
                        site_ids.append(site_id)

        return list(site_ids)

    @property
    def sc_df(self):
        '''
        index:
            region_id, site_id, [predicted] phase (w/o No Activity)
            incl. special site_id __avg__
                F1: micro (or option for macro)
                TIoU: ~micro over all truth-prediction pairs, skipping
                    undetected truth sites
                TE(p): micro
                confusion: micro
        columns:
            F1, TIoU, TE, TEp, [true] phase (incl. No Activity)

        confusion matrix and f1 scores apprently ignore subsites,
        so we must do the same
        https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/24
        MWE:
        >>> from sklearn.metrics import f1_score, confusion_matrix
        >>> f1 = f1_score(['a,a', 'a'], ['a,a', 'b'], labels=['a', 'b'],
        >>>               average=None)
        >>> confusion_matrix(['a,a', 'a'], ['a,a', 'b'], labels=['a', 'b'])
        array([[0, 1],
               [0, 0]])
        '''
        sc_dpath, sites = self.sc_dpath, self.site_ids

        df = pd.DataFrame(
            index=pd.MultiIndex.from_product((list(sites) + ['__avg__'], phases[1:]), names=['site', 'phase']),
            columns=(['F1', 'TIoU', 'TE', 'TEp'] + phases)
        )

        # per-site metrics
        # TODO parallelize, scales with no. of detected sites

        for site in sites:

            if (site_df := _read(sc_dpath / f'ac_f1_{site}.csv')) is not None:
                df.loc[(site, site_df.columns), 'F1'] = site_df.loc['F1 score'].values

            if (cm_df := _read(sc_dpath / f'ac_confusion_matrix_{site}.csv')) is not None:
                df.loc[(site, cm_df.index), cm_df.columns] = cm_df.values

            if (te_df := _read(sc_dpath / f'ac_temporal_error_{site}.csv')) is not None:
                df.loc[(site, te_df.columns), 'TE'] = te_df.iloc[0].values.flatten()

            if (tep_df := _read(sc_dpath / f'ap_temporal_error_{site}.csv')) is not None:
                df.loc[(site, tep_df.columns), 'TEp'] = tep_df.iloc[0].values.flatten()

        # already-calculated avg metrics

        # only defined for SP, AC
        if (f1_df := _read(sc_dpath / 'ac_f1_all_sites.csv')) is not None:
            df.loc[('__avg__', f1_df.columns), 'F1'] = f1_df.loc['F1 micro average'].values

        # TODO use phasetable instead to get true no activity?
        if (cm_df := _read(sc_dpath / 'ac_confusion_matrix_all_sites.csv')) is not None:
            df.loc[('__avg__', cm_df.index), cm_df.columns] = cm_df.values

        if (te_df := _read(sc_dpath / 'ac_temporal_error.csv')) is not None:
            df.loc[('__avg__', te_df.columns), 'TE'] = te_df.iloc[0].values.flatten()

        if (tep_df := _read(sc_dpath / 'ap_temporal_error.csv')) is not None:
            df.loc[('__avg__', tep_df.columns), 'TEp'] = tep_df.iloc[0].values.flatten()

        # TODO differentiate between zero and missing values in merge instead of fillna?
        # TODO doesn't handle oversegmentation of a truth site (n_pred > n_true)
        if (tiou_df := _read(sc_dpath / 'ac_tiou.csv')) is not None:
            tiou_df.index = tiou_df.index.str.replace('site truth ', '').str.split(' vs. ').str[0]
            df.loc[(tiou_df.index, tiou_df.columns), 'TIoU'] = tiou_df.values.flatten()
        df.loc['__avg__', 'TIoU'] = df.loc[sites, 'TIoU'].fillna(0).groupby(level='phase', sort=False).mean()

        df['region_id'] = self.region_id
        df = df.reset_index().set_index(['region_id', 'site', 'phase'])

        df[['F1', 'TIoU', 'TE', 'TEp']] = df[['F1', 'TIoU', 'TE', 'TEp']].astype(float)
        df[phases] = df[phases].astype("Int64")
        return df

    @property
    def sc_te_df(self):
        '''
        More detailed temporal error results; main value is included in sc_df.
        index:
            region_id, (site | __micro__), (ac | ap), phase

        columns:
            mean days (all detections)  <-- main value
            std days (all)
            mean days (early detections)
            std days (early)
            mean days (late detections)
            std days (late)
            all detections
            early
            late
            perfect
            missing proposals
            missing truth sites
        '''
        raise NotImplementedError

    @property
    def sc_phasetable(self):
        '''
        Currently used only for Gantt chart viz. Could be used to recalculate
        all SC metrics for micro-average.

        This excludes gt sites with no matched proposals and proposals with no
        matched gt sites.
        '''
        region_id, sc_dpath = self.region_id, self.sc_dpath

        delim = ' vs. '

        if (df := _read(sc_dpath / 'ac_phase_table.csv')) is not None:

            # df['date'] = pd.to_datetime(df['date'])
            if 'date' in df:  # could already be an idx
                df.set_index('date')
            df = df.fillna('NA' + delim + 'NA').astype('string')
            df = df.melt(ignore_index=False)
            df_sites = pd.DataFrame(
                df['variable'].str.split(delim).tolist(),
                columns=['site', 'site_candidate'],
                index=df.index)
            df_sites['site'] = df_sites['site'].str.replace('site truth ', '')
            df_sites['site_candidate'] = df_sites['site_candidate'].str.replace('site model ', '')
            df_slices = pd.DataFrame(
                df['value'].str.split(delim).tolist(),
                columns=['true', 'pred'],
                index=df.index)
            df = pd.concat((df_sites, df_slices), axis=1)
            df['region_id'] = region_id
            df = df.reset_index().set_index(['region_id', 'site', 'site_candidate', 'date'])
            df = (df.replace(['NA', '[]', None], pd.NA)
                    .astype('string')
                    .apply(lambda s: s.str.replace("'", '').str.strip('{}')))
            # df = df.apply(lambda s: s.str.split(', '))

            return df


def merge_bas_metrics_results(bas_results: List[RegionResult], fbetas: List[float]):
    '''
    Merge BAS results and return as a pd.DataFrame

    with MultiIndex([region_id', 'rho', 'tau'])
    incl. special region_ids __micro__, __macro__

    and columns:
        min_area                  int64
        tp sites                  int64
        tp exact                  int64
        tp under                  int64
        tp under (IoU)            int64
        tp under (IoT)            int64
        tp over                   int64
        fp sites                  int64
        fp area                 float64
        ffpa                    float64
        proposal area           float64
        fpa                     float64
        fn sites                  int64
        truth annotations         int64
        truth sites               int64
        proposed annotations      int64
        proposed sites            int64
        total sites               int64
        truth slices              int64
        proposed slices           int64
        precision               float64
        recall (PD)             float64
        F1                      float64
        spatial FAR             float64
        temporal FAR            float64
        images FAR              float64
    '''

    #
    # --- Helper functions for FAR ---
    #

    def area(regions):
        # ref: metrics-and-test-framework.evaluation.GeometryUtil
        def scale_area(lat):
            """
            Find square meters per degree for a given latitude based on
            EPSG:4326 :param lat: average latitude

            note that both latitude and longitude scales are dependent on
            latitude only
            https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree

            :return: square meters per degree for latitude coordinate
            """

            lat *= np.pi / 180.0  # convert to radians
            lat_scale = (111132.92 - (559.82 * np.cos(2 * lat)) +
                         (1.175 * np.cos(4 * lat)) -
                         (0.0023 * np.cos(6 * lat)))
            lon_scale = ((111412.84 * np.cos(lat)) - (93.5 * np.cos(3 * lat)) +
                         (0.118 * np.cos(5 * lat)))

            return lat_scale * lon_scale

        def area_sqkm(poly):
            avg_lat = (poly.bounds[1] + poly.bounds[3]) / 2.0
            return poly.area * scale_area(avg_lat) / 1e6

        polys = []
        for region in regions:

            region_feats = [
                feat for feat in region['features']
                if feat['properties']['type'] == 'region'
            ]
            assert len(region_feats) == 1
            region_feat = region_feats[0]

            polys.append(shapely.geometry.shape(region_feat['geometry']))

        regions_poly = shapely.ops.unary_union(polys)

        if isinstance(regions_poly, shapely.geometry.Polygon):
            return area_sqkm(regions_poly)
        elif isinstance(regions_poly, shapely.geometry.MultiPolygon):
            return sum(map(area_sqkm, regions_poly.geoms))
        else:
            raise TypeError(regions_poly)

    def n_dates(regions):
        dates = []
        for region in regions:

            region_feats = [
                feat for feat in region['features']
                if feat['properties']['type'] == 'region'
            ]
            assert len(region_feats) == 1
            region_feat = region_feats[0]

            dates.append(
                pd.date_range(region_feat['properties']['start_date'],
                              region_feat['properties']['end_date']))

            # return len(dates[0].union_many(dates[1:])) - 1
            return len(dates[0].union(dates[1:])) - 1

    def n_unique_images(sites):
        sources = set.union(*[{
            feat['properties']['source']
            for feat in site['features']
            if feat['properties']['type'] == 'observation'
        } for site in sites])
        return len(sources)

    #
    # --- Main logic ---
    #

    # all regions

    concat_df = pd.concat([r.bas_df for r in bas_results])

    sum_cols = [
        'tp sites', 'fp sites', 'fn sites', 'truth sites', 'proposed sites',
        'total sites', 'truth slices', 'proposed slices',
        'tp exact', 'tp over', 'tp under', 'tp under (IoT)', 'tp under (IoU)',
        'proposed annotations', 'truth annotations',
        'proposal area', 'fp area',
    ]
    mean_cols = [
        'precision', 'recall (PD)', 'F1',
        'spatial FAR', 'temporal FAR', 'images FAR',
        'fpa', 'ffpa',
    ]
    # alternate sweep param w/ rho, tau. range(0, 50000, 1000)
    # drop_cols = [ 'min_area', ]

    #
    # micro-average over sites
    #

    group_keys = list((ub.oset(concat_df.index.names) | concat_df.columns) & ['rho', 'tau', 'min_area'])

    micro_df = concat_df.groupby(group_keys)[sum_cols].sum()
    micro_df.loc[:, 'region_id'] = '__micro__'
    micro_df = micro_df.reset_index().set_index(['region_id', 'rho', 'tau'])

    # ref: metrics-and-test-framework.evaluation.Metric
    (_, tp), (_, fp), (_, fn) = micro_df[
        ['tp sites', 'fp sites', 'fn sites']].iteritems()
    micro_df['precision'] = np.where(tp > 0, tp / (tp + fp), 0)
    micro_df['recall (PD)'] = np.where(tp > 0, tp / (tp + fn), 0)
    micro_df['F1'] = np.where(tp > 0, tp / (tp + 0.5 * (fp + fn)), 0)

    all_regions = [r.region_model for r in bas_results]
    all_area = area(all_regions)
    # ref: metrics-and-test-framework.evaluation.Evaluation.build_scoreboard
    micro_df['spatial FAR'] = fp.astype(float) / all_area
    micro_df['temporal FAR'] = fp.astype(float) / n_dates(all_regions)

    # this is not actually how Images FAR is calculated!
    # https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/23
    #
    # all_sites = list(itertools.chain.from_iterable([
    #     r.site_models for r in bas_results]))
    # micro_df['images FAR'] = fp.astype(float) / n_unique_images(all_sites)
    #
    # instead, images in multiple proposed site stacks are double-counted.
    # take advantage of this to merge this metric with a simple average.
    n_images = (concat_df['fp sites'] /
                concat_df['images FAR']).groupby('region_id').mean().sum()
    micro_df['images FAR'] = fp.astype(float) / n_images

    # assume proposals are disjoint between regions
    micro_df['fpa'] = micro_df['proposal area'] / all_area
    micro_df['ffpa'] = micro_df['fp area'] / all_area

    #
    # compute fbeta scores
    #

    for fbeta in fbetas:
        (_, tp), (_, fp), (_, fn) = concat_df[
            ['tp sites', 'fp sites', 'fn sites']].iteritems()
        ftp = (1 + fbeta**2) * tp
        concat_df[f'F{fbeta:.2f}'] = np.where(tp > 0, (ftp / (ftp + (fbeta**2 * fn) + fp)), 0)

        (_, tp), (_, fp), (_, fn) = micro_df[
            ['tp sites', 'fp sites', 'fn sites']].iteritems()
        ftp = (1 + fbeta**2) * tp
        micro_df[f'F{fbeta:.2f}'] = np.where(tp > 0, (ftp / (ftp + (fbeta**2 * fn) + fp)), 0)

        mean_cols.append(f'F{fbeta:.2f}')

    #
    # macro-average over regions
    #

    macro_df = pd.concat(
        (concat_df.groupby(group_keys)[sum_cols].sum(),
         concat_df.groupby(group_keys)[mean_cols].mean()),
        axis=1
    )
    macro_df.loc[:, 'region_id'] = '__macro__'
    macro_df = macro_df.reset_index().set_index(['region_id', 'rho', 'tau'])

    df = pd.concat(
        (concat_df, micro_df, macro_df),
        axis=0
    )
    return df


def merge_sc_metrics_results(sc_results: List[RegionResult]):
    '''
    Merge SC results and return as a pd.DataFrame

    with MultiIndex(['region_id', 'phase'])
    incl. special region_ids
    __micro__: micro-avg over regions (normalize by n_sites per region)
    __macro__: macro-avg over regions
    In neither case do we weight by the length/size of individual sites.

    and columns:
        F1                     float64
        TIoU                   float64
        TE                     float64
        TEp                    float64
        No Activity              int64
        Site Preparation         int64
        Active Construction      int64
        Post Construction        int64

    Notes:
        - For confusion matrix, rows are pred and cols are true.
        - Confusion matrix is never normalized, so macro == micro.
        - F1 is only defined for SP and AC.
        - TEp is temporal error of next predicted phase
        - merged TE(p) is RMSE, so nonnegative, but regions' TE(p) can be
            negative.
        - TE is temporal error of current phase
        - TEp is temporal error of next predicted phase
    '''
    dfs = [r.sc_df for r in sc_results]
    concat_df = pd.concat(dfs, axis=0)
    # concat_df = concat_df.sort_values('date')

    sites_df = concat_df.query('site != "__avg__"')
    avg_df = concat_df.query('site == "__avg__"')

    def merge(df, region_id):
        g = df.groupby(level='phase', sort=False)
        merged_df = pd.concat(
            (g[['F1', 'TIoU']].agg(lambda s: s.fillna(0).mean()),
             g[['TE', 'TEp']].agg(lambda s: np.sqrt(np.mean(s.dropna() ** 2))),
             g[phases].sum()),
            axis=1
        )
        merged_df['region_id'] = region_id
        merged_df = merged_df.reset_index().set_index(['region_id', 'phase'])
        return merged_df

    macro_df = merge(avg_df, '__macro__')
    micro_df = merge(sites_df, '__micro__')

    df = pd.concat(
        (avg_df.droplevel('site'), macro_df, micro_df),
        axis=0
    )
    return df


def merge_metrics_results(region_dpaths, true_site_dpath, true_region_dpath, fbetas):
    '''
    Merge metrics results from multiple regions.

    Args:
        region_dpaths: List of directories containing the subdirs
            bas/
            phase_activity/ [optional]
        true_site_dpath, true_region_dpath: Path to GT annotations repo
        merge_dpath: Directory to save merged results.

    Returns:
        (bas_df, sc_df)
        Two pd.DataFrames that are saved as
            {out_dpath}/(bas|sc)_df.pkl
    '''
    # assert merge_dpath not in region_dpaths
    # merge_dpath.delete().ensuredir()

    results = []
    for pth in region_dpaths:
        try:
            results.append(
                RegionResult.from_dpath_and_anns_root(
                    pth, true_site_dpath, true_region_dpath)
            )
        except FileNotFoundError:
            print(f'warning: missing region {pth}')

    # merge BAS
    bas_results = [r for r in results if r.bas_dpath]
    bas_df = merge_bas_metrics_results(bas_results, fbetas)

    # merge SC
    sc_results = [r for r in results if r.sc_dpath]
    sc_df = merge_sc_metrics_results(sc_results)

    # create and print a BAS and SC summary
    min_rho, max_rho = 0.5, 0.5
    min_tau, max_tau = 0.2, 0.2

    group = bas_df.query(
        f'{min_rho} <= rho <= {max_rho} and {min_tau} <= tau <= {max_tau}'
    ).groupby('region_id')
    best_bas_rows = bas_df.loc[group['F1'].idxmax()]

    concise_best_bas_rows = best_bas_rows.rename(
        {'tp sites': 'tp',
         'fp sites': 'fp',
         'fn sites': 'fn',
         'truth sites': 'truth',
         'proposed sites': 'proposed',
         'total sites': 'total'}, axis=1)
    concise_best_bas_rows = concise_best_bas_rows[
        ['tp', 'fp', 'fn', 'truth', 'proposed', 'ffpa'] +
        [c for c in best_bas_rows.columns if c.startswith('F')]
    ]
    print(concise_best_bas_rows.to_string())

    concise_sc = sc_df.query('phase in ["Site Preparation", "Active Construction"]').copy()
    # there has to be some way to do this using
    # concise_sc.loc[concise_sc.index.map(???)], selecting series of
    # (column_label == row_label) and (column_label in (phases - row_label)).
    # Oh well.
    concise_sc['tp'] = np.stack([
            concise_sc.loc[(slice(None), 'Site Preparation'), 'Site Preparation'].values,
            concise_sc.loc[(slice(None), 'Active Construction'), 'Active Construction'].values,
           ], axis=1
    ).reshape(-1)
    concise_sc['fp'] = np.stack([
            concise_sc.loc[(slice(None), 'Site Preparation'), ['No Activity', 'Active Construction', 'Post Construction']].sum(axis=1).values,
            concise_sc.loc[(slice(None), 'Active Construction'), ['No Activity', 'Site Preparation', 'Post Construction']].sum(axis=1).values,
            ], axis=1
    ).reshape(-1)
    concise_sc = concise_sc[['F1', 'TIoU', 'TE', 'tp', 'fp']]
    print(concise_sc.to_string())

    json_data = {}
    json_data['best_bas_rows'] = json.loads(best_bas_rows.to_json(orient='table', indent=2))
    json_data['sc_df'] = json.loads(sc_df.to_json(orient='table', indent=2))
    return json_data, bas_df, sc_df, best_bas_rows


def _devcheck():
    """
    rsync -avprLPR --exclude '.succ' --exclude 'tmp' $HOME/data/dvc-repos/smart_expt_dvc/_testpipe/aggregate/./agg_params_ffmpktiwwpbx horologic:data/dvc-repos/smart_expt_dvc/_testpipe/aggregate
    """
    import geowatch
    region_dpaths = ['/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/aggregate/agg_params_ffmpktiwwpbx/KR_R001/KR_R001/']
    fbetas = []
    data_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_data')
    gt_dpath = data_dvc_dpath / 'annotations'
    true_region_dpath = gt_dpath / 'region_models'
    true_site_dpath =  gt_dpath / 'site_models'
    json_data, bas_df, sc_df, best_bas_rows = merge_metrics_results(region_dpaths, true_site_dpath, true_region_dpath, fbetas)
    import rich
    primary_columns = ['tp sites', 'fp sites', 'fn sites', 'ffpa', 'F1', 'tp over', 'tp exact', 'tp under']

    region_id = bas_df.index.levels[0][0]
    group = bas_df.loc[[region_id]]

    # other_cols = group.columns.difference(primary_columns)

    toshow = group.loc[:, primary_columns]
    rich.print(toshow.to_string())

    import kwplot
    sns = kwplot.autosns()
    plt = kwplot.autoplt()
    kwplot.figure(fnum=1, docla=1)
    sns.histplot(data=group, x='rho', y='tau', hue='F1')
    # sns.kdeplot(data=group, x='rho', y='tau', hue='F1')
    ax = plt.gca()
    ax.set_xlim(-0.1, 0.91)
    ax.set_ylim(-0.1, 0.51)
    ax.set_title(region_id)
    ax.scatter(0.5, 0.2, marker='*', s=400, color='orange', alpha=0.98)
    sns.scatterplot(data=group, x='rho', y='tau', hue='F1')

    sns.lineplot(data=group, x='rho', y='F1', hue='tau')
    sns.lineplot(data=group, x='tau', y='F1', hue='rho')


def iarpa_bas_color_legend():
    """
    Ignore:
        import kwplot
        from geowatch.tasks.metrics.merge_iarpa_metrics import *  # NOQA
        img = iarpa_bas_color_legend()
        kwplot.autompl()
        kwplot.imshow(img)
    """
    import kwplot
    from geowatch import heuristics
    colors = heuristics.IARPA_CONFUSION_COLORS
    img = kwplot.make_legend_img(colors)
    return img
