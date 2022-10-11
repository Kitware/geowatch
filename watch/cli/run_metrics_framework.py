#!/usr/bin/env python
import os
import sys
import json
import shlex
import pandas as pd
import numpy as np
import shapely.geometry
import shapely.ops
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from typing import List, Dict, Optional, Literal
import ubelt as ub
import scriptconfig as scfg
from packaging import version
import safer


class MetricsConfig(scfg.DataConfig):
    """
    Score IARPA site model GeoJSON files using IARPA's metrics-and-test-framework
    """
    pred_sites = scfg.Value(None, required=True, nargs='*', help=ub.paragraph(
        '''
        List of paths to predicted v2 site models. Or a path to a single text
        file containing the a list of paths to predicted site models.
        All region_ids from these sites will be scored, and it will be assumed
        that there are no other sites in these regions.
        '''))
    gt_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        Path to a local copy of the ground truth annotations,
        https://smartgitlab.com/TE/annotations.  If None, use the
        environment variable DVC_DATA_DPATH to find
        $DVC_DATA_DPATH/annotations.
        '''))

    true_site_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        Directory containing true site models. Defaults to
        gt_dpath / site_models
        '''))

    true_region_dpath = scfg.Value(None, help=ub.paragraph(
        '''
        Directory containing true region models. Defaults to
        gt_dpath / region_models
        '''))

    out_dir = scfg.Value(None, help=ub.paragraph(
        '''
        Output directory where scores will be written. Each
        region will have. Defaults to ./iarpa-metrics-output/
        '''))
    merge = scfg.Value(False, isflag=1, help=ub.paragraph(
        '''
        Merge BAS and SC metrics from all regions and output to
        {out_dir}/merged/
        '''))
    merge_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        Forces the merge summary to be written to a specific
        location.
        '''))
    merge_fbetas = scfg.Value([], help=ub.paragraph(
        '''
        A list of BAS F-scores to compute besides F1.
        '''))
    tmp_dir = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, will write temporary data here instead of
        using a     non-persistent directory
        '''))
    enable_viz = scfg.Value(False, isflag=1, help=ub.paragraph(
        '''
        If true, enables iarpa visualizations
        '''))
    name = scfg.Value('unknown', help=ub.paragraph(
        '''
        Short name for the algorithm used to generate the model
        '''))
    inputs_are_paths = scfg.Value(False, isflag=1, help=ub.paragraph(
        '''
        If given, the sites inputs will always be interpreted as
        paths and not raw json text.
        '''))
    use_cache = scfg.Value(False, isflag=1, help=ub.paragraph(
        '''
        IARPA metrics code currently contains a cache bug, do not
        enable the cache until this is fixed.
        '''))


phases = ['No Activity', 'Site Preparation', 'Active Construction', 'Post Construction']


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
        df[phases] = df[phases].astype(int)
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
            df = df.replace(['NA', '[]', None], pd.NA
                           ).astype('string'
                                   ).apply(
                                       lambda s: s.str.replace("'", '').str.strip('{}'))
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


import seaborn as sns
import matplotlib.pyplot as plt
# matplotlib.use('Agg')


def viz_sc(sc_results, save_dpath):

    def viz_sc_gantt(df, plot_title, save_fpath):
        # ignore subsites
        # TODO how to pick site boundary?
        df = df.apply(lambda s: s.str.split(', ').str[0])

        df['pred'] = df['pred'].fillna(method='ffill')
        df = df[~df['true'].isna()]
        # df['pred'] = df['pred'].fillna('Unknown')
        df['pred'] = df['pred'].fillna(method='bfill')
        df = df.reset_index()
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.melt(id_vars=['date'])

        # order hack for relplot
        phases_type = pd.api.types.CategoricalDtype(
            (['Unknown'] + phases)[::-1], ordered=True)
        df['value'] = df['value'].astype(phases_type)
        df = df.sort_values(by='value')

        # TODO threadsafe
        grid = sns.relplot(
            data=df,
            x='date',
            y='value',
            hue='variable',
            size='variable'
        )
        grid.savefig(save_fpath)

    def viz_sc_multi(df, plot_title, save_fpath):

        # df.index = [df.index.map('{0[0]} {0[1]} {0[2]}'.format), df.index.get_level_values(3)]

        # ignore subsites
        # TODO how to pick site boundary?
        df = df.apply(lambda s: s.str.split(', ').str[0])

        # could do just region_id for error bars after fillna
        df = df.reset_index()
        df['group'] = df[['region_id', 'site', 'site_candidate']].astype('string').agg('-'.join, axis=1)
        df = df.drop(['region_id', 'site', 'site_candidate'], axis=1)

        # TODO phase transition consistency check util?
        df['pred'] = df.groupby('group')['pred'].fillna(method='ffill').fillna('No Activity')
        # df['pred'] = df['pred'].fillna('Unknown')
        # df['pred'] = df['pred'].fillna(method='bfill')

        df = df[~df['true'].isna()]
        df = df[df['true'] != 'Unknown']  # Unk should be gone from df after this

        df['date'] = pd.to_datetime(df['date'])#.dt.date

        df['date'] = pd.to_datetime(df['date'])#.dt.date

        # must do this before searchsorted
        phases_type = pd.api.types.CategoricalDtype(
            ['Unknown'] + phases, ordered=True)
        df['pred'] = df['pred'].astype(phases_type)
        df['true'] = df ['true'].astype(phases_type)
        assert df.groupby('group')['true'].is_monotonic_increasing.all()
        # assert df.groupby('group')['pred'].is_monotonic_increasing.all()
        df['diff'] = df['pred'].cat.codes - df['true'].cat.codes

        if 0:  # absolute date
            df['date'] = df['date'].dt.date
        elif 1:  # relative date since start
            df['date'] = df.groupby('group')['date'].diff()
            df['date'] = df['date'].dt.days.fillna(0)
        else:  # relative date since last NA; requires 1 NA to exist before SP
            def align_start(grp, phase='Site Preparation', before=True):
                grp = grp.sort_values(by='date')
                assert grp['true'].iloc[0] != phase, grp['true']
                grp['date'] -= grp.iloc[grp['true'].searchsorted(phase) - int(before)]['date']
                return grp
            df = df.groupby('group').apply(align_start)
            df['date'] = df['date'].dt.days
        # df = df.melt(id_vars=['date'])

        import watch
        palette = {c['name']: c['color'] for c in watch.heuristics.CATEGORIES}

        from matplotlib.collections import LineCollection
        from matplotlib.colors import to_rgba

        jitter = 0.1
        df['diff'] += np.random.uniform(low=-jitter, high=jitter, size=len(df['diff']))

        df['group_phase'] = df[['group', 'true']].agg('_'.join, axis=1)

        # TODO threadsafe
        grid = sns.relplot(
            kind='scatter',
            data=df,
            x='date',
            y='diff',
            hue='true',
            palette=palette,
        )
        # need args instead of kwargs because of grid.map() weirdness
        def add_colored_linesegments(x, y, hue, units, **kwargs):
            # sns.lineplot(x=x, y=y,
                         # estimator=None,
                         # units=units,
                         # hue=hue,
                         # palette=palette,
            # )
            _df = pd.DataFrame(dict(xy=zip(x,y), hue=pd.Series(hue).map(palette).astype('string').map(to_rgba), units=units))

            lines = []
            colors = []
            for _, grp in _df.groupby('units'):
                phases = grp['hue'].unique()
                ixs = grp['hue'].searchsorted(phases)
                for start, end, phase in zip(ixs, (np.array(ixs[1:]) + 1).tolist() + [None], phases):
                    lines.append(grp['xy'][start:end])
                    colors.append(phase)

            import xdev; xdev.embed()
            lc = LineCollection(lines, alpha=0.5, colors=colors)
            ax.add_collection(lc)

        grid.map(add_colored_linesegments,
                 'date',
                 'diff',
                 'true',
                 'group',
        )
        # grid.set_axis_labels(x_var='days since final No Activity', y_var='pred phases ahead of true phase')
        grid.set_axis_labels(x_var='days since start', y_var='pred phases ahead of true phase')

        ax = plt.gca()
        ax
            # order=_phases[::-1],
            # kwargs=dict(
            # **dict(
                # jitter=False,
                # dodge=False,
                # size='variable'
                # s=[20, 10],
            # ),
        grid.savefig(save_fpath)

    phs = list(filter(lambda ph: ph is not None, (r.sc_phasetable for r in sc_results)))

    for ph in phs:

        rid = ph.index.get_level_values('region_id')[0]

        # site-level viz
        # for (site, site_cand), df in ph.groupby(['site', 'site_candidate']):
            # viz_sc_gantt(
                # df.droplevel([0, 1, 2]),
                # ' vs. '.join((site, site_cand)),
                # ((save_dpath / rid).ensuredir() / ('_'.join((site, site_cand)) + '.png'))
            # )

        # region-level viz
        viz_sc_multi(
            ph, rid,
            (save_dpath / f'sc_{rid}.png')
        )

    # merged viz
    merged_df = pd.concat(phs, axis=0)
    viz_sc_multi(
        merged_df,
        ' '.join(merged_df.index.get_level_values('region_id')),
        (save_dpath / 'sc_merged.png')
    )


def merge_metrics_results(region_dpaths, true_site_dpath, true_region_dpath,
                          merge_dpath, merge_fpath, fbetas, parent_info, info,
                          sc_viz=True):
    '''
    Merge metrics results from multiple regions.

    Args:
        region_dpaths: List of directories containing the subdirs
            bas/
            phase_activity/ [optional]
        true_site_dpath, true_region_dpath: Path to GT annotations repo
        merge_dpath: Directory to save merged results.
            Existing contents will be removed.

    Returns:
        (bas_df, sc_df)
        Two pd.DataFrames that are saved as
            {out_dpath}/(bas|sc)_df.pkl
    '''
    merge_dpath = ub.Path(merge_dpath).ensuredir()
    # assert merge_dpath not in region_dpaths
    # merge_dpath.delete().ensuredir()

    results = [
        RegionResult.from_dpath_and_anns_root(
            pth, true_site_dpath, true_region_dpath)
        for pth in region_dpaths
    ]

    # merge BAS
    bas_results = [r for r in results if r.bas_dpath]
    bas_df = merge_bas_metrics_results(bas_results, fbetas)

    # merge SC
    sc_results = [r for r in results if r.sc_dpath]
    sc_df = merge_sc_metrics_results(sc_results)

    bas_df.to_pickle(merge_dpath / 'bas_df.pkl')
    sc_df.to_pickle(merge_dpath / 'sc_df.pkl')

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
        ['tp', 'fp', 'fn', 'truth', 'proposed'] +
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

    # write BAS and SC summary in readable form
    with safer.open(merge_dpath / 'summary.csv', 'w') as f:
        best_bas_rows.to_csv(f)
        f.write('\n')
        sc_df.to_csv(f)

    json_data = {}
    # TODO: parent info should probably belong to info itself
    json_data['info'] = info
    json_data['parent_info'] = parent_info
    json_data['best_bas_rows'] = json.loads(best_bas_rows.to_json(orient='table', indent=2))
    json_data['sc_df'] = json.loads(sc_df.to_json(orient='table', indent=2))

    with safer.open(merge_fpath, 'w', temp_file=True) as f:
        json.dump(json_data, f, indent=4)

    # Symlink to visualizations
    region_viz_dpath = (merge_dpath / 'region_viz_overall').ensuredir()

    for dpath in region_dpaths:
        overall_dpath = dpath / 'overall'
        viz_dpath = overall_dpath / 'bas' / 'region'

        for viz_fpath in viz_dpath.iterdir():
            viz_link = viz_fpath.augment(dpath=region_viz_dpath)
            ub.symlink(viz_fpath, viz_link, verbose=1)

    # viz SC
    if sc_viz:
        viz_sc(sc_results, region_viz_dpath)

    return bas_df, sc_df


def ensure_thumbnails(image_root, region_id, sites):
    '''
    Symlink and organize images in the format the metrics framework expects

    For the region visualizations:
    > image_list = glob(f"{self.image_path}/
    >   {self.region_model.id.replace('_', '/')}/images/*/*/*.jp2")

    For the site visualizations:
    > image_list = glob(f"{self.image_path}/
    >   {gt_ann_id.replace('_', '/')}/crops/*.tif")

    Which becomes:
    {country_code}/
        {region_num}/
            images/
                */
                    */
                        *.jp2
            {site_num}/
                crops/
                    *.tif

    Args:
        image_root: root directory to save under
        region_id: ex. 'KR_R001'
        sites: proposed sites with image paths in the 'source' field
            TODO change to 'misc_info' field
    '''
    image_root = ub.Path(image_root)

    # gather images and dates
    site_img_date_dct = dict()
    for site in sites:

        img_date_dct = dict()

        for feat in site['features']:
            props = feat['properties']

            if props['type'] == 'observation':

                img_path = ub.Path(props['source'])
                if img_path.is_file():
                    img_date_dct[img_path] = props['observation_date']
                else:
                    print(f'warning: image {img_path}' ' is not a valid path')

            elif props['type'] == 'site':
                site_id = props['site_id']

            else:
                raise ValueError(props['type'])

        site_img_date_dct[site_id] = img_date_dct

    # build region viz
    region_root = image_root.joinpath(*region_id.split('_')) / 'images' / 'a' / 'b'
    region_root.mkdir(parents=True, exist_ok=True)
    for img_path, img_date in ub.dict_union(
            *site_img_date_dct.values()).items():
        link_path = (region_root / '_'.join(
            (img_date.replace('-', ''), img_path.with_suffix('.jp2').name)))
        ub.symlink(img_path, link_path, verbose=0)

    # build site viz
    for site_id, img_date_dct in site_img_date_dct.items():
        site_root = image_root.joinpath(*site_id.split('_')) / 'crops'
        site_root.mkdir(parents=True, exist_ok=True)
        for img_path, img_date in img_date_dct.items():
            # TODO crop
            link_path = (site_root / '_'.join(
                (img_date.replace('-', ''), img_path.with_suffix('.tif').name)))
            ub.symlink(img_path, link_path, verbose=0)


def main(cmdline=True, **kwargs):
    """
    Note currently depends on:
        https://smartgitlab.com/jon.crall/metrics-and-test-framework/-/tree/autogen-on-te

    Example:
        >>> # xdoctest: +REQUIRES(module:iarpa_smart_metrics)
        >>> from watch.cli.run_metrics_framework import *  # NOQA
        >>> from iarpa_smart_metrics.demo.generate_demodata import generate_demo_metrics_framework_data
        >>> cmdline = 0
        >>> base_dpath = ub.Path.appdir('watch', 'tests', 'test-iarpa-metrics2')
        >>> data_dpath = base_dpath / 'inputs'
        >>> dpath = base_dpath / 'outputs'
        >>> demo_info1 = generate_demo_metrics_framework_data(
        >>>     roi='DR_R001',
        >>>     num_sites=5, num_observations=10, noise=2, p_observe=0.5,
        >>>     p_transition=0.3, drop_noise=0.5, drop_limit=0.5)
        >>> demo_info2 = generate_demo_metrics_framework_data(
        >>>     roi='DR_R002',
        >>>     num_sites=7, num_observations=10, noise=1, p_observe=0.5,
        >>>     p_transition=0.1, drop_noise=0.8, drop_limit=0.5)
        >>> demo_info3 = generate_demo_metrics_framework_data(
        >>>     roi='DR_R003',
        >>>     num_sites=11, num_observations=10, noise=3, p_observe=0.5,
        >>>     p_transition=0.2, drop_noise=0.3, drop_limit=0.5)
        >>> print('demo_info1 = {}'.format(ub.repr2(demo_info1, nl=1)))
        >>> print('demo_info2 = {}'.format(ub.repr2(demo_info2, nl=1)))
        >>> print('demo_info3 = {}'.format(ub.repr2(demo_info3, nl=1)))
        >>> out_dpath = dpath / 'region_metrics'
        >>> merge_fpath = dpath / 'merged.json'
        >>> out_dpath.delete()
        >>> kwargs = {
        >>>     'pred_sites': demo_info1['pred_site_dpath'],
        >>>     'true_region_dpath': demo_info1['true_region_dpath'],
        >>>     'true_site_dpath': demo_info1['true_site_dpath'],
        >>>     'merge': True,
        >>>     'merge_fpath': merge_fpath,
        >>>     'out_dir': out_dpath,
        >>> }
        >>> main(cmdline=False, **kwargs)
        >>> # TODO: visualize
    """

    from watch.utils import util_path
    # from watch.utils import util_pattern

    config = MetricsConfig.legacy(cmdline=cmdline, data=kwargs)
    args = config

    config['pred_sites'] = util_path.coerce_patterned_paths(
        config['pred_sites'], expected_extension='*.geojson')

    # args, _ = parser.parse_known_args(args)
    config_dict = config.asdict()
    print('config = {}'.format(ub.repr2(config_dict, nl=2)))

    # load pred_sites
    pred_sites = []
    if len(args.pred_sites) == 0:
        raise Exception('No input sites were given')

    try:
        # Do we have the latest and greatest?
        import iarpa_smart_metrics
        METRICS_VERSION = version.Version(iarpa_smart_metrics.__version__)
    except Exception:
        raise AssertionError(
            'The iarpa_smart_metrics package should be pip installed '
            'in your virtualenv')
    assert METRICS_VERSION >= version.Version('0.2.0')

    # Record information about this process
    info = []
    from kwcoco.util import util_json
    from watch.utils import process_context

    # Args will be serialized in kwcoco, so make sure it can be coerced to json
    jsonified_args = util_json.ensure_json_serializable(config_dict)
    walker = ub.IndexableWalker(jsonified_args)
    for problem in util_json.find_json_unserializable(jsonified_args):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)

    proc_context = process_context.ProcessContext(
        type='process',
        name='watch.cli.run_metrics_framework',
        args=jsonified_args,
        extra={'iarpa_smart_metrics_version': iarpa_smart_metrics.__version__},
    )
    proc_context.start()

    parent_info = []
    for site_data in args.pred_sites:
        in_fpath = ub.Path(site_data)

        if args.inputs_are_paths:
            if not in_fpath.exists():
                raise FileNotFoundError(str(in_fpath))
            with open(in_fpath, 'r') as file:
                site_or_result = json.load(file)

            is_track_result = (
                isinstance(site_or_result, dict) and
                site_or_result.get('type', None) == 'tracking_result'
            )

            if is_track_result:
                track_result = site_or_result
                # The input was a track result json which contains pointers to
                # the actual sites
                parent_info.extend(track_result.get('info', []))
                for site_fpath in track_result['files']:
                    with open(site_fpath, 'r') as file:
                        site = json.load(file)
                    pred_sites.append(site)
            else:
                # It was just a site json file.
                site = site_or_result
                pred_sites.append(site)
        else:
            # TODO:
            # Deprecate passing raw json on the CLI, it has a limited length
            # What would be best is a single file that points to all of the
            # site jsons we care about, so we don't need to glob.
            try:
                if in_fpath.is_file():
                    with open(site_data, 'r') as file:
                        site = json.load(file)
                else:
                    raise NotImplementedError('deprecated to pass json as str')
                    site = json.loads(site_data)
            except json.JSONDecodeError as e:  # TODO split out as decorator?
                raise json.JSONDecodeError(e.msg + ' [cut for length]',
                                           e.doc[:100] + '...', e.pos)
            pred_sites.append(site)

    name = args.name
    true_site_dpath = args.true_site_dpath
    true_region_dpath = args.true_region_dpath

    if true_region_dpath is None or true_site_dpath is None:
        # normalize paths
        if args.gt_dpath is not None:
            gt_dpath = ub.Path(args.gt_dpath).absolute()
        else:
            import watch
            data_dvc_dpath = watch.find_dvc_dpath(tags='phase2_data')
            gt_dpath = data_dvc_dpath / 'annotations'
            print(f'gt_dpath unspecified, defaulting to {gt_dpath=}')

        if true_region_dpath is None:
            assert gt_dpath.is_dir(), gt_dpath
            true_region_dpath = gt_dpath / 'region_models'
        if true_site_dpath is None:
            assert gt_dpath.is_dir(), gt_dpath
            true_site_dpath =  gt_dpath / 'site_models'

    true_region_dpath = ub.Path(true_region_dpath)
    true_site_dpath = ub.Path(true_site_dpath)

    if args.tmp_dir is not None:
        tmp_dpath = ub.Path(args.tmp_dir)
    else:
        temp_dir = TemporaryDirectory(suffix='iarpa-metrics-tmp')
        tmp_dpath = ub.Path(temp_dir.name)

    # split sites by region
    out_dirs = []
    grouped_sites = ub.group_items(
        pred_sites, lambda site: site['features'][0]['properties']['region_id'])

    main_out_dir = ub.Path(args.out_dir or './iarpa-metrics-output')
    main_out_dir.ensuredir()

    full_invocation_text = ub.codeblock(
        '''
        #!/bin/bash
        __doc__="
        This is an auto-generated file that records the command used to
        generate this evaluation of multiple regions.
        "
        ''') + chr(10) + shlex.join(sys.argv) + chr(10)
    (main_out_dir / 'invocation.sh').write_text(full_invocation_text)

    # First build up all of the commands and prepare necessary data for them.
    commands = []
    for region_id, region_sites in ub.ProgIter(sorted(grouped_sites.items()),
                                               desc='prepare regions for eval'):

        site_dpath = (tmp_dpath / 'site' / region_id).ensuredir()
        image_dpath = (tmp_dpath / 'image').ensuredir()

        if args.use_cache:
            cache_dpath = (tmp_dpath / 'cache' / region_id).ensuredir()
        else:
            cache_dpath = 'None'

        out_dir = (main_out_dir / region_id).ensuredir()
        out_dirs.append(out_dir)

        # doctor site_dpath for expected structure
        pred_site_sub_dpath = site_dpath / 'latest' / region_id
        pred_site_sub_dpath.ensuredir()

        # copy site models to site_dpath
        for site in region_sites:
            geojson_fpath = pred_site_sub_dpath / (
                site['features'][0]['properties']['site_id'] + '.geojson'
            )
            with safer.open(geojson_fpath, 'w', temp_file=True) as f:
                json.dump(site, f)

        ensure_thumbnails(image_dpath, region_id, region_sites)

        if args.enable_viz:
            viz_flags = []
        else:
            viz_flags = [
                # '--no-viz-region',  # we do want this enabled
                '--no-viz-slices',
                '--no-viz-detection-table',
                '--no-viz-comparison-table',
                '--no-viz-associate-metrics',
                '--no-viz-activity-metrics',
            ]

        run_eval_command = [
            'python', '-m', 'iarpa_smart_metrics.run_evaluation',
            '--roi', region_id,
            '--gt_dir', os.fspath(true_site_dpath),
            '--rm_dir', os.fspath(true_region_dpath),
            '--sm_dir', os.fspath(pred_site_sub_dpath),
            '--image_dir', os.fspath(image_dpath),
            '--output_dir', os.fspath(out_dir),
            '--cache_dir', os.fspath(cache_dpath),
            ## Restrict to make this faster
            #'--tau', '0.2',
            #'--rho', '0.5',
            '--activity', 'overall',
            #'--loglevel', 'error',
            '--name', name,
            '--serial',
            # '--no-db',
            '--sequestered_id', 'seq',  # default None broken on autogen branch
        ]
        run_eval_command += viz_flags
        # run metrics framework
        cmd = shlex.join(run_eval_command)
        region_invocation_text = ub.codeblock(
            '''
            #!/bin/bash
            __doc__="
            This is an auto-generated file that records the command used to
            generate this evaluation of this particular region.
            "
            ''') + chr(10) + cmd + chr(10)
        # Dump this command to disk for reference and debugging.
        (out_dir / 'invocation.sh').write_text(region_invocation_text)
        commands.append(cmd)

    if 0:
        import cmd_queue
        queue = cmd_queue.Queue.create(backend='serial')
        for cmd in commands:
            queue.submit(cmd)
            # TODO: make command queue stop on the first failure?
            queue.run()
        # if queue.read_state()['failed']:
        #     raise Exception('jobs failed')
    else:
        import subprocess
        for cmd in commands:
            try:
                ub.cmd(cmd, verbose=3, check=True, shell=True)
            except subprocess.CalledProcessError:
                print('error in metrics framework, probably due to zero '
                      'TP site matches.')

    print('out_dirs = {}'.format(ub.repr2(out_dirs, nl=1)))
    if args.merge and out_dirs:
        merge_dpath = main_out_dir / 'merged'

        if args.merge_fpath is None:
            merge_fpath = merge_dpath / 'summary2.json'
        else:
            merge_fpath = ub.Path(args.merge_fpath)

        region_dpaths = out_dirs

        info.append(proc_context.stop())

        merge_metrics_results(region_dpaths, true_site_dpath,
                              true_region_dpath, merge_dpath, merge_fpath,
                              args.merge_fbetas, parent_info, info)
        print('merge_fpath = {!r}'.format(merge_fpath))


if __name__ == '__main__':
    main()
