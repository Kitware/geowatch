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
from typing import List, Dict
import ubelt as ub
import subprocess
import scriptconfig as scfg
from packaging import version


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

    virtualenv_cmd = scfg.Value(['true'], nargs='+', help=ub.paragraph(
        '''
        Command to run before calling the metrics framework in a subshell.

        Only necessary if the metrics framework is installed in a different
        virtual env from the current one. (Or maybe if you don't auto start it?
        Not sure. TODO: figure out if this inherits or not).
        '''))
    out_dir = scfg.Value(None, help=ub.paragraph(
        '''
        Output directory where scores will be written. Each
        region will have. Defaults to ./iarpa-metrics-output/
        '''))
    merge = scfg.Value(False, help=ub.paragraph(
        '''
        Merge BAS and SC metrics from all regions and output to
        {out_dir}/merged/
        '''))
    merge_fpath = scfg.Value(None, help=ub.paragraph(
        '''
        Forces the merge summary to be written to a specific
        location.
        '''))
    tmp_dir = scfg.Value(None, help=ub.paragraph(
        '''
        If specified, will write temporary data here instead of
        using a     non-persistant directory
        '''))
    enable_viz = scfg.Value(False, help=ub.paragraph(
        '''
        If true, enables iarpa visualizations
        '''))
    name = scfg.Value('unknown', help=ub.paragraph(
        '''
        Short name for the algorithm used to generate the model
        '''))
    inputs_are_paths = scfg.Value(False, help=ub.paragraph(
        '''
        If given, the sites inputs will always be interpreted as
        paths and not raw json text.
        '''))
    use_cache = scfg.Value(False, help=ub.paragraph(
        '''
        IARPA metrics code currently contains a cache bug, do not
        enable the cache until this is fixed.
        '''))


@dataclass(frozen=True)
class RegionResult:
    region_id: str  # 'KR_R001'
    region_model: Dict
    site_models: List[Dict]
    bas_dpath: ub.Path = None  # 'path/to/scores/latest/KR_R001/bas/'
    sc_dpath: ub.Path = None  # 'path/to/scores/latest/KR_R001/phase_activity/'

    @classmethod
    def from_dpath_and_anns_root(cls, region_dpath, true_site_dpath, true_region_dpath):
        region_dpath = ub.Path(region_dpath)
        region_id = region_dpath.name
        bas_dpath = region_dpath / 'completed' / 'bas'
        bas_dpath = bas_dpath if bas_dpath.is_dir() else None
        sc_dpath = region_dpath / 'completed' / 'phase_activity'
        sc_dpath = sc_dpath if sc_dpath.is_dir() else None
        region_fpath = true_region_dpath / (region_id + '.geojson')
        with open(region_fpath, 'r') as f:
            region_model = json.load(f)

        site_fpaths = sorted(true_site_dpath.glob(f'{region_id}_*.geojson'))
        site_models = [
            json.loads(open(pth).read())
            for pth in site_fpaths
        ]
        return cls(region_id, region_model, site_models, bas_dpath, sc_dpath)


def merge_bas_metrics_results(bas_results: List[RegionResult]):
    '''
    Merge BAS results and return as a pd.DataFrame

    with MultiIndex(['rho', 'tau']) (a parameter sweep)

    and columns:
        tp sites             int64
        fp sites             int64
        fn sites             int64
        truth sites          int64
        proposed sites       int64
        total sites          int64
        truth slices         int64
        proposed slices      int64
        precision          float64
        recall (PD)        float64
        F1                 float64
        spatial FAR        float64
        temporal FAR       float64
        images FAR         float64
    '''

    #
    # --- Helper functions for FAR ---
    #

    def area(regions):
        # ref: metrics-and-test-framework.evaluation.GeometryUtil
        def scale_area(lat):
            """
            Find square meters per degree for a given latitude based on EPSG:4326
            :param lat: average latitude
                note that both latitude and longitude scales are dependent on latitude only
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

    def to_df(bas_dpath, region_id):
        # scoreboard_fpaths = sorted(
        #     glob(os.path.join(bas_dpath, 'scoreboard_rho=*.csv')))
        # bas_dpath / 'F1.csv'
        scoreboard_fpath = (bas_dpath / 'scoreboard.csv')
        scoreboard = pd.read_csv(scoreboard_fpath)
        scoreboard = scoreboard.iloc[:, 1:].copy()
        scoreboard['region_id'] = region_id
        scoreboard = scoreboard.set_index(['region_id', 'rho', 'tau'])
        # f1_csv_fpath = (bas_dpath / 'F1.csv')
        # f1_csv = pd.read_csv(f1_csv_fpath)
        # f1_csv = f1_csv.set_index('tau')
        # f1_csv = f1_csv.rename(lambda x: x.split('rho=')[-1], axis=1)
        # f1_csv.columns.name = 'rho'
        # f1_csv = f1_csv.melt(ignore_index=False, value_name='F1').reset_index()
        # f1_csv['region_id'] = region_id
        return scoreboard

        # load each per-rho scoreboard and concat them
        # rho_parser = parse.Parser('scoreboard_rho={rho:f}.csv')
        # dfs = []
        # for pth in scoreboard_fpaths:
        #     rho = rho_parser.parse(os.path.basename(pth)).named['rho']
        #     df = pd.read_csv(pth)
        #     df['rho'] = rho
        #     df['region_id'] = region_id
        #     # MultiIndex with rho, tau and region_id
        #     df = df.set_index(['region_id', 'rho', 'tau'])
        #     dfs.append(df)
        # return pd.concat(dfs)

    dfs = [to_df(r.bas_dpath, r.region_id) for r in bas_results]

    concat_df = pd.concat(dfs)

    sum_cols = [
        'tp sites', 'fp sites', 'fn sites', 'truth sites', 'proposed sites',
        'total sites', 'truth slices', 'proposed slices'
    ]
    merged_df = concat_df.groupby(['rho', 'tau'])[sum_cols].sum()
    merged_df.loc[:, 'region_id'] = '__merged__'
    merged_df = merged_df.reset_index().set_index(['region_id', 'rho', 'tau'])

    # # ref: metrics-and-test-framework.evaluation.Metric
    (_, tp), (_, fp), (_, fn) = merged_df[['tp sites', 'fp sites',
                                           'fn sites']].iteritems()
    merged_df['precision'] = np.where(tp > 0, tp / (tp + fp), 0)
    merged_df['recall (PD)'] = np.where(tp > 0, tp / (tp + fn), 0)
    merged_df['F1'] = np.where(tp > 0, tp / (tp + 0.5 * (fp + fn)), 0)

    all_regions = [r.region_model for r in bas_results]
    # # ref: metrics-and-test-framework.evaluation.Evaluation.build_scoreboard
    merged_df['spatial FAR'] = fp.astype(float) / area(all_regions)
    merged_df['temporal FAR'] = fp.astype(float) / n_dates(all_regions)

    # this is not actually how Images FAR is calculated!
    # https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/23
    #
    # all_sites = list(itertools.chain.from_iterable([
    #     r.site_models for r in bas_results]))
    # merged_df['images FAR'] = fp.astype(float) / n_unique_images(all_sites)
    #
    # instead, images in multiple proposed site stacks are double-counted.
    # take advantage of this to merge this metric with a simple average.
    n_images = (concat_df['fp sites'] /
                concat_df['images FAR']).groupby('region_id').mean().sum()
    merged_df['images FAR'] = fp.astype(float) / n_images

    bas_merged_df, bas_concat_df = merged_df, concat_df
    return bas_merged_df, bas_concat_df


def _to_sc_df(sc_dpath, region_id):
    '''
    confusion matrix and f1 scores apprently ignore subsites,
    so we must do the same
    https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/24
    MWE:
    >>> from sklearn.metrics import confusion_matrix
    >>> confusion_matrix(['a,a', 'a'], ['a,a', 'b'], labels=['a', 'b'])
    array([[0, 1],
           [0, 0]])
    '''

    delim = ' vs. '
    sc_dpath = ub.Path(sc_dpath)

    df = pd.read_csv(sc_dpath / 'ac_phase_table.csv')

    # terr_df = pd.read_csv(sc_dpath / 'ac_temporal_error.csv')
    # f1_df = pd.read_csv(sc_dpath / 'ac_f1_all_sites.csv')
    # pd.read_csv(sc_dpath / 'ap_temporal_error.csv')

    # df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.fillna(pd.NA).astype('string')

    site_names = df.columns.values.tolist()

    df = df.applymap(lambda cell: delim.join((cell, cell))
                     if not pd.isna(cell) and delim not in cell else cell)

    parts = [df[col].str.split(delim, expand=True) for col in site_names]
    df = pd.concat(parts, axis=1, ignore_index=True)

    df.columns = pd.MultiIndex.from_product(
        #  ([region_id], site_names, ['truth', 'proposed']),
        #  names=['region_id', 'site', 'type'])
        (site_names, ['true', 'pred']),
        names=['site', 'type'])

    def fix_cell(c):
        if isinstance(c, str):
            if c.startswith('{') and c.count(',') == 0:
                return c.replace('{', '').replace('}', '')
            if c == '[]':
                return None
        return c

    df = df.applymap(fix_cell)

    return df


def merge_sc_metrics_results(sc_results: List[RegionResult]):
    '''

    Note:
        Handled:
        * ac_phase_table.csv
        * ac_tiou.csv
        * ac_temporal_error.csv

        Not yet handled:
        * ac_confusion_matrix_all_sites.csv
        * ac_f1_all_sites.csv
        * ap_temporal_error.csv

    Returns:
        a list of pd.DataFrames
        activity_table: F1 score, mean TIoU, temporal error
        confusion_matrix: confusion matrix
    '''

    from sklearn.metrics import f1_score, confusion_matrix

    # for r in sc_results:
    #     sc_dpath = r.sc_dpath
    #     region_id = r.region_id
    #     to_df(sc_dpath, region_id)
    # if 0:
    #     r = sc_results[0]
    #     sc_dpath, region_id = r.sc_dpath, r.region_id

    # Handle ac_phase_table.csv
    dfs = [_to_sc_df(r.sc_dpath, r.region_id) for r in sc_results]
    df = pd.concat(dfs, axis=1).sort_values('date')

    # phase activity categories
    phase_classifications = [
        "No Activity",
        "Site Preparation",
        "Active Construction",
        "Post Construction",
    ]

    sites = df.columns.levels[0]

    # Not sure abou this
    def propogate(labels):
        import pandas as pd
        prev = 'No Activity'
        new = []
        for item in labels:
            if isinstance(item, str):
                if item.lower() == 'nan':
                    item = None
            elif item is not None:
                if pd.isnull(item):
                    item = None
            if item is None:
                item = prev
            item = item.replace("'", '')
            item = item.replace('"', '')
            new.append(item)
            prev = item
        return new

    phase_true = []
    phase_pred = []
    for site in sites:
        true = propogate(df[site, 'true'])
        pred = propogate(df[site, 'pred'])
        phase_pred.extend(pred)
        phase_true.extend(true)

    # Can't drop NA. Need to propogate
    phase_true = np.array(phase_true)
    phase_pred = np.array(phase_pred)

    f1 = f1_score(phase_true, phase_pred,
                  labels=phase_classifications,
                  average=None)

    # TIoU is only ever evaluated per-site, so we can safely average these
    # per-site and call it a new metric mTIoU.
    tiou_dfs = []
    for r in sc_results:
        ac_tiou_fpath = r.sc_dpath / 'ac_tiou.csv'
        table = pd.read_csv(ac_tiou_fpath, index_col=0)
        missing = sorted(set(phase_classifications) - set(table.columns))
        table.loc[:, missing] = np.nan
        table = table.loc[:, phase_classifications]
        tiou_dfs.append(table)

    tious = pd.concat(tiou_dfs, axis=0)
    mtiou = tious.mean(axis=0, skipna=True)
    mtiou_vals = [mtiou.get(c, default=np.nan) for c in phase_classifications]

    # these are averaged using the mean over sites for each phase.
    # So the correct average over regions is to weight by (sites/region)
    temporal_errs = []
    for r in sc_results:
        ac_temporal_err_fpath = r.sc_dpath / 'ac_temporal_error.csv'
        tdf = pd.read_csv(ac_temporal_err_fpath)
        missing = sorted(set(phase_classifications) - set(tdf.columns))
        tdf.loc[:, missing] = np.nan
        if len(tdf) > 0:
            tdf = tdf.loc[tdf.index[0], phase_classifications]  # mean days (all detections)
        temporal_errs.append(tdf.astype(float).values)

    n_sites = [df.shape[1] for df in dfs]
    try:
        temporal_err = np.average(temporal_errs, weights=n_sites, axis=0)
    except ValueError:
        temporal_err = [np.nan] * len(phase_classifications)

    sc_df = pd.DataFrame(
        {
            'F1 score': f1,
            'mean TIoU': mtiou_vals,
            'Temporal Error (days)': temporal_err
        },
        index=phase_classifications).T
    sc_df = sc_df.rename_axis('Activity Classification', axis='columns')

    _cmval = confusion_matrix(phase_true, phase_pred,
                              labels=phase_classifications)

    sc_cm = pd.DataFrame(_cmval,
                         columns=phase_classifications,
                         index=phase_classifications)
    sc_cm = sc_cm.rename_axis("truth phase")
    sc_cm = sc_cm.rename_axis("predicted phase", axis="columns")

    return sc_df, sc_cm


def _make_merge_metrics(region_dpaths, true_site_dpath, true_region_dpath):
    results = [
        RegionResult.from_dpath_and_anns_root(pth, true_site_dpath, true_region_dpath)
        for pth in region_dpaths
    ]

    # merge BA
    bas_results = [r for r in results if r.bas_dpath]
    bas_merged_df, bas_concat_df = merge_bas_metrics_results(bas_results)

    # merge SC
    sc_results = [r for r in results if r.sc_dpath]
    sc_df, sc_cm = merge_sc_metrics_results(sc_results)

    return bas_concat_df, bas_merged_df, sc_df, sc_cm


def _make_summary_info(bas_concat_df, bas_merged_df, sc_cm, sc_df, parent_info, info):
    # Find best bas row in combined results

    min_rho, max_rho = 0.5, 0.5
    min_tau, max_tau = 0.2, 0.2

    bas_merged_df = bas_merged_df.reset_index()
    rho = bas_merged_df['rho']
    tau = bas_merged_df['tau']
    rho_flags = (min_rho <= rho) & (rho <= max_rho)
    tau_flags = (min_tau <= tau) & (tau <= max_tau)
    flags = tau_flags & rho_flags
    candidate_merged_bas_df = bas_merged_df[flags]

    bas_concat_df = bas_concat_df.reset_index()
    rho = bas_concat_df['rho']
    tau = bas_concat_df['tau']
    rho_flags = (min_rho <= rho) & (rho <= max_rho)
    tau_flags = (min_tau <= tau) & (tau <= max_tau)
    flags = tau_flags & rho_flags
    candidate_bas_concat_df = bas_concat_df[flags]

    # Find best merged bas row
    best_merged_row = candidate_merged_bas_df.loc[[candidate_merged_bas_df['F1'].idxmax()]]
    # Find best per-region bas row
    best_ids = candidate_bas_concat_df.groupby('region_id')['F1'].idxmax()
    best_per_region = candidate_bas_concat_df.loc[best_ids]
    # best_bas_row_ = pd.concat({'__merged__': best_bas_row}, names=['region_id'])
    # best_bas_row_.loc[:, 'region_id'] = '__merged__'
    # Get a best row for each region and the "merged" region
    best_bas_rows = pd.concat([best_per_region, best_merged_row])
    concise_best_bas_rows = best_bas_rows.rename(
        {'tp sites': 'tp',
         'fp sites': 'fp',
         'fn sites': 'fn',
         'truth sites': 'truth',
         'proposed sites': 'proposed',
         'total sites': 'total'}, axis=1)
    concise_best_bas_rows = concise_best_bas_rows.drop([
        'truth slices',
        'proposed slices', 'precision', 'recall (PD)', 'spatial FAR',
        'temporal FAR', 'images FAR'], axis=1)

    json_data = {}
    # TODO: parent info should probably belong to info itself
    json_data['info'] = info
    json_data['parent_info'] = parent_info
    json_data['best_bas_rows'] = json.loads(best_bas_rows.to_json(orient='table', indent=2))
    json_data['sc_cm'] = json.loads(sc_cm.to_json(orient='table', indent=2))
    json_data['sc_df'] = json.loads(sc_df.to_json(orient='table', indent=2))

    return json_data, concise_best_bas_rows, best_bas_rows


def merge_metrics_results(region_dpaths, true_site_dpath, true_region_dpath, merge_dpath, merge_fpath,
                          parent_info, info):
    '''
    Merge metrics results from multiple regions.

    Args:
        region_dpaths: List of directories containing the subdirs
            bas/
            phase_activity/ [optional]
            time_activity/ [TBD, not scored yet]
        true_site_dpath, true_region_dpath: Path to GT annotations repo
        merge_dpath: Directory to save merged results.
            Existing contents will be removed.

    Returns:
        (bas_df, sc_df)
        Two pd.DataFrames that are saved as
            {out_dpath}/(bas|sc)_scoreboard_df.pkl
    '''
    import safer
    merge_dpath = ub.Path(merge_dpath).ensuredir()
    # assert merge_dpath not in region_dpaths
    # merge_dpath.delete().ensuredir()

    bas_concat_df, bas_df, sc_df, sc_cm = _make_merge_metrics(region_dpaths, true_site_dpath, true_region_dpath)
    bas_df.to_pickle(merge_dpath / 'bas_scoreboard_df.pkl')
    sc_df.to_pickle(merge_dpath / 'sc_activity_df.pkl')
    sc_cm.to_pickle(merge_dpath / 'sc_confusion_df.pkl')

    json_data, concise_best_bas_rows, best_bas_rows = _make_summary_info(bas_concat_df, bas_df, sc_cm, sc_df, parent_info, info)
    print(concise_best_bas_rows.to_string())

    region_viz_dpath = (merge_dpath / 'region_viz_overall').ensuredir()

    # Symlink to visualizations
    for dpath in region_dpaths:
        overall_dpath = dpath / 'overall'
        viz_dpath = overall_dpath / 'bas' / 'region'

        for viz_fpath in viz_dpath.iterdir():
            viz_link = viz_fpath.augment(dpath=region_viz_dpath)
            ub.symlink(viz_fpath, viz_link, verbose=1)

    # write summary in readable form
    #
    summary_path = merge_dpath / 'summary.csv'
    with open(summary_path, 'w') as f:
        best_bas_rows.to_csv(f)
        f.write('\n')
        sc_df.to_csv(f)
        f.write('\n')
        sc_cm.to_csv(f)

    with safer.open(merge_fpath, 'w', temp_file=True) as f:
        json.dump(json_data, f, indent=4)

    return bas_df, sc_df, sc_cm


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
    import safer

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
    import socket
    # Args will be serailized in kwcoco, so make sure it can be coerced to json
    jsonified_args = util_json.ensure_json_serializable(config_dict)
    walker = ub.IndexableWalker(jsonified_args)
    for problem in util_json.find_json_unserializable(jsonified_args):
        bad_data = problem['data']
        walker[problem['loc']] = str(bad_data)
    start_timestamp = ub.timestamp()
    info.append({
        'type': 'process',
        'properties': {
            'name': 'watch.cli.run_metrics_framework',
            'args': jsonified_args,
            'hostname': socket.gethostname(),
            'cwd': os.getcwd(),
            'userhome': ub.userhome(),
            'iarpa_smart_metrics_version': iarpa_smart_metrics.__version__,
            'timestamp': start_timestamp,
        }
    })

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

    # validate virtualenv command
    virtualenv_cmd = ' '.join(args.virtualenv_cmd)
    try:
        ub.cmd(virtualenv_cmd, verbose=1, check=True, shell=True)
    except Exception as ex:
        raise ValueError('The given virtualenv command is invalid') from ex

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
    for region_id, region_sites in ub.ProgIter(list(grouped_sites.items()),
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
            '--name', name,
            '--no-db',
        ]
        run_eval_command += viz_flags
        # run metrics framework
        cmd = f'{virtualenv_cmd} && ' + shlex.join(run_eval_command)
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

    if 1:
        import cmd_queue
        queue = cmd_queue.Queue.create(backend='serial')
        for cmd in commands:
            queue.submit(cmd)
            # TODO: make command queue stop on the first failure?
            queue.run()
        # if queue.read_state()['failed']:
        #     raise Exception('jobs failed')
    else:
        # Original way to invoke
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

        merge_metrics_results(region_dpaths, true_site_dpath,
                              true_region_dpath, merge_dpath, merge_fpath,
                              parent_info, info)
        print('merge_fpath = {!r}'.format(merge_fpath))


def _hack_remerge_data():
    """
    Hack to redo the merge with a different F1 maximization criterion

    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc --hardware="hdd")
    cd "$DVC_DPATH"
    ls models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json
    ls models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json
    """
    import watch
    data_dvc_dpath = watch.find_dvc_dpath(hardware='hdd')
    globstr = str(data_dvc_dpath / 'models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json')
    from watch.utils import util_path
    from watch.utils import simple_dvc
    summary_metrics = util_path.coerce_patterned_paths(globstr)
    import json
    import safer
    dvc = simple_dvc.SimpleDVC(data_dvc_dpath)

    # for merge_fpath in summary_metrics:
    #     if dvc.is_tracked(merge_fpath):
    #     pass

    dvc.unprotect(summary_metrics)

    for merge_fpath in ub.ProgIter(summary_metrics, desc='rewrite merge metrics'):
        region_dpaths = [p for p in list(merge_fpath.parent.parent.glob('*')) if p.name != 'merged']
        anns_root = data_dvc_dpath / 'annotations'
        true_site_dpath = anns_root / 'site_models'
        true_region_dpath = anns_root / 'region_models'

        # merge_dpath = merge_fpath.parent

        json_data = json.loads(merge_fpath.read_text())
        parent_info = json_data['parent_info']

        bas_concat_df, bas_df, sc_df, sc_cm = _make_merge_metrics(region_dpaths, true_site_dpath, true_region_dpath)
        new_json_data, *_ = _make_summary_info(bas_concat_df, bas_df, sc_cm, sc_df, parent_info)

        with safer.open(merge_fpath, 'w', temp_file=True) as f:
            json.dump(new_json_data, f, indent=4)

    dvc.add(summary_metrics)
    dvc.git_commitpush('Fixup merged iarpa metrics')
    dvc.push(summary_metrics, remote='aws')


if __name__ == '__main__':
    main()
