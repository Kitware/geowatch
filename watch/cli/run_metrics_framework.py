#!/usr/bin/env python
import argparse
import sys
import os
import json
import parse
import pandas as pd
import numpy as np
import shapely.geometry
import shapely.ops
from tempfile import TemporaryDirectory
from dataclasses import dataclass
from glob import glob
from typing import List, Dict
import ubelt as ub
import subprocess
from packaging import version


@dataclass(frozen=True)
class RegionResult:
    region_id: str  # 'KR_R001'
    region_model: Dict
    site_models: List[Dict]
    bas_dpath: str = None  # 'path/to/scores/latest/KR_R001/bas/'
    sc_dpath: str = None  # 'path/to/scores/latest/KR_R001/phase_activity/'

    @classmethod
    def from_dpath_and_anns_root(cls, region_dpath, anns_root):
        region_id = os.path.basename(os.path.normpath(region_dpath))
        bas_dpath = os.path.join(region_dpath, 'bas')
        bas_dpath = bas_dpath if os.path.isdir(bas_dpath) else None
        sc_dpath = os.path.join(region_dpath, 'phase_activity')
        sc_dpath = sc_dpath if os.path.isdir(sc_dpath) else None
        region_model = json.load(
            open(
                os.path.join(anns_root, 'region_models',
                             region_id + '.geojson')))
        site_models = [
            json.load(open(pth)) for pth in sorted(
                glob(
                    os.path.join(anns_root, 'site_models',
                                 f'{region_id}_*.geojson')))
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

    as in bas/scoreboard_rho=*.csv
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
        scoreboard_fpaths = sorted(
            glob(os.path.join(bas_dpath, 'scoreboard_rho=*.csv')))

        # load each per-rho scoreboard and concat them
        rho_parser = parse.Parser('scoreboard_rho={rho:f}.csv')
        dfs = []
        for pth in scoreboard_fpaths:
            rho = rho_parser.parse(os.path.basename(pth)).named['rho']
            df = pd.read_csv(pth)
            df['rho'] = rho
            df['region_id'] = region_id
            # MultiIndex with rho, tau and region_id
            df = df.set_index(['region_id', 'rho', 'tau'])
            dfs.append(df)

        return pd.concat(dfs)

    dfs = [to_df(r.bas_dpath, r.region_id) for r in bas_results]

    concat_df = pd.concat(dfs)
    result_df = pd.DataFrame(index=dfs[0].droplevel('region_id').index)

    sum_cols = [
        'tp sites', 'fp sites', 'fn sites', 'truth sites', 'proposed sites',
        'total sites', 'truth slices', 'proposed slices'
    ]
    result_df[sum_cols] = concat_df.groupby(['rho', 'tau'])[sum_cols].sum()

    # ref: metrics-and-test-framework.evaluation.Metric
    (_, tp), (_, fp), (_, fn) = result_df[['tp sites', 'fp sites',
                                           'fn sites']].iteritems()
    result_df['precision'] = np.where(tp > 0, tp / (tp + fp), 0)
    result_df['recall (PD)'] = np.where(tp > 0, tp / (tp + fn), 0)
    result_df['F1'] = np.where(tp > 0, tp / (tp + 0.5 * (fp + fn)), 0)

    all_regions = [r.region_model for r in bas_results]
    # ref: metrics-and-test-framework.evaluation.Evaluation.build_scoreboard
    result_df['spatial FAR'] = fp.astype(float) / area(all_regions)
    result_df['temporal FAR'] = fp.astype(float) / n_dates(all_regions)

    # this is not actually how Images FAR is calculated!
    # https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/23
    #
    # all_sites = list(itertools.chain.from_iterable([
    #     r.site_models for r in bas_results]))
    # result_df['images FAR'] = fp.astype(float) / n_unique_images(all_sites)
    #
    # instead, images in multiple proposed site stacks are double-counted.
    # take advantage of this to merge this metric with a simple average.
    n_images = (concat_df['fp sites'] /
                concat_df['images FAR']).groupby('region_id').mean().sum()
    result_df['images FAR'] = fp.astype(float) / n_images

    return result_df, concat_df


def merge_sc_metrics_results(sc_results: List[RegionResult]):
    '''
    Returns:
        a list of pd.DataFrames
        activity_table: F1 score, mean TIoU, temporal error
        confusion_matrix: confusion matrix
    '''

    from sklearn.metrics import f1_score, confusion_matrix

    def to_df(sc_dpath, region_id):
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

        df = pd.read_csv(os.path.join(sc_dpath, 'activity_phase_table.csv'))

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.fillna(pd.NA).astype('string')

        site_names = df.columns.values.tolist()

        df = df.applymap(lambda cell: delim.join((cell, cell))
                         if not pd.isna(cell) and delim not in cell else cell)
        df = pd.concat(
            [df[col].str.split(delim, expand=True) for col in site_names],
            axis=1,
            ignore_index=True)

        df.columns = pd.MultiIndex.from_product(
            #  ([region_id], site_names, ['truth', 'proposed']),
            #  names=['region_id', 'site', 'type'])
            (site_names, ['truth', 'proposed']),
            names=['site', 'type'])

        return df

    dfs = [to_df(r.sc_dpath, r.region_id) for r in sc_results]
    df = pd.concat(dfs, axis=1).sort_values('date')

    # phase activity categories
    phase_classifications = [
        # "No Activity",
        "Site Preparation",
        "Active Construction",
        "Post Construction",
    ]

    sites = df.columns.levels[0]
    phase_true = np.concatenate(
        [df[site, 'truth'].dropna().to_numpy() for site in sites])
    phase_pred = np.concatenate(
        [df[site, 'proposed'].dropna().to_numpy() for site in sites])

    f1 = f1_score(phase_true,
                  phase_pred,
                  labels=phase_classifications,
                  average=None)

    # TIoU is only ever evaluated per-site, so we can safely average these
    # per-site and call it a new metric mTIoU.
    mtiou = pd.concat([
        pd.read_csv(os.path.join(r.sc_dpath, 'activity_tiou_table.csv')).drop(
            'TIoU', axis=1) for r in sc_results
    ],
                      axis=1).mean(axis=1, skipna=True).values

    # these are averaged using the mean over sites for each phase.
    # So the correct average over regions is to weight by (sites/region)
    temporal_errs = [
        pd.read_csv(os.path.join(
            r.sc_dpath,
            'activity_prediction_table.csv')).loc[0][1:].astype(float).values
        for r in sc_results
    ]
    n_sites = [df.shape[1] for df in dfs]
    try:
        temporal_errs, n_sites = zip(*filter(
            # TODO how to handle merging partial predictions?
            lambda tn: len(tn[0]) == 3,
            zip(temporal_errs, n_sites)))
        temporal_err = np.average(temporal_errs, weights=n_sites, axis=0)
    except ValueError:
        temporal_err = [np.nan, np.nan, np.nan]

    # import xdev; xdev.embed()
    activity_table = pd.DataFrame(
        {
            'F1 score': f1,
            'mean TIoU': mtiou,
            'Temporal Error (days)': temporal_err
        },
        index=phase_classifications).T
    activity_table = activity_table.rename_axis('Activity Classification',
                                                axis='columns')

    confusion_matrix = pd.DataFrame(confusion_matrix(
        phase_true, phase_pred, labels=phase_classifications),
                                    columns=phase_classifications,
                                    index=phase_classifications)
    confusion_matrix = confusion_matrix.rename_axis("truth phase")
    confusion_matrix = confusion_matrix.rename_axis("predicted phase",
                                                    axis="columns")

    return activity_table, confusion_matrix


def _hack_remerge_data():
    """
    Hack to redo the merge with a different F1 maximization criterion

    DVC_DPATH=$(WATCH_PREIMPORT=none python -m watch.cli.find_dvc --hardware="hdd")
    cd "$DVC_DPATH"
    ls models/fusion/eval3_candidates/eval/*/*/*/*/eval/curves/measures2.json
    ls models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json
    """
    import watch
    dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
    globstr = str(dvc_dpath / 'models/fusion/eval3_candidates/eval/*/*/*/*/eval/tracking/*/iarpa_eval/scores/merged/summary2.json')
    from watch.utils import util_path
    from watch.utils import simple_dvc
    summary_metrics = util_path.coerce_patterned_paths(globstr)
    import json
    import safer
    dvc = simple_dvc.SimpleDVC(dvc_dpath)

    # for merge_fpath in summary_metrics:
    #     if dvc.is_tracked(merge_fpath):
    #     pass

    dvc.unprotect(summary_metrics)

    for merge_fpath in ub.ProgIter(summary_metrics, desc='rewrite merge metrics'):
        region_dpaths = [p for p in list(merge_fpath.parent.parent.glob('*')) if p.name != 'merged']
        anns_root = dvc_dpath / 'annotations'

        # merge_dpath = merge_fpath.parent

        json_data = json.loads(merge_fpath.read_text())
        parent_info = json_data['parent_info']

        bas_concat_df, bas_df, sc_df, sc_cm = _make_merge_metrics(region_dpaths, anns_root)
        new_json_data, *_ = _make_summary_info(bas_concat_df, bas_df, sc_cm, sc_df, parent_info)

        with safer.open(merge_fpath, 'w', temp_file=True) as f:
            json.dump(new_json_data, f, indent=4)

    dvc.add(summary_metrics)
    dvc.git_commitpush('Fixup merged iarpa metrics')
    dvc.push(summary_metrics, remote='aws')


def _make_merge_metrics(region_dpaths, anns_root):
    results = [
        RegionResult.from_dpath_and_anns_root(pth, anns_root)
        for pth in region_dpaths
    ]

    # merge BAS
    bas_results = [r for r in results if r.bas_dpath]
    bas_df, bas_concat_df = merge_bas_metrics_results(bas_results)

    # merge SC
    sc_df, sc_cm = merge_sc_metrics_results([r for r in results if r.sc_dpath])

    return bas_concat_df, bas_df, sc_df, sc_cm


def _make_summary_info(bas_concat_df, bas_df, sc_cm, sc_df, parent_info, info):
    # Find best bas row in combined results

    min_rho, max_rho = 0.5, 0.5
    min_tau, max_tau = 0.2, 0.2

    bas_df = bas_df.reset_index()
    rho = bas_df['rho']
    tau = bas_df['tau']
    rho_flags = (min_rho <= rho) & (rho <= max_rho)
    tau_flags = (min_tau <= tau) & (tau <= max_tau)
    flags = tau_flags & rho_flags
    candidate_bas_df = bas_df[flags]

    bas_concat_df = bas_concat_df.reset_index()
    rho = bas_concat_df['rho']
    tau = bas_concat_df['tau']
    rho_flags = (min_rho <= rho) & (rho <= max_rho)
    tau_flags = (min_tau <= tau) & (tau <= max_tau)
    flags = tau_flags & rho_flags
    candidate_bas_concat_df = bas_concat_df[flags]

    # Find best merged bas row
    best_bas_row = candidate_bas_df.loc[[candidate_bas_df['F1'].idxmax()]]
    # Find best per-region bas row
    best_ids = candidate_bas_concat_df.groupby('region_id')['F1'].idxmax()
    best_per_region = candidate_bas_concat_df.loc[best_ids]
    best_bas_row_ = pd.concat({'merged':  best_bas_row}, names=['region_id'])
    # Get a best row for each region and the "merged" region
    best_bas_rows = pd.concat([best_per_region, best_bas_row_])
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

    return json_data, concise_best_bas_rows, best_bas_row_


def merge_metrics_results(region_dpaths, anns_root, merge_dpath, merge_fpath,
                          parent_info, info):
    '''
    Merge metrics results from multiple regions.

    Args:
        region_dpaths: List of directories containing the subdirs
            bas/
            phase_activity/ [optional]
            time_activity/ [TBD, not scored yet]
        anns_root: Path to GT annotations repo
        merge_dpath: Directory to save merged results.
            Existing contents will be removed.

    Returns:
        (bas_df, sc_df)
        Two pd.DataFrames that are saved as
            {out_dpath}/(bas|sc)_scoreboard_df.pkl
    '''
    import safer
    merge_dpath = ub.Path(merge_dpath)
    # assert merge_dpath not in region_dpaths
    # merge_dpath.delete().ensuredir()
    merge_dpath.ensuredir()

    bas_concat_df, bas_df, sc_df, sc_cm = _make_merge_metrics(region_dpaths, anns_root)
    bas_df.to_pickle(merge_dpath / 'bas_scoreboard_df.pkl')
    sc_df.to_pickle(merge_dpath / 'sc_activity_df.pkl')
    sc_cm.to_pickle(merge_dpath / 'sc_confusion_df.pkl')

    json_data, concise_best_bas_rows, best_bas_row = _make_summary_info(bas_concat_df, bas_df, sc_cm, sc_df, parent_info, info)
    print(concise_best_bas_rows.to_string())

    # write summary in readable form
    #
    summary_path = merge_dpath / 'summary.csv'
    with open(summary_path, 'w') as f:
        best_bas_row.to_csv(f)
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


def main(args):
    import safer
    parser = argparse.ArgumentParser(
        description='Score IARPA site model GeoJSON files using IARPA\'s '
        'metrics-and-test-framework')
    parser.add_argument('sites',
                        nargs='*',
                        help='''
        List of paths or serialized JSON strings containg v2 site models.
        All region_ids from these sites will be scored, and it will be assumed
        that there are no other sites in these regions.
        ''')
    parser.add_argument('--gt_dpath',
                        help='''
        Path to a local copy of the ground truth annotations,
        https://smartgitlab.com/TE/annotations.
        If None, use the environment variable DVC_DPATH to find
        $DVC_DPATH/annotations.
        ''')
    parser.add_argument('--metrics_dpath',
                        help='''
        Path to a local copy of the metrics framework,
        https://smartgitlab.com/TE/metrics-and-test-framework.
        If None, use the environment variable METRICS_DPATH.
        DEPRECATED. Simply ensure iarpa_smart_metrics is pip installed
                    in your virutalenv.
        ''')
    # https://stackoverflow.com/a/49351471
    parser.add_argument(
        '--virtualenv_cmd',
        default=['true'],  # no-op bash command
        nargs='+',  # hack for spaces
        help='''
        Command to run before calling the metrics framework in a subshell.
        The metrics framework should be installed in a different virtual env
        from WATCH, using eg conda or pyenv.
        ''')
    parser.add_argument('--out_dir',
                        help='''
        Output directory where scores will be written. Each region will have
        Defaults to ./output/
        ''')
    parser.add_argument('--merge',
                        action='store_true',
                        help='''
        Merge BAS and SC metrics from all regions and output to
        {out_dir}/merged/
        ''')

    parser.add_argument('--merge_fpath',
                        help='''
        Forces the merge summary to be written to a specific location.
        ''')

    parser.add_argument('--tmp_dir',
                        help='''
        If specified, will write temporary data here instead of using a
        non-persistant directory
        ''')

    parser.add_argument('--enable_viz',
                        help='''
        If true, enables iarpa visualizations
        ''')

    parser.add_argument('--name', default='unknown', help=(
        'Short name for the algorithm used to generate the model'))

    parser.add_argument(
        '--inputs_are_paths', action='store_true', help=ub.paragraph(
            '''
            If given, the sites inputs will always be interpreted as paths
            and not raw json text.
            '''))

    parser.add_argument(
        '--use_cache', default=False, action='store_true', help=ub.paragraph(
            '''
            IARPA metrics code currently contains a cache bug, do not enable
            the cache until this is fixed.
            '''))

    args, _ = parser.parse_known_args(args)
    print('args.__dict__ = {}'.format(ub.repr2(args.__dict__, nl=2)))

    # load sites
    sites = []
    if len(args.sites) == 0:
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
    jsonified_args = util_json.ensure_json_serializable(args.__dict__)
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
    for site_data in args.sites:
        if args.inputs_are_paths:
            in_fpath = ub.Path(site_data)
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
                    sites.append(site)
            else:
                # It was just a site json file.
                site = site_or_result
                sites.append(site)
        else:
            # TODO:
            # Deprecate passing raw json on the CLI, it has a limited length
            # What would be best is a single file that points to all of the
            # site jsons we care about, so we don't need to glob.
            try:
                if os.path.isfile(site_data):
                    with open(site_data, 'r') as file:
                        site = json.load(file)
                else:
                    site = json.loads(site_data)
            except json.JSONDecodeError as e:  # TODO split out as decorator?
                raise json.JSONDecodeError(e.msg + ' [cut for length]',
                                           e.doc[:100] + '...', e.pos)
            sites.append(site)

    name = args.name

    # normalize paths
    if args.gt_dpath is not None:
        gt_dpath = ub.Path(args.gt_dpath).absolute()
    else:
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
        gt_dpath = dvc_dpath / 'annotations'
        print(f'gt_dpath unspecified, defaulting to {gt_dpath=}')
    assert gt_dpath.is_dir(), gt_dpath
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    if args.tmp_dir is not None:
        tmp_dpath = ub.Path(args.tmp_dir)
    else:
        temp_dir = TemporaryDirectory(suffix='iarpa-metrics-tmp')
        tmp_dpath = ub.Path(temp_dir.name)

    # validate virtualenv command
    virtualenv_cmd = ' '.join(args.virtualenv_cmd)
    ub.cmd(virtualenv_cmd, verbose=1, check=True, shell=True)

    # split sites by region
    out_dirs = []
    grouped_sites = ub.group_items(
        sites, lambda site: site['features'][0]['properties']['region_id'])

    main_out_dir = ub.Path(args.out_dir or '.')

    main_out_dir = ub.Path(args.out_dir or '.')

    for region_id, region_sites in grouped_sites.items():

        site_dpath = (tmp_dpath / 'site' / region_id).ensuredir()
        image_dpath = (tmp_dpath / 'image').ensuredir()

        if args.use_cache:
            cache_dpath = (tmp_dpath / 'cache' / region_id).ensuredir()
        else:
            # Hack to disable cache by using a different directory each time
            _cache_dir = TemporaryDirectory(suffix='iarpa-metrics-cache')
            cache_dpath = ub.Path(_cache_dir.name)

        out_dir = (main_out_dir / region_id).ensuredir()

        # doctor site_dpath for expected structure
        site_sub_dpath = site_dpath / 'latest' / region_id
        site_sub_dpath.ensuredir()

        # copy site models to site_dpath
        for site in region_sites:
            geojson_fpath = site_sub_dpath / (
                site['features'][0]['properties']['site_id'] + '.geojson'
            )
            with safer.open(geojson_fpath, 'w', temp_file=True) as f:
                json.dump(site, f)

        ensure_thumbnails(image_dpath, region_id, region_sites)

        if not args.use_cache:
            cache_dpath = 'None'

        from scriptconfig.smartcast import smartcast
        if smartcast(args.enable_viz):
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

        viz_flags = ' '.join(viz_flags)
        import shlex
        run_eval_command = [
            'python', '-m', 'iarpa_smart_metrics.run_evaluation',
            '--roi', region_id,
            '--gt_dir', gt_dpath / 'site_models',
            '--rm_dir', gt_dpath / 'region_models',
            '--sm_dir', site_sub_dpath,
            '--image_dir', image_dpath,
            '--output_dir', out_dir if args.out_dir else 'None',
            '--cache_dir', cache_dpath,
            '--name', shlex.quote(name)
        ]
        run_eval_command += viz_flags
        # run metrics framework
        cmd = '{virtualenv_cmd} &&' + ' '.join(list(map(str, run_eval_command)))
        # ub.codeblock(fr'''
        #     {virtualenv_cmd} &&
        #     python -m iarpa_smart_metrics.run_evaluation \
        #         --roi {region_id} \
        #         --gt_dir {gt_dpath / 'site_models'} \
        #         --rm_dir {gt_dpath / 'region_models'} \
        #         --sm_dir {site_sub_dpath} \
        #         --image_dir {image_dpath} \
        #         --output_dir {out_dir if args.out_dir else None} \
        #         --cache_dir {cache_dpath} \
        #         --name {shlex.quote(name)} \
        #         {viz_flags}
        #     ''')
        (out_dir / 'invocation.sh').write_text(cmd)

        try:
            ub.cmd(cmd, verbose=3, check=True, shell=True)
            out_dirs.append(out_dir)
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
        merge_metrics_results(out_dirs, gt_dpath, merge_dpath, merge_fpath,
                              parent_info, info)
        # print('wrote {!r}'.format(summary_path2))


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
