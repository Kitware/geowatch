#!/usr/bin/env python
import argparse
import sys
import os
import json
import parse
import itertools
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


def merge_bas_metrics_results(results: List[RegionResult]):
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
            return sum(map(area_sqkm, regions_poly))
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

            return len(dates[0].union_many(dates[1:])) - 1

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

    dfs = [to_df(r.bas_dpath, r.region_id) for r in results]

    merged_df = pd.concat(dfs)
    result_df = pd.DataFrame(index=dfs[0].droplevel('region_id').index)

    sum_cols = [
        'tp sites', 'fp sites', 'fn sites', 'truth sites', 'proposed sites',
        'total sites', 'truth slices', 'proposed slices'
    ]
    result_df[sum_cols] = merged_df.groupby(['rho', 'tau'])[sum_cols].sum()

    # ref: metrics-and-test-framework.evaluation.Metric
    (_, tp), (_, fp), (_, fn) = result_df[['tp sites', 'fp sites',
                                           'fn sites']].iteritems()
    result_df['precision'] = np.where(tp > 0, tp / (tp + fp), 0)
    result_df['recall (PD)'] = np.where(tp > 0, tp / (tp + fn), 0)
    result_df['F1'] = np.where(tp > 0, tp / (tp + 0.5 * (fp + fn)), 0)

    all_regions = [r.region_model for r in results]
    # ref: metrics-and-test-framework.evaluation.Evaluation.build_scoreboard
    result_df['spatial FAR'] = fp.astype(float) / area(all_regions)
    result_df['temporal FAR'] = fp.astype(float) / n_dates(all_regions)

    # this is not actually how Images FAR is calculated!
    # https://smartgitlab.com/TE/metrics-and-test-framework/-/issues/23
    #
    # all_sites = list(itertools.chain.from_iterable([
    #     r.site_models for r in results]))
    # result_df['images FAR'] = fp.astype(float) / n_unique_images(all_sites)
    #
    # instead, images in multiple proposed site stacks are double-counted.
    # take advantage of this to merge this metric with a simple average.
    n_images = (merged_df['fp sites'] /
                merged_df['images FAR']).groupby('region_id').mean().sum()
    result_df['images FAR'] = fp.astype(float) / n_images

    return result_df


def merge_sc_metrics_results(results: List[RegionResult]):
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
        TODO submit this as an issue
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
            axis=1, ignore_index=True)

        df.columns = pd.MultiIndex.from_product(
            #  ([region_id], site_names, ['truth', 'proposed']),
            #  names=['region_id', 'site', 'type'])
            (site_names, ['truth', 'proposed']),
            names=['site', 'type'])

        return df

    dfs = [to_df(r.sc_dpath, r.region_id) for r in results]
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
            'TIoU', axis=1) for r in results
    ],
                      axis=1).mean(axis=1, skipna=True).values

    # these are averaged using the mean over sites for each phase.
    # So the correct average over regions is to weight by (sites/region)
    temporal_errs = [
        pd.read_csv(os.path.join(
            r.sc_dpath,
            'activity_prediction_table.csv')).loc[0][1:].astype(float).values
        for r in results
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


def merge_metrics_results(region_dpaths, anns_root, out_dpath=None):
    '''
    Merge metrics results from multiple regions.

    Args:
        region_dpaths: List of directories containing the subdirs
            bas/
            phase_activity/ [optional]
            time_activity/ [TBD, not scored yet]
        anns_root: Path to GT annotations repo
        out_dpath: Directory to save merged results. Existing contents will
            be removed.
            Default is {common root of region_dpaths}/merged/

    Returns:
        (bas_df, sc_df)
        Two pd.DataFrames that are saved as
            {out_dpath}/(bas|sc)_scoreboard_df.pkl
    '''

    if out_dpath is None:
        out_dpath = os.path.join(os.path.commonpath(region_dpaths), 'merged')
    assert out_dpath not in region_dpaths
    os.system(f'rm -r {out_dpath}')
    os.makedirs(out_dpath, exist_ok=True)

    results = [
        RegionResult.from_dpath_and_anns_root(pth, anns_root)
        for pth in region_dpaths
    ]

    #
    # merge BAS
    #

    bas_df = merge_bas_metrics_results([r for r in results if r.bas_dpath])
    bas_df.to_pickle(os.path.join(out_dpath, 'bas_scoreboard_df.pkl'))
    #
    # merge SC
    #

    sc_df, sc_cm = merge_sc_metrics_results([r for r in results if r.sc_dpath])
    sc_df.to_pickle(os.path.join(out_dpath, 'sc_activity_df.pkl'))
    sc_cm.to_pickle(os.path.join(out_dpath, 'sc_confusion_df.pkl'))

    #
    # write summary in readable form
    #

    summary_path = os.path.join(out_dpath, 'summary.csv')
    if os.path.isfile(summary_path):
        os.remove(summary_path)
    with open(summary_path, 'a+') as f:
        best_bas_row = bas_df[bas_df['F1'] == bas_df['F1'].max()].iloc[[-1]]
        best_bas_row.to_csv(f)
        f.write('\n')
        sc_df.to_csv(f)
        f.write('\n')
        sc_cm.to_csv(f)

    return bas_df, sc_df, sc_cm


def ensure_thumbnails(image_path, ann_root, region_id, coco_dset):
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
        image_path: root directory to save under
        ann_root: $DVC_DPATH/annotations/ == smartgitlab.com/TE/annotations/
        region_id: ex. 'KR_R001'
        coco_dset: containing a video named {region_id}
    '''

    return NotImplemented


def main(args):
    parser = argparse.ArgumentParser(
        description='Score IARPA site model GeoJSON files using IARPA\'s '
        'metrics-and-test-framework')
    parser.add_argument('sites',
                        nargs='+',
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
        ''')
    # https://stackoverflow.com/a/49351471
    parser.add_argument('--virtualenv_cmd',
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

    parser.add_argument('--tmp_dir',
                        help='''
        If specified, will write temporary data here instead of using a
        non-persistant directory
        ''')

    if 0:  # TODO
        parser.add_argument('--keep_thumbnails',
                            action='store_true',
                            help='''
        Output thumbnails of region and ground truth sites to
        {out_dir}/thumbnails/
        ''')

    args = parser.parse_args(args)

    # load sites
    sites = []
    for site in args.sites:
        try:
            if os.path.isfile(site):
                site = json.load(open(site))
            else:
                site = json.loads(site)
        except json.JSONDecodeError as e:  # TODO split out as decorator?
            raise json.JSONDecodeError(e.msg + ' [cut for length]',
                                       e.doc[:100] + '...', e.pos)
        sites.append(site)

    # normalize paths
    if args.gt_dpath is not None:
        gt_dpath = os.path.abspath(args.gt_dpath)
    else:
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
        gt_dpath = dvc_dpath / 'annotations'
        print(f'gt_dpath unspecified, defaulting to {gt_dpath=}')
    assert os.path.isdir(gt_dpath), gt_dpath
    if args.metrics_dpath is not None:
        metrics_dpath = os.path.abspath(args.metrics_dpath)
    else:
        metrics_dpath = os.environ['METRICS_DPATH']
        print(f'metrics_dpath unspecified, defaulting to {metrics_dpath=}')
    assert os.path.isdir(metrics_dpath), metrics_dpath
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

    for region_id, region_sites in grouped_sites.items():

        site_dpath = (tmp_dpath / 'site' / region_id).ensuredir()
        image_dpath = (tmp_dpath / 'image' / region_id).ensuredir()
        cache_dpath = (tmp_dpath / 'cache' / region_id).ensuredir()

        if True:
            # doctor site_dpath for expected structure
            site_sub_dpath = os.path.join(site_dpath, 'latest', region_id)
            os.makedirs(site_sub_dpath, exist_ok=True)

            # copy site models to site_dpath
            for site in region_sites:
                with open(
                        os.path.join(
                            site_sub_dpath,
                            (site['features'][0]['properties']['site_id'] +
                             '.geojson')), 'w') as f:
                    json.dump(site, f)

            if 1:

                # link rgb images to image_dpath for viz
                img_date_dct = dict()
                for site in sites:
                    for feat in site['features'][1:]:
                        img_path = feat['properties']['source']
                        if os.path.isfile(img_path):
                            img_date_dct[img_path] = feat['properties'][
                                'observation_date']
                        else:
                            print(f'warning: image {img_path}'
                                  ' is not a valid path')
                for img_path, img_date in img_date_dct.items():
                    # use filename expected by metrics framework
                    ub.symlink(
                        img_path,
                        os.path.join(
                            image_dpath, '_'.join(
                                (img_date, os.path.basename(img_path)))),
                        overwrite=True)

            else:  # TODO finish updating this

                ensure_thumbnails()

            # run metrics framework
            if args.out_dir is not None:
                out_dir = os.path.join(args.out_dir, region_id)
            else:
                out_dir = None
            out_dirs.append(out_dir)
            # cache_dpath is always empty to work around bugs
            cmd = ub.paragraph(fr'''
                {virtualenv_cmd} &&
                python {os.path.join(metrics_dpath, 'run_evaluation.py')}
                    --roi {region_id}
                    --gt_path {os.path.join(gt_dpath, 'site_models')}
                    --rm_path {os.path.join(gt_dpath, 'region_models')}
                    --sm_path {site_dpath}
                    --image_dir {image_dpath}
                    --output_dir {out_dir}
                    --cache_dir {cache_dpath}
                ''')
            try:
                ub.cmd(cmd, verbose=3, check=True, shell=True)
            except subprocess.CalledProcessError:
                print('error in metrics framework, probably due to zero '
                      'TP site matches.')

    if args.merge:
        merge_metrics_results(out_dirs, gt_dpath)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
