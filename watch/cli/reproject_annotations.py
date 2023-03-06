#!/usr/bin/env python
r"""
Assigns geospace annotation to image pixels and frames

CommandLine:
    # Update a dataset with new annotations
    DVC_DATA_DPATH=$(smartwatch_dvc --tags='phase2_data' --hardware=auto)

    # You dont need to run this, but if you plan on running the reproject
    # annotations script multiple times, preloading this work will make it
    # faster
    python -m watch add_fields \
        --src $DVC_DATA_DPATH/Drop6/data.kwcoco.json \
        --dst $DVC_DATA_DPATH/Drop6/data.kwcoco.json \
        --overwrite=warp --workers 10

    # Update to whatever the state of the annotations submodule is
    python -m watch reproject_annotations \
        --src $DVC_DATA_DPATH/Drop6/data.kwcoco.json \
        --dst $DVC_DATA_DPATH/Drop6/data.kwcoco.json \
        --viz_dpath $DVC_DATA_DPATH/Drop6/_viz_propogate \
        --site_models="$DVC_DATA_DPATH/annotations/drop6/site_models/*.geojson"

    python -m watch visualize \
        --src $DVC_DATA_DPATH/Drop6/data.kwcoco.json \
        --space="video" \
        --num_workers=avail \
        --any3="only" --draw_anns=True --draw_imgs=False --animate=True


Notes:
    To add iMerit regions, we will need to recognize site-summaries from
    region models instead of site-models themselves.  Code to do this is in:
        https://gitlab.kitware.com/smart/watch/-/blob/master/watch/cli/kwcoco_to_geojson.py#L476
    in `add_site_summary_to_kwcoco`.
"""
import dateutil
import kwcoco
import kwimage
import ubelt as ub
import numpy as np
import scriptconfig as scfg
import io
import warnings
import math
from watch.utils import kwcoco_extensions
from watch.utils import util_kwplot
from watch.utils import util_time
from watch.utils.util_environ import envflag


class ReprojectAnnotationsConfig(scfg.Config):
    """
    Projects annotations from geospace onto a kwcoco dataset and optionally
    propogates them forward in time.

    References:
        https://smartgitlab.com/TE/annotations/-/wikis/Alternate-Site-Type
    """
    default = {
        'src': scfg.Value(help=ub.paragraph(
            '''
            path to the kwcoco file to propagate labels in
            '''), position=1),

        'dst': scfg.Value('propagated_data.kwcoco.json', help=ub.paragraph(
            '''
            Where the output kwcoco file with propagated labels is saved
            '''), position=2),

        'site_models': scfg.Value(None, help=ub.paragraph(
            '''
            Geospatial geojson "site" annotation files. Either a path to a
            file, or a directory.
            ''')),

        'region_models': scfg.Value(None, help=ub.paragraph(
            '''
            Geospatial geojson "region" annotation files. Containing site
            summaries.  Either a path to a file, or a directory.
            ''')),

        'viz_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            if specified, visualizations will be written to this directory
            ''')),

        'verbose': scfg.Value(1, help=ub.paragraph(
            '''
            use this to print details
            ''')),

        'role': scfg.Value(None, help=ub.paragraph(
            '''
            if specified, the value is assigned as the role of each new
            annotation.
            ''')),

        'clear_existing': scfg.Value(True, help=ub.paragraph(
            '''
            if True, clears existing annotations before reprojecting the new ones.
            ''')),

        'propogate_strategy': scfg.Value('SMART', help='strategy for how to interpolate annotations over time'),

        'geo_preprop': scfg.Value('auto', help='force if we check geo properties or not'),

        'geospace_lookup': scfg.Value('auto', help='if False assumes region-ids can be used to lookup association'),

        'workers': scfg.Value(0, help='number of workers for geo-preprop if done'),

        'status_to_catname': scfg.Value(None, help=ub.paragraph(
            '''
            Can be yaml or a path to a yaml file containing a mapping from
            status to kwcoco category names. This partially overwrites behavior
            in heuristics.PHASE_STATUS_TO_KWCOCO_CATNAME, so only the
            difference mapping needs to be specified.
            E.g. "{positive_excluded: positive}".
            '''))
    }


DUMMY_START_DATE = '1970-01-01'
DUMMY_END_DATE = '2101-01-01'


def main(cmdline=False, **kwargs):
    r"""
    Example:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.cli.reproject_annotations import *  # NOQA
        >>> import watch
        >>> dvc_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
        >>> coco_fpath = dvc_dpath / 'Drop6/imgonly-KR_R001.kwcoco.json'
        >>> dpath = ub.Path.appdir('watch/tests/project_annots').ensuredir()
        >>> cmdline = False
        >>> output_fpath = dpath / 'test_project_data.kwcoco.json'
        >>> viz_dpath = (dpath / 'viz').ensuredir()
        >>> kwargs = {
        >>>     'src': coco_fpath,
        >>>     'dst': output_fpath,
        >>>     'viz_dpath': viz_dpath,
        >>>     'site_models': dvc_dpath / 'annotations/site_models',
        >>>     'region_models': dvc_dpath / 'annotations/region_models',
        >>> }
        >>> main(cmdline=cmdline, **kwargs)
    """
    import geopandas as gpd  # NOQA
    from watch.utils import util_gis
    from watch.utils import util_parallel
    from watch.utils import util_yaml
    from watch import heuristics
    config = ReprojectAnnotationsConfig(data=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    output_fpath = config['dst']
    if output_fpath is None:
        raise AssertionError

    # Load the coco dataset with all of the images
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])

    geo_preprop = config['geo_preprop']
    geospace_lookup = config['geospace_lookup']
    if geo_preprop == 'auto':
        if len(coco_dset.index.imgs):
            coco_img = coco_dset.coco_image(ub.peek(coco_dset.index.imgs.keys()))
            geo_preprop = not any('geos_corners' in obj for obj in coco_img.iter_asset_objs())
            print('auto-choose geo_preprop = {!r}'.format(geo_preprop))

    workers = util_parallel.coerce_num_workers(config['workers'])

    if geo_preprop:
        kwcoco_extensions.coco_populate_geo_heuristics(
            coco_dset, overwrite={'warp'}, workers=workers,
            keep_geotiff_metadata=False,
        )

    # Read the external CRS84 annotations from the site models

    HACK_HANDLE_DUPLICATE_SITE_ROWS = envflag(
        'HACK_HANDLE_DUPLICATE_SITE_ROWS', default=True)

    site_model_infos = list(util_gis.coerce_geojson_datas(
        config['site_models'], desc='load site models', allow_raw=True,
        workers=workers))

    status_to_catname_default = ub.udict(heuristics.PHASE_STATUS_TO_KWCOCO_CATNAME)
    status_to_catname = util_yaml.coerce_yaml(config['status_to_catname'])
    if status_to_catname is not None:
        status_to_catname = status_to_catname_default | status_to_catname

    sites = []
    for info in site_model_infos:
        gdf = info['data']
        is_site = gdf['type'] == 'site'
        if HACK_HANDLE_DUPLICATE_SITE_ROWS:
            if is_site.sum() > 1:
                # There are some site models that contain duplicate site rows.
                # Fix them here.
                site_rows = gdf[is_site]
                assert ub.allsame(site_rows['site_id'])
                gdf = gdf.drop(site_rows.iloc[1:].index, axis=0)
        sites.append(gdf)

    regions = []
    if config['region_models'] is not None:
        region_model_infos = list(util_gis.coerce_geojson_datas(
            config['region_models'], desc='load geojson region-models',
            allow_raw=True, workers=workers))
        for info in region_model_infos:
            gdf = info['data']
            regions.append(gdf)

    if config['clear_existing']:
        coco_dset.clear_annotations()

    region_id_to_sites = expand_site_models_with_site_summaries(sites, regions)

    propogate_strategy = config['propogate_strategy']
    viz_dpath = config['viz_dpath']
    want_viz = bool(viz_dpath)

    propogated_annotations, all_drawable_infos = assign_sites_to_images(
        coco_dset, region_id_to_sites, propogate_strategy,
        geospace_lookup=geospace_lookup, want_viz=want_viz,
        status_to_catname=status_to_catname)

    if config['role'] is not None:
        _role = config['role']
        for ann in propogated_annotations:
            ann['role'] = _role
        del _role

    for ann in propogated_annotations:
        coco_dset.add_annotation(**ann)

    kwcoco_extensions.warp_annot_segmentations_from_geos(coco_dset)

    if output_fpath != 'return':
        # print('dump coco_dset.fpath = {!r}'.format(coco_dset.fpath))
        coco_dset.fpath = output_fpath
        coco_dset.dump(coco_dset.fpath)

    if viz_dpath == 'auto':
        viz_dpath = (ub.Path(coco_dset.fpath).parent / '_viz_reproject_anns')
    if viz_dpath:
        import kwplot
        kwplot.autoplt()
        viz_dpath = ub.Path(viz_dpath).ensuredir()
        print('viz_dpath = {!r}'.format(viz_dpath))
        for fnum, info in enumerate(ub.ProgIter(all_drawable_infos, desc='draw region site propogation', verbose=3)):
            drawable_region_sites = info['drawable_region_sites']
            region_id = info['region_id']
            region_image_dates = info['region_image_dates']
            fig = kwplot.figure(fnum=fnum)
            ax = fig.gca()
            plot_image_and_site_times(coco_dset, region_image_dates,
                                      drawable_region_sites, region_id, ax=ax)

            fig.set_size_inches(np.array([16, 9]) * 1)
            plot_fpath = viz_dpath / f'time_propogation_{region_id}.png'
            fig.tight_layout()
            fig.savefig(plot_fpath)

    if output_fpath == 'return':
        return coco_dset


def check_sitemodel_assumptions(sites):
    """
    For debugging and checking assumptions about site models
    """
    try:
        for site_df in ub.ProgIter(sites, desc='checking site assumptions'):
            first = site_df.iloc[0]
            rest = site_df.iloc[1:]
            assert first['type'] == 'site', (
                f'first row must have type of site, got {first["type"]}')
            assert first['region_id'] is not None, (
                f'first row must have a region id. Got {first["region_id"]}')
            assert rest['type'].apply(lambda x: x == 'observation').all(), (
                f'rest of row must have type observation. Instead got: {rest["type"].unique()}')
            # assert rest['region_id'].apply(lambda x: x is None).all(), (
            #     f'rest of row must have region_id=None. Instead got {rest["region_id"].unique()}')
    except AssertionError:
        print(site_df)
        raise


def separate_region_model_types(regions):
    """
    Split up each region model into its region info and site summary info
    """
    region_id_to_site_summaries = {}
    region_id_region_row = {}
    for region_df in ub.ProgIter(regions, desc='checking region assumptions', verbose=3):
        is_region = region_df['type'] == 'region'
        region_part = region_df[is_region]
        assert len(region_part) == 1, 'must have exactly one region in each region file'
        assert region_part['region_id'].apply(lambda x: x is not None).all(), 'regions must have region ids'

        region_row = region_part.iloc[0]
        region_id = region_row['region_id']
        region_id_region_row[region_id] = region_row
        assert region_id not in region_id_to_site_summaries

        # Hack to set all region-ids
        region_df.loc[:, 'region_id'] = region_id
        sites_part = region_df[~is_region]

        FIX_LEADING_SPACE = 1
        if FIX_LEADING_SPACE:
            sites_part.loc[sites_part['type'] == ' site_summary', 'type'] = 'site_summary'

        try:
            assert (sites_part['type'] == 'site_summary').all(), 'rest of data must be site summaries'
            assert sites_part['region_id'].apply(lambda x: (x is None) or x == region_id).all(), (
                'site-summaries do not have region ids (unless we make them)')
        except AssertionError:
            print(sites_part['type'].unique())
            embed_if_requested()
            raise
        region_id_to_site_summaries[region_id] = sites_part

        # Check datetime errors
        sitesum_start_dates = sites_part['start_date'].apply(lambda x: util_time.coerce_datetime(x or DUMMY_START_DATE))
        sitesum_end_dates = sites_part['end_date'].apply(lambda x: util_time.coerce_datetime(x or DUMMY_END_DATE))
        has_bad_time_range = sitesum_start_dates > sitesum_end_dates
        bad_dates = sites_part[has_bad_time_range]
        if len(bad_dates):
            print('BAD DATES')
            print(bad_dates)
    return region_id_to_site_summaries, region_id_region_row


def expand_site_models_with_site_summaries(sites, regions):
    """
    Takes all site summaries from region models that do not have a
    corresponding site model and makes a "pseudo-site-model" for use in BAS.

    Args:
        sites (List[GeoDataFrame]):
        regions (List[GeoDataFrame]):

    Returns:
        Dict[str, List[DataFrame]] : region_id_to_sites  :
            a mapping from region names to a list of site models both
            real and/or pseudo.
    """
    import pandas as pd
    # import watch

    if __debug__:
        check_sitemodel_assumptions(sites)

    region_id_to_site_summaries, region_id_region_row = separate_region_model_types(regions)

    region_id_to_sites = ub.group_items(sites, lambda x: x.iloc[0]['region_id'])

    VERYVERBOSE = 0

    region_id_to_num_sitesumms = ub.map_vals(len, region_id_to_site_summaries)
    region_id_to_num_sites = ub.map_vals(len, region_id_to_sites)
    print('region_id_to_num_sitesumms = {}'.format(ub.repr2(region_id_to_num_sitesumms, nl=1, sort=0)))
    if VERYVERBOSE:
        print('region_id_to_num_sites = {}'.format(ub.repr2(region_id_to_num_sites, nl=1, sort=0)))

    site_rows1 = []
    for region_id, region_sites in region_id_to_sites.items():
        for site in region_sites:
            site_sum_rows = site[site['type'] == 'site']
            assert len(site_sum_rows) == 1
            site_rows1.append(site_sum_rows)

    site_rows2 = []
    for region_id, site_summaries in region_id_to_site_summaries.items():
        site_rows2.append(site_summaries)

    expected_keys = ['index', 'observation_date', 'source', 'sensor_name', 'type',
                     'current_phase', 'is_occluded', 'is_site_boundary', 'region_id',
                     'site_id', 'version', 'status', 'mgrs', 'score', 'start_date',
                     'end_date', 'model_content', 'originator', 'validated', 'geometry',
                     'misc_info' ]

    if site_rows1:
        site_df1 = pd.concat(site_rows1).reset_index()
        assert len(set(site_df1['site_id'])) == len(site_df1), 'site ids must be unique'
        site_df1 = site_df1.set_index('site_id', drop=False, verify_integrity=True).drop('index', axis=1)
        if 'misc_info' not in site_df1.columns:
            site_df1['misc_info'] = None
    else:
        site_df1 = pd.DataFrame([], columns=expected_keys)

    if site_rows2:
        site_df2 = pd.concat(site_rows2).reset_index()
        if len(set(site_df2['site_id'])) != len(site_df2):
            counts = site_df2['site_id'].value_counts()
            duplicates = counts[counts > 1]
            warnings.warn('Site summaries contain duplicate site_ids:\n{}'.format(duplicates))
            # Filter to unique sites
            unique_idx = np.unique(site_df2['site_id'], return_index=True)[1]
            site_df2 = site_df2.iloc[unique_idx]

        site_df2 = site_df2.set_index('site_id', drop=False, verify_integrity=True).drop('index', axis=1)

        if 'misc_info' not in site_df2.columns:
            site_df2['misc_info'] = None
    else:
        site_df2 = pd.DataFrame([], columns=expected_keys)

    common_site_ids = sorted(set(site_df1['site_id']) & set(site_df2['site_id']))
    common1 = site_df1.loc[common_site_ids]
    common2 = site_df2.loc[common_site_ids]

    common_columns = common1.columns.intersection(common2.columns)
    common_columns = common_columns.drop(['type', 'region_id'])

    col_to_flags = {}
    for col in common_columns:
        error_flags = ~(
            (common1[col] == common2[col]) |
            (common1[col].isnull() & common2[col].isnull()))
        col_to_flags[col] = error_flags

    print('col errors: ' + repr(ub.map_vals(sum, col_to_flags)))
    any_error_flag = np.logical_or.reduce(list(col_to_flags.values()))
    total_error_rows = any_error_flag.sum()
    print('total_error_rows = {!r}'.format(total_error_rows))
    if total_error_rows:
        error1 = common1[any_error_flag]
        error2 = common2[any_error_flag]
        columns = ['site_id', 'version', 'mgrs', 'start_date', 'end_date', 'status', 'originator', 'score', 'model_content', 'validated']

        error1 = reorder_columns(error1, columns)
        error2 = reorder_columns(error2, columns)
        print('Disagree rows for site models')
        print(error1.drop(['type', 'region_id', 'misc_info'], axis=1))
        print('Disagree rows for region models')
        print(error2.drop(['type', 'region_id', 'validated'], axis=1))

    # Find sites that only have a site-summary
    summary_only_site_ids = sorted(set(site_df2['site_id']) - set(site_df1['site_id']))
    region_id_to_only_site_summaries = dict(list(site_df2.loc[summary_only_site_ids].groupby('region_id')))

    if VERYVERBOSE:
        region_id_to_num_only_sitesumms = ub.map_vals(len, region_id_to_only_site_summaries)
        print('region_id_to_num_only_sitesumms = {}'.format(ub.repr2(region_id_to_num_only_sitesumms, nl=1, sort=0)))

    # Transform site-summaries without corresponding sites into pseudo-site
    # observations
    # https://smartgitlab.com/TE/standards/-/wikis/Site-Model-Specification

    # if 0:
    #     # Use the json schema to ensure we are coding this correctly
    #     import jsonref
    #     from watch.rc.registry import load_site_model_schema
    #     site_model_schema = load_site_model_schema()
    #     # Expand the schema
    #     site_model_schema = jsonref.loads(jsonref.dumps(site_model_schema))
    #     site_model_schema['definitions']['_site_properties']['properties'].keys()
    #     list(ub.flatten([list(p['properties'].keys()) for p in site_model_schema['definitions']['unassociated_site_properties']['allOf']]))
    #     list(site_model_schema['definitions']['unassociated_site_properties']['properties'].keys())
    #     list(ub.flatten([list(p['properties'].keys()) for p in site_model_schema['definitions']['associated_site_properties']['allOf']]))
    #     list(site_model_schema['definitions']['associated_site_properties']['properties'].keys())
    #     site_model_schema['definitions']['observation_properties']['properties']

    # resolver = jsonschema.RefResolver.from_schema(site_model_schema)
    # site_model_schema[

    region_id_to_num_sites = ub.map_vals(len, region_id_to_sites)
    # print('BEFORE region_id_to_num_sites = {}'.format(ub.repr2(region_id_to_num_sites, nl=1)))

    for region_id, sitesummaries in region_id_to_only_site_summaries.items():
        region_row = region_id_region_row[region_id]
        pseud_sites = make_pseudo_sitemodels(region_row, sitesummaries)
        region_id_to_sites[region_id].extend(pseud_sites)

    region_id_to_num_sites = ub.map_vals(len, region_id_to_sites)
    if VERYVERBOSE:
        print('AFTER (sitesummary) region_id_to_num_sites = {}'.format(ub.repr2(region_id_to_num_sites, nl=1)))

    def is_nonish(x):
        return x is None or isinstance(x, float) and math.isnan(x)

    # Fix out of order observations
    FIX_OBS_ORDER = True
    if FIX_OBS_ORDER:
        new_region_id_to_sites = {}
        for region_id, region_sites in region_id_to_sites.items():
            _sites = []
            for site_gdf in region_sites:
                site_gdf['observation_date'].argsort()
                is_obs = site_gdf['type'] == 'observation'
                obs_rows = site_gdf[is_obs]
                site_rows = site_gdf[~is_obs]
                obs_rows = obs_rows.iloc[obs_rows['observation_date'].argsort()]

                assert not is_nonish(site_rows['status'])
                if obs_rows['observation_date'].apply(is_nonish).any():
                    obs_rows.loc[obs_rows['observation_date'].apply(is_nonish).values, 'observation_date'] = DUMMY_END_DATE
                assert not obs_rows['observation_date'].apply(is_nonish).any()

                # raise Exception
                # site_gdf

                site_gdf = pd.concat([site_rows.reset_index(), obs_rows.reset_index()], axis=0).reset_index()
                _sites.append(site_gdf)
            new_region_id_to_sites[region_id] = _sites
        region_id_to_sites = new_region_id_to_sites

    if 0:
        site_high_level_summaries = []
        for region_id, region_sites in region_id_to_sites.items():
            print('=== {} ==='.format(region_id))
            for site_gdf in region_sites:
                site_summary_row = site_gdf.iloc[0]
                site_rows = site_gdf.iloc[1:]
                track_id = site_summary_row['site_id']
                status = site_summary_row['status']
                status = status.lower().strip()
                summary = {
                    'region_id': region_id,
                    'track_id': track_id,
                    'status': status,
                    'start_date': site_summary_row['start_date'],
                    'end_date': site_summary_row['end_date'],
                    'unique_phases': site_rows['current_phase'].unique(),
                }
                # print('summary = {}'.format(ub.repr2(summary, nl=0)))
                site_high_level_summaries.append(summary)

        df = pd.DataFrame(site_high_level_summaries)
        for region_id, subdf in df.groupby('region_id'):
            print('=== {} ==='.format(region_id))
            subdf = subdf.sort_values('status')
            print(subdf.to_string())

    if __debug__:
        for region_id, region_sites in ub.ProgIter(region_id_to_sites.items(), desc='validate sites'):
            for site_df in region_sites:
                validate_site_dataframe(site_df)

    return region_id_to_sites


def make_pseudo_sitemodels(region_row, sitesummaries):
    import geojson
    import json
    from watch.utils import util_gis

    # observation_properties = [
    #     'type', 'observation_date', 'source', 'sensor_name',
    #     'current_phase', 'is_occluded', 'is_site_boundary', 'score',
    #     'misc_info'
    # ]
    site_properites = [
        'type', 'version', 'mgrs', 'status', 'model_content', 'start_date',
        'end_date', 'originator', 'score', 'validated', 'misc_info',
        'region_id', 'site_id']
    # Use region start/end date if the site does not have them
    region_start_date = region_row['start_date'] or DUMMY_START_DATE
    region_end_date = region_row['end_date'] or DUMMY_END_DATE

    region_start_date, region_end_date = sorted(
        [region_start_date, region_end_date], key=util_time.coerce_datetime)

    pseudo_sites = []
    for _, site_summary in sitesummaries.iterrows():
        geom = site_summary['geometry']
        if geom is None:
            print('warning got none geom')
            continue

        try:
            poly_json = kwimage.Polygon.from_shapely(geom.convex_hull).to_geojson()
        except Exception as ex:
            print(f'ex={ex}')
            embed_if_requested()
            raise
        mpoly_json = kwimage.MultiPolygon.from_shapely(geom).to_geojson()

        has_keys = site_summary.index.intersection(site_properites)
        # missing_keys = pd.Index(site_properites).difference(site_summary.index)
        psudo_site_prop = site_summary[has_keys].to_dict()
        psudo_site_prop['type'] = 'site'
        # TODO: how to handle missing start / end dates?
        start_date = site_summary['start_date'] or region_start_date
        end_date = site_summary['end_date'] or region_end_date

        # hack
        start_date, end_date = sorted(
            [start_date, end_date], key=util_time.coerce_datetime)

        psudo_site_prop['start_date'] = start_date
        psudo_site_prop['end_date'] = end_date

        start_datetime = util_time.coerce_datetime(start_date)
        end_datetime = util_time.coerce_datetime(end_date)

        assert start_datetime <= end_datetime

        score = site_summary.get('score', None)
        try:
            score = float(score)
        except TypeError:
            ...

        observation_prop_template = {
            'type': 'observation',
            'observation_date': None,
            # 'source': None,
            # 'sensor_name': None,
            # 'current_phase': None,
            # 'is_occluded': None,
            # 'is_site_boundary': None,
            'score': score,
            # 'misc_info': None,
        }

        psudo_site_features = [
            geojson.Feature(
                properties=psudo_site_prop, geometry=poly_json,
            )
        ]
        psudo_site_features.append(
            geojson.Feature(
                properties=ub.dict_union(observation_prop_template, {
                    'observation_date': start_date,
                    'current_phase': None,
                }),
                geometry=mpoly_json)
        )
        psudo_site_features.append(
            geojson.Feature(
                properties=ub.dict_union(observation_prop_template, {
                    'observation_date': end_date,
                    'current_phase': None,
                }),
                geometry=mpoly_json)
        )
        psudo_site_model = geojson.FeatureCollection(psudo_site_features)
        pseudo_gpd = util_gis.load_geojson(io.StringIO(json.dumps(psudo_site_model)))
        pseudo_sites.append(pseudo_gpd)
    return pseudo_sites

    # if 1:
    #     from watch.rc.registry import load_site_model_schema
    #     site_model_schema = load_site_model_schema()
    #     real_site_model = json.loads(ub.Path('/media/joncrall/flash1/smart_watch_dvc/annotations/site_models/BR_R002_0009.geojson').read_text())
    #     ret = jsonschema.validate(real_site_model, schema=site_model_schema)
    #     import jsonschema
    #     ret = jsonschema.validate(psudo_site_model, schema=site_model_schema)


def validate_site_dataframe(site_df):
    from watch.utils import util_time
    import numpy as np
    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'
    first = site_df.iloc[0]
    rest = site_df.iloc[1:]
    assert first['type'] == 'site', 'first row must have type of site'
    assert first['region_id'] is not None, 'first row must have a region id'
    assert rest['type'].apply(lambda x: x == 'observation').all(), (
        'rest of row must have type observation')
    # assert rest['region_id'].apply(lambda x: x is None).all(), (
    #     'rest of row must have region_id=None')

    site_start_date = first['start_date'] or dummy_start_date
    site_end_date = first['end_date'] or dummy_end_date
    import math
    if isinstance(site_end_date, float) and math.isnan(site_end_date):
        site_end_date = dummy_end_date
    if isinstance(site_start_date, float) and math.isnan(site_start_date):
        site_start_date = dummy_start_date

    site_start_datetime = util_time.coerce_datetime(site_start_date)
    site_end_datetime = util_time.coerce_datetime(site_end_date)

    if site_end_datetime < site_start_datetime:
        print('\n\nBAD SITE DATES:')
        print(first)

    status = {}
    # Check datetime errors in observations
    try:
        obs_dates = [None if x is None else util_time.coerce_datetime(x) for x in rest['observation_date']]
        obs_isvalid = [x is not None for x in obs_dates]
        valid_obs_dates = list(ub.compress(obs_dates, obs_isvalid))
        if not all(valid_obs_dates):
            # null_obs_sites.append(first[['site_id', 'status']].to_dict())
            pass
        valid_deltas = np.array([d.total_seconds() for d in np.diff(valid_obs_dates)])
        if not (valid_deltas >= 0).all():
            raise AssertionError('observations are not sorted temporally')
            # warnings.warn('observations are not sorted temporally')
    except AssertionError as ex:
        print('ex = {!r}'.format(ex))
        print(site_df)
        raise

    return status


def assign_sites_to_images(coco_dset, region_id_to_sites, propogate_strategy,
                           geospace_lookup='auto', want_viz=1,
                           status_to_catname=None):
    """
    Given a coco dataset (with geo information) and a list of geojson sites,
    determines which images each site-annotations should go on.

    Args:
        coco_dset (kwcoco.CocoDataset):
        region_id_to_sites (Dict[str, List[DataFrame]]):
        propogate_strategy: (str): a code that describes how we
           are going to past/future propogate
        geospace_lookup: (str):
        want_viz (bool):

    Returns:
        Tuple[List[Dict[str, Any]], Dict[str, Any]]:
            A tuple: propogated_annotations, all_drawable_infos
    """
    from shapely.ops import unary_union
    import pandas as pd
    from watch.utils import util_gis
    # Create a geopandas data frame that contains the CRS84 extent of all images
    img_gdf = kwcoco_extensions.covered_image_geo_regions(coco_dset)

    # Group all images by their video-id and take a union of their geometry to
    # get a the region covered by the video.
    video_gdfs = []
    vidid_to_imgdf = {}
    for vidid, subdf in img_gdf.groupby('video_id'):
        subdf = subdf.sort_values('frame_index')
        video_gdf = subdf.dissolve()
        video_gdf = video_gdf.drop(['date_captured', 'name', 'image_id', 'frame_index', 'height', 'width'], axis=1)
        combined = unary_union(list(subdf.geometry.values))
        video_gdf['geometry'].iloc[0] = combined
        video_gdfs.append(video_gdf)
        vidid_to_imgdf[vidid] = subdf

    if len(video_gdfs) > 0:
        videos_gdf = pd.concat(video_gdfs, ignore_index=True)
    else:
        videos_gdf = None

    # Ensure colors and categories
    from watch import heuristics
    status_to_color = {d['name']: kwimage.Color(d['color']).as01()
                       for d in heuristics.HUERISTIC_STATUS_DATA}
    # print('coco_dset categories = {}'.format(ub.repr2(coco_dset.dataset['categories'], nl=2)))
    for cat in heuristics.CATEGORIES:
        coco_dset.ensure_category(**cat)
    # hack in heuristic colors
    heuristics.ensure_heuristic_coco_colors(coco_dset)
    # handle any other colors
    kwcoco_extensions.category_category_colors(coco_dset)
    # print('coco_dset categories = {}'.format(ub.repr2(coco_dset.dataset['categories'], nl=2)))

    all_drawable_infos = []  # helper if we are going to draw

    if geospace_lookup == 'auto':
        coco_video_names = set(coco_dset.index.name_to_video.keys())
        region_ids = set(region_id_to_sites.keys())
        geospace_lookup = not coco_video_names.issubset(region_ids)
        print('geospace_lookup = {!r}'.format(geospace_lookup))

    # Find the video associated with each region
    # If this assumption is not valid, we could refactor to loop through
    # each site, do the geospatial lookup, etc...
    # but this is faster if we know regions are consistent
    video_id_to_region_id = {}
    if geospace_lookup:
        # Association via geospace lookup
        video_id_to_region_ids = ub.ddict(list)
        for region_id, region_sites in region_id_to_sites.items():
            video_ids = []
            for site_gdf in region_sites:
                # determine which video it the site belongs to
                video_overlaps = util_gis.geopandas_pairwise_overlaps(site_gdf, videos_gdf)
                overlapping_video_indexes = set(np.hstack(list(video_overlaps.values())))
                if len(overlapping_video_indexes) > 0:
                    # assert len(overlapping_video_indexes) == 1, 'should only belong to one video'
                    overlapping_video_indexes = list(overlapping_video_indexes)
                    # overlapping_video_index = ub.peek(overlapping_video_indexes)
                    # video_name = coco_dset.index.videos[video_id]['name']
                    # assert site_gdf.iloc[0].region_id == video_name, 'sanity check'
                    # assert site_gdf.iloc[0].region_id == region_id, 'sanity check'
                    video_ids.extend(videos_gdf.iloc[overlapping_video_indexes]['video_id'].tolist())
            video_ids = sorted(set(video_ids))
            if len(video_ids) > 1:
                warnings.warn('A site exists in more than one video')
            # assert ub.allsame(video_ids)
            if len(video_ids) == 0:
                print('No geo-space match for region_id={}'.format(region_id))
                continue
            for video_id in video_ids:
                video_id_to_region_ids[video_id].append(region_id)

            for video_id, region_ids in video_id_to_region_ids.items():
                if len(region_ids) != 1:
                    # FIXME: This should not be the case, but it seems it is
                    # due to super regions maybe? If it is super regions this
                    # hack of just choosing one of them, should be mostly ok?
                    msg = f'a video {video_id=} contains more than one region {region_ids=}, not a handled case yet. We are punting and just choosing 1'
                    warnings.warn(msg)
                    # raise AssertionError(msg)
                video_id_to_region_id[video_id] = region_ids[0]

    else:
        # Association via video name
        for region_id, region_sites in region_id_to_sites.items():
            try:
                video = coco_dset.index.name_to_video[region_id]
            except KeyError:
                VERYVERBOSE = 0
                if VERYVERBOSE:
                    print('No region-id match for region_id={}'.format(region_id))
                continue
            video_id = video['id']
            video_id_to_region_id[video_id] = region_id

    print('Found Association: video_id_to_region_id = {}'.format(ub.repr2(video_id_to_region_id, nl=1)))
    propogated_annotations = []
    for video_id, region_id in video_id_to_region_id.items():
        region_sites = region_id_to_sites[region_id]
        print(f'{region_id=} {video_id=} #sites={len(region_sites)}')
        # Grab the images data frame for that video
        subimg_df = vidid_to_imgdf[video_id]
        region_image_dates = np.array(list(map(dateutil.parser.parse, subimg_df['date_captured'])))
        region_image_indexes = np.arange(len(region_image_dates))
        region_gids = subimg_df['image_id'].values

        if geospace_lookup:
            # Note: this was built for when videos really did correspond to
            # regions in the case where videos correspond to tracks, this might
            # not work as well. To mitigate, we can filter down to overlapping
            # geospatial sites in this region here.
            video_poly = subimg_df.geometry.unary_union
            filtered_region_sites = []
            for site_gdf in region_sites:
                site_poly = site_gdf.geometry.unary_union
                if video_poly.intersects(site_poly):
                    # iou = video_poly.intersection(site_poly).area / video_poly.union(site_poly).area
                    # print('iou = {!r}'.format(iou))
                    filtered_region_sites.append(site_gdf)
            region_sites = filtered_region_sites
            print(f'{region_id=} {video_id=} #filtered(sites)={len(region_sites)}')

        drawable_region_sites = []

        # For each site in this region
        for site_gdf in region_sites:
            site_anns, drawable_summary = propogate_site(
                coco_dset, site_gdf, subimg_df, propogate_strategy,
                region_image_dates, region_image_indexes, region_gids,
                status_to_color, want_viz, status_to_catname=status_to_catname)
            propogated_annotations.extend(site_anns)
            if want_viz:
                drawable_region_sites.append(drawable_summary)

        if want_viz:
            drawable_region_sites = sorted(
                drawable_region_sites,
                key=lambda drawable_summary: (
                    min([r['site_row_datetime'] for r in drawable_summary]).timestamp()
                    if len(drawable_summary) else
                    float('inf')
                )
            )

        all_drawable_infos.append({
            'drawable_region_sites': drawable_region_sites,
            'region_id': region_id,
            'region_image_dates': region_image_dates,
        })

    return propogated_annotations, all_drawable_infos


def propogate_site(coco_dset, site_gdf, subimg_df, propogate_strategy,
                   region_image_dates, region_image_indexes, region_gids,
                   status_to_color, want_viz, status_to_catname):
    """
    Given a set of site observations determines how to propogate them onto
    potential images in the assigned region.
    """
    from watch.utils import util_gis
    from watch import heuristics

    if __debug__ and 0:
        # Sanity check, the sites should have spatial overlap with each image in the video
        image_overlaps = util_gis.geopandas_pairwise_overlaps(site_gdf, subimg_df)
        num_unique_overlap_frames = set(ub.map_vals(len, image_overlaps).values())
        assert len(num_unique_overlap_frames) == 1

    site_summary_row = site_gdf.iloc[0]
    site_rows = site_gdf.iloc[1:]
    track_id = site_summary_row['site_id']
    status = site_summary_row['status']
    status = status.lower().strip()

    if status == 'pending':
        # hack for QFabric
        status = 'positive_pending'

    start_date = coerce_datetime2(site_summary_row['start_date'])
    end_date = coerce_datetime2(site_summary_row['end_date'])

    FIX_BACKWARDS_DATES = True
    if FIX_BACKWARDS_DATES:
        # Some sites have backwards dates. Unfortunately we don't
        # have any control to fix them, so we have to handle them.
        if start_date is not None and end_date is not None:
            if start_date > end_date:
                warnings.warn(
                    'A site has flipped start/end dates. '
                    'Fixing here, but it should be fixed in the site model itself.')
                start_date, end_date = end_date, start_date

    flags = ~site_rows['observation_date'].isnull()
    valid_site_rows = site_rows[flags]

    observation_dates = np.array([
        coerce_datetime2(x) for x in valid_site_rows['observation_date']
    ])

    # This check doesn't seem generally necessary.
    if start_date is not None and observation_dates[0] != start_date:
        print('WARNING: inconsistent start')
        print(site_gdf)
        print(f'start_date = {start_date}')
        print(f'end_date   = {end_date}')
        print('observation_dates = {}'.format(ub.repr2(observation_dates.tolist(), nl=1)))
    if end_date is not None and observation_dates[-1] != end_date:
        print('WARNING: inconsistent end date')
        print(site_gdf)
        print(f'start_date = {start_date}')
        print(f'end_date   = {end_date}')
        print('observation_dates = {}'.format(ub.repr2(observation_dates.tolist(), nl=1)))

    # Assuming observations are sorted by date
    assert all([d.total_seconds() >= 0 for d in np.diff(observation_dates)])

    # Determine the first image each site-observation will be
    # associated with and then propogate them forward as necessary.

    # NOTE: https://github.com/Erotemic/misc/blob/main/learn/viz_searchsorted.py if you
    # need to remember or explain how searchsorted works

    site_rows_list = site_rows.to_dict(orient='records')

    PROJECT_ENDSTATE = True
    BACKPROJECT_START_STATES = 0  # turn off back-projection

    if propogate_strategy == 'SMART':
        # TODO: This should be replaced with NEW-SMART
        # The original, but inflexible, propogate strategy

        # To future-propogate:
        # (1) assign each observation to its nearest image (temporally)
        # without "going over" (i.e. the assigned image must be at or after
        # the observation)
        # (2) Splitting the image observations and taking all but the first
        # gives all current-and-future images for each observation that
        # happen before the next observation.
        try:
            found_forward_idxs = np.searchsorted(region_image_dates, observation_dates, 'left')
        except TypeError:
            # handle  can't compare offset-naive and offset-aware datetimes
            region_image_dates = [util_time.ensure_timezone(dt)
                                  for dt in region_image_dates]
            observation_dates = [util_time.ensure_timezone(dt)
                                 for dt in observation_dates]
            found_forward_idxs = np.searchsorted(region_image_dates, observation_dates, 'left')

        image_index_bins = np.split(region_image_indexes, found_forward_idxs)
        forward_image_idxs_per_observation = image_index_bins[1:]

        # To past-propogate:
        # (1) assign each observation to its nearest image (temporally)
        # without "going under" (i.e. the assigned image must be at or
        # before the observation)
        # (2) Splitting the image observations and taking all but the last
        # gives all current-and-past-only images for each observation that
        # happen after the previous observation.
        # NOTE: we only really need to backward propogate the first label
        # (if we even want to do that at all)
        found_backward_idxs = np.searchsorted(region_image_dates, observation_dates, 'right')
        backward_image_idxs_per_observation = np.split(region_image_indexes, found_backward_idxs)[:-1]

        # Convert this into the "new" simpler format where each site row is
        # simply associated with the frames it will propogate to.
        obs_associated_gxs = []
        _iter = zip(forward_image_idxs_per_observation,
                    backward_image_idxs_per_observation,
                    site_rows_list)
        for annot_idx, (forward_gxs, backward_gxs, site_row) in enumerate(_iter):
            propogate_gxs = []
            catname = site_row['current_phase']
            if catname is None:
                # Based on the status choose a kwcoco category name
                # using the watch heuristics
                catname = status_to_catname[status]
            current_and_forward_gxs = sorted(
                forward_gxs,
                key=lambda gx: util_time.coerce_datetime(coco_dset.imgs[region_gids[gx]]['date_captured']))
            # Always propogate at least to the nearest frame?
            current_gxs = current_and_forward_gxs[0:1]
            propogate_gxs.extend(current_gxs)
            # Determine if this subsite propogates forward and/or backward
            if PROJECT_ENDSTATE or catname not in heuristics.HEURISTIC_END_STATES:
                propogate_gxs.extend(forward_gxs)

            if BACKPROJECT_START_STATES:
                # Only need to backpropogate the first label (and maybe even not that?)
                if annot_idx == 0 and catname in heuristics.HEURISTIC_START_STATES:
                    propogate_gxs.extend(backward_gxs)
            obs_associated_gxs.append(propogate_gxs)

    elif propogate_strategy == "NEW-PAST":
        image_times = [t.timestamp() for t in region_image_dates]
        key_infos = [{'time': t.timestamp(), 'applies': 'past'} for t in observation_dates]
        obs_associated_gxs = keyframe_interpolate(image_times, key_infos)

    elif propogate_strategy == "NEW-SMART":
        raise NotImplementedError(
            'TODO: use the new logic, but with smart heuristics for the behavior. '
            'Need to vet that the following code is equivalent to the above code'
        )

        # For each annotation determine how it propogates in time.
        site_rows_list
        image_times = [t.timestamp() for t in region_image_dates]
        key_infos = []
        for annot_idx, (dt, site_row) in enumerate(zip(observation_dates, site_rows_list)):
            # SMART annotations apply to the future by default.
            applies = 'future'
            # But we may change that based on category
            catname = site_row['current_phase']
            if catname is None:
                catname = status_to_catname[status]
            if not PROJECT_ENDSTATE:
                if catname in heuristics.HEURISTIC_END_STATES:
                    raise NotImplementedError(
                        'need a applies strategy for only the next frame'
                    )
            if BACKPROJECT_START_STATES:
                if annot_idx == 0 and catname in heuristics.HEURISTIC_START_STATES:
                    applies = 'past'

            key_infos.append({
                'time': dt.timestamp(),
                'applies': applies,
            })
        obs_associated_gxs = keyframe_interpolate(image_times, key_infos)
    else:
        raise KeyError(propogate_strategy)

    ###
    # Note:
    # There is a current assumption that an observation implicitly
    # propogates forward in time, but never backwards. This is a
    # project-specific assumption and a more general system would have
    # this be configurable. The current propogate_strategy gives a minor
    # degree of control, but this should be specified at the observation
    # level in the annotation file. This is really the annotation state
    # interpolation problem.

    # Create annotations on each frame we are associated with
    site_anns = []
    drawable_summary = []
    _iter = zip(obs_associated_gxs, site_rows_list)
    for annot_idx, (propogate_gxs, site_row) in enumerate(_iter):

        site_row_datetime = coerce_datetime2(site_row['observation_date'])
        assert site_row_datetime is not None

        catname = site_row['current_phase']

        if isinstance(catname, float) and math.isnan(catname):
            catname = None

        if catname is None:
            # Based on the status choose a kwcoco category name
            # using the watch heuristics
            catname = status_to_catname[status]

        if catname is None:
            HACK_TO_PASS = 1
            if HACK_TO_PASS:
                # We should find out why this is happening
                warnings.warn(f'Positive annotation without a class label: status={status}, {annot_idx}, {site_row}')
                continue
            raise AssertionError(f'status={status}, {annot_idx}, {site_row}')

        propogated_on = []
        category_colors = []
        categories = []

        # Handle multi-category per-row logic
        site_catnames = [c.strip() for c in catname.split(',')]
        row_summary = {
            'track_id': track_id,
            'site_row_datetime': site_row_datetime,
            'propogated_on': propogated_on,
            'category_colors': category_colors,
            'categories': categories,
            'status': status,
            'color': status_to_color[status],
        }

        site_polygons = [
            p.to_geojson()
            for p in kwimage.MultiPolygon.from_shapely(site_row['geometry']).to_multi_polygon().data
        ]
        assert len(site_polygons) == len(site_catnames)

        # Propogate each subsite
        for subsite_catname, poly in zip(site_catnames, site_polygons):
            propogate_gids = region_gids[propogate_gxs]
            for gid in propogate_gids:
                img = coco_dset.imgs[gid]
                img_datetime = util_time.coerce_datetime(img['date_captured'])

                propogated_on.append(img_datetime)

                cid = coco_dset.ensure_category(subsite_catname)
                cat = coco_dset.index.cats[cid]
                category_colors.append(cat.get('color', None))
                categories.append(subsite_catname)
                ann = {
                    'image_id': gid,
                    'segmentation_geos': poly,
                    'status': status,
                    'category_id': cid,
                    'track_id': track_id,
                }
                misc_info = site_row.get('misc_info', {})
                if misc_info:
                    ann['misc_info'] = misc_info
                site_anns.append(ann)

        if want_viz:
            drawable_summary.append(row_summary)

    return site_anns, drawable_summary


def keyframe_interpolate(image_times, key_infos):
    """
    New method for keyframe interapolation.

    Given a set of image times and a set of key frames with information on how
    they propogate, determine which images will be assigned which keyframes.

    Not yet used for the SMART method, but could be in the future.
    The keyframe propogation behavior is also currently very simple and may be
    expanded in the future.

    Args:
        image_times (ndarray):
            contains just the times of the underying images we will
            associate with the keyframes.

        key_infos (Dict):
            contains each keyframe time and information about its behavior.

    Returns:
        List[List]:
            a list of associated image indexes for each key frame.

    Example:
        >>> from watch.cli.reproject_annotations import *  # NOQA
        >>> image_times = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> # TODO: likely also needs a range for a maximum amount of time you will
        >>> # apply the observation for.
        >>> key_infos = [
        >>>     {'time': 1.2, 'applies': 'future'},
        >>>     {'time': 3.4, 'applies': 'both'},
        >>>     {'time': 6, 'applies': 'past'},
        >>>     {'time': 9, 'applies': 'past'},
        >>> ]
        >>> key_assignment = keyframe_interpolate(image_times, key_infos)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(fnum=1, doclf=1)
        >>> key_times = [d['time'] for d in key_infos]
        >>> key_times = np.array(key_times)
        >>> plot_poc_keyframe_interpolate(image_times, key_times, key_assignment)
    """
    key_times = [d['time'] for d in key_infos]
    key_times = np.array(key_times)

    # For each image find the closest keypoint index at the current time or in
    # the future.
    forward_idxs = np.searchsorted(key_times, image_times, 'left')

    # Add a padding which makes the array math easier.
    padded_key_times = np.hstack([[-np.inf], key_times, [np.inf]])
    padded_forward_idxs = forward_idxs + 1

    # Determine if the image is exactly associated with a keyframe or not
    associated_times = padded_key_times[padded_forward_idxs]
    has_exact_keyframe = (associated_times == image_times)

    # For the image that have an exact keyframe, denote that.
    padded_curr_idxs = np.full_like(image_times, fill_value=0, dtype=int)
    padded_curr_idxs[has_exact_keyframe] = padded_forward_idxs[has_exact_keyframe]

    # For each image denote the index of the keyframe before it
    padded_prev_idxs = padded_forward_idxs - 1

    # For each image denote the index of the keyframe after it
    # for locations where the times match exactly add one because
    # it the forward idx represents the current and not the next.
    padded_next_idxs = padded_forward_idxs
    padded_next_idxs[has_exact_keyframe] += 1

    # Set the locations of invalid indices to -1 (+1 for padding)
    padded_next_idxs[padded_next_idxs > len(key_times)] = 0

    # For each image, it will either have:
    # * exactly one keyframe after it and maybe one directly on it.
    # * a keyframe before and after it and maybe one directly on it.
    # * exactly one keyframe before it and maybe one directly on it.
    #
    # We encode this by pointing to the index of the keyframe
    # before the image, on the image, or after the image. If one of these
    # keyframes doesn't exist we use a -1
    prev_idxs = padded_prev_idxs - 1
    curr_idxs = padded_curr_idxs - 1
    next_idxs = padded_next_idxs - 1

    if 0:
        import pandas as pd
        print(pd.DataFrame({
            'image_time': image_times,
            'prev': prev_idxs,
            'curr': curr_idxs,
            'next': next_idxs,
        }))

    # Note that the config of prev, curr, next forms a grouping where each
    # unique row has the same operation applied to it.
    ### --- <opaque group logic>
    import kwarray
    pcn_rows = np.stack([prev_idxs, curr_idxs, next_idxs], axis=1)
    arr = pcn_rows
    dtype_view = np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))
    arr_view = arr.view(dtype_view)
    _, uidx, uinv = np.unique(arr_view, return_inverse=True, return_index=True)
    row_groupids = uidx[uinv]
    unique_rowxs, groupxs = kwarray.group_indices(row_groupids)
    ### --- </opaque group logic>

    keyidx_to_imageidxs = [[] for _ in range(len(key_infos))]
    # Now we have groups of images corresponding to each unique keyframe case.
    for rowx, image_groupx in zip(unique_rowxs, groupxs):
        # In each iteration we have a group of images that all have the same
        # relationship to a set of up to 3 keyframes. We first gather what
        # keyframes these are.
        prev_idx, curr_idx, next_idx = pcn_rows[rowx]

        prev_keyinfo = key_infos[prev_idx] if prev_idx >= 0 else None
        curr_keyinfo = key_infos[curr_idx] if curr_idx >= 0 else None
        next_keyinfo = key_infos[next_idx] if next_idx >= 0 else None

        if curr_keyinfo is not None:
            # The current key always takes priority
            keyidx_to_imageidxs[curr_idx].extend(image_groupx)
        else:
            prev_is_relevant = prev_keyinfo is not None and (
                prev_keyinfo['applies'] in {'future', 'both'})
            next_is_relevant = next_keyinfo is not None and (
                next_keyinfo['applies'] in {'past', 'both'})

            if prev_is_relevant and next_is_relevant:
                # This is a conflicting case, and we need to guess which one to
                # use, use the nearest in time.
                #
                # TODO: If a keyframe has a duration property filter to the set
                # of images in this group that it applies to.
                # key_duration = next_keyinfo.get('duration', float('inf'))
                group_times = image_times[image_groupx]
                d1 = np.abs(group_times - prev_keyinfo['time'])
                d2 = np.abs(group_times - next_keyinfo['time'])
                partitioner = d1 < d2
                prev_groupx = image_groupx[partitioner]
                next_groupx = image_groupx[~partitioner]
                keyidx_to_imageidxs[prev_idx].extend(prev_groupx)
                keyidx_to_imageidxs[next_idx].extend(next_groupx)
            elif next_is_relevant:
                # Simple case, only have a relevant next keyframe
                #
                # TODO: If a keyframe has a duration property filter to the set
                # of images in this group that it applies to.
                # key_duration = next_keyinfo.get('duration', float('inf'))
                # group_times = image_times[image_groupx]
                keyidx_to_imageidxs[next_idx].extend(image_groupx)
            elif prev_is_relevant:
                # Simple case, only have a relevant prev keyframe
                #
                # TODO: If a keyframe has a duration property filter to the set
                # of images in this group that it applies to.
                # key_duration = next_keyinfo.get('duration', float('inf'))
                # group_times = image_times[image_groupx]
                keyidx_to_imageidxs[prev_idx].extend(image_groupx)
            else:
                # It is ok if neither keyframe is relevant that is a hole in
                # the track.
                ...
    return keyidx_to_imageidxs


def plot_poc_keyframe_interpolate(image_times, key_times, key_assignment):
    """
    Helper to visualize the keyframe interpolation algorithm.
    """
    import kwplot
    import matplotlib as mpl
    plt = kwplot.autoplt()
    ax = plt.gca()

    ylocs = {
        'image': 2,
        'key': 1,
    }
    xlocs = {
        'image': image_times,
        'key': key_times,
    }
    segments = {
        key: [(x, ylocs[key]) for x in xs]
        for key, xs in xlocs.items()
    }
    key1 = 'image'
    key2 = 'key'

    colors = {
        key1: 'darkblue',
        key2: 'orange',
        'association': 'purple',
    }
    xlocs1 = xlocs[key1]
    # xlocs2 = xlocs[key2]
    segment1 = segments[key1]
    segment2 = segments[key2]

    association_segments = []
    for idx2, idx1s in enumerate(key_assignment):
        pt2 = segment2[idx2]
        for idx1 in idx1s:
            if idx1 < 0:
                pt1 = [xlocs1[0] - 1, ylocs[key1]]
            elif idx1 == len(segments[key1]):
                pt1 = [xlocs1[-1] + 1, ylocs[key1]]
            else:
                pt1 = segment1[idx1]
            association_segments.append([pt1, pt2])

    circles = [mpl.patches.Circle(xy, radius=0.1) for xy in segment1]
    data_points = mpl.collections.PatchCollection(circles, color=colors[key1], alpha=0.7)
    data_lines = mpl.collections.LineCollection([segment1], color=colors[key1], alpha=0.5)
    ax.add_collection(data_lines)
    ax.add_collection(data_points)

    circles = [mpl.patches.Circle(xy, radius=0.1) for xy in segment2]
    data_points = mpl.collections.PatchCollection(circles, color=colors[key2], alpha=0.7)
    data_lines = mpl.collections.LineCollection([segment1], color=colors[key2], alpha=0.5)
    ax.add_collection(data_lines)
    ax.add_collection(data_points)

    found_association_lines = mpl.collections.LineCollection(association_segments, color=colors['association'], alpha=0.5)
    ax.add_collection(found_association_lines)

    ax.autoscale_view()
    ax.set_aspect('equal')

    kwplot.phantom_legend(colors)

    ax.set_ylim(min(ylocs.values()) - 1, max(ylocs.values()) + 1)


def coerce_datetime2(data):
    """ Is this a monad 🦋 ? """
    return None if data is None or (isinstance(data, float) and math.isnan(data)) else util_time.coerce_datetime(data)


def plot_image_and_site_times(coco_dset, region_image_dates, drawable_region_sites, region_id, ax=None):
    """
    References:
        .. [HandleDates] https://stackoverflow.com/questions/44642966/how-to-plot-multi-color-line-if-x-axis-is-date-time-index-of-pandas
    """
    import kwplot
    if ax is None:
        plt = kwplot.autoplt()
        ax = plt.gca()

    ax.cla()

    from watch import heuristics
    import matplotlib as mpl
    hueristic_status_data = heuristics.HUERISTIC_STATUS_DATA

    status_to_color = {d['name']: kwimage.Color(d['color']).as01()
                       for d in hueristic_status_data}
    # region_status_labels = {site_gdf.iloc[0]['status'] for site_gdf in region_sites}
    # for status in region_status_labels:
    #     if status not in status_to_color:
    #         hueristic_status_data.append({
    #             'name': status,
    #             'color': kwimage.Color.random().as01(),
    #         })
    # status_to_color = {d['name']: kwimage.Color(d['color']).as01()
    #                    for d in hueristic_status_data}

    bounds_segments = []
    for t in ub.ProgIter(region_image_dates, desc='plot bounds'):
        y2 = len(drawable_region_sites) + 1
        x1 = mpl.dates.date2num(t)
        xy1 = (x1, 0)
        xy2 = (x1, y2)
        segment = [xy1, xy2]
        bounds_segments.append(segment)
        # ax.plot([t, t], [0, y2], color='darkblue', alpha=0.5)
    line_group = mpl.collections.LineCollection(
        bounds_segments, color='darkblue', alpha=0.5)
    ax.add_collection(line_group)

    if 1:
        ax.autoscale_view()
        ax.xaxis_date()

    all_times = []

    propogate_attrs = {
        'segments': [],
        'colors': [],
    }
    for summary_idx, drawable_summary in enumerate(ub.ProgIter(drawable_region_sites, desc='plot region')):
        site_dates = [r['site_row_datetime'] for r in drawable_summary]
        all_times.extend(site_dates)
        yloc = summary_idx
        status_color = drawable_summary[0]['color']

        # Draw propogations
        for row in drawable_summary:
            t1 = row['site_row_datetime']
            cat_colors = row['category_colors']
            yoffsets = np.linspace(0.5, 0.75, len(cat_colors))[::-1]

            # Draw a line for each "part" of the side at this timestep
            # Note: some sites seem to have a ton of parts that could be
            # consolodated? Is this real or is there a bug?
            for yoff, color in zip(yoffsets, cat_colors):
                for tp in row['propogated_on']:
                    x1 = mpl.dates.date2num(t1)
                    x2 = mpl.dates.date2num(tp)
                    y1 = yloc
                    y2 = yloc + yoff
                    segment = [(x1, y1), (x2, y2)]
                    propogate_attrs['segments'].append(segment)
                    propogate_attrs['colors'].append(color)
                    # ax.plot([x1, x2], [y1, y2], '-', color=color)

        # Draw site keyframes
        ax.plot(site_dates, [yloc] * len(site_dates), '-o', color=status_color, alpha=0.5)

    propogate_group = mpl.collections.LineCollection(
        propogate_attrs['segments'],
        color=propogate_attrs['colors'],
        alpha=0.5)
    ax.add_collection(propogate_group)

    propogate_attrs['segments']
    propogate_attrs['colors']

    ax.set_xlim(min(all_times), max(all_times))
    ax.set_ylim(0, len(drawable_region_sites))

    cat_to_color = {cat['name']: cat['color'] for cat in coco_dset.cats.values()}

    util_kwplot.phantom_legend(status_to_color, ax=ax, legend_id=1, loc=0)
    util_kwplot.phantom_legend(cat_to_color, ax=ax, legend_id=3, loc=3)

    ax.set_xlabel('Time')
    ax.set_ylabel('Site Index')
    ax.set_title('Site & Image Timeline: ' + region_id)
    return ax


def draw_geospace(dvc_dpath, sites):
    from watch.utils import util_gis
    import geopandas as gpd
    import kwplot
    kwplot.autompl()
    region_fpaths = util_gis.coerce_geojson_paths(dvc_dpath / 'drop1/region_models')
    regions = []
    for info in util_gis.coerce_geojson_datas(region_fpaths):
        gdf = info['data']
        regions.append(gdf)

    wld_map_gdf = gpd.read_file(
        gpd.datasets.get_path('naturalearth_lowres')
    )
    ax = wld_map_gdf.plot()

    for gdf in regions:
        centroid = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
        centroid.plot(ax=ax, marker='o', facecolor='orange', alpha=0.5)
        gdf.plot(ax=ax, facecolor='none', edgecolor='orange', alpha=0.5)

    for gdf in sites:
        centroid = gdf.to_crs('+proj=cea').centroid.to_crs(gdf.crs)
        centroid.plot(ax=ax, marker='o', facecolor='red', alpha=0.5)
        gdf.plot(ax=ax, facecolor='none', edgecolor='red', alpha=0.5)


_SubConfig = ReprojectAnnotationsConfig


def embed_if_requested(n=0):
    """
    Calls xdev.embed conditionally based on the environment.

    Useful in cases where you want to leave the embed call around, but you dont
    want it to trigger in normal circumstances.

    Specifically, embed is only called if the environment variable XDEV_EMBED
    exists or if --xdev-embed is in sys.argv.
    """
    import os
    import ubelt as ub
    import xdev
    if os.environ.get('XDEV_EMBED', '') or ub.argflag('--xdev-embed'):
        xdev.embed(n=n + 1)


def reorder_columns(df, columns):
    remain = df.columns.difference(columns)
    return df.reindex(columns=(columns + list(remain)))


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/reproject_annotations.py
    """
    main(cmdline=True)
