#!/usr/bin/env python
r"""
Assigns geospace annotation to image pixels and frames

CommandLine:
    # Update a dataset with new annotations
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc

    # You dont need to run this, but if you plan on running the project
    # annotations script multiple times, preloading this work will make it
    # faster
    python -m watch add_fields \
        --src $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json \
        --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json \
        --overwrite=warp --workers 10

    # Update to whatever the state of the annotations submodule is
    DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
    python -m watch project_annotations \
        --src $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
        --dst $DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
        --viz_dpath $DVC_DPATH/Drop2-Aligned-TA1-2022-02/_viz_propogate \
        --site_models="$DVC_DPATH/annotations/site_models/*.geojson"

    python -m watch visualize \
        --src $DVC_DPATH/Drop2-Aligned-TA1-2022-01/data.kwcoco.json \
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
from watch.utils import kwcoco_extensions
from watch.utils import util_kwplot
from watch.utils import util_path
from watch.utils import util_time


class ProjectAnnotationsConfig(scfg.Config):
    """
    Projects annotations from geospace onto a kwcoco dataset and optionally
    propogates them in

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

        'clear_existing': scfg.Value(True, help=ub.paragraph(
            '''
            if True, clears existing annotations before projecting the new ones.
            ''')),

        'propogate': scfg.Value(True, help='if True does forward propogation in time'),

        'geo_preprop': scfg.Value('auto', help='force if we check geo properties or not'),

        'geospace_lookup': scfg.Value('auto', help='if False assumes region-ids can be used to lookup association'),

        'workers': scfg.Value(0, help='number of workers for geo-preprop if done'),
    }


def main(cmdline=False, **kwargs):
    """
    CommandLine:
        python -m watch.cli.project_annotations \
            --src

    Ignore:
        >>> # xdoctest: +REQUIRES(env:DVC_DPATH)
        >>> from watch.cli.project_annotations import *  # NOQA
        >>> from watch.utils import util_data
        >>> import tempfile
        >>> dvc_dpath = util_data.find_smart_dvc_dpath()
        >>> #kwcoco_fpath = dvc_dpath / 'Drop1-Aligned-L1-2022-01/data.kwcoco.json'
        >>> #kwcoco_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
        >>> kwcoco_fpath = dvc_dpath / 'Aligned-Drop2-TA1-2022-03-07/data.kwcoco.json'
        >>> dpath = ub.Path(ub.ensure_app_cache_dir('watch/tests/project_annots'))
        >>> cmdline = False
        >>> output_fpath = dpath / 'data.kwcoco.json'
        >>> viz_dpath = (dpath / 'viz').ensuredir()
        >>> kwargs = {
        >>>     'src': kwcoco_fpath,
        >>>     'dst': output_fpath,
        >>>     'viz_dpath': viz_dpath,
        >>>     'site_models': dvc_dpath / 'annotations/site_models',
        >>>     'region_models': dvc_dpath / 'annotations/region_models',
        >>> }
        >>> main(**kwargs)
    """
    import geopandas as gpd  # NOQA
    from watch.utils import util_gis
    config = ProjectAnnotationsConfig(default=kwargs, cmdline=cmdline)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))

    output_fpath = config['dst']
    if output_fpath is None:
        raise AssertionError

    # Load the coco dataset with all of the images
    coco_dset = kwcoco.CocoDataset.coerce(config['src'])
    site_geojson_fpaths = util_path.coerce_patterned_paths(config['site_models'], '.geojson')

    geo_preprop = config['geo_preprop']
    geospace_lookup = config['geospace_lookup']
    if geo_preprop == 'auto':
        if len(coco_dset.index.imgs):
            coco_img = coco_dset.coco_image(ub.peek(coco_dset.index.imgs.keys()))
            geo_preprop = not any('geos_corners' in obj for obj in coco_img.iter_asset_objs())
            print('auto-choose geo_preprop = {!r}'.format(geo_preprop))

    if geo_preprop:
        kwcoco_extensions.coco_populate_geo_heuristics(
            coco_dset, overwrite={'warp'}, workers=config['workers'],
            keep_geotiff_metadata=False,
        )

    # Read the external CRS84 annotations from the site models
    sites = []

    HACK_HANDLE_DUPLICATE_SITE_ROWS = True

    for fpath in ub.ProgIter(site_geojson_fpaths, desc='load geojson site-models'):
        gdf = util_gis.read_geojson(fpath)
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
        region_geojson_fpaths = util_path.coerce_patterned_paths(config['region_models'], '.geojson')
        for fpath in ub.ProgIter(region_geojson_fpaths, desc='load geojson region-models'):

            if fpath.stem == 'IN_C000':
                # HACK: Remove when shi region is fixed.
                continue

            gdf = util_gis.read_geojson(fpath)
            regions.append(gdf)

            # if 1:
            #     region_rows = gdf[gdf['type'] == 'region']
            #     assert len(region_rows) == 1
            #     region_row = region_rows.iloc[0]
            #     if region_row['region_id'] != fpath.stem:
            #         print(gdf)
            #         print(region_row['region_id'])
            #         print('fpath = {!r}'.format(fpath))
            #         raise AssertionError

    if config['clear_existing']:
        coco_dset.clear_annotations()

    region_id_to_sites = expand_site_models_with_site_summaries(sites, regions)

    propogate = config['propogate']

    viz_dpath = config['viz_dpath']
    want_viz = bool(viz_dpath)

    propogated_annotations, all_drawable_infos = assign_sites_to_images(
        coco_dset, region_id_to_sites, propogate,
        geospace_lookup=geospace_lookup, want_viz=want_viz)

    for ann in propogated_annotations:
        coco_dset.add_annotation(**ann)
    kwcoco_extensions.warp_annot_segmentations_from_geos(coco_dset)

    coco_dset.fpath = output_fpath
    print('dump coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    coco_dset.dump(coco_dset.fpath)

    if viz_dpath == 'auto':
        viz_dpath = (ub.Path(coco_dset.fpath).parent / '_viz_project_anns')
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


def expand_site_models_with_site_summaries(sites, regions):
    """
    Takes all site summaries from region models that do not have a
    corresponding site model and makes a "pseudo-site-model" for use in BAS.

    Returns:
        Dict[str, List[DataFrame]] : region_id_to_sites  :
            a mapping from region names to a list of site models both
            real and/or pseudo.
    """
    import pandas as pd
    import geojson
    # import watch
    import json
    from watch.utils import util_gis

    dummy_start_date = '1970-01-01'
    dummy_end_date = '2101-01-01'

    if __debug__:
        # Check assumptions about site models
        for site_df in ub.ProgIter(sites, desc='checking site assumptions'):
            first = site_df.iloc[0]
            rest = site_df.iloc[1:]
            import xdev
            with xdev.embed_on_exception_context:
                assert first['type'] == 'site', (
                    f'first row must have type of site, got {first["type"]}')
                assert first['region_id'] is not None, (
                    f'first row must have a region id. Got {first["region_id"]}')
                assert rest['type'].apply(lambda x: x == 'observation').all(), (
                    f'rest of row must have type observation. Instead got: {rest["type"].unique()}')
                assert rest['region_id'].apply(lambda x: x is None).all(), (
                    f'rest of row must have region_id=None. Instead got {rest["region_id"].unique()}')

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
        assert (sites_part['type'] == 'site_summary').all(), 'rest of data must be site summaries'
        assert sites_part['region_id'].apply(lambda x: (x is None) or x == region_id).all(), (
            'site-summaries do not have region ids (unless we make them)')
        region_id_to_site_summaries[region_id] = sites_part

        # Check datetime errors
        sitesum_start_dates = sites_part['start_date'].apply(lambda x: util_time.coerce_datetime(x or dummy_start_date))
        sitesum_end_dates = sites_part['end_date'].apply(lambda x: util_time.coerce_datetime(x or dummy_end_date))
        has_bad_time_range = sitesum_start_dates > sitesum_end_dates
        bad_dates = sites_part[has_bad_time_range]
        if len(bad_dates):
            print('BAD DATES')
            print(bad_dates)

    region_id_to_sites = ub.group_items(sites, lambda x: x.iloc[0]['region_id'])

    region_id_to_num_sitesumms = ub.map_vals(len, region_id_to_site_summaries)
    region_id_to_num_sites = ub.map_vals(len, region_id_to_sites)
    print('region_id_to_num_sitesumms = {}'.format(ub.repr2(region_id_to_num_sitesumms, nl=1, sort=0)))
    print('region_id_to_num_sites = {}'.format(ub.repr2(region_id_to_num_sites, nl=1, sort=0)))

    if 1:
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
            assert len(set(site_df2['site_id'])) == len(site_df2), 'site ids must be unique'
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

            def reorder_columns(df, columns):
                remain = df.columns.difference(columns)
                return df.reindex(columns=(columns + list(remain)))
            error1 = reorder_columns(error1, columns)
            error2 = reorder_columns(error2, columns)
            print('Disagree rows for site models')
            print(error1.drop(['type', 'region_id', 'misc_info'], axis=1))
            print('Disagree rows for region models')
            print(error2.drop(['type', 'region_id', 'validate'], axis=1))

        # Find sites that only have a site-summary
        summary_only_site_ids = sorted(set(site_df2['site_id']) - set(site_df1['site_id']))
        region_id_to_only_site_summaries = dict(list(site_df2.loc[summary_only_site_ids].groupby('region_id')))

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

        # observation_properties = [
        #     'type', 'observation_date', 'source', 'sensor_name',
        #     'current_phase', 'is_occluded', 'is_site_boundary', 'score',
        #     'misc_info'
        # ]
        site_properites = [
            'type', 'version', 'mgrs', 'status', 'model_content', 'start_date',
            'end_date', 'originator', 'score', 'validated', 'misc_info',
            'region_id', 'site_id']

        # resolver = jsonschema.RefResolver.from_schema(site_model_schema)
        # site_model_schema[

        region_id_to_num_sites = ub.map_vals(len, region_id_to_sites)
        print('BEFORE region_id_to_num_sites = {}'.format(ub.repr2(region_id_to_num_sites, nl=1)))

        for region_id, sitesummaries in region_id_to_only_site_summaries.items():
            region_row = region_id_region_row[region_id]

            # Use region start/end date if the site does not have them
            region_start_date = region_row['start_date'] or dummy_start_date
            region_end_date = region_row['end_date'] or dummy_end_date

            region_start_date, region_end_date = sorted(
                [region_start_date, region_end_date], key=util_time.coerce_datetime)

            for _, site_summary in sitesummaries.iterrows():
                geom = site_summary['geometry']

                poly_json = kwimage.Polygon.from_shapely(geom.convex_hull).to_geojson()
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

                observation_prop_template = {
                    'type': 'observation',
                    'observation_date': None,
                    # 'source': None,
                    # 'sensor_name': None,
                    # 'current_phase': None,
                    # 'is_occluded': None,
                    # 'is_site_boundary': None,
                    'score': float(site_summary['score']),
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
                pseudo_gpd = util_gis.read_geojson(io.StringIO(json.dumps(psudo_site_model)))
                region_id_to_sites[region_id].append(pseudo_gpd)
                # if 1:
                #     from watch.rc.registry import load_site_model_schema
                #     site_model_schema = load_site_model_schema()
                #     real_site_model = json.loads(ub.Path('/media/joncrall/flash1/smart_watch_dvc/annotations/site_models/BR_R002_0009.geojson').read_text())
                #     ret = jsonschema.validate(real_site_model, schema=site_model_schema)
                #     import jsonschema
                #     ret = jsonschema.validate(psudo_site_model, schema=site_model_schema)

        region_id_to_num_sites = ub.map_vals(len, region_id_to_sites)
        print('AFTER (sitesummary) region_id_to_num_sites = {}'.format(ub.repr2(region_id_to_num_sites, nl=1)))

    if 0:
        site_high_level_summaries = []
        for region_id, region_sites in region_id_to_sites.items():
            print('=== {} ==='.format(region_id))
            for site_gdf in region_sites:
                site_summary_row = site_gdf.iloc[0]
                site_rows = site_gdf.iloc[1:]
                track_id = site_summary_row['site_id']
                status = site_summary_row['status']
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
                # import xdev
                # with xdev.embed_on_exception_context:
                validate_site_dataframe(site_df)

    return region_id_to_sites


def validate_site_dataframe(site_df):
    from dateutil.parser import parse
    import numpy as np
    dummy_start_date = '1970-01-01'  # hack, could be more robust here
    dummy_end_date = '2101-01-01'
    first = site_df.iloc[0]
    rest = site_df.iloc[1:]
    assert first['type'] == 'site', 'first row must have type of site'
    assert first['region_id'] is not None, 'first row must have a region id'
    assert rest['type'].apply(lambda x: x == 'observation').all(), (
        'rest of row must have type observation')
    assert rest['region_id'].apply(lambda x: x is None).all(), (
        'rest of row must have region_id=None')

    site_start_date = first['start_date'] or dummy_start_date
    site_end_date = first['end_date'] or dummy_end_date
    site_start_datetime = parse(site_start_date)
    site_end_datetime = parse(site_end_date)

    if site_end_datetime < site_start_datetime:
        print('\n\nBAD SITE DATES:')
        print(first)

    status = {}
    # Check datetime errors in observations
    try:
        obs_dates = [None if x is None else parse(x) for x in rest['observation_date']]
        obs_isvalid = [x is not None for x in obs_dates]
        valid_obs_dates = list(ub.compress(obs_dates, obs_isvalid))
        if not all(valid_obs_dates):
            # null_obs_sites.append(first[['site_id', 'status']].to_dict())
            pass
        valid_deltas = np.array([d.total_seconds() for d in np.diff(valid_obs_dates)])
        assert (valid_deltas >= 0).all(), 'observations must be sorted temporally'
    except AssertionError as ex:
        print('ex = {!r}'.format(ex))
        print(site_df)
        raise

    return status


def assign_sites_to_images(coco_dset, region_id_to_sites, propogate, geospace_lookup='auto', want_viz=1):
    """
    Given a coco dataset (with geo information) and a list of geojson sites,
    determines which images each site-annotations should go on.
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
    videos_gdf = pd.concat(video_gdfs, ignore_index=True)

    PROJECT_ENDSTATE = True

    # Ensure colors and categories
    from watch import heuristics
    status_to_color = {d['name']: kwimage.Color(d['color']).as01()
                       for d in heuristics.HUERISTIC_STATUS_DATA}
    print('coco_dset categories = {}'.format(ub.repr2(coco_dset.dataset['categories'], nl=2)))
    for cat in heuristics.CATEGORIES:
        coco_dset.ensure_category(**cat)
    # hack in heuristic colors
    heuristics.ensure_heuristic_coco_colors(coco_dset)
    # handle any other colors
    kwcoco_extensions.category_category_colors(coco_dset)
    print('coco_dset categories = {}'.format(ub.repr2(coco_dset.dataset['categories'], nl=2)))

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
                # import xdev
                # with xdev.embed_on_exception_context:
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
                print('No region-id match for region_id={}'.format(region_id))
                continue
            video_id = video['id']
            video_id_to_region_id[video_id] = region_id

    def coerce_datetime2(data):
        """ Is this a monad 🦋 ? """
        return None if data is None else util_time.coerce_datetime(data)

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
            if __debug__ and 0:
                # Sanity check, the sites should have spatial overlap with each image in the video
                image_overlaps = util_gis.geopandas_pairwise_overlaps(site_gdf, subimg_df)
                # import xdev
                # with xdev.embed_on_exception_context:
                num_unique_overlap_frames = set(ub.map_vals(len, image_overlaps).values())
                assert len(num_unique_overlap_frames) == 1

            site_summary_row = site_gdf.iloc[0]
            site_rows = site_gdf.iloc[1:]
            track_id = site_summary_row['site_id']
            status = site_summary_row['status']

            start_date = coerce_datetime2(site_summary_row['start_date'])
            end_date = coerce_datetime2(site_summary_row['end_date'])

            ALLOW_BACKWARDS_DATES = True
            if ALLOW_BACKWARDS_DATES:
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

            if start_date is not None and observation_dates[0] != start_date:
                raise AssertionError(f'start_date={start_date}, obs[0]={observation_dates[0]}')
            if end_date is not None and observation_dates[-1] != end_date:
                raise AssertionError(f'end_date={end_date}, obs[-1]={observation_dates[-1]}')

            # Assuming observations are sorted by date
            assert all([d.total_seconds() >= 0 for d in np.diff(observation_dates)])

            # Determine the first image each site-observation will be
            # associated with and then propogate them forward as necessary.

            # NOTE: github.com/Erotemic/misc/learn/viz_searchsorted.py if you
            # need to remember or explain how searchsorted works

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

            # TODO: use heuristic module
            HEURISTIC_END_STATES = {
                'Post Construction'
            }

            BACKPROJECT_START_STATES = 0  # turn off back-projection
            HEURISTIC_START_STATES = {
                'No Activity',
            }

            # Create annotations on each frame we are associated with
            site_anns = []
            drawable_summary = []
            _iter = zip(forward_image_idxs_per_observation,
                        backward_image_idxs_per_observation,
                        site_rows.to_dict(orient='records'))
            for annot_idx, (forward_gxs, backward_gxs, site_row) in enumerate(_iter):

                site_row_datetime = coerce_datetime2(site_row['observation_date'])
                assert site_row_datetime is not None

                catname = site_row['current_phase']
                if catname is None:
                    # Based on the status choose a kwcoco category name
                    # using the watch heuristics
                    catname = heuristics.PHASE_STATUS_TO_KWCOCO_CATNAME[status]

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

                # A bit hacky, clean up logic later
                current_and_forward_gids = region_gids[forward_gxs]
                backward_gids = region_gids[backward_gxs]
                forward_gids = []
                current_gids = []
                current_and_forward_gids = sorted(
                    current_and_forward_gids,
                    key=lambda gid: util_time.coerce_datetime(coco_dset.imgs[gid]['date_captured']))

                # Always propogate at least to the nearest frame?
                # TODO: could have better rules about what counts as a frame
                # the annotation "belongs" to and what counts as a forward
                # propogation frame.
                current_gids = current_and_forward_gids[0:1]
                forward_gids = current_and_forward_gids[1:None]

                # Propogate each subsite
                for subsite_catname, poly in zip(site_catnames, site_polygons):

                    # Determine if this subsite propogates forward and/or backward
                    propogate_gids = []
                    propogate_gids.extend(current_gids)
                    if PROJECT_ENDSTATE or catname not in HEURISTIC_END_STATES:
                        propogate_gids.extend(forward_gids)

                    if BACKPROJECT_START_STATES:
                        # Only need to backpropogate the first label (and maybe even not that?)
                        if annot_idx == 0 and catname in HEURISTIC_START_STATES:
                            propogate_gids.extend(backward_gids)

                    for gid in propogate_gids:
                        img = coco_dset.imgs[gid]
                        img_datetime = util_time.coerce_datetime(img['date_captured'])

                        propogated_on.append(img_datetime)

                        cid = coco_dset.ensure_category(subsite_catname)
                        cat = coco_dset.index.cats[cid]
                        category_colors.append(cat['color'])
                        categories.append(subsite_catname)
                        ann = {
                            'image_id': gid,
                            'segmentation_geos': poly,
                            'status': status,
                            'category_id': cid,
                            'track_id': track_id,
                        }
                        site_anns.append(ann)

                if want_viz:
                    drawable_summary.append(row_summary)

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

    from watch.tasks.fusion import heuristics
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
    region_fpaths = util_path.coerce_patterned_paths(dvc_dpath / 'drop1/region_models', '.geojson')
    regions = []
    for fpath in ub.ProgIter(region_fpaths, desc='load geojson annots'):
        gdf = util_gis.read_geojson(fpath)
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


_SubConfig = ProjectAnnotationsConfig


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/project_annotations.py
    """
    main(cmdline=True)
