r"""
Demodata for a simple region and site model

See Also:
    ../cli/stac_search.py
    ../stac/stac_search_builder.py


CommandLine:

    ###
    # Create the demo dataset
    ###

    # Create a demo region file
    xdoctest -m geowatch.demo.demo_region demo_khq_region_fpath

    DATASET_SUFFIX=DemoKHQ-2022-09-19-V7
    DEMO_DPATH=$HOME/.cache/geowatch/demo/datasets
    REGION_FPATH="$HOME/.cache/geowatch/demo/annotations/KHQ_R001.geojson"
    SITE_GLOBSTR="$HOME/.cache/geowatch/demo/annotations/KHQ_R001_sites/*.geojson"
    REGION_ID=$(jq -r '.features[] | select(.properties.type=="region") | .properties.region_id' "$REGION_FPATH")
    RESULT_FPATH=$DEMO_DPATH/all_sensors_kit/${REGION_ID}.input

    mkdir -p "$DEMO_DPATH"

    # Define SMART_STAC_API_KEY
    source "$HOME"/code/watch/secrets/secrets

    # Delete this to prevent duplicates
    rm -f "$RESULT_FPATH"

    # Construct the TA2-ready dataset
    python -m geowatch.cli.prepare_ta2_dataset \
        --dataset_suffix=$DATASET_SUFFIX \
        --cloud_cover=100 \
        --stac_query_mode=auto \
        --sensors "L2-L8" \
        --api_key=env:SMART_STAC_API_KEY \
        --collated False \
        --requester_pays=True \
        --dvc_dpath="$DEMO_DPATH" \
        --aws_profile=iarpa \
        --region_globstr="$REGION_FPATH" \
        --site_globstr="$SITE_GLOBSTR" \
        --fields_workers=8 \
        --convert_workers=8 \
        --align_workers=26 \
        --cache=0 \
        --ignore_duplicates=1 \
        --separate_region_queues=1 \
        --separate_align_jobs=1 \
        --target_gsd=30 \
        --visualize=True \
        --serial=True --run=1

    # Package up for release on IPFS
    DATASET_DPATH=$DEMO_DPATH/Aligned-$DATASET_SUFFIX

    rm $DATASET_DPATH/img*kwcoco.json
    rm -rf $DATASET_DPATH/_viz512
    rm -rf $DATASET_DPATH/_cache

    7z a $DATASET_DPATH.zip  $DATASET_DPATH

    # Pin the data to IPFS
    DATASET_CID=$(ipfs add -Q -w $DATASET_DPATH.zip --cid-version=1 -s size-1048576)
    echo "On Remote machines run: "
    echo "ipfs pin add $DATASET_CID"

    # Look at the contents of the underlying folder to build scripts.
    echo "DATASET_CID = $DATASET_CID"
    echo "DATASET_SUFFIX=$DATASET_SUFFIX"
    ipfs ls "$DATASET_CID"


    # Pin on a remote service
    ipfs pin remote add --service=web3.storage.erotemic --name="$DATASET_SUFFIX" $DATASET_CID --background
    ipfs pin remote ls --service=web3.storage.erotemic --cid=$DATASET_CID --status=queued,pinning,pinned,failed

    DATASET_CID=bafybeigdkhphpa3n3rdv33w7g6tukmprdnch7g4bp4hc6ebmcr76y6yhwu
    ipfs pin remote ls --service=web3.storage.erotemic --cid=$DATASET_CID --status=queued,pinning,pinned,failed

"""


def demo_khq_annots():
    """
    A small demo region around KHQ while it is being built

    Notes:
        The dates for this demo are rough. THis is the information we have.

            * Some cleanup of the site was happening early in 2017 (and perhaps
              even as early as 2016

            * Land clearing photo 2017-10-06

            * Constructions began in 2017.

            * In late 2017 the first structures went up (the stairwells)

            * Stairwell photo dated 2017-11-19

            * Photo of final building construction stages on 2018-11-29

            * Landscaping continued beyond building construction

            * We moved in in late 2018
    """
    import mgrs
    import kwimage
    import geojson
    import geopandas as gpd
    import ubelt as ub
    from geowatch.utils import util_gis
    from kwutil import util_time

    crs84 = util_gis._get_crs84()

    region_id = 'KHQ_R001'
    site_id = 'KHQ_R001_0000'

    # Boundary of the KHQ construction site
    khq_site_geos = {
        "type": "Polygon",
        "coordinates": [
            [[-73.77200379967688, 42.864783745778894],
             [-73.77177715301514, 42.86412514733195],
             [-73.77110660076141, 42.8641654498268],
             [-73.77105563879013, 42.86423720786224],
             [-73.7710489332676, 42.864399400374786],
             [-73.77134531736374, 42.8649134986743],
             [-73.77200379967688, 42.864783745778894]]
        ]
    }

    sites = []
    if 1:
        # Detailed observation on KHQ construction
        obs_property_defaults = {
            'type': 'observation',
            'observation_date': None,
            'source': None,
            'sensor_name': None,
            'current_phase': None,
            'is_occluded': None,
            'is_site_boundary': None,
            'score': 1.0,
        }

        observations = []
        observations.append(geojson.Feature(
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2016-12-01",
                "current_phase": 'No Activity',
                "source": 'guess',
            }),
            geometry=khq_site_geos,
        ))
        observations.append(geojson.Feature(
            geometry=khq_site_geos,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2017-09-13",
                "current_phase": 'Site Preparation',
                'sensor_name': 'S2_L2A',
                "source": 'sentinel-2',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geos,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2017-10-03",
                "current_phase": 'Site Preparation',
                'sensor_name': 'S2_L2A',
                "source": 'sentinel-2',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geos,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2017-10-06",
                "current_phase": 'Site Preparation',
                "source": 'ground-photo',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geos,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2017-11-19",
                "current_phase": 'Active Construction',
                "source": 'ground-photo',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geos,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2018-11-29",
                "current_phase": 'Active Construction',
                "source": 'ground-photo',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geos,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2019-01-01",
                "current_phase": 'Post Construction',
                "source": 'guess',
            })))
        obs_df = gpd.GeoDataFrame(observations, crs=crs84)
        # site_boundary_poly = kwimage.Polygon.coerce(obs_df.geometry.unary_union)
        # Use context manager to do all transforms in UTM space
        with util_gis.UTM_TransformContext(obs_df) as context:
            site_boundary_poly = context.geoms_utm.unary_union
            context.finalize(site_boundary_poly)
        site_boundary_geom = context.final_geoms_crs84.iloc[0]
        site_boundary_poly = kwimage.Polygon.coerce(site_boundary_geom)
        site_boundary_geos = site_boundary_poly.to_geojson()

        obs_dates = [
            util_time.coerce_datetime(obs['properties']['observation_date'])
            for obs in observations
        ]
        start_date = min(obs_dates)
        end_date = max(obs_dates)

        lon, lat = kwimage.Polygon.from_geojson(khq_site_geos).centroid
        mgrs_code = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)
        khq_sitesum = geojson.Feature(
            properties={
                "type": "site",
                "region_id": region_id,
                "site_id": site_id,
                "version": "2.0.0",
                "status": "positive_annotated",
                "mgrs": mgrs_code,
                "score": 1.0,
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "model_content": "annotation",
                "originator": "kit-demo",
                "validated": "False",
            },
            geometry=site_boundary_geos
        )
        khq_site = geojson.FeatureCollection(
            [khq_sitesum] + observations
        )
        sites.append(khq_site)

    # Build site summaries for each site (there is only one site)
    site_summaries = []
    for site in sites:
        # The property type changes from site to site summary.
        # Not sure why, but dems da specs
        site_header = site['features'][0]
        site_summary = geojson.Feature(
            properties=ub.dict_union(
                site_header['properties'], {'type': 'site_summary'}),
            geometry=site_header['geometry'],
        )
        site_summaries.append(site_summary)

    # Aggregate information across sites to build info for the region
    site_geoms = []
    site_start_dates = []
    site_end_dates = []
    for site in site_summaries:
        site_geoms.append(
            kwimage.Polygon.coerce(site['geometry']).to_shapely()
        )
        start_date = site['properties']['start_date']
        if start_date is not None:
            site_start_dates.append(util_time.coerce_datetime(start_date))
        end_date = site['properties']['end_date']
        if end_date is not None:
            site_end_dates.append(util_time.coerce_datetime(end_date))

    # Custom region geom beyond that provided by sites
    khq_region_geom = kwimage.Polygon.coerce({
        "type": "Polygon",
        "coordinates": [
            [[-73.77379417419434, 42.86254939745846],
             [-73.76715302467346, 42.86361104246733],
             [-73.76901984214783, 42.86713400027327],
             [-73.77529621124268, 42.865978051904726],
             [-73.7755537033081, 42.86542759269259],
             [-73.7750494480133, 42.862525805139775],
             [-73.77379417419434, 42.86254939745846]]
        ]
    }).to_shapely()
    site_geoms.append(khq_region_geom)

    # Use context manager to do all transforms in UTM space
    with util_gis.UTM_TransformContext(site_geoms) as context:
        tmp_geom_utm = context.geoms_utm.unary_union
        tmp_poly_utm = kwimage.Polygon.coerce(tmp_geom_utm)
        agg_geom_utm = tmp_poly_utm.scale(2.0, about='center').to_shapely()
        context.finalize(agg_geom_utm)
    khq_region_poly = context.final_geoms_crs84.iloc[0]
    khq_region_geom = kwimage.Polygon.coerce(khq_region_poly).to_geojson()

    delta_pad = util_time.coerce_timedelta('1460days')

    khq_region_start_time = min(site_start_dates) - delta_pad
    khq_region_end_time = max(site_end_dates) + delta_pad

    # Enlarge the region
    khq_region_feature = geojson.Feature(
        properties={
            "type": "region",
            "region_id": "KHQ_R001",
            "version": "2.4.3",
            "mgrs": mgrs_code,
            "start_date": khq_region_start_time.date().isoformat(),
            "end_date": khq_region_end_time.date().isoformat(),
            "originator": "kit-demo",
            "model_content": "annotation",
            "comments": 'geowatch-demo-data',
        },
        geometry=khq_region_geom
    )
    region = geojson.FeatureCollection([khq_region_feature] + site_summaries)
    return region, sites


def demo_smart_annots():
    """
    A small demo region in an area with a lot of data coverage
    """
    import mgrs
    import kwimage
    import geojson
    import geopandas as gpd
    import ubelt as ub
    from geowatch.utils import util_gis
    from kwutil import util_time

    crs84 = util_gis._get_crs84()

    region_id = 'SDEMO_R001'
    site_id = 'SDEMO_R001_0000'

    # Boundary of the KHQ construction site
    site_geos = {'type': 'Polygon',
                 'coordinates': [[[-81.70861006, 30.3741017],
                                  [-81.70745055, 30.28932539],
                                  [-81.58419451, 30.29015089],
                                  [-81.58346045, 30.37588478],
                                  [-81.70861006, 30.3741017]]]}

    sites = []
    WITH_SITES = 0
    if WITH_SITES:
        # obs_property_defaults = {
        #     'type': 'observation',
        #     'observation_date': None,
        #     'source': None,
        #     'sensor_name': None,
        #     'current_phase': None,
        #     'is_occluded': None,
        #     'is_site_boundary': None,
        #     'score': 1.0,
        # }

        # No observations in this case
        observations = []
        obs_df = gpd.GeoDataFrame(observations, crs=crs84)
        # site_boundary_poly = kwimage.Polygon.coerce(obs_df.geometry.unary_union)
        # Use context manager to do all transforms in UTM space
        with util_gis.UTM_TransformContext(obs_df) as context:
            site_boundary_poly = context.geoms_utm.unary_union
            context.finalize(site_boundary_poly)
        site_boundary_geom = context.final_geoms_crs84.iloc[0]
        site_boundary_poly = kwimage.Polygon.coerce(site_boundary_geom)
        site_boundary_geos = site_boundary_poly.to_geojson()

        obs_dates = [
            util_time.coerce_datetime(obs['properties']['observation_date'])
            for obs in observations
        ]
        start_date = min(obs_dates)
        end_date = max(obs_dates)

        lon, lat = kwimage.Polygon.from_geojson(site_geos).centroid
        mgrs_code = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)
        sitesum = geojson.Feature(
            properties={
                "type": "site",
                "region_id": region_id,
                "site_id": site_id,
                "version": "2.0.0",
                "status": "positive_annotated",
                "mgrs": mgrs_code,
                "score": 1.0,
                "start_date": start_date.date().isoformat(),
                "end_date": end_date.date().isoformat(),
                "model_content": "annotation",
                "originator": "kit-demo",
                "validated": "False",
            },
            geometry=site_boundary_geos
        )
        site = geojson.FeatureCollection(
            [sitesum] + observations
        )
        sites.append(site)

    # Build site summaries for each site (there is only one site)
    site_summaries = []
    for site in sites:
        # The property type changes from site to site summary.
        # Not sure why, but dems da specs
        site_header = site['features'][0]
        site_summary = geojson.Feature(
            properties=ub.dict_union(
                site_header['properties'], {'type': 'site_summary'}),
            geometry=site_header['geometry'],
        )
        site_summaries.append(site_summary)

    # Aggregate information across sites to build info for the region
    site_geoms = []
    site_start_dates = []
    site_end_dates = []
    for site in site_summaries:
        site_geoms.append(
            kwimage.Polygon.coerce(site['geometry']).to_shapely()
        )
        start_date = site['properties']['start_date']
        if start_date is not None:
            site_start_dates.append(util_time.coerce_datetime(start_date))
        end_date = site['properties']['end_date']
        if end_date is not None:
            site_end_dates.append(util_time.coerce_datetime(end_date))

    # Custom region geom beyond that provided by sites
    region_geom = kwimage.Polygon.coerce(site_geos).scale(1.5, about='centroid').to_shapely()
    site_geoms.append(region_geom)

    # Use context manager to do all transforms in UTM space
    with util_gis.UTM_TransformContext(site_geoms) as context:
        tmp_geom_utm = context.geoms_utm.unary_union
        tmp_poly_utm = kwimage.Polygon.coerce(tmp_geom_utm)
        agg_geom_utm = tmp_poly_utm.scale(2.0, about='center').to_shapely()
        context.finalize(agg_geom_utm)
    region_poly = context.final_geoms_crs84.iloc[0]
    region_geom = kwimage.Polygon.coerce(region_poly).to_geojson()

    delta_pad = util_time.coerce_timedelta('128days')

    site_start_dates.append(util_time.coerce_datetime('2017-01-01'))
    site_end_dates.append(util_time.coerce_datetime('2017-01-01'))

    region_start_time = min(site_start_dates) - delta_pad
    region_end_time = max(site_end_dates) + delta_pad

    lon, lat = kwimage.Polygon.coerce(region_geom).centroid
    mgrs_code = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)

    # Enlarge the region
    # TODO: see RegionModel in ~/code/watch/geowatch/cli/cluster_sites.py
    region_feature = geojson.Feature(
        properties={
            "type": "region",
            "region_id": region_id,
            "version": "2.4.3",
            "mgrs": mgrs_code,
            "start_date": region_start_time.date().isoformat(),
            "end_date": region_end_time.date().isoformat(),
            "originator": "kit-demo",
            "model_content": "annotation",
            "comments": 'geowatch-demo-data',
        },
        geometry=region_geom
    )
    region = geojson.FeatureCollection([region_feature] + site_summaries)
    return region, sites


def _configure_osm():
    """
    Configure open street map
    """
    import osmnx as ox
    import ubelt as ub
    import os
    osm_settings_dirs_varnames = [
        'data_folder',
        'logs_folder',
        'imgs_folder',
        'cache_folder',
    ]
    # Make osm dirs point at a standardized location
    osm_cache_root = ub.Path.appdir('osm')
    for varname in osm_settings_dirs_varnames:
        val = ub.Path(getattr(ox.settings, varname))
        if not val.is_absolute():
            new_val = os.fspath(osm_cache_root / os.fspath(val))
            setattr(ox.settings, varname, new_val)

    ox.settings.log_console = True
    ox.settings.log_console = True
    return ox


def _show_demo_annots_on_map():
    """
    Test to check the demo features on a map

    Requires:
        pip install osmnx

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> from geowatch.demo.demo_region import *  # NOQA
        >>> from geowatch.demo.demo_region import _show_demo_annots_on_map
        >>> import kwplot
        >>> kwplot.autompl()
    """
    import io
    import json
    import kwimage
    import kwplot
    import matplotlib
    from geowatch.utils import util_gis
    ox = _configure_osm()
    # https://geopandas.org/en/stable/gallery/plotting_basemap_background.html
    kwplot.autompl()

    region, sites = demo_khq_annots()
    region_gdf = util_gis.load_geojson(io.StringIO(json.dumps(region)))

    region_geom = region_gdf.geometry.iloc[0]
    osm_graph = ox.graph_from_polygon(region_geom)

    ax = kwplot.figure(fnum=1, docla=True).gca()
    fig, ax = ox.plot_graph(osm_graph, bgcolor='lawngreen', node_color='dodgerblue', edge_color='skyblue', ax=ax)
    print(matplotlib.get_backend())

    for geom in region_gdf.geometry:
        kwpoly = kwimage.Polygon.coerce(geom)
        kwpoly.draw(ax=ax, facecolor='none', fill=None, linewidth=1)

    # import contextily as cx
    # import mplcairo  # NOQA
    # cx.add_basemap(ax, url=cx.providers.OpenStreetMap.Mapnik)
    # cx.add_basemap(ax)
    # cx.add_basemap(ax, url=cx.providers.Esri.WorldImagery)


def demo_khq_region_fpath():
    import json
    import ubelt as ub
    region, sites = demo_khq_annots()
    annot_dpath = ub.Path.appdir('geowatch/demo/annotations').ensuredir()

    # Dump region file
    region_id = region['features'][0]['properties']['region_id']
    region_fpath = annot_dpath / (region_id + '.geojson')
    region_fpath.write_text(json.dumps(region))
    print(f'wrote region_fpath={region_fpath}')

    # Dump site file
    site_dpath = (annot_dpath / (region_id + '_sites')).ensuredir()
    for site in sites:
        site_id = site['features'][0]['properties']['site_id']
        site_fpath = site_dpath / (site_id + '.geojson')
        site_fpath.write_text(json.dumps(site))
        print(f'wrote site_fpath={site_fpath}')
    return region_fpath


def demo_smart_region_fpath():
    import json
    import ubelt as ub
    region, sites = demo_smart_annots()
    annot_dpath = ub.Path.appdir('geowatch/demo/annotations').ensuredir()

    # Dump region file
    region_id = region['features'][0]['properties']['region_id']
    region_fpath = annot_dpath / (region_id + '.geojson')
    region_fpath.write_text(json.dumps(region))
    print(f'wrote region_fpath={region_fpath}')

    # Dump site file
    site_dpath = (annot_dpath / (region_id + '_sites')).ensuredir()
    for site in sites:
        site_id = site['features'][0]['properties']['site_id']
        site_fpath = site_dpath / (site_id + '.geojson')
        site_fpath.write_text(json.dumps(site))
        print(f'wrote site_fpath={site_fpath}')
    return region_fpath


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/demo/demo_region.py
    """
    demo_khq_region_fpath()
