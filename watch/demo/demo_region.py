"""
Demodata for a simple region and site model
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
    from watch.utils import util_gis
    from watch.utils import util_time

    crs84 = util_gis._get_crs84()

    region_id = 'KHQ_R001'
    site_id = 'KHQ_R001_0000'

    # Boundary of the KHQ construction site
    khq_site_geom = {
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
            geometry=khq_site_geom,
        ))
        observations.append(geojson.Feature(
            geometry=khq_site_geom,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2017-10-06",
                "current_phase": 'Site Preparation',
                "source": 'ground-photo',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geom,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2017-11-19",
                "current_phase": 'Active Construction',
                "source": 'ground-photo',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geom,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2018-11-29",
                "current_phase": 'Active Construction',
                "source": 'ground-photo',
            })))
        observations.append(geojson.Feature(
            geometry=khq_site_geom,
            properties=ub.dict_union(obs_property_defaults, {
                "observation_date": "2019-01-01",
                "current_phase": 'Post Construction',
                "source": 'guess',
            })))
        obs_df = gpd.GeoDataFrame(observations, crs=crs84)
        site_boundary_poly = kwimage.Polygon.coerce(obs_df.geometry.unary_union)
        site_boundary_geom = site_boundary_poly.to_geojson()

        obs_dates = [
            util_time.coerce_datetime(obs['properties']['observation_date'])
            for obs in observations
        ]
        start_date = min(obs_dates)
        end_date = max(obs_dates)

        lon, lat = kwimage.Polygon.from_geojson(khq_site_geom).centroid
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
            geometry=site_boundary_geom
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

    khq_region_geom = {
        "type": "Polygon",
        "coordinates": [
            [[-73.77379417419434, 42.86254939745846],
             [-73.76715302467346, 42.86361104246733],
             [-73.76901984214783, 42.86713400027327],
             [ -73.77529621124268, 42.865978051904726],
             [ -73.7755537033081, 42.86542759269259],
             [ -73.7750494480133, 42.862525805139775],
             [ -73.77379417419434, 42.86254939745846]]
        ]
    }

    # Enlarge the region
    khq_region_poly = kwimage.Polygon.coerce(khq_region_geom)
    khq_region_geom = khq_region_poly.scale(2.0, about='center').to_geojson()

    khq_region_feature = geojson.Feature(
        properties={
            "type": "region",
            "region_id": "KHQ_R001",
            "version": "2.4.3",
            "mgrs": mgrs_code,
            "start_date": "2017-01-01",
            "end_date": "2020-01-01",
            "originator": "kit-demo",
            "model_content": "annotation",
            "comments": 'watch-demo-data',
        },
        geometry=khq_region_geom
    )
    region = geojson.FeatureCollection([khq_region_feature] + site_summaries)
    return region, sites


def _configure_osm():
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
        >>> from watch.demo.demo_region import *  # NOQA
        >>> from watch.demo.demo_region import _show_demo_annots_on_map
        >>> import kwplot
        >>> kwplot.autompl()
    """
    import io
    import json
    import kwimage
    import kwplot
    import matplotlib
    from watch.utils import util_gis
    ox = _configure_osm()
    # https://geopandas.org/en/stable/gallery/plotting_basemap_background.html
    kwplot.autompl()

    region, sites = demo_khq_annots()
    region_gdf = util_gis.read_geojson(io.StringIO(json.dumps(region)))

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
    annot_dpath = ub.Path.appdir('watch/demo/annotations').ensuredir()

    # Dump region file
    region_id = region['features'][0]['properties']['region_id']
    region_fpath = annot_dpath / (region_id + '.geojson')
    region_fpath.write_text(json.dumps(region))

    # Dump site file
    site_dpath = (annot_dpath / (region_id + '_sites')).ensuredir()
    for site in sites:
        site_id = site['features'][0]['properties']['site_id']
        site_fpath = site_dpath / (site_id + '.geojson')
        site_fpath.write_text(json.dumps(site))

    return region_fpath
