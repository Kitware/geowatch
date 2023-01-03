"""
Functions to creates random "true" region and site models for demodata.
"""
import geojson
import kwarray
import kwimage
import mgrs
import ubelt as ub
from datetime import datetime as datetime_cls
from shapely.ops import unary_union
from kwarray.distributions import TruncNormal
from watch.demo.metrics_demo import demo_utils


class RegionModelGenerator:
    """
    Note:
        A good refactor to this random data design would be to have a class
        which holds all the parameters to generate random site / region models.
        For now we are using this as a placeholder to store constants.

    TODO:
        Transfer the functionality of `random_region_model` here.
    """
    ...


class SiteModelGenerator:
    """
    TODO:
        Transfer the functionality of random_site_model here.
    """
    SITE_PHASES = [
        "No Activity",
        "Site Preparation",
        "Active Construction",
        "Post Construction",
    ]


def random_region_model(region_id=None, num_sites=3, num_observations=5,
                        p_observe=0.5, p_transition=0.15, rng=None):
    """
    Generate a random region model with random sites and observation support.

    The region model is generated simply by sampling a random polygon in
    geospace that isn't too big or too small.

    Then observation support simulates images that we have from this region.
    These supporting images might be from different simulated sensors or in
    different resolutions. They may contain partial coverage of the region.

    The sites are randomly sampled polygons inside the region and are assigned
    to specific observation as they would be if they were generated for real
    imagery. Sites have a simple model that allows them to evolve over time,
    change shape / size and their phase label.

    Args:
        region_name (str | None): Name of the region.
            If unspecified, a random one is created.

        num_sites (int): number of random sites

        num_observations (int): number of random observations

        p_observe (float):
            the probability that a site model is annotated in any particular
            observation.

        p_transition (float):
            truth phase transition model. Currently just the probability
            the phase changes on any particular observation.

        rng : seed or random number generator

    Returns:
        Tuple[geojson.FeatureCollection, List[geojson.FeatureCollection]]:
            A region model and its corresponding site models

    Example:
        >>> from watch.demo.metrics_demo.demo_truth import *  # NOQA
        >>> region, sites, renderables = random_region_model(num_sites=2, num_observations=5, p_observe=0.5, rng=0)
        >>> print('region = {}'.format(ub.repr2(region, nl=4, precision=6, sort=0)))
        region = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': ...
                    },
                    'properties': {
                        'type': 'region',
                        'region_id': 'DR_0684',
                        'version': '2.4.3',
                        'mgrs': '51PXM',
                        'start_date': '2011-05-28',
                        'end_date': '2018-09-13',
                        'originator': 'demo-truth',
                        'model_content': 'annotation',
                        'comments': 'demo-data',
                    },
                },
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': ...
                    },
                    'properties': {
                        'type': 'site_summary',
                        'status': 'positive_annotated',
                        'version': '2.0.1',
                        'site_id': 'DR_0684_000',
                        'mgrs': '51PXM',
                        'start_date': '2015-03-16',
                        'end_date': '2018-09-13',
                        'score': 1,
                        'originator': 'demo',
                        'model_content': 'annotation',
                        'validated': 'True',
                        'misc_info': {'color': [0.551139, 1.000000, 0.000000]},
                    },
                },
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': ...
                    },
                    'properties': {
                        'type': 'site_summary',
                        'status': 'positive_annotated',
                        'version': '2.0.1',
                        'site_id': 'DR_0684_001',
                        'mgrs': '51PXM',
                        'start_date': '2011-05-28',
                        'end_date': '2018-09-13',
                        'score': 1,
                        'originator': 'demo',
                        'model_content': 'annotation',
                        'validated': 'True',
                        'misc_info': {'color': [1.000000, 0.367780, 0.000000]},
                    },
                },
            ],
        }


    """
    rng = kwarray.ensure_rng(rng)

    if num_observations <= 0:
        raise ValueError('must have at least one observation')

    if region_id is None:
        region_id = 'DR_{:04d}'.format(rng.randint(0, 1000))

    region_poly = demo_utils.random_geo_polygon(max_rt_area=10_000, rng=rng)

    region_geom = region_poly.to_shapely()

    lon, lat = region_poly.centroid
    mgrs_code = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)

    # Determine how we made each observation
    observables = random_observables(num_observations, rng=rng)
    wld_polygon = region_poly
    for observation in observables:
        # Could be more complex here, but other code would have to change
        # Making assumptions everything is aligned in initial pass
        observation['mgrs_code'] = mgrs_code
        observation['wld_polygon'] = wld_polygon

    # Define the region feature
    region_feature = geojson.Feature(
        properties={
            "type": "region",
            "region_id": region_id,
            "version": "2.4.3",
            "mgrs": mgrs_code,
            "start_date": observables[0]['datetime'].date().isoformat(),
            "end_date": observables[-1]['datetime'].date().isoformat(),
            "originator": "demo-truth",
            "model_content": "annotation",
            "comments": "demo-data",
        },
        geometry=region_geom,
    )

    region_box = region_poly.bounding_box()
    region_corners = region_box.corners()

    # Create random site models within this region
    sites = []
    site_summaries = []
    for site_num in range(num_sites):
        site_id = f"{region_id}_{site_num:03d}"
        sitesum, site = random_site_model(
            region_id, site_id, region_corners, observables,
            p_observe=p_observe, p_transition=p_transition, rng=rng)
        site_summaries.append(sitesum)
        sites.append(site)

    region = geojson.FeatureCollection([region_feature] + site_summaries)

    # Create information about how we should render dummy images
    renderables = []
    for frame_idx, observable in enumerate(observables):

        # For each observed date, generate information about the toy images we
        # "observed" and what visible data is in them.
        image_box = kwimage.Boxes([[0, 0, 800, 600]], "xywh")
        image_corners = image_box.corners().astype(float)
        tf_image_from_region = kwimage.Affine.fit(region_corners, image_corners)

        datetime = observable["datetime"]

        visible_polys = []
        for sitesum in site_summaries:
            site_d1 = datetime_cls.fromisoformat(sitesum["properties"]["start_date"])
            site_d2 = datetime_cls.fromisoformat(sitesum["properties"]["end_date"])
            # TODO: more date range intersection query, can blend between geom
            # observations
            if site_d1 <= datetime and datetime <= site_d2:
                wld_site_poly = kwimage.Polygon.coerce(sitesum["geometry"])
                img_site_poly = wld_site_poly.warp(tf_image_from_region)
                img_site_poly.meta["color"] = sitesum["properties"]["misc_info"][
                    "color"
                ]
                visible_polys.append(img_site_poly)

        img_width = image_box.width.ravel()[0]
        img_height = image_box.height.ravel()[0]
        image_dsize = [img_width, img_height]
        renderable = {
            "image_dsize": image_dsize,
            "visible_polys": visible_polys,
            "date": datetime,
            "wld_polygon": observable["wld_polygon"],
            "sensor": observable["sensor_name"],
            "frame_idx": frame_idx,
        }
        renderables.append(renderable)

    return region, sites, renderables


def random_observables(num_observations, rng=None):
    """
    Create a random sequence of sensor observations

    Args:
        num_observations (int): number of observations
        rng : random seed or generator

    Returns:
        List[dict]: list of each item corresonding to a simulated observable

    Example:
        >>> from watch.demo.metrics_demo.demo_truth import *  # NOQA
        >>> num_observations = 2
        >>> observables = random_observables(1, rng=32)
        >>> print('observables = {}'.format(ub.repr2(observables, nl=2)))
        observables = [
            {
                'datetime': datetime.datetime(2018, 8, 3, 16, 55, 35, 398921),
                'mgrs_code': None,
                'sensor_name': 'demosat-2',
                'source': 'demosat-220180803T165535',
                'wld_polygon': None,
            },
        ]
    """
    rng = kwarray.ensure_rng(rng)
    observed_times = demo_utils.random_time_sequence(
        "2010-01-01", "2020-01-01", num_observations, rng=rng
    )
    # A list of simulated sensors we pretend an observation might be from
    demo_sensors = [
        'demosat-1',
        'demosat-2',
    ]

    # Determine how we made each observation
    observables = []
    sensor_name = "demosat"  # could be more complex
    for datetime in observed_times:
        sensor_name = rng.choice(demo_sensors)
        observables.append(
            {
                "datetime": datetime,
                "sensor_name": sensor_name,
                "source": sensor_name + datetime.strftime('%Y%m%dT%H%M%S'),
                "mgrs_code": None,  # set later
                "wld_polygon": None,  # set later
            }
        )
    return observables


def random_site_model(region_id, site_id, region_corners, observables,
                      p_observe=0.5, p_transition=0.15, rng=None):
    """
    Make a dummy sequence somewhere within a region's observed spacetime grid.

    Args:
        region_id (str): the name of the region we generate this site for

        site_id (str): the name of the site

        region_corners (ndarray):
            corners of the region to embed this site in.

        observables (List[Dict]):
            information about opportunities to generate an observation

        p_observe (float):
            the probability that a site model is annotated in any particular
            observation.

        p_transition (float):
            truth phase transition model. Currently just the probability
            the phase changes on any particular observation.

        rng : random state or seed

    Returns:
        Tuple[Dict, Dict]: site_summary, site

    Example:
        >>> from watch.demo.metrics_demo.demo_truth import *  # NOQA
        >>> region_id = 'DR_0042'
        >>> site_id = 'DR_0042_9001'
        >>> rng = kwarray.ensure_rng(0)
        >>> region_corners = kwimage.Boxes([[0, 0, 1, 1]], 'xywh').to_ltrb().corners()
        >>> observables = random_observables(1, rng=rng)
        >>> p_observe = 1.0
        >>> site_summary, site = random_site_model(region_id, site_id, region_corners, observables,
        >>>                                        p_observe=p_observe, rng=rng)
        >>> print('site = {}'.format(ub.repr2(site, nl=4, sort=0)))
        site = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [[[0.759303, 0.749121], [0.717993, 0.763785], [0.719567, 0.797059], [0.751856, 0.810715], [0.776689, 0.799334], [0.779587, 0.762869], [0.759303, 0.749121]]],
                    },
                    'properties': {
                        'type': 'site',
                        'status': 'positive_annotated',
                        'version': '2.0.1',
                        'site_id': 'DR_0042_9001',
                        'mgrs': None,
                        'start_date': '2015-06-28',
                        'end_date': '2015-06-28',
                        'score': 1,
                        'originator': 'demo',
                        'model_content': 'annotation',
                        'validated': 'True',
                        'misc_info': {'color': [0.0, 1.0, 0.0]},
                        'region_id': 'DR_0042',
                    },
                },
                {
                    'type': 'Feature',
                    'geometry': {
                        'type': 'MultiPolygon',
                        'coordinates': [[[[0.719567, 0.797059], [0.717993, 0.763785], [0.759303, 0.749121], [0.779587, 0.762869], [0.776689, 0.799334], [0.751856, 0.810715]]]],
                    },
                    'properties': {
                        'type': 'observation',
                        'observation_date': '2015-06-28',
                        'source': 'demosat-220150628T072421',
                        'sensor_name': 'demosat-2',
                        'current_phase': 'No Activity',
                        'is_occluded': 'False',
                        'is_site_boundary': 'True',
                        'score': 1.0,
                    },
                },
            ],
        }


    Example:
        >>> from watch.demo.metrics_demo.demo_truth import *  # NOQA
        >>> region_id = 'DR_0042'
        >>> site_id = 'DR_0042_9001'
        >>> rng = kwarray.ensure_rng(0)
        >>> region_corners = kwimage.Boxes([[0, 0, 1, 1]], 'xywh').to_ltrb().corners()
        >>> observables = random_observables(10, rng=rng)
        >>> p_observe = 1.0
        >>> site_summary, site = random_site_model(region_id, site_id, region_corners, observables,
        >>>                                        p_observe=p_observe, rng=rng)
        >>> print('site_summary = {}'.format(ub.repr2(site_summary, nl=-1, sort=0)))
        site_summary = {
            'type': 'Feature',
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [0.194188, 0.062901],
                        [0.173624, 0.064937],
                        [0.149949, 0.095669],
                        [0.178188, 0.128928],
                        [0.212713, 0.124288],
                        [0.215977, 0.08933],
                        [0.194188, 0.062901]
                    ]
                ]
            },
            'properties': {
                'type': 'site_summary',
                'status': 'positive_annotated',
                'version': '2.0.1',
                'site_id': 'DR_0042_9001',
                'mgrs': None,
                'start_date': '2013-11-01',
                'end_date': '2019-08-21',
                'score': 1,
                'originator': 'demo',
                'model_content': 'annotation',
                'validated': 'True',
                'misc_info': {
                    'color': [1.0, 0.36777954425013254, 0.0]
                }
            }
        }

    Example:
        >>> from watch.demo.metrics_demo.demo_truth import *  # NOQA
        >>> region_id = 'DR_0042'
        >>> site_id = 'DR_0042_9001'
        >>> rng = kwarray.ensure_rng(42232)
        >>> region_corners = kwimage.Boxes([[5, 7, 11, 13]], 'xywh').to_ltrb().corners()
        >>> observables = random_observables(10, rng=rng)
        >>> p_observe = 1.0
        >>> site_summary, site = random_site_model(region_id, site_id, region_corners, observables,
        >>>                                        p_observe=p_observe, rng=rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> import geopandas as gpd
        >>> # Draw our true and perterbed site model
        >>> site_gdf = gpd.GeoDataFrame.from_features(site[1:])
        >>> total_poly = site_gdf['geometry'].unary_union
        >>> minx, miny, maxx, maxy = total_poly.bounds
        >>> fig = kwplot.figure(doclf=True, fnum=1)
        >>> pnum_ = kwplot.PlotNums(nSubplots=len(site_gdf))
        >>> for idx in range(len(site_gdf)):
        >>>     row = site_gdf.iloc[idx: idx + 1]
        >>>     item = row.iloc[0]
        >>>     title = item['observation_date'] + ' ' + item['current_phase']
        >>>     fig = kwplot.figure(pnum=pnum_(), title=title)
        >>>     ax = fig.gca()
        >>>     row.plot(ax=ax, alpha=0.5, color='limegreen', edgecolor='black')
        >>>     row.centroid.plot(ax=ax, alpha=0.5, color='green')
        >>>     ax.set_xlim(minx, maxx)
        >>>     ax.set_ylim(miny, maxy)
        >>> kwplot.show_if_requested()
    """
    rng = kwarray.ensure_rng(rng)

    ### Toydata generation parameters

    # Factors about how the geometry evolves
    max_scale_change = 1.05
    scale_distri = TruncNormal(
        1.0, 0.3, 1 / max_scale_change, max_scale_change, rng=rng
    )

    # Factors into initial size of geometry
    max_scale_factor = 1 / 6
    max_translate = 1 - max_scale_factor

    # Generate a random site polygon in the unit 0-1 quadrent
    # Scale it to ensure its takes at most max_scale_factor of the space
    # And randomly translate within that space.
    offset = rng.rand(2) * max_translate
    site_unit_geom = kwimage.Polygon.random(rng=rng)
    site_unit_geom = site_unit_geom.scale(max_scale_factor).translate(offset)

    # Take a subset of the dates we observed this annotation
    n_observables = len(observables)
    observe_flags = rng.rand(n_observables) <= p_observe
    n_tries = 0
    while observe_flags.sum() < min(n_observables, 2):
        # Make sure we have 2 of them
        observe_flags = rng.rand(n_observables) <= p_observe
        n_tries += 1
        if n_tries > 1000:
            raise Exception('Cannot satisfy constraint')

    sampled_observables = list(ub.compress(observables, observe_flags))

    phases = SiteModelGenerator.SITE_PHASES

    # Assign a phase to each date
    curr_phase_idx = 0
    sampled_phases = []
    for _ in range(len(sampled_observables)):
        # could do a more complex transition model
        if curr_phase_idx >= len(phases):
            sampled_observables = sampled_observables[: len(sampled_phases)]
            break
        phase = phases[curr_phase_idx]
        sampled_phases.append(phase)
        if rng.rand() < p_transition:
            curr_phase_idx += 1

    obs_property_defaults = ub.udict(
        {
            "type": "observation",
            "observation_date": None,
            "source": None,
            "sensor_name": None,
            "current_phase": None,
            "is_occluded": "False",  # None,
            "is_site_boundary": "True",  # None,
            "score": 1.0,
        }
    )

    sampled_unit_geoms = []
    curr_unit_geom = site_unit_geom.copy()
    for _ in range(len(sampled_observables)):
        # randomly evolve the geometry over time
        scale = scale_distri.sample()
        curr_unit_geom = curr_unit_geom.scale(scale, about='centroid')
        sampled_unit_geoms.append(curr_unit_geom)

    # Build transform from unit to region space
    unit_corners = kwimage.Boxes([[0, 0, 1.0, 1.0]], "xywh").corners()
    tf_region_from_unit = kwimage.Affine.fit(unit_corners, region_corners)

    observations = []
    for unit_geom, observable, phase in zip(
        sampled_unit_geoms, sampled_observables, sampled_phases
    ):
        # Warp it into the region space
        datetime = observable["datetime"]
        site_geom = unit_geom.warp(tf_region_from_unit)
        observations.append(
            geojson.Feature(
                geometry=site_geom.to_multi_polygon().to_geojson(),
                properties=obs_property_defaults | {
                    "observation_date": datetime.date().isoformat(),
                    "current_phase": phase,
                    "sensor_name": observable["sensor_name"],
                    "source": observable["source"],
                },
            )
        )
    summary_unit_geom = unary_union([g.to_shapely() for g in sampled_unit_geoms]).convex_hull
    summary_unit_poly = kwimage.Polygon.coerce(summary_unit_geom)
    summary_geom = summary_unit_poly.warp(tf_region_from_unit).to_shapely()

    # Build site summary
    status = "positive_annotated"  # Could be more complex here

    mgrs_code = observables[0]["mgrs_code"]

    colors = kwimage.Color.distinct(10)
    idx = rng.randint(0, len(colors) - 1)
    color = colors[idx]

    site_summary = make_site_summary(
        observations, mgrs_code, site_id, status, summary_geom
    )
    if "misc_info" not in site_summary["properties"]:
        site_summary["properties"]["misc_info"] = {}
    site_summary["properties"]["misc_info"]["color"] = color

    site_header = site_summary.copy()
    site_header["properties"] = site_header["properties"].copy()
    site_header["properties"]["type"] = "site"
    site_header["properties"]["region_id"] = region_id
    site = geojson.FeatureCollection([site_header] + observations)
    return site_summary, site


def make_site_summary(observations, mgrs_code, site_id, status, summary_geom=None):
    """
    Consolodate site observations into a site summary
    """
    if summary_geom is None:
        summary_geom = unary_union(
            [kwimage.MultiPolygon.coerce(o["geometry"]).to_shapely() for o in observations]
        ).convex_hull
    start_date = observations[0]["properties"]["observation_date"]
    end_date = observations[-1]["properties"]["observation_date"]
    sitesum_props = {
        "type": "site_summary",
        "status": status,
        "version": "2.0.1",
        "site_id": site_id,
        "mgrs": mgrs_code,
        "start_date": start_date,
        "end_date": end_date,
        "score": 1,
        "originator": "demo",
        "model_content": "annotation",
        "validated": "True",
    }
    site_summary = geojson.Feature(
        properties=sitesum_props,
        geometry=kwimage.Polygon.coerce(summary_geom).to_geojson(),
    )
    return site_summary
