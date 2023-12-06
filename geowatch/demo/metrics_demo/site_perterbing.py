"""
Functions to perterb or "jitter" truth that gradually degrade the site models
in specified ways. This can be used to check the scoring metrics at various
levels of accuracy / inaccuracy.
"""
import numpy as np
import geojson
import kwarray
import kwimage
import ubelt as ub
import geopandas as gpd
from geowatch.demo.metrics_demo import demo_truth
from geowatch.demo.metrics_demo import demo_utils


def perterb_site_model(sites, rng=None, **kwargs):
    """
    Given a true site models from a region, perterb them to make "demo"
    predicted site models.

    Args:
        sites (List[geojson.FeatureCollection]): geojson site observations
        rng : random seed or generator
        **kwargs: factors to control per-site perterbations. See
            :func:`perterb_single_site_model` for available options.

    Returns:
        List[geojson.FeatureCollection]: modified site models

    Example:
        >>> from geowatch.demo.metrics_demo.site_perterbing import *  # NOQA
        >>> _, sites, _ = demo_truth.random_region_model(rng=12345)
        >>> pred_sites1 = perterb_site_model(sites, noise=1, rng=34567)
        >>> pred_sites2 = perterb_site_model(sites, noise=0, rng=34567)
    """
    rng = kwarray.ensure_rng(rng)

    # Shuffle our site models so the indices don't correspond.
    templates = sites.copy()
    rng.shuffle(templates)

    # TODO: drop sites to simulate false negatives and add new random sites to
    # generate false positives.

    pred_sites = []
    for idx, site in enumerate(templates):
        try:
            pred_site = perterb_single_site_model(site, idx=idx, rng=rng, **kwargs)
        except IndexError:
            continue  # drop the site
        pred_sites.append(pred_site)
    return pred_sites


class PerterbModel:
    """
    TODO:
        consume the functionality of perterb_single_site_model

    Example:
        >>> from geowatch.demo.metrics_demo.site_perterbing import *  # NOQA
        >>> self = PerterbModel()

    Example:
        >>> from geowatch.demo.metrics_demo.site_perterbing import *  # NOQA
        >>> from geowatch.demo.metrics_demo import demo_truth
        >>> region_id = 'DR_0042'
        >>> site_id = 'DR_0042_9001'
        >>> rng = kwarray.ensure_rng(4222)
        >>> region_corners = kwimage.Boxes([[5, 7, 11, 13]], 'xywh').to_ltrb().corners()
        >>> observables = demo_truth.random_observables(15, rng=rng)
        >>> p_observe = 0.5
        >>> site_summary, site = demo_truth.random_site_model(
        >>>     region_id, site_id, region_corners, observables,
        >>>     p_observe=p_observe, p_transition=0.4,  rng=rng)
        >>> kwargs = dict(noise=10, drop_limit=0.1)
        >>> self = PerterbModel(**kwargs)
        >>> pred_site = self.perterb_single_site(site)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> import geopandas as gpd
        >>> # Draw our true and perterbed site model
        >>> orig_site_gdf = gpd.GeoDataFrame.from_features(site)
        >>> pred_site_gdf = gpd.GeoDataFrame.from_features(pred_site)
        >>> total_poly1 = orig_site_gdf['geometry'].unary_union
        >>> total_poly2 = pred_site_gdf['geometry'].unary_union
        >>> total_poly = total_poly1.union(total_poly2)
        >>> minx, miny, maxx, maxy = total_poly.bounds
        >>> fig = kwplot.figure(doclf=True, fnum=1)
        >>> phases = demo_truth.SiteModelGenerator.SITE_PHASES
        >>> phase_to_color = ub.udict(ub.dzip(phases, kwimage.Color.distinct(len(phases))))
        >>> pnum_ = kwplot.PlotNums(nRows=4, nSubplots=len(observables) + 1)
        >>> boundary_colors = ub.udict({'true': 'limegreen', 'pred': 'dodgerblue'}).map_values(lambda c: kwimage.Color(c).as01())
        >>> for obs in observables:
        >>>     title = obs['datetime'].isoformat()
        >>>     trues = orig_site_gdf[orig_site_gdf['observation_date'] == obs['datetime'].date().isoformat()]
        >>>     preds = pred_site_gdf[pred_site_gdf['observation_date'] == obs['datetime'].date().isoformat()]
        >>>     fig = kwplot.figure(pnum=pnum_(), title=title)
        >>>     ax = fig.gca()
        >>>     if len(trues):
        >>>         colors = list(phase_to_color.take(trues['current_phase']))
        >>>         trues.plot(ax=ax, alpha=0.5, color=colors)
        >>>         trues.boundary.plot(ax=ax, alpha=0.5, color=boundary_colors['true'], linewidth=3)
        >>>         trues.centroid.plot(ax=ax, alpha=0.5, color=boundary_colors['true'])
        >>>     if len(preds):
        >>>         colors = list(phase_to_color.take(preds['current_phase']))
        >>>         preds.plot(ax=ax, alpha=0.5, color=colors)
        >>>         preds.boundary.plot(ax=ax, alpha=0.5, color=boundary_colors['pred'], linewidth=3)
        >>>         preds.centroid.plot(ax=ax, alpha=0.5, color=boundary_colors['pred'])
        >>>     ax.set_xlim(minx, maxx)
        >>>     ax.set_ylim(miny, maxy)
        >>> legend1 = kwimage.draw_header_text(kwplot.make_legend_img(phase_to_color), 'Face Key', fit=True)
        >>> legend2 = kwimage.draw_header_text(kwplot.make_legend_img(boundary_colors), 'Border Key', fit=True)
        >>> legend = kwimage.stack_images([legend1, legend2], axis=0, resize='smaller')
        >>> kwplot.imshow(legend, pnum=pnum_())
        >>> kwplot.show_if_requested()
    """

    def __init__(self, noise=0.0, performer_id='alice', rng=None, **kwargs):
        self.rng = rng
        self.performer_id = performer_id

        default_noise = ub.udict({
            'warp_noise': 1.0,
            'label_noise': 1.0,
            'drop_noise': 0.5,
        })
        default_limits = ub.udict({
            'drop_limit': 0.5,
            'scale_limit': 0.3,
            'theta_limit': np.pi / 16,
        })
        kwargs = ub.udict(kwargs)

        limits = default_limits | (kwargs & default_limits)
        kwargs -= limits
        limits = limits.map_keys(lambda k: k.split('_limit')[0])

        noises = default_noise | (kwargs & default_noise)
        kwargs -= noises
        noises = noises.map_keys(lambda k: k.split('_noise')[0])

        self.limits = limits
        self.noises = noises
        self.noise = noise

        if len(kwargs) > 0:
            raise ValueError('unknown kwargs={}'.format(kwargs))

        self.distris = None
        self._setup_distributions()

    def _setup_distributions(self):
        from kwarray import distributions as distri
        rng = self.rng
        noises = self.noises
        limits = self.limits
        max_scale = 1 + limits['scale'] * (noises['warp'] * self.noise)
        max_theta = 0 + limits['theta'] * (noises['warp'] * self.noise)
        p_randomize = max(0, np.tanh(noises['label'] * self.noise))
        p_drop = min(max(0, np.tanh(noises['drop'] * self.noise)), limits['drop'])

        distris = {}
        distris['scale'] = distri.Uniform(1, max_scale, rng=rng) ** distri.CategoryUniform([-1, 1])
        distris['theta'] = distri.Uniform(0, max_theta, rng=rng) * distri.CategoryUniform([-1, 1])
        distris['phase'] = distri.CategoryUniform(demo_truth.SiteModelGenerator.SITE_PHASES, rng=rng)
        distris['randomize_phase'] = distri.Bernoulli(p_randomize, rng=rng)
        distris['drop_frame'] = distri.Bernoulli(p_drop, rng=rng)
        self.distris = distris

    def perterb_single_site(self, site, idx=None):
        true_site_summary = site["features"][0]["properties"]
        mgrs_code = true_site_summary["mgrs"]

        # TODO: add the ability to pass in observables to add temporal noise

        # TODO: could be more complex here.
        valid_status_codes = [
            "system_proposed",
            "system_confirmed",
            "system_rejected",
        ]
        status = valid_status_codes[1]

        # Find the true site header
        orig_site_header = None
        orig_observations = []
        for feat in site["features"]:
            if feat['properties']['type'] == 'site':
                if orig_site_header is not None:
                    raise ValueError('More than one site header')
                orig_site_header = feat
            elif feat['properties']['type'] == 'observation':
                orig_observations.append(feat)
            else:
                raise TypeError('Unknown feature type, should be site or observation')
        if orig_site_header is None:
            raise ValueError(
                'A site feature collection does not have a summary header '
                'with type=site in its properties')

        region_id = orig_site_header['properties']['region_id']
        if idx is None:
            site_id = orig_site_header['properties']['site_id']
        else:
            site_id = f"{region_id}_{idx:03d}"

        orig_obs_crs84 = gpd.GeoDataFrame.from_features(
            orig_observations, crs=demo_utils.get_crs84())

        # Do any shape warping in the local UTM space.
        orig_obs_utm = demo_utils.project_gdf_to_local_utm(orig_obs_crs84)
        orig_geoms_utm = orig_obs_utm['geometry']

        new_geoms_utm = []
        new_properties = []
        for feat, geom_utm in zip(orig_observations, orig_geoms_utm):

            if self.distris['drop_frame'].sample():
                continue

            orig_props = ub.udict(feat["properties"])
            poly = kwimage.MultiPolygon.coerce(geom_utm)

            xy = [d[0] for d in poly.to_shapely().centroid.xy]
            rand_aff_params = {
                'scale': self.distris['scale'].sample(),
                'theta': self.distris['theta'].sample(),
            }
            aff = kwimage.Affine.affine(**rand_aff_params, about=xy)
            new_poly = poly.warp(aff)

            current_phase = orig_props["current_phase"]

            if self.distris['randomize_phase'].sample():
                current_phase = self.distris['phase'].sample()

            new_props = orig_props | {
                "id": site_id,
                "current_phase": current_phase,
                "is_site_boundary": True,
                "originator": self.performer_id,
                "validated": "False",
            }
            new_properties.append(new_props)
            new_geoms_utm.append(new_poly.to_shapely())

        # Convert back to CRS84
        new_geoms_utm = gpd.GeoDataFrame(geometry=new_geoms_utm, crs=orig_obs_utm.crs)
        new_summary_utm = gpd.GeoDataFrame(geometry=[new_geoms_utm.unary_union], crs=orig_obs_utm.crs)

        new_summary_crs84 = new_summary_utm.to_crs(orig_obs_crs84.crs)['geometry'].iloc[0]
        new_geoms_crs84 = new_geoms_utm.to_crs(orig_obs_crs84.crs)['geometry']

        # Build the geojson features
        new_observations = []
        for new_props, new_geom in zip(new_properties, new_geoms_crs84):
            new_poly = kwimage.MultiPolygon.coerce(new_geom)
            new = geojson.Feature(
                geometry=new_poly.to_geojson(),
                properties=new_props,
            )
            new_observations.append(new)

        new_site_header = demo_truth.make_site_summary(
            new_observations, mgrs_code, site_id, status,
            summary_geom=new_summary_crs84)
        new_site_header["properties"].update(
            {
                "type": "site",
                "site_id": site_id,
                "region_id": region_id,
            }
        )
        pred_site = geojson.FeatureCollection([new_site_header] + new_observations)
        return pred_site


def perterb_single_site_model(site, noise=0.0, warp_noise=1.0,
                              performer_id='alice', idx=None, rng=None, **kwargs):
    """
    Given a true site model, perterb it to make a "demo" predicted site model.

    Args:
        site (geojson.FeatureCollection): geojson site observations

        noise (float):
            Magnitude of all noise. This factor modulates all other noise
            values. Setting to 0 disables all perterbation. Typically 1
            is the largest "reasonable" value. Setting any noise magnitude
            higher than 1 may result in pathological perterbations.

        warp_noise (float):
            magnitude of random warp that we will use to perterb the boundary
            of the true site polygons.

        performer_id (str):
            used in the originator property

        idx (int): the index of this new site model.
            If unspecified, the same site id is used in truth and pred.

        rng : random seed or generator

        **kwargs : other PerterbModel params

    Returns:
        geojson.FeatureCollection: modified site model

    Example:
        >>> # Demo case with a lot of noise
        >>> from geowatch.demo.metrics_demo.site_perterbing import *  # NOQA
        >>> rng = kwarray.ensure_rng(43240830)
        >>> _, sites, _ = demo_truth.random_region_model(num_observations=20, rng=rng)
        >>> site = sites[0]
        >>> pred_site = perterb_single_site_model(site, noise=1.0, rng=rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> # Draw our true and perterbed site model
        >>> orig_site_gdf = gpd.GeoDataFrame.from_features(site)
        >>> pred_site_gdf = gpd.GeoDataFrame.from_features(pred_site)
        >>> fig = kwplot.figure(doclf=True, fnum=1)
        >>> ax = fig.gca()
        >>> total_poly = pred_site_gdf['geometry'].unary_union.union(orig_site_gdf['geometry'].unary_union)
        >>> minx, miny, maxx, maxy = total_poly.bounds
        >>> ax.set_xlim(minx, maxx)
        >>> ax.set_ylim(miny, maxy)
        >>> pred_site_gdf.plot(ax=ax, alpha=0.5, color='orange', edgecolor='purple')
        >>> orig_site_gdf.plot(ax=ax, alpha=0.5, color='limegreen', edgecolor='black')
        >>> orig_site_gdf.centroid.plot(ax=ax, alpha=0.5, color='green')
        >>> pred_site_gdf.centroid.plot(ax=ax, alpha=0.5, color='red')
        >>> kwplot.show_if_requested()

    Example:
        >>> # Demo case with almost zero noise
        >>> from geowatch.demo.metrics_demo.site_perterbing import *  # NOQA
        >>> rng = kwarray.ensure_rng(43240830)
        >>> _, sites, _ = demo_truth.random_region_model(num_observations=20, rng=rng)
        >>> site = sites[0]
        >>> pred_site = perterb_single_site_model(site, noise=1e-2, rng=rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:kwplot)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> # Draw our true and perterbed site model
        >>> orig_site_gdf = gpd.GeoDataFrame.from_features(site)
        >>> pred_site_gdf = gpd.GeoDataFrame.from_features(pred_site)
        >>> fig = kwplot.figure(doclf=True, fnum=2)
        >>> ax = fig.gca()
        >>> total_poly = pred_site_gdf['geometry'].unary_union.union(orig_site_gdf['geometry'].unary_union)
        >>> minx, miny, maxx, maxy = total_poly.bounds
        >>> ax.set_xlim(minx, maxx)
        >>> ax.set_ylim(miny, maxy)
        >>> pred_site_gdf.plot(ax=ax, alpha=0.5, color='orange', edgecolor='purple')
        >>> orig_site_gdf.plot(ax=ax, alpha=0.5, color='limegreen', edgecolor='black')
        >>> orig_site_gdf.centroid.plot(ax=ax, alpha=0.5, color='green')
        >>> pred_site_gdf.centroid.plot(ax=ax, alpha=0.5, color='red')
        >>> kwplot.show_if_requested()
    """

    perterber = PerterbModel(noise=noise, warp_noise=warp_noise,
                             performer_id=performer_id, rng=rng, **kwargs)
    pred_site = perterber.perterb_single_site(site, idx=idx)
    return pred_site
