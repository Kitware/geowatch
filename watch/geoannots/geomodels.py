"""
Geojson object oriented interface for region and site models
"""
import ubelt as ub
import geojson


class _FeatureCollectionMixin(ub.NiceRepr, geojson.FeatureCollection):
    type = 'FeatureCollection'

    def __nice__(self):
        return ub.urepr(self.info(), nl=2)

    def pandas(self):
        """
        Returns:
            geopandas.GeoDataFrame: the feature collection as data frame
        """
        import geopandas as gpd
        gdf = gpd.GeoDataFrame.from_features(self['features'])
        return gdf

    def deepcopy(self):
        import copy
        return copy.deepcopy(self)


class RegionModel(_FeatureCollectionMixin):
    """
    Wrapper around a geojson region model FeatureCollection

    Example:
        >>> from watch.geoannots.geomodels import *  # NOQA
        >>> self = RegionModel.random()
        >>> print(self)
        >>> self.validate()
    """

    def info(self):
        header = self.header
        prop = '<no region header>' if header is None else header['properties']
        info = {
            'num_site_summaries': len(list(self.site_summaries())),
            'properties': prop,
        }
        return info

    @property
    def header(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == 'region':
                return feat

    def site_summaries(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == 'site_summary':
                yield feat

    def pandas_summaries(self):
        """
        Returns:
            geopandas.GeoDataFrame: the site summaries as a data frame
        """
        import geopandas as gpd
        gdf = gpd.GeoDataFrame.from_features(list(self.site_summaries()))
        return gdf

    @classmethod
    def random(cls):
        from watch.demo.metrics_demo import demo_truth
        region, _, _ = demo_truth.random_region_model()
        # print('region = {}'.format(ub.urepr(region, nl=-1)))
        return cls(**region)

    @classmethod
    def coerce(self, data):
        if isinstance(data, RegionModel):
            return data
        elif isinstance(data, dict):
            return RegionModel(**data)
        else:
            raise TypeError

    def validate(self):
        feature_types = ub.dict_hist([f['properties']['type'] for f in self.features])
        assert feature_types.pop('region', 0) == 1
        assert set(feature_types).issubset({'site_summary'})
        import watch
        import jsonschema
        schema = watch.rc.registry.load_region_model_schema()
        jsonschema.validate(self, schema)

        header = self.header
        if header is not self.features[0]:
            raise AssertionError('Header should be the first feature')

        from dateutil.parser import parse
        dummy_start_date = '1970-01-01'  # hack, could be more robust here
        dummy_end_date = '2101-01-01'
        start_date = header['properties']['start_date'] or dummy_start_date
        end_date = header['properties']['end_date'] or dummy_end_date
        start_datetime = parse(start_date)
        end_datetime = parse(end_date)
        if end_datetime < start_datetime:
            raise AssertionError('bad date')


class SiteModel(_FeatureCollectionMixin):
    """
    Wrapper around a geojson site model FeatureCollection

    Example:
        >>> from watch.geoannots.geomodels import *  # NOQA
        >>> self = SiteModel.random()
        >>> print(self)
        >>> self.validate()
    """
    def info(self):
        header = self.header
        prop = '<no site header>' if header is None else header['properties']
        info = {
            'num_observations': len(list(self.observations())),
            'properties': prop,
        }
        return info

    @property
    def header(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == 'site':
                return feat

    def observations(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == 'observation':
                yield feat

    @classmethod
    def coerce(cls, data):
        if isinstance(data, SiteModel):
            return data
        elif isinstance(data, dict):
            return SiteModel(**data)
        else:
            raise TypeError

    @classmethod
    def random(cls):
        """
        """
        from watch.demo.metrics_demo import demo_truth
        _, sites, _ = demo_truth.random_region_model(num_sites=1)
        return cls(**sites[0])

    def as_summary(self):
        header = self.header
        if header is None:
            ...
        else:
            summary = header.copy()
            summary['properties'] = summary['properties'].copy()
            assert summary['properties']['type'] == 'site'
            summary['properties']['type'] = 'site_summary'
            return SiteSummary(**summary)

    @property
    def site_id(self):
        return self.header['properties']['site_id']

    @property
    def start_date(self):
        return self.header['properties']['start_date']

    @property
    def end_date(self):
        return self.header['properties']['end_date']

    @property
    def status(self):
        return self.header['properties']['status']

    def validate(self):
        header = self.header
        if header is not self.features[0]:
            raise AssertionError('Header should be the first feature')

        feature_types = ub.dict_hist([f['properties']['type'] for f in self.features])
        assert feature_types.pop('site', 0) == 1
        assert set(feature_types).issubset({'observation'})
        import watch
        import jsonschema
        schema = watch.rc.registry.load_site_model_schema()
        jsonschema.validate(self, schema)

        from dateutil.parser import parse
        dummy_start_date = '1970-01-01'  # hack, could be more robust here
        dummy_end_date = '2101-01-01'
        start_date = header['properties']['start_date'] or dummy_start_date
        end_date = header['properties']['end_date'] or dummy_end_date
        start_datetime = parse(start_date)
        end_datetime = parse(end_date)
        if end_datetime < start_datetime:
            raise AssertionError('bad date')


class _FeatureMixin(ub.NiceRepr, geojson.Feature):
    type = 'Feature'

    def __nice__(self):
        return ub.urepr(self.info(), nl=2)

    def info(self):
        info = {
            'properties': self['properties'],
        }
        return info


class Observation(_FeatureMixin):
    ...


class SiteSummary(_FeatureMixin):
    ...


class SiteHeader(_FeatureMixin):
    ...


class RegionHeader(_FeatureMixin):
    """
    A helper wrapper for the region model header features
    """

    @classmethod
    def coerce(cls, data):
        """
        Example:
            >>> data = RegionModel.random()
            >>> h1 = RegionHeader.coerce(data)
            >>> h2 = RegionHeader.coerce(data.header)
            >>> assert h1 == h2

            RegionHeader.coerce(orig_region_model)
        """
        if isinstance(data, cls):
            return data
        if isinstance(data, dict):
            if 'type' in data:
                if data['type'] == 'Feature':
                    return cls(**data)
                elif data['type'] == 'FeatureCollection':
                    return cls(**RegionModel(**data).header)
        raise TypeError(data)


# def _site_header_from_observations(observations, mgrs_code, site_id, status, summary_geom=None):
#     """
#     Consolodate site observations into a site summary
#     """
#     if summary_geom is None:
#         summary_geom = unary_union(
#             [kwimage.MultiPolygon.coerce(o["geometry"]).to_shapely() for o in observations]
#         ).convex_hull
#     start_date = observations[0]["properties"]["observation_date"]
#     end_date = observations[-1]["properties"]["observation_date"]
#     sitesum_props = {
#         "type": "site_summary",
#         "status": status,
#         "version": "2.0.1",
#         "site_id": site_id,
#         "mgrs": mgrs_code,
#         "start_date": start_date,
#         "end_date": end_date,
#         "score": 1,
#         "originator": "demo",
#         "model_content": "annotation",
#         "validated": "True",
#     }
#     site_summary = geojson.Feature(
#         properties=sitesum_props,
#         geometry=kwimage.Polygon.coerce(summary_geom).to_geojson(),
#     )
#     return site_summary


class ModelCollection(list):
    """
    A storage container for multiple site / region models
    """

    def validate(self, mode='process', workers=0):
        from watch.utils import util_progress
        pman = util_progress.ProgressManager()
        with pman:
            jobs = ub.JobPool(mode='process', max_workers=8)
            for s in pman.progiter(self, desc='submit validate models'):
                jobs.submit(s.validate)
            for job in pman.progiter(jobs.as_completed(), total=len(jobs), desc='collect validate models'):
                job.result()


class SiteModelCollection(ModelCollection):

    def as_region_model(self, region=None):
        """
        Convert a set of site models to a region model

        Args:
            region (RegonModel | RegionHeader):
                updates an existing region model with new site summaries
        """
        site_summaries = [s.as_summary() for s in self]

        if region is not None:
            region_header = RegionHeader.coerce(region)
        else:
            raise NotImplementedError
            region_header = geojson.Feature(
                properties={
                    "type": "region",
                    "region_id": None,
                    "version": "2.4.3",
                    "mgrs": None,
                    "start_date": None,
                    "end_date": None,
                    "originator": None,
                    "model_content": None,
                    "comments": "",
                },
                geometry=None,
            )

        region_geom = region_header['geometry']
        if region_geom is None:
            from shapely.ops import unary_union
            import shapely.geometry
            site_geoms = [shapely.geometry.shape(s['geometry']).buffer(0)
                          for s in site_summaries]
            region_geom = unary_union(site_geoms).convex_hull

        region_features = [region_header] + site_summaries
        region_model = RegionModel(features=region_features)
        return region_model
