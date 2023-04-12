"""
Geojson object oriented interface for region and site models

SeeAlso:
    ../rc/registry.py
"""
import ubelt as ub
import geopandas as gpd
import geojson
import jsonschema
import copy
from watch.utils import util_time
from watch.utils import util_progress


class _Model(ub.NiceRepr, geojson.FeatureCollection):
    type = 'FeatureCollection'
    _header_type = NotImplemented
    _body_type = NotImplemented

    def __nice__(self):
        return ub.urepr(self.info(), nl=2)

    def pandas(self):
        """
        Returns:
            geopandas.GeoDataFrame: the feature collection as data frame
        """
        gdf = gpd.GeoDataFrame.from_features(self['features'])
        return gdf

    def deepcopy(self):
        return copy.deepcopy(self)

    @classmethod
    def coerce(cls, data):
        if isinstance(data, cls):
            return data
        elif isinstance(data, dict):
            return cls.from_dict(data)
        elif isinstance(data, gpd.GeoDataFrame):
            return cls.from_dataframe(data)
        else:
            raise TypeError

    @classmethod
    def from_dataframe(cls, gdf):
        """
        Args:
            gdf (GeoDataFrame):
        """
        jsondct = gdf.__geo_interface__
        return cls(**jsondct)

    @classmethod
    def from_dict(cls, data):
        """
        Args:
            gdf (GeoDataFrame):
        """
        return cls(**data)

    @property
    def start_date(self):
        return util_time.coerce_datetime(self.header['properties']['start_date'])

    @property
    def end_date(self):
        return util_time.coerce_datetime(self.header['properties']['end_date'])

    def load_schema(self, strict=True):
        raise NotImplementedError('abstract')

    def body_features(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == self._body_type:
                yield feat

    @property
    def header(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == self._header_type:
                return feat

    def validate(self, strict=True):
        import rich
        header = self.header
        if header is not self.features[0]:
            raise AssertionError('Header should be the first feature')

        def print_validation_error_info(ex, depth=1):
            if ex.parent is not None:
                max_depth = print_validation_error_info(ex.parent, depth=depth + 1)
            else:
                max_depth = depth

            rich.print(f'[yellow] validation error depth = {depth} / {max_depth}')
            print('ex.__dict__ = {}'.format(ub.urepr(ex.__dict__, nl=3)))
            return depth

        feature_types = ub.dict_hist([f['properties']['type'] for f in self.features])
        assert feature_types.pop(self._header_type, 0) == 1
        assert set(feature_types).issubset({self._body_type})
        schema = self.load_schema(strict=strict)
        try:
            jsonschema.validate(self, schema)
        except jsonschema.ValidationError as e:
            ex = e
            rich.print('[red] JSON VALIDATION ERROR')
            print(f'self={self}')
            print_validation_error_info(ex)
            # ub.IndexableWalker(self)[ex.absolute_path]
            # ub.IndexableWalker(schema)[ex.schema_path]
            rich.print(ub.codeblock(
                '''
                [yellow] jsonschema validation notes:
                    * depsite our efforts, information to debug the issue may not be shown, double check your schema and instance manually.
                    * anyOf schemas may print the error, and not the part you intended to match.
                    * oneOf schemas may not explicitly say that you matched both.
                '''))
            rich.print('[red] JSON VALIDATION ERROR')
            raise

        start_date = self.start_date
        end_date = self.end_date
        if start_date is not None and end_date is not None:
            if end_date < start_date:
                raise AssertionError('bad date')


class RegionModel(_Model):
    """
    Wrapper around a geojson region model FeatureCollection

    Example:
        >>> from watch.geoannots.geomodels import *  # NOQA
        >>> self = RegionModel.random()
        >>> print(self)
        >>> self.validate(strict=False)

    """
    _header_type = 'region'
    _body_type = 'site_summary'

    def info(self):
        header = self.header
        prop = '<no region header>' if header is None else header['properties']
        info = {
            'num_site_summaries': len(list(self.site_summaries())),
            'properties': prop,
        }
        return info

    def load_schema(self, strict=True):
        import watch
        schema = watch.rc.registry.load_region_model_schema(strict=strict)
        return schema

    def site_summaries(self):
        yield from self.body_features()

    def pandas_summaries(self):
        """
        Returns:
            geopandas.GeoDataFrame: the site summaries as a data frame
        """
        gdf = gpd.GeoDataFrame.from_features(list(self.site_summaries()))
        return gdf

    @classmethod
    def random(cls):
        from watch.demo.metrics_demo import demo_truth
        region, _, _ = demo_truth.random_region_model()
        # print('region = {}'.format(ub.urepr(region, nl=-1)))
        return cls(**region)


class SiteModel(_Model):
    """
    Wrapper around a geojson site model FeatureCollection

    Example:
        >>> from watch.geoannots.geomodels import *  # NOQA
        >>> self = SiteModel.random()
        >>> print(self)
        >>> self.validate(strict=False)
    """
    _header_type = 'site'
    _body_type = 'observation'

    def info(self):
        header = self.header
        prop = '<no site header>' if header is None else header['properties']
        info = {
            'num_observations': len(list(self.observations())),
            'properties': prop,
        }
        return info

    def load_schema(self, strict=True):
        import watch
        schema = watch.rc.registry.load_site_model_schema(strict=strict)
        return schema

    @property
    def header(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == 'site':
                return feat

    def observations(self):
        yield from self.body_features()

    def pandas_observations(self):
        """
        Returns:
            geopandas.GeoDataFrame: the site summaries as a data frame
        """
        gdf = gpd.GeoDataFrame.from_features(list(self.observations()))
        return gdf

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
    def status(self):
        return self.header['properties']['status']

    def fix_geom(self):
        from shapely.geometry import shape
        from shapely.validation import make_valid
        from shapely.geometry import MultiPolygon
        for feat in self.features:
            geom = shape(feat['geometry'])
            if geom.geom_type in {'MultiPolygon', 'Polygon'}:
                make_valid(geom)
            else:
                geom = geom.buffer(3).convex_hull
                geom = MultiPolygon([geom])
            feat['geometry'] = geom.__geo_interface__

    def fixup(self):
        self.clamp_scores()
        # self.fix_geom()

    def clamp_scores(self):
        for feat in self.features:
            fprop = feat['properties']
            fprop['score'] = float(max(min(1, fprop['score']), 0))


class _Feature(ub.NiceRepr, geojson.Feature):
    type = 'Feature'

    def __nice__(self):
        return ub.urepr(self.info(), nl=2)

    def info(self):
        info = {
            'properties': self['properties'],
        }
        return info


class Observation(_Feature):
    ...


class SiteSummary(_Feature):
    ...


class SiteHeader(_Feature):
    ...


class RegionHeader(_Feature):
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

    def fixup(self):
        pman = util_progress.ProgressManager()
        with pman:
            for s in pman.progiter(self, desc='fixup'):
                s.fixup()

    def validate(self, mode='process', workers=0):
        import rich
        # pman = util_progress.ProgressManager(backend='progiter')
        pman = util_progress.ProgressManager()
        with pman:
            tries = 0
            while True:
                jobs = ub.JobPool(mode='process', max_workers=8 if tries == 0 else 0)
                for s in pman.progiter(self, desc='submit validate models'):
                    job = jobs.submit(s.validate)
                    job.s = s
                try:
                    for job in pman.progiter(jobs.as_completed(), total=len(jobs), desc='collect validate models'):
                        job.result()
                except Exception:
                    if tries > 0 or workers == 0:
                        raise Exception
                    tries += 1
                    rich.print('[red] ERROR: [yellow] Failed to validate: trying again with workers=0')
                else:
                    break


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
