"""
Geojson object oriented interface for region and site models.

This defines two classes ``SiteModel`` and ``RegionModel``, both of which
inherit from ``geojson.FeatureCollection``, so all geojson operations are
valid, but these classes contain extra convenience methods for loading,
dumping, manipulating, validating, and inspecting the data.

A non exhaustive list of convenience methods / properties of note are shared by
both site and region models are:

    * pandas - convert to a geopandas data frame

    * coerce_multiple - read multiple geojson files at once.

    * header - a quick way to access the singular header row (region for region models and site for site models).

    * body_features - any row that is not a header is a body feature (site_summaries for region models and observations for site models).

    * validate - checks the site/region model against the schema.

    * random - classmethod to make a random instance of the site / region model for testing

New region model specific convenience methods / properties are:

    * site_summaries
    * region_id
    * pandas_summaries
    * pandas_region

New site model specific convenience methods / properties are:

    * observations
    * pandas_observations
    * as_summary
    * region_id
    * site_id
    * status

SeeAlso:
    ../rc/registry.py

The following example illustrates how to read region / site models efficiently

Example:
    >>> # xdoctest: +REQUIRES(env:HAS_DVC)
    >>> import watch
    >>> dvc_data_dpath = watch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    >>> region_models_dpath = dvc_data_dpath / 'annotations/drop6/region_models'
    >>> site_models_dpath = dvc_data_dpath / 'annotations/drop6/site_models'
    >>> from watch.geoannots import geomodels
    >>> region_models = list(geomodels.RegionModel.coerce_multiple(region_models_dpath))
    >>> site_models = list(geomodels.SiteModel.coerce_multiple(site_models_dpath, workers=8))
    >>> print(f'Number of region models: {len(region_models)}')
    >>> print(f'Number of site models: {len(site_models)}')
    >>> # Quick demo of associating sites to regions
    >>> region_id_to_sites = ub.group_items(site_models, key=lambda s: s.header['properties']['region_id'])
    >>> region_id_to_num_sites = ub.udict(region_id_to_sites).map_values(len)
    >>> print('region_id_to_num_sites = {}'.format(ub.urepr(region_id_to_num_sites, nl=1)))
    >>> # It is also easy to convert these models to geopandas
    >>> region_model = region_models[0]
    >>> gdf = region_model.pandas()
    >>> print(gdf)
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

    def dumps(self, **kw):
        import json
        return json.dumps(self, **kw)

    @classmethod
    def coerce_multiple(cls, data, allow_raw=False, workers=0, mode='thread',
                        verbose=1, desc=None):
        """
        Load multiple geojson files

        Args:
            arg (str | PathLike | List[str | PathLike]):
                an argument that is coerceable to one or more geojson files.

            **kwargs: see :func:`util_gis.coerce_geojson_datas`

        Yields:
            Self
        """
        from watch.utils import util_gis
        infos = list(util_gis.coerce_geojson_datas(
            data, format='json', allow_raw=allow_raw, workers=workers,
            mode=mode, verbose=verbose, desc=desc))
        for info in infos:
            yield cls(**info['data'])

    @classmethod
    def coerce(cls, data):
        import os
        if isinstance(data, cls):
            return data
        elif isinstance(data, dict):
            return cls.from_dict(data)
        elif isinstance(data, list):
            if all(isinstance(d, dict) and d['type'] == 'Feature' for d in data):
                return cls.from_features(data)
            else:
                raise TypeError('lists must a list of Features')
            return cls.from_dict(data)
        elif isinstance(data, gpd.GeoDataFrame):
            return cls.from_dataframe(data)
        elif isinstance(data, (str, os.PathLike)):
            got = list(cls.coerce_multiple(data))
            assert len(got) == 1
            return got[0]
        else:
            raise TypeError

    @classmethod
    def from_features(cls, features):
        """
        Args:
            gdf (GeoDataFrame):
        """
        self = cls(features=features)
        return self

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

    def _validate_quick_checks(self):
        header = self.header
        if header is None:
            raise AssertionError('Geo Model has no header')

        if header is not self.features[0]:
            raise AssertionError('Header should be the first feature')

        if header['properties']['type'] != self._header_type:
            raise AssertionError('Header type is wrong')

        if self['type'] != 'FeatureCollection':
            raise AssertionError('GeoModels should be FeatureCollections')

        feature_types = ub.dict_hist([
            f['properties']['type'] for f in self.features])
        assert feature_types.pop(self._header_type, 0) == 1
        assert set(feature_types).issubset({self._body_type})

        start_date = self.start_date
        end_date = self.end_date
        if start_date is not None and end_date is not None:
            if end_date < start_date:
                raise AssertionError('bad date')

    def _validate_schema(self, strict=True):
        import rich

        def print_validation_error_info(ex, depth=1):
            if ex.parent is not None:
                max_depth = print_validation_error_info(ex.parent, depth=depth + 1)
            else:
                max_depth = depth
            rich.print(f'[yellow] error depth = {depth} / {max_depth}')
            print('ex.__dict__ = {}'.format(ub.urepr(ex.__dict__, nl=3)))
            return depth

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

    def validate(self, strict=True):
        self._validate_quick_checks()
        self._validate_schema(strict=strict)


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
        yield from (SiteSummary(**f) for f in self.body_features())

    @classmethod
    def coerce(cls, data):
        """
        Example:
            >>> from watch.geoannots.geomodels import *  # NOQA
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('watch/tests/geoannots/coerce').ensuredir()
            >>> region = RegionModel.random(with_sites=False, rng=0)
            >>> data = fpath = (dpath/ 'region.geojson')
            >>> fpath.write_text(region.dumps())
            >>> region_models = list(RegionModel.coerce_multiple(fpath))
            >>> region_model = RegionModel.coerce(fpath)
        """
        self = super().coerce(data)
        assert self.header['properties']['type'] == 'region'
        return self

    def pandas_summaries(self):
        """
        Returns:
            geopandas.GeoDataFrame: the site summaries as a data frame
        """
        gdf = gpd.GeoDataFrame.from_features(list(self.site_summaries()))
        return gdf

    def pandas_region(self):
        """
        Returns:
            geopandas.GeoDataFrame: the region header as a data frame
        """
        gdf = gpd.GeoDataFrame.from_features([self.header])
        return gdf

    @classmethod
    def random(cls, with_sites=False, **kwargs):
        """
        Args:
            with_sites (bool):
                also returns site models if True
            **kwargs : passed to :func:`demo_truth.random_region_model`

        Returns:
            RegionModel | Tuple[RegionModel, SiteModelCollection]

        Example:
            >>> from watch.geoannots.geomodels import *  # NOQA
            >>> region1 = RegionModel.random(with_sites=False, rng=0)
            >>> region2, sites2 = RegionModel.random(with_sites=True, rng=0)
            >>> assert region1 == region2, 'rngs should be the same'
        """
        from watch.demo.metrics_demo import demo_truth

        region, sites, _ = demo_truth.random_region_model(
            **kwargs, with_renderables=False)

        region = cls(**region)

        if with_sites:
            sites = SiteModelCollection([SiteModel(**s) for s in sites])
            return region, sites
        else:
            return region

    @property
    def region_id(self):
        return self.header['properties']['region_id']


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
    def random(cls, rng=None, **kwargs):
        """
        """
        from watch.demo.metrics_demo import demo_truth
        _, sites, _ = demo_truth.random_region_model(num_sites=1, rng=rng, **kwargs)
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
    def region_id(self):
        return self.header['properties']['region_id']

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


class _SiteOrSummary(_Feature):
    """
    Site summaries and site headers are nearly the same
    """

    @property
    def start_date(self):
        return util_time.coerce_datetime(self['properties']['start_date'])

    @property
    def end_date(self):
        return util_time.coerce_datetime(self['properties']['end_date'])

    @property
    def site_id(self):
        return self['properties']['site_id']


class SiteSummary(_SiteOrSummary):
    ...


class SiteHeader(_SiteOrSummary):
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
