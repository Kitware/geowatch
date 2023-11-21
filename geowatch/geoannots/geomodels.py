"""
Geojson object oriented interface for region and site models.

This defines two classes ``SiteModel`` and ``RegionModel``, both of which
inherit from ``geojson.FeatureCollection``, so all geojson operations are
valid, but these classes contain extra convenience methods for loading,
dumping, manipulating, validating, and inspecting the data.

A non exhaustive list of convenience methods / properties of note are shared by
both site and region models are:

    * dumps - convert to a geojson string

    * pandas - convert to a geopandas data frame

    * coerce_multiple - read multiple geojson files at once.

    * header - a quick way to access the singular header row (region for region models and site for site models).

    * body_features - any row that is not a header is a body feature (site_summaries for region models and observations for site models).

    * validate - checks the site/region model against the schema.

    * random - classmethod to make a random instance of the site / region model for testing


Official T&E Terminology:

A Region Model gives an overview of entire region and summarizes all sites it contains. It consists of:

* A single header feature with type="region" that defines the region spacetime bounds

* Multiple body features with type="site_summary" that correspond to the bounds of an entire site. (i.e. there is one for each site in the region). A site summary has a "status" that applies to the entire temporal range of the site. (i.e. positive, negative, ignore)

A Site Model gives a detailed account of a single site within a region. It consists of:

* A single header feature with type="site" that roughly corresponds to one of the "site_summary" features in the region model. It also contains the holistic "status" field.

* Multiple body features with type="observation". This represents a single keyframe at a single point in time within the site's activity sequence. It contains a "current_phase" label that describes the specific phase of an activity at that current point in time.


Note: A site summary may exist on its own (i.e. without a corresponding site model) that gives a rough overview with holistic status, rough spatial bounds and a start / end date.


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
    >>> import geowatch
    >>> dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    >>> region_models_dpath = dvc_data_dpath / 'annotations/drop6/region_models'
    >>> site_models_dpath = dvc_data_dpath / 'annotations/drop6/site_models'
    >>> from geowatch.geoannots import geomodels
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


XDEV_PROFILE=1 xdoctest ~/code/watch/geowatch/geoannots/geomodels.py


For testing the following example shows how to generate and inspect a random
site / region model.


Example:
    >>> from geowatch.geoannots.geomodels import *
    >>> # Generate a region model and also return its sites
    >>> region, sites = RegionModel.random(with_sites=True, rng=0)
    >>> # A region model consists of a region header
    >>> region_header = region.header
    >>> # And multiple site summaries. (We take the first one here)
    >>> site_summary = list(region.site_summaries())[0]
    >>> print('region_header.properties = {}'.format(ub.urepr(region_header['properties'], nl=1)))
    region_header.properties = {
        'type': 'region',
        'region_id': 'DR_R684',
        'version': '2.4.3',
        'mgrs': '51PXM',
        'start_date': '2011-05-28',
        'end_date': '2018-09-13',
        'originator': 'demo-truth',
        'model_content': 'annotation',
        'comments': 'demo-data',
    }
    >>> print('site_summary.properties = {}'.format(ub.urepr(site_summary['properties'], nl=1)))
    site_summary.properties = {
        'type': 'site_summary',
        'status': 'positive_annotated',
        'version': '2.0.1',
        'site_id': 'DR_R684_0000',
        'mgrs': '51PXM',
        'start_date': '2011-05-28',
        'end_date': '2018-09-13',
        'score': 1,
        'originator': 'demo',
        'model_content': 'annotation',
        'validated': 'True',
        'cache': {'color': [0.5511393746687864, 1.0, 0.0]},
    }
    >>> # A site model consists of a site header that roughly corresponds to a
    >>> # site summary in the region file
    >>> site = sites[0]
    >>> site_header = site.header
    >>> # It also contains one or more observations
    >>> site_obs = list(site.observations())[0]
    >>> print('site_header.properties = {}'.format(ub.urepr(site_header['properties'], nl=1)))
    site_header.properties = {
        'type': 'site',
        'status': 'positive_annotated',
        'version': '2.0.1',
        'site_id': 'DR_R684_0000',
        'mgrs': '51PXM',
        'start_date': '2011-05-28',
        'end_date': '2018-09-13',
        'score': 1,
        'originator': 'demo',
        'model_content': 'annotation',
        'validated': 'True',
        'cache': {'color': [0.5511393746687864, 1.0, 0.0]},
        'region_id': 'DR_R684',
    }
    >>> print('site_obs.properties = {}'.format(ub.urepr(site_obs['properties'], nl=1)))
    site_obs.properties = {
        'type': 'observation',
        'observation_date': '2011-05-28',
        'source': 'demosat-220110528T132754',
        'sensor_name': 'demosat-2',
        'current_phase': 'No Activity',
        'is_occluded': 'False',
        'is_site_boundary': 'True',
        'score': 1.0,
    }


"""
import ubelt as ub
import geopandas as gpd
import geojson
import jsonschema
import copy
import json
from kwutil import util_time
from kwutil import util_progress

_VALID_SITE_OBSERVATION_FIELDS = {"type",
                                  "observation_date",
                                  "source",
                                  "sensor_name",
                                  "current_phase",
                                  "score",
                                  "misc_info",
                                  "is_occluded",
                                  "is_site_boundary"}


class _Model(ub.NiceRepr, geojson.FeatureCollection):
    """
    A base class for :class:`RegionModel` and :class:`SiteModel`.

    Note that because this extends :class:`geojson.FeatureCollection`, this is
    a dictionary.
    """
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
        return json.dumps(self, **kw)

    @classmethod
    def coerce_multiple(cls, data, allow_raw=False, workers=0, mode='thread',
                        verbose=1, desc=None, parse_float=None):
        """
        Load multiple geojson files.

        Args:
            arg (str | PathLike | List[str | PathLike]):
                an argument that is coerceable to one or more geojson files.

            **kwargs: see :func:`util_gis.coerce_geojson_datas`

        Yields:
            Self

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> import ubelt as ub
            >>> #
            >>> ### Setup demo data
            >>> dpath = ub.Path.appdir('geowatch/tests/geoannots/coerce_multiple')
            >>> dpath.delete().ensuredir()
            >>> regions, sites = [], []
            >>> for i in range(3):
            >>>     rm, sms = RegionModel.random(with_sites=True, rng=i)
            >>>     regions.append(rm)
            >>>     sites.extend(sms)
            >>> region_dpath = (dpath / 'region_models').ensuredir()
            >>> site_dpath = (dpath / 'site_models').ensuredir()
            >>> for region in regions:
            >>>     region_fpath = region_dpath / f'{region.region_id}.geojson'
            >>>     region_fpath.write_text(region.dumps())
            >>> for site in sites:
            >>>     site_fpath = site_dpath / f'{site.site_id}.geojson'
            >>>     site_fpath.write_text(site.dumps())
            >>> #
            >>> # Test coercing from a directory
            >>> regions2 = list(RegionModel.coerce_multiple(region_dpath))
            >>> sites2 = list(SiteModel.coerce_multiple(site_dpath))
            >>> assert len(regions2) == len(regions)
            >>> assert len(sites2) == len(sites)
            >>> #
            >>> # Test coercing from a glob pattern
            >>> regions3 = list(RegionModel.coerce_multiple(region_dpath / (regions[0].region_id + '*')))
            >>> sites3 = list(SiteModel.coerce_multiple(site_dpath / ('*.geojson')))
            >>> assert len(regions3) == 1
            >>> assert len(sites3) == len(sites)
            >>> #
            >>> # Test coercing from existing data
            >>> # Broken
            >>> # regions4 = list(RegionModel.coerce_multiple(regions))
            >>> # sites4 = list(SiteModel.coerce_multiple(sites))
            >>> # assert len(regions4) == len(regions)
            >>> # assert len(sites4) == len(sites)

        """
        from geowatch.utils import util_gis
        infos = list(util_gis.coerce_geojson_datas(
            data, format='json', allow_raw=allow_raw, workers=workers,
            mode=mode, verbose=verbose, desc=desc, parse_float=parse_float))
        for info in infos:
            item = cls(**info['data'])
            # Can we enrich each item with the path it was read from without
            # breaking dumps?
            # if 'fpath' in info:
            #     item.fpath = info['fpath']
            yield item

    @classmethod
    def coerce(cls, data, parse_float=None):
        """
        Coerce a :class:`RegionModel` or :class:`SiteModel` from some input.
        """
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
            got = list(cls.coerce_multiple(data, parse_float=parse_float, verbose=0))
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

    @property
    def geometry(self):
        """
        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> RegionModel.random().geometry
            >>> SiteModel.random().geometry
        """
        from shapely import geometry
        return geometry.shape(self.header['geometry'])

    def load_schema(self, strict=True):
        raise NotImplementedError('abstract')

    def body_features(self):
        for feat in self['features']:
            prop = feat['properties']
            if prop['type'] == self._body_type:
                yield feat

    def strip_body_features(self):
        """
        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> self = RegionModel.random()
            >>> assert len(list(self.body_features())) > 0
            >>> self.strip_body_features()
            >>> assert len(list(self.body_features())) == 0
        """
        self['features'] = [self.header]

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
            import warnings
            warnings.warn('Header should be the first feature')

        if header['properties']['type'] != self._header_type:
            raise AssertionError('Header type is wrong')

        if self['type'] != 'FeatureCollection':
            raise AssertionError('GeoModels should be FeatureCollections')

        feature_types = ub.dict_hist([
            f['properties']['type'] for f in self.features])
        assert feature_types.pop(self._header_type, 0) == 1, 'Missing header'
        assert set(feature_types).issubset({self._body_type}), f'Unexpected feature types: {feature_types}'

        start_date = self.start_date
        end_date = self.end_date
        if start_date is not None and end_date is not None:
            if end_date < start_date:
                raise AssertionError(f'bad date: start_date={start_date} end_date={end_date}')

    def _validate_schema(self, strict=True, verbose=1, parts=True):
        schema = self.load_schema(strict=strict)
        try:
            jsonschema.validate(self, schema)
        except jsonschema.ValidationError as _full_ex:
            full_ex = _full_ex
            if verbose:
                print(f'self={self}')
                _report_jsonschema_error(full_ex)
            if parts:
                try:
                    self._validate_parts(strict=strict, verbose=verbose)
                except Exception as _part_ex:
                    part_ex = _part_ex
                    part_ex.full_ex = full_ex
                    raise part_ex
            raise full_ex

    def validate(self, strict=True, verbose=1, parts=True):
        """
        Validates that the model conforms to its schema and does a decent job
        of localizing where errors are.

        Args:
            strict (bool):
                if False, SMART-specific fields have their restrictions
                loosened. Defaults to True.

            verbose (bool):
                if True prints out extra information on an errors

            parts (bool):
                if True, attempts to determine what part of the data is causing
                the error.
        """
        self._validate_quick_checks()
        self._validate_schema(strict=strict, verbose=verbose, parts=parts)

    def _validate_parts(self, strict=True, verbose=1):
        """
        Runs jsonschema validation checks on each part of the feature
        collection independently to better localize where the errors are.

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> self = RegionModel.random(rng=0)
            >>> self._validate_parts(strict=False)
            >>> self = SiteModel.random(rng=0)
            >>> self._validate_parts(strict=False)
        """
        import jsonschema
        schema = ub.udict(self.load_schema(strict=strict))
        schema - {'properties', 'required', 'title', 'type'}
        defs = schema[chr(36) + 'defs']
        header_schema = schema | (defs[self._header_type + '_feature'])
        body_schema = schema | (defs[self._body_type + '_feature'])
        try:
            jsonschema.validate(self.header, header_schema)
        except jsonschema.ValidationError as e:
            if verbose:
                _report_jsonschema_error(e)
            raise
        for obs_feature in self.body_features():
            try:
                jsonschema.validate(obs_feature, body_schema)
            except jsonschema.ValidationError as e:
                if verbose:
                    _report_jsonschema_error(e)
                raise

    def _update_cache_key(self):
        """
        Ensure we are using the up to date schema cache.

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> self = RegionModel.random(rng=0)
            >>> feat = list(self.site_summaries())[0]
            >>> self._update_cache_key()
            >>> assert 'annotation_cache' not in feat['properties']
            >>> feat['properties']['annotation_cache'] = {'foo': 'bar'}
            >>> self._update_cache_key()
            >>> # An old cache key, updates the new one.
            >>> assert 'cache' in feat['properties']
            >>> assert feat['properties']['cache']['foo'] == 'bar'
            >>> # But it wont overwrite.
            >>> feat['properties']['annotation_cache'] = {'foo': 'baz'}
            >>> self._update_cache_key()
            >>> assert 'cache' in feat['properties']
            >>> assert feat['properties']['cache']['foo'] == 'bar'

            self.header['properties']['cache'] = None
            self.fixup()
            self.validate(strict=0)
            assert self.header['properties']['cache'] == {}
        """
        for feat in self['features']:
            prop = feat['properties']
            _update_propery_cache(prop)

    def ensure_isodates(self):
        """
        Ensure that dates are provided as dates and not datetimes

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> region = RegionModel.random()
            >>> region.header['properties']['start_date'] = '1970-01-01T000000'
            >>> region.ensure_isodates()
            >>> assert region.header['properties']['start_date'] == '1970-01-01'
        """
        date_keys = ['start_date', 'end_date']
        for feat in self['features']:
            props = feat['properties']
            for key in date_keys:
                if key in props:
                    oldval = props[key]
                    if oldval is not None:
                        dt = util_time.coerce_datetime(oldval)
                        try:
                            newval = dt.date().isoformat()
                        except Exception:
                            print('ERROR: oldval = {}'.format(ub.urepr(oldval, nl=1)))
                        props[key] = newval

    def fix_backwards_dates(self):
        """
        If start and end dates are backwards, flip them.

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> ss = SiteSummary.random()
            >>> ss['properties']['start_date'] = '1970-01-01T000000'
            >>> ss.ensure_isodates()
            >>> assert ss['properties']['start_date'] == '1970-01-01'
        """
        start_date = self.start_date
        end_date = self.end_date
        if start_date is not None and end_date is not None:
            if end_date < start_date:
                _new_start = self.header['properties']['end_date']
                _new_end = self.header['properties']['start_date']
                self.header['properties']['start_date'] = _new_start
                self.header['properties']['end_date'] = _new_end

    @property
    def model_type(self):
        return self.header['properties']['type']

    @property
    def model_id(self):
        header_id_key = self._header_type + '_id'
        return self.header['properties'][header_id_key]


def _report_jsonschema_error(ex):
    import rich

    def print_validation_error_info(ex, depth=1):
        if ex.parent is not None:
            max_depth = print_validation_error_info(ex.parent, depth=depth + 1)
        else:
            max_depth = depth
        rich.print(f'[yellow] error depth = {depth} / {max_depth}')
        print('ex.__dict__ = {}'.format(ub.urepr(ex.__dict__, nl=3)))
        return depth

    rich.print('[red] JSON VALIDATION ERROR')
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


class RegionModel(_Model):
    """
    Wrapper around a geojson region model FeatureCollection

    TODO:
        Rename to Region?

    Example:
        >>> from geowatch.geoannots.geomodels import *  # NOQA
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

    @classmethod
    def load_schema(cls, strict=True):
        import geowatch
        schema = geowatch.rc.registry.load_region_model_schema(strict=strict)
        return schema

    def site_summaries(self):
        yield from (SiteSummary(**f) for f in self.body_features())

    @classmethod
    def coerce(cls, data, parse_float=None):
        """
        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('geowatch/tests/geoannots/coerce').ensuredir()
            >>> region = RegionModel.random(with_sites=False, rng=0)
            >>> data = fpath = (dpath/ 'region.geojson')
            >>> fpath.write_text(region.dumps())
            >>> region_models = list(RegionModel.coerce_multiple(fpath))
            >>> region_model = RegionModel.coerce(fpath)
        """
        self = super().coerce(data, parse_float=parse_float)
        assert self.header['properties']['type'] == 'region'
        return self

    def pandas_summaries(self):
        """
        Returns:
            geopandas.GeoDataFrame: the site summaries as a data frame

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> self = RegionModel.random()
            >>> gdf = self.pandas_summaries()
            >>> print(gdf)
            >>> # Test empty pandas summary
            >>> self = RegionModel.random(num_sites=0)
            >>> gdf = self.pandas_summaries()
            >>> print(gdf)
            >>> assert len(gdf) == 0
        """
        from geowatch.utils import util_gis
        crs84 = util_gis.get_crs84()
        site_summaries = list(self.site_summaries())
        if len(site_summaries):
            gdf = gpd.GeoDataFrame.from_features(site_summaries, crs=crs84)
        else:
            # TODO: could infer more columns here.
            gdf = gpd.GeoDataFrame.from_features(
                [], crs=crs84, columns=['geometry'])
        return gdf

    def pandas_region(self):
        """
        Returns:
            geopandas.GeoDataFrame: the region header as a data frame

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> self = RegionModel.random()
            >>> print(self.pandas_region())
        """
        from geowatch.utils import util_gis
        crs84 = util_gis.get_crs84()
        gdf = gpd.GeoDataFrame.from_features([self.header], crs=crs84)
        return gdf

    @classmethod
    def random(cls, with_sites=False, **kwargs):
        """
        Args:
            with_sites (bool):
                also returns site models if True

            **kwargs :
                passed to
                :func:`geowatch.demo.metrics_demo.demo_truth.random_region_model`.
                Some of these args are:
                    num_sites
                    num_observations
                    start_time
                    end_time
                    region_poly
                    rng

        Returns:
            RegionModel | Tuple[RegionModel, SiteModelCollection]

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> region1 = RegionModel.random(with_sites=False, rng=0)
            >>> region2, sites2 = RegionModel.random(with_sites=True, rng=0)
            >>> assert region1 == region2, 'rngs should be the same'
        """
        from geowatch.demo.metrics_demo import demo_truth

        region, sites, _ = demo_truth.random_region_model(
            **kwargs, with_renderables=False)

        region = cls(**region)

        if with_sites:
            sites = SiteModelCollection([SiteModel(**s) for s in sites])
            return region, sites
        else:
            return region

    def add_site_summary(self, summary):
        """
        Add a site summary to the region.

        Args:
            summary (SiteSummary | SiteModel):
                a site summary or site model. If given as a site model
                it is converted to a site summary and then added.

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> region = RegionModel.random(num_sites=False)
            >>> site1 = SiteModel.random(region=region)
            >>> site2 = SiteModel.random(region=region)
            >>> site3 = SiteModel.random(region=region)
            >>> summary = site2.as_summary()
            >>> region.add_site_summary(site1)
            >>> region.add_site_summary(summary)
            >>> region.add_site_summary(dict(site3.as_summary()))
            >>> import pytest
            >>> with pytest.raises(TypeError):
            ...     region.add_site_summary(dict(site3))
            >>> assert len(list(region.site_summaries())) == 3
        """
        if isinstance(summary, SiteModel):
            summary = summary.as_summary()
        if summary['type'] != 'Feature' or summary['properties']['type'] != 'site_summary':
            raise TypeError('Input was not a site summary or coercable type')
        self['features'].append(summary)

    @property
    def region_id(self):
        return self.header['properties']['region_id']

    def fixup(self):
        """
        Fix common issues with this region model

        Returns:
            RegionModel
        """
        self._update_cache_key()
        self.remove_invalid_properties()
        self.ensure_isodates()
        self.fix_backwards_dates()
        return self

    def remove_invalid_properties(self):
        """
        Remove invalid properties from this region model
        """
        props = self.header['properties']
        bad_region_header_properties = ['validated', 'score', 'site_id', 'status', 'socre']
        for key in bad_region_header_properties:
            props.pop(key, None)

        bad_sitesum_features = ['region_id', 'validate', 'validated',
                                'predicted_phase_transition',
                                'predicted_phase_transition_date']
        for sitesum in self.body_features():
            siteprops = sitesum['properties']
            for key in bad_sitesum_features:
                siteprops.pop(key, None)

    def ensure_comments(self):
        props = self.header['properties']

        props['comments'] = props.get('comments', '')

    def infer_header(self, region_header=None):
        """
        Infer any missing header information from site summaries.

        If this region model does not have a header, but it contains site
        summaries, then use that information to infer a header value. Useful
        when converting site summaries to full region models.

        Args:
            region_header (RegionHeader):
                if specified, use this information when possible and then
                infer the rest.

        SeeAlso:

            * :func:`SiteModelCollection.as_region_model`

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> # Make a region without a header
            >>> self = RegionModel.random()
            >>> self.features.remove(self.header)
            >>> assert self.header is None
            >>> # Infer the header using site summaries
            >>> self.infer_header()
            >>> assert self.header is not None
        """
        current_header = self.header

        if region_header is not None:
            if current_header is not None:
                raise ValueError('cannot specify a region header when one already exists')
            region_header = RegionHeader.coerce(region_header)
            region_header = copy.deepcopy(region_header)
        else:
            if current_header is not None:
                region_header = current_header
            else:
                region_header = RegionHeader.empty()

        site_summaries = list(self.site_summaries())
        region_header = _infer_region_header_from_site_summaries(
            region_header, site_summaries)

        if region_header is not current_header:
            assert current_header is None
            self.features.insert(0, region_header)


class SiteModel(_Model):
    """
    Wrapper around a geojson site model FeatureCollection

    TODO:
        Rename to Site?

    Example:
        >>> from geowatch.geoannots.geomodels import *  # NOQA
        >>> self = SiteModel.random()
        >>> print(self)
        >>> self.validate(strict=False)
    """
    _header_type = 'site'
    _body_type = 'observation'

    def info(self):
        header = self.header
        prop = '<no site header>' if header is None else header['properties']
        info = {}
        info['num_observations'] = len(list(self.observations()))
        if header is not None:
            info['header_geom_type'] = header['geometry']['type']
        info['properties'] = prop
        return info

    @classmethod
    def load_schema(cls, strict=True):
        import geowatch
        schema = geowatch.rc.registry.load_site_model_schema(strict=strict)
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

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> self = SiteModel.random()
            >>> gdf = self.pandas_observations()
            >>> print(gdf)
            >>> # Test empty pandas summary
            >>> del self.features[1:]
            >>> gdf = self.pandas_observations()
            >>> print(gdf)
            >>> assert len(gdf) == 0
        """
        from geowatch.utils import util_gis
        crs84 = util_gis.get_crs84()
        features = list(self.observations())
        if len(features):
            gdf = gpd.GeoDataFrame.from_features(features, crs=crs84)
        else:
            gdf = gpd.GeoDataFrame.from_features(features, crs=crs84,
                                                 columns=['geometry'])
        return gdf

    def pandas_site(self):
        """
        Returns:
            geopandas.GeoDataFrame: the region header as a data frame

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> self = SiteModel.random()
            >>> print(self.pandas_site())
        """
        from geowatch.utils import util_gis
        crs84 = util_gis.get_crs84()
        gdf = gpd.GeoDataFrame.from_features([self.header], crs=crs84)
        return gdf

    @classmethod
    def random(cls, rng=None, region=None, site_poly=None, **kwargs):
        """
        Args:
            rng (int | str | RandomState | None) :
                seed or random number generator

            region (RegionModel | None):
                if specified generate a new site in this region model.
                (This will overwrite some of the kwargs).

            site_poly (kwimage.Polygon | shapely.geometry.Polygon | None):
                if specified, this polygon is used as the geometry for new site
                models. Note: all site models will get this geometry, so
                typically this is only used when num_sites=1.

            **kwargs :
                passed to :func:`geowatch.demo.metrics_demo.demo_truth.random_region_model`.

        Returns:
            SiteModel

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> region1 = RegionModel.random(with_sites=False, rng=0)
            >>> region2, sites2 = RegionModel.random(with_sites=True, rng=0)
            >>> assert region1 == region2, 'rngs should be the same'

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> region = RegionModel.random(with_sites=False, rng=0)
            >>> site = SiteModel.random(region=region)
            >>> assert region.region_id == site.region_id

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> import kwimage
            >>> region = RegionModel.random(with_sites=False, rng=0)
            >>> # Test specification of the site geometry.
            >>> site_poly = kwimage.Polygon.coerce(region.geometry)
            >>> site = SiteModel.random(region=region, site_poly=site_poly)
            >>> assert abs(region.geometry.area - site.geometry.area) < 1e-7
            >>> site = SiteModel.random(region=region, site_poly=site_poly.scale(10))
            >>> assert abs(region.geometry.area - site.geometry.area) > 1e-7
        """
        from geowatch.demo.metrics_demo import demo_truth
        kwargs.setdefault('with_renderables', False)
        kwargs['site_poly'] = site_poly
        if region is not None:
            kwargs['region_poly'] = region.header.geometry
            kwargs['region_id'] = region.region_id
        _, sites, _ = demo_truth.random_region_model(num_sites=1, rng=rng, **kwargs)
        return cls(**sites[0])

    def as_summary(self):
        """
        Modify and return this site header feature as a site-summary body
        feature for a region model.

        Returns:
            SiteSummary
        """
        header = self.header
        if header is None:
            raise IndexError('Site model has no header')
        else:
            header = SiteHeader(**header)
            summary = header.as_summary()
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

    def fix_sensor_names(self):
        for feat in self.observations():
            prop = feat['properties']
            if prop.get('sensor_name') == 'WorldView 1':
                prop['sensor_name'] = 'WorldView'

    def fix_current_phase_salient(self):
        for feat in self.observations():
            prop = feat['properties']
            current_phase = prop.get('current_phase')
            if current_phase is not None:
                if 'salient' in current_phase:
                    prop['current_phase'] = prop['current_phase'].replace(
                        'salient', 'Active Construction')

    def fixup(self):
        """
        Fix common issues with this site model

        Returns:
            SiteModel
        """
        self._update_cache_key()
        self.clamp_scores()
        self.fix_sensor_names()
        self.ensure_isodates()
        self.fix_current_phase_salient()
        self.fix_backwards_dates()
        self.fix_old_schema_properties()
        # self.fix_geom()
        return self

    def fix_old_schema_properties(self):
        """
        If an old schema property exists and is not null, move it to
        the cache.
        """
        old_keys = ['comments']
        for feat in self.features:
            props = feat['properties']
            for key in old_keys:
                if key in props:
                    old_value = props.pop(key)
                    if old_value is not None:
                        if 'cache' not in props:
                            props['cache'] = {}
                        # Dont overwrite an existing key with the same name
                        # in this case we just drop the bad value
                        if key not in props['cache']:
                            props['cache'][key] = old_value

    def ensure_isodates(self):
        """
        Ensure that dates are provided as dates and not datetimes

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> site = SiteModel.random()
            >>> # Set props as datetimes
            >>> site.header['properties']['start_date'] = '1970-01-01T000000'
            >>> site.features[1]['properties']['observation_date'] = '1970-01-01T000000'
            >>> site.ensure_isodates()
            >>> # The fixup ensure dates
            >>> assert site.features[1]['properties']['observation_date'] == '1970-01-01'
            >>> assert site.header['properties']['start_date'] == '1970-01-01'
        """
        date_keys = ['start_date', 'end_date', 'predicted_phase_transition_date']
        feat = self.header
        props = feat['properties']
        for key in date_keys:
            if key in props:
                old_val = props[key]
                if old_val is not None:
                    props[key] = util_time.coerce_datetime(old_val).date().isoformat()

        date_keys = ['observation_date']
        for feat in self.body_features():
            props = feat['properties']
            for key in date_keys:
                if key in props:
                    old_val = props[key]
                    if old_val is not None:
                        props[key] = util_time.coerce_datetime(old_val).date().isoformat()

    def clamp_scores(self):
        for feat in self.features:
            fprop = feat['properties']
            fprop['score'] = float(max(min(1, fprop['score']), 0))

    def remove_invalid_properties(self):
        """
        Remove invalid properties from this site model
        """
        # T&E site schema no longer allows extraneous keys to be
        # included in region / site models; removing all unsupported
        # keys (could consider putting in 'misc_info' rather than
        # deleting, though not clear if 'misc_info' will be supported
        # in the future)
        for obs in self.observations:
            oprop = obs['properties']

            to_remove = set()
            for k in oprop.keys():
                if k not in _VALID_SITE_OBSERVATION_FIELDS:
                    to_remove.add(k)

            for k in to_remove:
                del oprop[k]

    def _manual_validation(self):
        """
        Hard coded checks. The jsonschema is pretty bad at identifing where
        errors are, so these are some hard coded checks that hit some simple
        errors we have seen before.
        """
        features = self.features
        if len(features) < 2:
            raise AssertionError('should have at least two features')

        type_to_expected_fields = {
            'feature': {
                'required': {'type', 'properties', 'geometry'},
                'optional': set(),
            },
            'site': {
                'required': {
                    'type', 'site_id', 'region_id', 'version', 'mgrs', 'model_content',
                    'start_date', 'end_date', 'status', 'originator'},
                'optional': {
                    'misc_info', 'validated', 'score',
                    'predicted_phase_transition_date',
                    'predicted_phase_transition'
                }
            },
            'observation': {
                'required': {
                    'type', 'observation_date', 'source', 'sensor_name',
                    'current_phase', 'is_occluded', 'is_site_boundary'
                },
                'optional': {
                    'misc_info', 'score',
                }
            }
        }

        type_to_expected_geoms = {
            'site': {'Polygon'},
            'observation': {'Polygon', 'MultiPolygon'},
        }

        def check_expected_fields(have, type):
            expected = type_to_expected_fields[type]
            missing = expected['required'] - have
            extra = have - (expected['required'] | expected['optional'])
            if extra:
                yield {
                    'msg': f'Extra fields: {extra}'
                }
            if missing:
                yield {
                    'msg': f'Missing fields: {missing}'
                }
            return errors

        def check_expected_geom(geom, type):
            allowed_types = type_to_expected_geoms[type]
            if geom.geom_type not in allowed_types:
                yield {
                    'msg': f'{type} must be in {allowed_types}: got {geom.geom_type}'
                }

        from shapely.geometry import shape
        errors = []
        for feat in features:
            have = set(feat.keys())
            errors += list(check_expected_fields(have, type='feature'))
            geom = shape(feat['geometry'])
            props = feat['properties']
            proptype = props['type']
            if proptype == 'site':
                have = set(props.keys())
                errors += list(check_expected_fields(have, type='site'))
                errors += list(check_expected_geom(geom, type='site'))
            elif proptype == 'observation':
                have = set(props.keys())
                errors += list(check_expected_fields(have, type='observation'))
                errors += list(check_expected_geom(geom, type='observation'))
            else:
                errors += {
                    'msg': f'Unknown site type: {proptype}',
                }

        if len(errors):
            print('errors = {}'.format(ub.urepr(errors, nl=1)))
            raise AssertionError


class _Feature(ub.NiceRepr, geojson.Feature):
    """
    Base class for geojson features that conform to an IARPA geomodel spec

    Example:
        >>> # Test the class variables for subclasses are defined correctly
        >>> assert RegionHeader._feat_type == 'region'
        >>> assert SiteSummary._feat_type == 'site_summary'
        >>> assert SiteHeader._feat_type == 'site'
        >>> assert Observation._feat_type == 'observation'
        >>> assert RegionHeader._model_cls is RegionModel
        >>> assert SiteSummary._model_cls is RegionModel
        >>> assert SiteHeader._model_cls is SiteModel
        >>> assert Observation._model_cls is SiteModel
    """
    type = 'Feature'
    _model_cls = NotImplemented
    _feat_type = NotImplemented

    def __nice__(self):
        return ub.urepr(self.info(), nl=2)

    def info(self):
        info = {
            'properties': self['properties'],
        }
        return info

    @property
    def properties(self):
        return self['properties']

    @classmethod
    def load_schema(cls, strict=True):
        """
        Return the sub-schema for the approprite header / body feature
        based on the declaration of _model_cls and _feat_type
        """
        assert cls._model_cls is not NotImplemented
        assert cls._feat_type is not NotImplemented
        region_schema = cls._model_cls.load_schema(strict=strict)
        schema = ub.udict(region_schema)
        schema - {'properties', 'required', 'title', 'type'}
        defs = schema[chr(36) + 'defs']
        feat_schema = schema | (defs[cls._feat_type + '_feature'])
        return feat_schema

    def validate(self, strict=True, verbose=1):
        """
        Validate this sub-schema
        """
        feat_schema = self.load_schema(strict=strict)
        try:
            jsonschema.validate(self, feat_schema)
        except jsonschema.ValidationError as e:
            if verbose:
                _report_jsonschema_error(e)
            raise

    def ensure_isodates(self):
        """
        Ensure that dates are provided as dates and not datetimes

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> ss = SiteSummary.random()
            >>> ss['properties']['start_date'] = '1970-01-01T000000'
            >>> ss.ensure_isodates()
            >>> assert ss['properties']['start_date'] == '1970-01-01'
        """
        date_keys = ['start_date', 'end_date']
        props = self['properties']
        for key in date_keys:
            if key in props:
                oldval = props[key]
                if oldval is not None:
                    dt = util_time.coerce_datetime(oldval)
                    try:
                        newval = dt.date().isoformat()
                    except Exception:
                        print('ERROR: oldval = {}'.format(ub.urepr(oldval, nl=1)))
                    props[key] = newval

    def infer_mgrs(self, strict=True):
        """

        Args:
            strict (bool): if False, do not error if this fails

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> ss = SiteSummary.random()
            >>> ss.infer_mgrs()
        """

        from shapely.geometry import shape
        import mgrs
        if self.geometry is None:
            handle_error('Cannot infer mgrs, missing geometry',
                         extype=Exception, strict=strict)
        else:
            _geom = shape(self.geometry)
            lon = _geom.centroid.xy[0][0]
            lat = _geom.centroid.xy[1][0]
            mgrs_code = mgrs.MGRS().toMGRS(lat, lon, MGRSPrecision=0)
            self.properties['mgrs'] = mgrs_code
        return self


class _SiteOrSummaryMixin:
    """
    Site summaries and site headers are nearly the same
    """

    # Data for conversion between site / site-summaries
    _cache_keys = {
        'site_summary': 'annotation_cache',
        'site': 'misc_info',
    }
    # Record non-common properties between the two similar schemas
    _only_properties = {
        'site_summary': [
            'comments'
        ],
        'site': [
            'predicted_phase_transition_date',
            'predicted_phase_transition',
            'region_id',
        ]
    }

    @property
    def start_date(self):
        return util_time.coerce_datetime(self['properties']['start_date'])

    @property
    def end_date(self):
        return util_time.coerce_datetime(self['properties']['end_date'])

    @property
    def site_id(self):
        return self['properties']['site_id']

    def _update_cache_key(self):
        """
        Ensure we are using the up to date schema cache.
        """
        prop = self['properties']
        _update_propery_cache(prop)

    def _convert(self, new_cls):
        """
        Common logic for converting site <-> site_summary

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> site = SiteModel.random()
            >>> site.validate(strict=False)
            >>> region = RegionModel.random()
            >>> region.validate(strict=False)
            >>> site1 = SiteHeader(**site.header)
            >>> site1.validate(strict=False)
            >>> summary1 = SiteSummary(**ub.peek(region.body_features()))
            >>> summary1.validate(strict=False)
            >>> summary2 = site1.as_summary()
            >>> summary2.validate(strict=False)
            >>> import pytest
            >>> with pytest.raises(Exception):
            >>>     site2 = summary1.as_site()
            >>> summary1['properties']['cache']['region_id'] = region.region_id
            >>> site2 = summary1.as_site()
            >>> site2.validate(strict=False)
            >>> # Check the round-trip conversion
            >>> summary3 = site2.as_summary()
            >>> site3 = summary2.as_site()
            >>> summary1_ = SiteSummary(**summary1.copy())
            >>> summary1_._update_cache_key()
            >>> site1_ = SiteHeader(**site1.copy())
            >>> site1_._update_cache_key()
            >>> assert summary3 == summary1_ and summary3 is not summary1
            >>> assert site3 == site1_ and site3 is not site1
            >>> # Revalidate everything to ensure no memory issues happened
            >>> summary3.validate(strict=0)
            >>> summary2.validate(strict=0)
            >>> summary1.validate(strict=0)
            >>> site3.validate(strict=0)
            >>> site2.validate(strict=0)
            >>> site1.validate(strict=0)
            >>> site.validate(strict=0)
            >>> region.validate(strict=0)
        """
        old_type = self._feat_type
        new_type = new_cls._feat_type
        old_cache_key = self._cache_keys[old_type]
        old_only_props = self._only_properties[old_type]
        new_cache_key  = self._cache_keys[new_type]
        new_only_props  = self._only_properties[new_type]

        feat = self.copy()
        props = feat['properties'].copy()

        if 1:
            # Use new scheme
            _update_propery_cache(props)
            old_cache_key = 'cache'
            new_cache_key = 'cache'

        feat['properties'] = props
        assert props['type'] == old_type
        props['type'] = new_type
        if old_cache_key in props:
            props[new_cache_key] = props.pop(old_cache_key)
        cache = props.get(new_cache_key, {})
        for key in new_only_props:
            if key in cache:
                props[key] = cache.pop(key)
        for key in old_only_props:
            if key in props:
                cache[key] = props.pop(key)
        if cache:
            props[new_cache_key] = cache

        if old_type == 'site_summary':
            if 'region_id' not in props:
                raise Exception(ub.paragraph(
                    '''
                    Cannot convert a site-summary to a site header when the
                    region-id is unknown. As a workaround you can set the
                    .properties.annotation_cache.region_id
                    '''))

        new = new_cls(**feat)
        return new


class RegionHeader(_Feature):
    """
    The region header feature of a region model.
    """
    _model_cls = RegionModel
    _feat_type = RegionModel._header_type

    @classmethod
    def empty(cls):
        """
        Create an empty region header
        """
        self = cls(
            properties={
                "type": "region",
                "region_id": None,
                "version": "2.4.3",
                "mgrs": None,
                "start_date": None,
                "end_date": None,
                "originator": None,
                "model_content": None,
            },
            geometry=None,
        )
        return self

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

    def ensure_isodates(self):
        date_keys = ['start_date', 'end_date', 'predicted_phase_transition_date']
        feat = self
        props = feat['properties']
        for key in date_keys:
            if key in props:
                old_val = props[key]
                if old_val is not None:
                    props[key] = util_time.coerce_datetime(old_val).date().isoformat()
        return self


class SiteSummary(_Feature, _SiteOrSummaryMixin):
    """
    The site-summary body feature of a region model.
    """
    _model_cls = RegionModel
    _feat_type = RegionModel._body_type

    @classmethod
    def from_geopandas_frame(cls, df, drop_id=True):
        json_text = df.to_json(drop_id=drop_id)
        json_data = json.loads(json_text)
        for feat in json_data['features']:
            if feat['properties']['type'] == 'site_summary':
                yield cls(**feat)

    def as_site(self):
        """
        Modify and return this site summary feature as a site header feature
        for a site model.

        Returns:
            SiteHeader

        Example:
            >>> # Convert a RegionModel to a collection of SiteModels
            >>> from geowatch.geoannots import geomodels
            >>> region = geomodels.RegionModel.random()
            >>> sites = []
            >>> for sitesum in region.site_summaries():
            >>>     # Current hacky way to pass along region ids
            >>>     sitesum['properties']['cache']['region_id'] = region.region_id
            >>>     # This only produces a site header, we may need to add
            >>>     # observations to the site model itself as well
            >>>     site_header = sitesum.as_site()
            >>>     site = SiteModel(features=[site_header])
            >>>     sites.append(site)
        """
        new_cls = SiteHeader
        return self._convert(new_cls)

    def fixup(self):
        """
        Fixup the site summary
        """
        self._update_cache_key()
        # self.ensure_isodates()
        return self

    @classmethod
    def coerce(cls, data):
        if isinstance(data, cls):
            self = data
        elif isinstance(data, dict):
            assert data['type'] == 'Feature'
            assert data['properties']['type'] == 'site_summary'
            self = cls(**data)
        else:
            raise TypeError(type(data))
        return self

    @classmethod
    def random(cls, rng=None, region=None, site_poly=None, **kwargs):
        """
        Args:
            rng (int | str | RandomState | None) :
                seed or random number generator

            region (RegionModel | None):
                if specified generate a new site in this region model.
                (This will overwrite some of the kwargs).

            site_poly (kwimage.Polygon | shapely.geometry.Polygon | None):
                if specified, this polygon is used as the geometry for new site
                models. Note: all site models will get this geometry, so
                typically this is only used when num_sites=1.

            **kwargs :
                passed to :func:`geowatch.demo.metrics_demo.demo_truth.random_region_model`.

        Returns:
            SiteSummary

        Example:
            >>> from geowatch.geoannots.geomodels import *  # NOQA
            >>> sitesum = SiteSummary.random(rng=0)
            >>> print('sitesum = {}'.format(ub.urepr(sitesum, nl=2)))
        """
        site = SiteModel.random(rng=rng, region=region, site_poly=site_poly, **kwargs)
        return site.as_summary()


class SiteHeader(_Feature, _SiteOrSummaryMixin):
    """
    The site header feature of a site model.
    """
    _model_cls = SiteModel
    _feat_type = SiteModel._header_type

    @classmethod
    def empty(cls):
        """
        Create an empty region header

        Example:
            from geowatch.geoannots.geomodels import *  # NOQA
            self = SiteHeader.empty()
            ...
        """
        self = cls(
            properties={
                "type": "site",
                "version": "2.4.3",
                "mgrs": None,
                "status": None,
                "model_content": None,
                "score": None,
                "start_date": None,
                "end_date": None,
                "originator": None,
                "validated": 'False',
            },
            geometry=None,
        )
        return self

    def as_summary(self):
        """
        Modify and return this site header feature as a site-summary body
        feature for a region model.

        Returns:
            SiteSummary
        """
        new_cls = SiteSummary
        return self._convert(new_cls)

    @classmethod
    def coerce(cls, data):
        if isinstance(data, cls):
            self = data
        elif isinstance(data, dict):
            assert data['type'] == 'Feature'
            assert data['properties']['type'] == 'site'
            self = cls(**data)
        else:
            raise TypeError(type(data))
        return self


class Observation(_Feature):
    """
    The observation body feature of a site model.
    """
    _model_cls = SiteModel
    _feat_type = SiteModel._body_type

    @classmethod
    def coerce(cls, data):
        if isinstance(data, cls):
            self = data
        elif isinstance(data, dict):
            assert data['type'] == 'Feature'
            assert data['properties']['type'] == 'observation'
            self = cls(**data)
        else:
            raise TypeError(type(data))
        return self

    @property
    def observation_date(self):
        return util_time.coerce_datetime(self['properties']['observation_date'])


# def _site_header_from_observations(observations, mgrs_code, site_id, status, summary_geom=None):
#     """
#     Consolodate site observations into a site header
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
            for model in pman.progiter(self, desc='fixup'):
                model.fixup()
        return self

    def validate(self, strict=False, stop_on_failure=True, verbose=1, mode='process', workers=0):
        """
        Validate multiple models in parallel
        """
        import rich
        # pman = util_progress.ProgressManager(backend='progiter')
        pman = util_progress.ProgressManager()
        with pman:
            jobs = ub.JobPool(mode='process', max_workers=workers)
            for model in pman.progiter(self, desc='submit validate models'):
                job = jobs.submit(model.validate, strict=strict, verbose=verbose)
                job.model = model
            num_passed = 0
            errors = []
            prog = pman.progiter(jobs.as_completed(), total=len(jobs), desc='collect validate models')
            for job in prog:
                try:
                    job.result()
                except Exception as ex:
                    rich.print(f'[red] ERROR: failed to validate {job.model.model_id} : {job.model.model_type} in a collection')
                    errors.append((ex, job.model))
                    prog.set_extra(f'Passed: {num_passed}, Failed: {len(errors)}')
                    if stop_on_failure:
                        raise
                else:
                    num_passed += 1
                prog.set_extra(f'Passed: {num_passed}, Failed: {len(errors)}')
            if errors:
                num_failed = len(errors)
                num_total = len(jobs)
                failed_model_ids = [model.model_id for ex, model in errors]
                rich.print(f'[red] ERROR: failed to validate {num_failed} / {num_total} models')
                rich.print('failed_model_ids = {}'.format(ub.urepr(failed_model_ids, nl=1)))
                raise Exception(f'Failed to validate {num_failed} / {num_total} models')


class SiteModelCollection(ModelCollection):

    def as_region_model(self, region_header=None, region_id=None, strict=True):
        """
        Convert a set of site models to a region model

        Args:
            region (RegonModel | RegionHeader | None):
                If specified, use this information to generate the new region
                header. If unspecified, we attempt to infer this from the site
                models.

            region_id (str | None):
                if specified, use this as the region id

            strict (bool):
                if False, ignore missing uninferable information.

        Returns:
            RegonModel: a new region model where each site in this collection
                appears as a site summary.

        Example:
            >>> from geowatch.geoannots.geomodels import RegionModel
            >>> region, sites = RegionModel.random(with_sites=True, rng=0)
            >>> self = SiteModelCollection(sites)
            >>> self.as_region_model()
        """
        site_summaries = [s.as_summary() for s in self]
        site_header_properties = [site.header['properties'] for site in self]

        if region_header is not None:
            region_header = RegionHeader.coerce(region_header)
            region_header = copy.deepcopy(region_header)
        else:
            region_header = RegionHeader.empty()

        if region_id is not None:
            region_header['properties']['region_id'] = region_id

        region_props = region_header['properties']
        # note: region_id does not appear in a site summary, but it does in the
        # site model.
        key = 'region_id'
        if region_props.get(key, None) is None:
            if len(self) == 0:
                handle_error(f'No sites. Unable to infer {key}.', strict=strict)
            else:
                region_props[key] = _rectify_keys(key, site_header_properties)

        region_header = _infer_region_header_from_site_summaries(
            region_header, site_summaries, strict)

        region_features = [region_header] + site_summaries
        region_model = RegionModel(features=region_features)
        return region_model


def _infer_region_header_from_site_summaries(region_header, site_summaries, strict=True):
    """
    Given a RegionHeader use site_summaries to fill missing data.
    """
    if region_header is None:
        region_header = RegionHeader.empty()
    region_props = region_header.get('properties', None)

    if region_props.get('type', None) is None:
        region_props['type'] = 'region'

    site_summary_properties = [sitesum['properties'] for sitesum in site_summaries]

    shared_unique_properties = ['originator', 'model_content', 'mgrs']

    for key in shared_unique_properties:
        if region_props.get(key, None) is None:
            try:
                if len(site_summaries) == 0:
                    handle_error(f'No sites. Unable to infer {key}.', strict=strict)
                else:
                    region_props[key] = _rectify_keys(key, site_summary_properties)
            except ValueError:
                # Allow MGRS to fail. We can use region geometry to get the
                # right one.
                if key != 'mgrs':
                    raise

    if region_props.get('start_date', None) is None:
        if len(site_summaries) == 0:
            handle_error('No sites. Unable to infer start_date.', strict=strict)
        dates = [p['start_date'] for p in site_summary_properties if p['start_date'] is not None]
        if len(dates) == 0:
            handle_error('No sites with start dates', strict=strict)
        else:
            region_props['start_date'] = min(dates)

    if region_props.get('end_date', None) is None:
        if len(site_summaries) == 0:
            handle_error('No sites. Unable to infer end_date.', strict=strict)
        dates = [p['end_date'] for p in site_summary_properties if p['end_date'] is not None]
        if len(dates) == 0:
            handle_error('No sites with end dates', strict=strict)
        else:
            region_props['end_date'] = max(dates)

    if region_header.get('geometry', None) is None:
        if len(site_summaries) == 0:
            handle_error(f'No sites. Unable to infer {key}.', strict=strict)
            # region_header['geometry'] = {'type': 'Point', 'coordinates': []}
        else:
            from shapely.ops import unary_union
            import kwimage
            import shapely.geometry
            site_geoms = [shapely.geometry.shape(s['geometry']).buffer(0)
                          for s in site_summaries]
            sh_geom = unary_union(site_geoms).envelope
            dct_geom = kwimage.Polygon.from_shapely(sh_geom).to_geojson()
            region_header['geometry'] = dct_geom

    if region_props.get('mgrs', None) is None:
        RegionHeader(**region_header).infer_mgrs(strict=strict)

    return region_header


def _rectify_keys(key, properties_list):
    """
    Given a key and a list of dictionaries, extract the value for that key in
    all dictionaries and check they are all the same.

    Args:
        key (str): key of interest
        properties_list (List[Dict[str, T]]): multiple property dictionaries

    Returns:
        T: value from properties dictionaries.
    """
    if len(properties_list) == 0:
        raise ValueError(f'No sites. Unable to infer {key}.')
    unique_values = {p[key] for p in properties_list}
    if len(unique_values) > 1:
        msg = (f'More than one key={key!r} in with unique_values={unique_values!r}')
        print(msg)
        raise ValueError(msg)
    value = list(unique_values)[0]
    return value


def handle_error(msg, extype=ValueError, strict=True):
    import rich
    if strict:
        raise extype(msg)
    else:
        rich.print(f'[yellow]WARNING: {msg}')


def _update_propery_cache(prop):
    """
    Move to the new cache schema
    """
    if 'annotation_cache' in prop or 'misc_info' in prop:
        cache = prop.get('cache', {})
        cache = ub.udict.union(prop.pop('annotation_cache', {}), cache)
        cache = ub.udict.union(prop.pop('misc_info', {}), cache)
        if cache:
            prop['cache'] = cache
    if 'cache' in prop:
        if prop['cache'] is None:
            prop['cache'] = {}


def coerce_site_or_region_model(model_data):
    """
    Args:
        model_data (dict): A geojson FeatureCollection that should correspond
            to a SiteModel or RegionModel.

    Returns:
        SiteModel | RegionModel - return type depends on the input data
    """
    assert isinstance(model_data, dict)
    assert model_data['type'] == 'FeatureCollection'
    for feat in model_data['features']:
        assert feat['type'] == 'Feature'
        if feat['properties']['type'] == 'region':
            return RegionModel(**model_data)
        elif feat['properties']['type'] == 'site':
            return SiteModel(**model_data)
    raise AssertionError('Did not find a region or site header')
