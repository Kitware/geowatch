"""
Utilities for geopandas and other geographic information system tools
"""
import ubelt as ub
import numpy as np
import json
import os


def geopandas_pairwise_overlaps(gdf1, gdf2, predicate='intersects'):
    """
    Find pairwise relationships between each geometries

    Args:
        gdf1 (GeoDataFrame): query geo data
        gdf2 (GeoDataFrame): database geo data (builds spatial index)
        predicate (str, default='intersects'): a DE-9IM [1] predicate.
           (e.g. if intersection finds intersections between geometries)

    References:
        ..[1] https://en.wikipedia.org/wiki/DE-9IM

    TODO:
        - [ ] This can move to watch.utils

    Returns:
        dict:
            mapping from integer-indexes in gdf1 to
            overlapping integer-indexes in gdf2

    Example:
        >>> from watch.utils.util_gis import *  # NOQA
        >>> import geopandas as gpd
        >>> gpd.GeoDataFrame()
        >>> gdf1 = gpd.read_file(
        >>>     gpd.datasets.get_path('naturalearth_lowres')
        >>> )
        >>> gdf2 = gpd.read_file(
        >>>     gpd.datasets.get_path('naturalearth_cities')
        >>> )
        >>> gdf1 = gdf1.set_index('name')
        >>> gdf2 = gdf2.set_index('name')
        >>> mapping = geopandas_pairwise_overlaps(gdf1, gdf2)

    Benchmark:
        import timerit
        ti = timerit.Timerit(10, bestof=3, verbose=2)
        for timer in ti.reset('with sindex O(N * log(M))'):
            with timer:
                fast_mapping = geopandas_pairwise_overlaps(gdf1, gdf2)
        for timer in ti.reset('without sindex O(N * M)'):
            with timer:
                from collections import defaultdict
                slow_mapping = defaultdict(list)
                for idx1, geom1 in enumerate(gdf1.geometry):
                    slow_mapping[idx1] = []
                    for idx2, geom2 in enumerate(gdf2.geometry):
                        if geom1.intersects(geom2):
                            slow_mapping[idx1].append(idx2)
        # check they are the same
        assert set(slow_mapping) == set(fast_mapping)
        for idx1 in slow_mapping:
            slow_idx2s = slow_mapping[idx1]
            fast_idx2s = fast_mapping[idx1]
            assert sorted(fast_idx2s) == sorted(slow_idx2s)
    """
    assert gdf1.index.is_unique, 'GeoDataFrame indexes must be unique'
    assert gdf2.index.is_unique, 'GeoDataFrame indexes must be unique'

    # Construct the spatial index (requires pygeos and/or rtree)
    sindex2 = gdf2.sindex
    # For each query polygon, lookup intersecting polygons in the spatial index
    idx1_to_idxs2 = {}
    for idx1, (rowid, row1) in enumerate(gdf1.iterrows()):
        idxs2 = sindex2.query(row1.geometry, predicate=predicate)
        # Record result indexes that "match" given the geometric predicate
        idx1_to_idxs2[idx1] = idxs2
    return idx1_to_idxs2


def latlon_text(lat, lon, precision=6):
    """
    Make a lat,lon string suitable for a filename.

    Pads with leading zeros so file names will align nicely at the same level
    of prcision.

    Args:
        lat (float): degrees latitude

        lon (float): degrees longitude

        precision (float, default=6):
            Number of trailing decimal places. As rule of thumb set this to:
                6 - for ~10cm accuracy,
                5 - for ~1m accuracy,
                2 - for ~1km accuracy,

    TODO:
        - [ ] This can move to watch.utils

    Notes:
        1 degree of latitude is *very* roughly the order of 100km, so the
        default precision of 6 localizes down to ~0.1 meters, which will
        usually be sufficient for satellite applications, but be mindful of
        using this text in applications that require more precision. Note 1
        degree of longitude will vary, but will always be at least as precise
        as 1 degree of latitude.

    Example:
        >>> lat = 90
        >>> lon = 180
        >>> print(latlon_text(lat, lon))
        N90.000000E180.000000

        >>> lat = 0
        >>> lon = 0
        >>> print(latlon_text(lat, lon))
        N00.000000E000.000000

    Example:
        >>> print(latlon_text(80.123, 170.123))
        >>> print(latlon_text(10.123, 80.123))
        >>> print(latlon_text(0.123, 0.123))
        N80.123000E170.123000
        N10.123000E080.123000
        N00.123000E000.123000

        >>> print(latlon_text(80.123, 170.123, precision=2))
        >>> print(latlon_text(10.123, 80.123, precision=2))
        >>> print(latlon_text(0.123, 0.123, precision=2))
        N80.12E170.12
        N10.12E080.12
        N00.12E000.12

        >>> print(latlon_text(80.123, 170.123, precision=5))
        >>> print(latlon_text(10.123, 80.123, precision=5))
        >>> print(latlon_text(0.123, 0.123, precision=5))
        N80.12300E170.12300
        N10.12300E080.12300
        N00.12300E000.12300
    """
    def _build_float_precision_fmt(num_leading, num_trailing):
        num2 = num_trailing
        # 2 extra for radix and leading sign
        num1 = num_leading + num_trailing + 2
        fmtparts = ['{:+0', str(num1), '.', str(num2), 'F}']
        fmtstr = ''.join(fmtparts)
        return fmtstr

    assert -90 <= lat <= 90, 'invalid lat'
    assert -180 <= lon <= 180, 'invalid lon'

    # Ensure latitude had 2 leading places and longitude has 3
    latfmt = _build_float_precision_fmt(2, precision)
    lonfmt = _build_float_precision_fmt(3, precision)

    lat_str = latfmt.format(lat).replace('+', 'N').replace('-', 'S')
    lon_str = lonfmt.format(lon).replace('+', 'E').replace('-', 'W')
    text = lat_str + lon_str
    return text


def demo_regions_geojson_text():
    import ubelt as ub
    geojson_text = ub.codeblock(
        '''
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"type": "region", "region_model_id": "US_Jacksonville_R01", "version": "1.0.1", "mgrs": "17RMP", "start_date": "2009-05-09", "end_date": "2020-01-26" },
                    "geometry": {"type": "Polygon", "coordinates": [[[-81.6953, 30.3652], [-81.6942, 30.2984], [-81.5975, 30.2992], [-81.5968, 30.3667], [-81.6953, 30.3652]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"type": "site", "site_id": "17RMP_US_Jacksonville_R01_0000", "start_date": "2016-02-14", "end_date": "2017-11-01"},
                    "geometry": {"type": "Polygon", "coordinates": [[[-81.6364, 30.3209], [-81.6364, 30.3236], [-81.6397, 30.3236], [-81.6397, 30.3209], [-81.6364, 30.3209]]]}
                },
                {
                    "type": "Feature",
                    "properties": {"type": "site", "site_id": "17RMP_US_Jacksonville_R01_0001", "start_date": "2016-07-13", "end_date": "2020-01-26" },
                    "geometry": {"type": "Polygon", "coordinates": [[[-81.6085, 30.3568], [-81.6085, 30.3600], [-81.6120, 30.3600], [-81.6120, 30.3568], [-81.6085, 30.3568]]]}
                }
            ]
        }
        ''')
    return geojson_text


def load_geojson(file, default_axis_mapping='OAMS_TRADITIONAL_GIS_ORDER'):
    """
    Args:
        file (str | file): path or file object containing geojson data.

        axis_mapping (str, default='OAMS_TRADITIONAL_GIS_ORDER'):
            The axis-ordering of the geojson file on disk.  This is assumed to
            be traditional ordering by default according to the geojson spec.

    Returns:
        GeoDataFrame : a dataframe with geo info.
            Note: geopandas always stores data in traditional XY, although its
            CRS does seem to hold axis order info?

        # OLD: This will ALWAYS return
        # with an OAMS_AUTHORITY_COMPLIANT wgs84 crs (i.e. lat,lon) even
        # though the on disk order is should be OAMS_TRADITIONAL_GIS_ORDER.

    References:
        https://geopandas.org/docs/user_guide/projections.html#the-axis-order-of-a-crs

    Example:
        >>> import io
        >>> from watch.utils.util_gis import *  # NOQA
        >>> geojson_text = demo_regions_geojson_text()
        >>> file = io.StringIO()
        >>> file.write(geojson_text)
        >>> file.seek(0)
        >>> region_df = load_geojson(file)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> import geopandas as gpd
        >>> kwplot.autompl()
        >>> wld_map_gdf = gpd.read_file(
        >>>     gpd.datasets.get_path('naturalearth_lowres')
        >>> ).to_crs('crs84')
        >>> ax = wld_map_gdf.plot()
        >>> region_df.plot(ax=ax, edgecolor='orange', alpha=0.8)
        >>> # https://gis.stackexchange.com/questions/372564/userwarning-when-trying-to-get-centroid-from-a-polygon-geopandas
        >>> centroid = region_df.to_crs('+proj=cea').centroid.to_crs(region_df.crs)
        >>> centroid.plot(ax=ax, edgecolor='orange', alpha=0.8)

        print('region_df.crs = {!r}'.format(region_df.crs))
        print('wld_map_gdf.crs = {!r}'.format(wld_map_gdf.crs))
    """
    import geopandas as gpd
    valid_axis_mappings = {
        'OAMS_TRADITIONAL_GIS_ORDER',
        'OAMS_AUTHORITY_COMPLIANT',
    }
    if default_axis_mapping not in valid_axis_mappings:
        raise Exception

    # Read custom ROI regions
    region_df = gpd.read_file(file)

    # TODO: can we construct a pyproj.CRS from wgs84, but with traditional
    # order?

    # import pyproj
    # wgs84 = pyproj.CRS.from_epsg(4326)
    # z = pyproj.Transformer.from_crs(4326, 4326, always_xy=True)
    # crs1 = region_df.crs
    # pyproj.CRS.from_dict(crs1.to_json_dict())
    # z = region_df.crs
    # z.to_json_dict()

    # Use a CRS that actually reflects the underlying data
    if default_axis_mapping == 'OAMS_TRADITIONAL_GIS_ORDER':
        crs84 = _get_crs84()
        # this is much faster and the only reason this is ok is because the
        # input is xy-wgs84 so the transform (which is slow) would be a noop
        region_df._crs = crs84
        # region_df = region_df.to_crs(crs84)
    else:
        raise NotImplementedError('geopandas only deals with traditional lon/lat')
    return region_df


read_geojson = load_geojson  # backwards compat


@ub.memoize
def get_crs84():
    """
    Constructing the CRS84 is slow.
    This function memoizes it so it only happens once.

    Returns:
        pyproj.crs.crs.CRS
    """
    from pyproj import CRS
    crs84 = CRS.from_user_input('crs84')
    return crs84


_get_crs84 = get_crs84


def _flip(x, y):
    return (y, x)


def shapely_flip_xy(geom):
    from shapely import ops
    return ops.transform(_flip, geom)


def project_gdf_to_local_utm(gdf_crs84, max_utm_zones=1):
    """
    Find the local UTM zone for a geo data frame and project to it.

    Assumes geometry is in CRS-84.

    All geometry in the GDF must be in the same UTM zone.

    Args:
        gdf_crs84 (geopandas.GeoDataFrame):
            The data with CRS-84 geometry to project into a local UTM

        max_utm_zones (int):
            If the data spans more than this many UTM zones, error.
            Otherwise, we take the first one.

    Returns:
        geopandas.GeoDataFrame

    Example:
        >>> import geopandas as gpd
        >>> from watch.utils.util_gis import *  # NOQA
        >>> import kwarray
        >>> import kwimage
        >>> rng = kwarray.ensure_rng(0)
        >>> # Gen lat/lons between 0 and 1, which is in UTM zone 31N
        >>> gdf_crs84 = gpd.GeoDataFrame({'geometry': [
        >>>     kwimage.Polygon.random(rng=rng).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).to_shapely(),
        >>> ]}, crs='crs84')
        >>> gdf_utm = project_gdf_to_local_utm(gdf_crs84)
        >>> assert gdf_utm.crs.name == 'WGS 84 / UTM zone 31N'

    Example:
        >>> import geopandas as gpd
        >>> from watch.utils.util_gis import *  # NOQA
        >>> import kwarray
        >>> import kwimage
        >>> # If the data is too big for a single UTM zone,
        >>> rng = kwarray.ensure_rng(0)
        >>> gdf_crs84 = gpd.GeoDataFrame({'geometry': [
        >>>     kwimage.Polygon.random(rng=rng).scale(90).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).scale(90).to_shapely(),
        >>>     kwimage.Polygon.random(rng=rng).scale(90).to_shapely(),
        >>> ]}, crs='crs84')
        >>> import pytest
        >>> with pytest.raises(ValueError):
        >>>     gdf_utm = project_gdf_to_local_utm(gdf_crs84)

    # TODO: Gracefully handle cases where the UTM zones are different
    # but all neighbors. Find a good example of this.
    # Example:
    #     >>> import geopandas as gpd
    #     >>> from watch.utils.util_gis import *  # NOQA
    #     >>> import kwarray
    #     >>> # If the data is too big for a single UTM zone,
    #     >>> rng = kwarray.ensure_rng(0)
    #     >>> # Corner case in Madagascar where the extent isn't too big, but it
    #     >>> # spans multiple UTM zones
    #     >>> poly = kwimage.Polygon.coerce({
    #     >>>     "type": "Polygon",
    #     >>>     "coordinates": [
    #     >>>         [[-73.77200379967688, 42.864783745778894],
    #     >>>          [-73.77177715301514, 42.86412514733195],
    #     >>>          [-73.77110660076141, 42.8641654498268],
    #     >>>          [-73.77105563879013, 42.86423720786224],
    #     >>>          [-73.7710489332676, 42.864399400374786],
    #     >>>          [-73.77134531736374, 42.8649134986743],
    #     >>>          [ -73.77200379967688, 42.864783745778894]]]
    #     >>> })
    #     >>> gdf_crs84 = gpd.GeoDataFrame({
    #     >>>     'geometry': [poly.to_shapely()]}, crs='crs84')
    """
    # if gdf_crs84.crs.name != 'WGS 84 (CRS84)':
    #     raise AssertionError('expected CRS-84 input')
    epsg_zones = []
    for geom_crs84 in gdf_crs84.geometry:
        epsg_zone = find_local_meter_epsg_crs(geom_crs84)
        epsg_zones.append(epsg_zone)
    if not ub.allsame(epsg_zones):
        unique_utm = set(epsg_zones)
        if len(unique_utm) > max_utm_zones:
            raise ValueError(ub.paragraph(
                '''
                Input data spanned multiple UTM zones.
                This is currently not allowed. {}
                '''
            ).format(unique_utm))
    # TODO: if there are more than one is there a way to get "the best one?"
    epsg_zone = epsg_zones[0]
    gdf_utm = gdf_crs84.to_crs(epsg_zone)
    return gdf_utm


def utm_epsg_from_latlon(lat, lon):
    """
    Find a reasonable UTM CRS for a given lat / lon

    The purpose of this function is to get a reasonable CRS for computing
    distances in meters. If the region of interest is very large, this may not
    be valid.

    Args:
        lat (float): degrees in latitude
        lon (float): degrees in longitude

    Returns:
        int : the ESPG code of the UTM zone

    References:
        https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
        https://gis.stackexchange.com/questions/365584/convert-utm-zone-into-epsg-code

    Example:
        >>> from watch.utils.util_gis import *  # NOQA
        >>> epsg_code = utm_epsg_from_latlon(0, 0)
        >>> print('epsg_code = {!r}'.format(epsg_code))
        epsg_code = 32631
    """
    import utm
    # easting, northing, zone_num, zone_code = utm.from_latlon(min_lat, min_lon)
    zone_num = utm.latlon_to_zone_number(lat, lon)

    # Construction of EPSG code from UTM zone number
    south = lat < 0
    epsg_code = 32600
    epsg_code += int(zone_num)
    if south is True:
        epsg_code += 100
    return epsg_code


class UTM_TransformContext:
    """
    Helper to project into UTM space, perform some transform, and then project
    back to the original CRS.

    Currently only supports CRS84

    Example:
        >>> import kwimage
        >>> from watch.utils.util_gis import *  # NOQA
        >>> data_crs84 = kwimage.Polygon.random()
        >>> with UTM_TransformContext(data_crs84) as self:
        >>>     orig_utm_poly = kwimage.Polygon.coerce(self.geoms_utm.iloc[0])
        >>>     new_utm_poly = orig_utm_poly.scale(2, about='center')
        >>>     self.finalize(new_utm_poly)
        >>> final_result = kwimage.Polygon.coerce(self.final_geoms_crs84.iloc[0])
        >>> naive_result = data_crs84.scale(2, about='center')
        >>> # Note the subtle difference in the naive vs context result
        >>> print(f'final_result={final_result}')
        >>> print(f'naive_result={naive_result}')
    """

    def __init__(self, data_crs84):
        """
        Args:
            data_crs84 (Coercable[GeoSeries]):
                something we know how to transform into a GeoSeries
        """
        from watch.utils import util_gis
        self.crs84 = util_gis._get_crs84()
        self.geoms_crs84 = self._coerce_geo_series(data_crs84, self.crs84)
        self.crs_utm = None
        self.gdf_utm = None
        self.final_geoms_utm = None
        self.final_geoms_crs84 = None

    def __enter__(self):
        from watch.utils import util_gis
        import geopandas as gpd
        gdf_crs84 = gpd.GeoDataFrame(geometry=self.geoms_crs84)
        gdf_utm = util_gis.project_gdf_to_local_utm(gdf_crs84)
        self.geoms_utm = gdf_utm.geometry
        self.crs_utm = self.geoms_utm.crs
        return self

    def _coerce_geo_series(self, data, default_crs=None):
        import shapely
        import geopandas as gpd
        import kwimage
        if isinstance(data, list):
            geoms = gpd.GeoSeries(data, crs=default_crs)
        else:
            if isinstance(data, shapely.geometry.base.BaseGeometry):
                geoms = gpd.GeoSeries([data], crs=default_crs)
            elif isinstance(data, kwimage.Polygon):
                geoms = gpd.GeoSeries([data.to_shapely()], crs=default_crs)
            elif isinstance(data, gpd.GeoDataFrame):
                geoms = data.geometry
            elif isinstance(data, gpd.GeoSeries):
                geoms = data
            else:
                raise TypeError(type(data))
        return geoms

    def finalize(self, final_utm):
        """
        Args:
            final_utm (Coercable[GeoSeries]):
                something coercable to geometry in UTM coordinates
        """
        final_geoms_utm = self._coerce_geo_series(
            final_utm, default_crs=self.crs_utm)
        self.final_geoms_utm = final_geoms_utm

    def __exit__(self, a, b, c):
        if self.final_geoms_utm is None:
            raise RuntimeError('Need to call finalize')
        self.final_geoms_crs84 = self.final_geoms_utm.to_crs(self.crs84)


def _demo_convert_latlon_to_utm():
    # Pretend we have these lon/lat (CRS84 corners)
    lat_y = 40.060759
    lon_x = 116.613095
    lat_y_off = 0.0001
    lat_x_off = 0.0001
    # hard code so north is up
    wld_corners = np.array([
        [lon_x - lat_x_off, lat_y + lat_y_off],
        [lon_x - lat_x_off, lat_y - lat_y_off],
        [lon_x + lat_x_off, lat_y - lat_y_off],
        [lon_x + lat_x_off, lat_y + lat_y_off],
    ])
    import kwimage
    wld_poly = kwimage.Polygon(exterior=wld_corners)
    wld_poly_sh = wld_poly.to_shapely()

    lon = wld_poly_sh.centroid.x
    lat = wld_poly_sh.centroid.y
    utm_code = utm_epsg_from_latlon(lat, lon)

    import geopandas as gpd
    gdf_crs84 = gpd.GeoDataFrame({'geometry': [wld_poly_sh]}, crs='crs84')
    gdf_utm = gdf_crs84.to_crs(utm_code)
    utm_poly_sh = gdf_utm.iloc[0]['geometry']

    utm_poly = kwimage.Polygon.from_shapely(utm_poly_sh)
    utm_corners = utm_poly.data['exterior'].data

    min_x = utm_corners.T[0].min()
    max_x = utm_corners.T[0].max()
    min_y = utm_corners.T[1].min()
    max_y = utm_corners.T[1].max()

    # Note: UTM bottom should be the min value, so be careful here

    def _torch_meshgrid(*basis_dims):
        """
        References:
            https://zhaoyu.li/post/how-to-implement-meshgrid-in-pytorch/
        """
        basis_lens = list(map(len, basis_dims))
        new_dims = []
        for i, basis in enumerate(basis_dims):
            # Probably a more efficent way to do this, but its right
            newshape = [1] * len(basis_dims)
            reps = list(basis_lens)
            newshape[i] = -1
            reps[i] = 1
            dd = basis.view(*newshape).repeat(*reps)
            new_dims.append(dd)
        return new_dims

    import torch
    image_width = 256
    image_height = 256

    utm_x_basis = torch.linspace(min_x, max_x, image_width)
    utm_y_basis = torch.linspace(max_y, min_y, image_height)

    xgrid, ygrid = _torch_meshgrid(utm_x_basis, utm_y_basis)


def find_local_meter_epsg_crs(geom_crs84):
    """
    Find the "best" meter based CRS for a smallish geographic region.

    Currently this only returns UTM zones. Might be better to return an Albers
    projection if the geometry spans more than one UTM zone.

    Args:
        geom_crs84 (shapely.geometry.base.BaseGeometry):
            shapely geometry in CRS84 (lon/lat wgs84)

    Returns:
        int: epsg code

    References:
        [1] https://gis.stackexchange.com/questions/148181/choosing-projection-crs-for-short-distance-based-analysis/148187
        [2] http://projfinder.com/

    Example:
        >>> from watch.utils.util_gis import *  # NOQA
        >>> import kwimage
        >>> geom_crs84 = kwimage.Polygon.random().translate(-0.5).scale((180, 90)).to_shapely()
        >>> epsg_zone = find_local_meter_epsg_crs(geom_crs84)

    SeeAlso:
        GeoDataFrame.estimate_utm_crs

    TODO:
        - [ ] Albers?
        - [ ] Better UTM zone intersection
        - [ ] Fix edge cases
    """

    if 0:
        # TODO: probably can do this via: GeoDataFrame.estimate_utm_crs
        # But ours is faster. Not sure if its more correct though.
        import geopandas as gpd
        utm_crs = gpd.array.from_shapely([geom_crs84], 'crs84').estimate_utm_crs()
        epsg_zone = utm_crs.to_epsg()

    lonmin, latmin, lonmax, latmax = geom_crs84.bounds
    # Hack: this doesnt work on boundries (or for larger regions)
    # correct way of doing this would be lookup candiate CRS zones,
    # and find the one with highest intersection area weighted by distance
    # to the center of the valid region.
    latmid = (latmin + latmax) / 2
    lonmid = (lonmin + lonmax) / 2
    candidate_utm_codes = [
        utm_epsg_from_latlon(latmin, lonmin),
        utm_epsg_from_latlon(latmax, lonmax),
        utm_epsg_from_latlon(latmax, lonmin),
        utm_epsg_from_latlon(latmin, lonmax),
        utm_epsg_from_latlon(latmid, lonmid),
    ]
    epsg_zone = ub.argmax(ub.dict_hist(candidate_utm_codes))
    return epsg_zone


def check_latlons(lat, lon):
    """
    Quick check to see if latitudes and longitudes are valid.

    Longitude (x) is always between -180 and 180 (degrees east)
    Latitude (y) is always between -90 and 90 (degrees north)

    Example:
        >>> from watch.utils.util_gis import *  # NOQA
        >>> import pytest
        >>> assert not check_latlons(1000, 1000)
        >>> assert check_latlons(0, 0)
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    bad_lat = (lat < -90).any() or (lat > 90).any()
    bad_lon = (lon < -180).any() or (lon > 180).any()
    in_ranges = not (bad_lon or bad_lat)
    return in_ranges


def coerce_geojson_datas(arg, format='dataframe', allow_raw=False, workers=0,
                         mode='thread', verbose=1, desc=None):
    """
    Attempts to resolve an argument into multiple geojson datas.

    Multiple threads / processes are used to load the specified information and
    the function generates dictionaries of information containing the file path
    and the loaded data as they become available.

    The argument can be:

            1. A path to a geojson file (or a list of them)

            2. A glob string specifying multiple geojson files (or a list of them)

            3. A path to a json manifest file.

            4. If allow_raw is True, then the input can be a raw json string,
               dict, or GeoDataFrame.

    Args:
        arg (str | PathLike | List[str | PathLike]):
            an argument that is coerceable to one or more GeoDataFrames.

        format (str):
            Indicates the returned format of the data. Can be 'dataframe' where
            the 'data' key will be a GeoDataFrame, or 'dict' where the raw json
            data will be returned.

        allow_raw (bool):
            if True, we will also check if the arguments are raw json /
            geopandas data that can be loaded. In general try not to enable
            this.

        workers (int):
            number of io workers

        mode (str):
            concurrent executor mode. Can be 'serial', 'thread', or 'process'.

        desc (str):
            custom message for progress bar.

    Yields:
        List[Dict[str, Any | GeoDataFrame | Dict]]:
            A list of dictionaries formated with the keys:

                * fpath (str): the file path the data was loaded from (
                    if applicable)

                * data (GeoDataFrame | dict):
                    the data loaded in the requested format

    SeeAlso:
        * load_site_or_region_dataframes - the function that does the loading
            after the arguments are coerced.

    Example:
        >>> # xdoctest: +SKIP("failing on CI. unsure why")
        >>> from watch.utils.util_gis import *  # NOQA
        >>> from watch.demo.metrics_demo import generate_demodata
        >>> info1 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R001')
        >>> info2 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R002')
        >>> info3 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R003')
        >>> info4 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R012')
        >>> info5 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R022')
        >>> #
        >>> region_fpaths = sorted(info1['true_region_dpath'].glob('*.geojson'))
        >>> site_fpaths = sorted(info1['true_site_dpath'].glob('*.geojson'))
        >>> #
        >>> import json
        >>> manifest_fpath1 =  info1['output_dpath'] / 'demo_manifest1.json'
        >>> manifest_fpath2 =  info1['output_dpath'] / 'demo_manifest2.json'
        >>> manifest_data1 = {
        >>>     'files': [str(p) for p in region_fpaths[0:2]]
        >>> }
        >>> manifest_data2 = {
        >>>     'files': [str(p) for p in region_fpaths[3:4]]
        >>> }
        >>> manifest_fpath1.write_text(json.dumps(manifest_data1))
        >>> manifest_fpath2.write_text(json.dumps(manifest_data2))
        >>> variants = []
        >>> # ==========
        >>> # Test Cases
        >>> # ==========
        >>> #
        >>> # List of region files
        >>> arg = region_fpaths
        >>> result = list(coerce_geojson_datas(arg))
        >>> assert len(result) == 5
        >>> #
        >>> # Glob for region files
        >>> arg = str(info1['true_region_dpath']) + '/*R*2*.geojson'
        >>> result = list(coerce_geojson_datas(arg))
        >>> assert len(result) == 3
        >>> #
        >>> # Manifest file
        >>> arg = manifest_fpath1
        >>> result = list(coerce_geojson_datas(arg))
        >>> assert len(result) == 2
        >>> #
        >>> # Manifest file glob
        >>> arg = str(manifest_fpath1.parent / '*.json')
        >>> result = list(coerce_geojson_datas(arg))
        >>> assert len(result) == 3
        >>> #
        >>> # Manifest file glob and a region path
        >>> arg = [manifest_fpath2, region_fpaths[0]]
        >>> result = list(coerce_geojson_datas(arg))
        >>> assert len(result) == 2
        >>> #
        >>> # Site glob and a manifest glob
        >>> arg = [str(info1['true_site_dpath']) + '/DR_R002_*.geojson',
        ...        str(manifest_fpath1 + '*')]
        >>> result = list(coerce_geojson_datas(arg))
        >>> assert len(result) == 9
        >>> #
        >>> # Site directory and manifest file.
        >>> arg = [str(info1['true_site_dpath']),
        ...        str(manifest_fpath1 + '*')]
        >>> result = list(coerce_geojson_datas(arg))
        >>> assert len(result) == 31

        >>> # Test raw loading and format swapping
        >>> from watch.utils import util_gis
        >>> arg = util_gis.demo_regions_geojson_text()
        >>> result1 = list(coerce_geojson_datas(arg, allow_raw=False))
        >>> assert len(result1) == 0
        >>> result2 = list(coerce_geojson_datas(arg, allow_raw=True))
        >>> assert len(result2) == 1
        >>> arg = result2
        >>> result3 = list(coerce_geojson_datas(arg, format='dataframe', allow_raw=True))
        >>> assert result3 == result2
        >>> result4 = list(coerce_geojson_datas(arg, format='json', allow_raw=True))
        >>> assert isinstance(result4[0]['data'], dict)
        >>> result5 = list(coerce_geojson_datas(
        >>>     result4, format='dataframe', allow_raw=True))
        >>> assert isinstance(result4[0]['data'], dict)

        >>> #
        >>> # Test nothing case
        >>> assert len(list(coerce_geojson_datas([], allow_raw=True))) == 0
    """
    if format not in {'json', 'dataframe'}:
        raise KeyError(format)

    if allow_raw:
        # Normally the function assumes we are only inputing things that are
        # coercable to paths, and then to geojson. But sometimes we might want
        # to pass around that data directly. In this case, grab those items
        # first, and then resolve the rest of them.
        raw_items = []
        other_items = []
        for item in ([arg] if not isinstance(arg, list) else arg):
            was_raw, item = _coerce_raw_geojson(item, format)
            if was_raw:
                raw_items.append(item)
            else:
                other_items.append(item)
        path_coercable = other_items
    else:
        path_coercable = arg

    # Handle the normal better-defined case of coercing arguments into paths
    geojson_fpaths = coerce_geojson_paths(path_coercable)

    if allow_raw:
        if verbose:
            if raw_items or len(geojson_fpaths) == 0:
                print(f'Coerced {len(raw_items)} raw geojson item')
            if raw_items:
                if len(geojson_fpaths) == 0:
                    # Disable path verbosity if there were raw items, but no
                    # paths.
                    verbose = 0

    # Now all of resolved accumulator items should be geojson files.
    # Submit the data to be loaded.
    geojson_fpaths = list(ub.unique(geojson_fpaths))
    data_gen = load_geojson_datas(
        geojson_fpaths, workers=workers, mode=mode, desc=desc,
        format=format, verbose=verbose, yield_after_submit=True)

    # Start the background workers
    next(data_gen)

    if allow_raw:
        # yield the raw data before the generated data
        yield from raw_items

    # Finish the main generator
    yield from data_gen


def coerce_geojson_paths(data, return_manifests=False):
    """
    Resolves the argument to a list of geojson paths.  The argument can be a
    full path, a glob string, a path to a manifest file or any combination of
    the previous in a list.

    Args:
        data : argument to coerce
        return_manifests (bool): if True additionally returns paths to
            any intermediate manifest files.

    Example:
        >>> from watch.utils.util_gis import *  # NOQA
        >>> import json
        >>> from watch.demo.metrics_demo import generate_demodata
        >>> # Setup a bunch of geojson files
        >>> outdir = ub.Path.appdir("watch/tests/gis/coerce_geojson")
        >>> info1 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R001', outdir=outdir)
        >>> info2 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R002', outdir=outdir)
        >>> info3 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R003', outdir=outdir)
        >>> info4 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R012', outdir=outdir)
        >>> info5 = generate_demodata.generate_demo_metrics_framework_data(roi='DR_R022', outdir=outdir)
        >>> region_fpaths = sorted(info1['true_region_dpath'].glob('*.geojson'))
        >>> site_fpaths = sorted(info1['true_site_dpath'].glob('*.geojson'))
        >>> manifest_fpath1 =  info1['output_dpath'] / 'demo_manifest1.json'
        >>> manifest_data1 = {'files': [str(p) for p in region_fpaths[0:2]]}
        >>> geojson_dpath = info1['true_site_dpath']
        >>> manifest_fpath1.write_text(json.dumps(manifest_data1))
        >>> # Test manifest case
        >>> geojson_fpaths = coerce_geojson_paths(manifest_fpath1)
        >>> assert len(geojson_fpaths) == 2
        >>> # Test directory case
        >>> geojson_fpaths = coerce_geojson_paths(geojson_dpath)
        >>> assert len(geojson_fpaths) == 15
        >>> # Test glob case
        >>> geojson_fpaths = coerce_geojson_paths(geojson_dpath / '*R001_*')
        >>> assert len(geojson_fpaths) == 3
        >>> # Test list of files and globstr
        >>> data = [geojson_dpath / '*R002_*'] + geojson_fpaths
        >>> geojson_fpaths = coerce_geojson_paths(data)
        >>> assert len(geojson_fpaths) == 6
        >>> # Test manifest case2
        >>> info = coerce_geojson_paths(manifest_fpath1, return_manifests=True)
        >>> assert len(info['manifest_fpaths']) == 1
        >>> assert len(info['geojson_fpaths']) == 2
    """
    from watch.utils import util_path
    paths = util_path.coerce_patterned_paths(data, '.geojson')
    geojson_fpaths = []
    manifest_fpaths = []
    for p in paths:
        resolved = None
        if isinstance(p, (str, os.PathLike)) and str(p).endswith('.json'):
            # Check to see if this is a manifest file
            peeked = json.loads(p.read_text())
            if isinstance(peeked, dict) and 'files' in peeked:
                manifest_fpaths.append(p)
                resolved = peeked['files']
        if resolved is None:
            resolved = [p]
        geojson_fpaths.extend(resolved)

    if return_manifests:
        return {
            'manifest_fpaths': manifest_fpaths,
            'geojson_fpaths': geojson_fpaths,
        }
    else:
        return geojson_fpaths


def _coerce_raw_geojson(item, format):
    """
    Helper for the coerce method
    """
    import geopandas as gpd
    was_raw = False

    if isinstance(item, dict):
        # Allow the item to be a wrapped dict returned by this func
        was_raw = True
        if set(item.keys()) == {'fpath', 'data', 'format'}:
            item = item['data']

    if isinstance(item, str):
        # Allow the item to be unparsed
        try:
            item = json.loads(item)
        except json.JSONDecodeError:
            ...  # not json data
        else:
            was_raw = True

    if isinstance(item, (os.PathLike, str)):
        ...  # not raw
    elif isinstance(item, dict):
        # Allow the item to be parsed json
        was_raw = True
        assert item.get('type', None) == 'FeatureCollection'
        if format == 'dataframe':
            item = gpd.GeoDataFrame.from_features(item['features'])
            # Hack in CRS-84
            crs84 = _get_crs84()
            # this is much faster and the only reason this is ok is because the
            # input is xy-wgs84 so the transform (which is slow) would be a noop
            item._crs = crs84
    elif isinstance(item, gpd.GeoDataFrame):
        # Allow the item to be a GeoDataFrame
        was_raw = True
        if format == 'json':
            item = json.loads(item.to_json())
    else:
        raise TypeError(type(item))
    if was_raw:
        item = {
            'fpath': None,
            'data': item,
            'format': format,
        }
    return was_raw, item


def _load_json_from_path(path):
    with open(path, 'r') as file:
        return json.load(file)


def load_geojson_datas(geojson_fpaths, format='dataframe', workers=0,
                       mode='thread', verbose=1, desc=None,
                       yield_after_submit=False):
    """
    Generator that loads sites (and the path they loaded from) in parallel

    Args:
        geojson_fpaths (Iterable[PathLike]):
            geojson paths to load

        workers (int):
            number of background loading workers

        mode (str):
            concurrent executor mode

        desc (str): overwrite message for the progress bar

        yield_after_submit (bool):
            backend argument that will yield None after the data is submitted
            to force the data loading to start processing in the background.

    Yields:
        Dict:
            containing keys, 'fpath' and 'gdf'.

    SeeAlso:
        * coerce_geojson_datas - the coercable version of this function.
        * coerce_geojson_paths - only coerces paths

    Example:
        >>> from watch.utils.util_gis import *  # NOQA
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('watch', 'tests', 'util_gis', 'load_geojson_data')
        >>> dpath.ensuredir()
        >>> fpath = dpath / 'data.geojson'
        >>> fpath.write_text(demo_regions_geojson_text())
        >>> # Test both format loaders work correctly.
        >>> gdf = list(load_geojson_datas([fpath], format='dataframe'))[0]['data']
        >>> dct = list(load_geojson_datas([fpath], format='json'))[0]['data']
        >>> import geopandas as gpd
        >>> assert isinstance(gdf, gpd.GeoDataFrame)
        >>> assert isinstance(dct, dict)
    """
    from watch.utils import util_gis
    # sites = []
    if desc is None:
        desc = 'load geojson datas'

    jobs = ub.JobPool(mode=mode, max_workers=workers)
    submit_progkw = {
        'desc': 'submit ' + desc,
        'verbose': (workers > 0) and verbose
    }

    if format == 'dataframe':
        loader = util_gis.load_geojson
    elif format == 'json':
        loader = _load_json_from_path
    else:
        raise KeyError(format)

    for fpath in ub.ProgIter(geojson_fpaths, **submit_progkw):
        job = jobs.submit(loader, fpath)
        job.fpath = fpath

    if yield_after_submit:
        yield None

    result_progkw = {
        'verbose': verbose,
    }
    for job in jobs.as_completed(desc=desc, progkw=result_progkw):
        data = job.result()
        info = {
            'fpath': job.fpath,
            'data': data,
            'format': format,
        }
        yield info
