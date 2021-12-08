import ubelt as ub


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
        dict: mapping from indexes in gdf1 to overlapping indexes in gdf2

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
    # Construct the spatial index (requires pygeos and/or rtree)
    sindex2 = gdf2.sindex
    # For each query polygon, lookup intersecting polygons in the spatial index
    idx1_to_idxs2 = {}
    for idx1, row1 in gdf1.iterrows():
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


def read_geojson(file, default_axis_mapping='OAMS_TRADITIONAL_GIS_ORDER'):
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
        >>> region_df = read_geojson(file)
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


@ub.memoize
def _get_crs84():
    """ This call can be fairly slow, so we cache it. """
    from pyproj import CRS
    crs84 = CRS.from_user_input('crs84')
    return crs84


def _flip(x, y):
    return (y, x)


def shapely_flip_xy(geom):
    from shapely import ops
    return ops.transform(_flip, geom)
