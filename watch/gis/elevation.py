"""
Tools for querying for elevation data.
"""
import requests
import time
import random
import ubelt as ub


def query_open_elevation(lat, lon, cache=True, attempts=10, verbose=0):
    """
    Use open-elevation to query the elevation for a lat/lon point.

    This issues a web request, so it can be slow.

    Args:
        lat (float): degrees in latitude
        lon (float): degrees in longitude
        cache (bool): if True uses on-disk caching
        attempts (int): number of attempts before giving up
        verbose (int): verbosity flag

    Returns:
        float : elevation in meters

    Notes:
        TODO: should download elevation maps locally:
            https://data.kitware.com/#collection/59eb64168d777f31ac6477e7/folder/59fb784d8d777f31ac6480fb
            https://www.google.com/url?sa=j&url=https%3A%2F%2Fwww.usgs.gov%2Fcenters%2Feros%2Fscience%2Fusgs-eros-archive-digital-elevation-global-30-arc-second-elevation-gtopo30%3Fqt-science_center_objects%3D0%23qt-science_center_objects&uct=1599876275&usg=jBvv8w64RCBJd2SyQA3kUtKhMQ4.&source=chat

    References:
        https://gis.stackexchange.com/questions/338392/getting-elevation-for-multiple-lat-long-coordinates-in-python
        https://gis.stackexchange.com/questions/212106/seeking-alternative-to-google-maps-elevation-api
        https://open-elevation.com/
        https://www.freemaptools.com/elevation-finder.htm

    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> from watch.gis.elevation import *  # NOQA
        >>> lat = 37.65026538818887
        >>> lon = 128.81096081618637
        >>> elevation = query_open_elevation(lat, lon, verbose=3)
        >>> print('elevation = {!r}'.format(elevation))
        elevation = 449
    """
    url = 'https://api.open-elevation.com/api/v1/lookup?'
    suffix = 'locations={},{}'.format(float(lat), float(lon))
    query_url = url + suffix

    cacher = ub.Cacher('elevation', depends=query_url,
                       appname='smart-watch/elevation_query', verbose=verbose)
    body = cacher.tryload()
    if body is None:
        for i in range(attempts):
            result = requests.get(query_url)
            if result.status_code != 200:
                if verbose:
                    print('REQUEST FAILED')
                    print(result.text)
                    print('RETRY')
                time.sleep(3 + random.random() * 3)
            else:
                body = result.json()
                break
        if body is None:
            raise Exception('Failed to query')
        cacher.save(body)
    elevation = body['results'][0]['elevation']
    return elevation
