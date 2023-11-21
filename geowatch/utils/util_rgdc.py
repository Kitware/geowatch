import numpy as np
from dateutil.parser import isoparse


def group_tiles(query, timediff_sec=300):
    """
    Given a RGDC search query result, nest results so that scenes
    that can be merged are grouped together

    TODO when RGDC STAC endpoint is available, calculate overlap with AOI here

    Args:
        query: output of rgdc.search
        timediff_sec: max allowed time between adjacent scenes.
            Should nominally be 23 seconds for Landsat 7 and 8. Some buffer is included.
            TODO what is it for Sentinel?
    Example:
        >>> # xdoctest: +SKIP
        >>> from rgd_client import Rgdc
        >>> client = Rgdc('username', 'password')
        >>> query = (client.search(**kwargs, instrumentation='ETM') +
        >>>          client.search(**kwargs, instrumentation='OLI_TIRS'))
        >>> # query == [scene1, scene2, scene3, scene4]
        >>> query = group_tiles(query)
        >>> # query == [[scene1], [scene2, scene3], [scene4]]
    """
    # ensure we're only working with one satellite at a time
    query = sorted(query, key=lambda q: isoparse(q['acquisition_date']))
    sensors, ixs = np.unique([q['instrumentation'] for q in query],
                             return_index=True)
    assert set(sensors).issubset({'ETM', 'OLI_TIRS', 'S2A', 'S2B'}), f'unsupported sensor in {sensors}'
    to_process = np.split(query, ixs[1:])

    def _path_row(q):
        """
        Returns LS path and row as strings. eg ('115', '034')
        """
        path_row = q['subentry_name'].split('_')[2]
        path, row = path_row[:3], path_row[3:]
        return path, row

    result = []
    for sensor, query in zip(sensors, to_process):
        # extract and split by datetimes
        dts = [isoparse(q['acquisition_date']) for q in query]
        diffs = np.diff([dt.timestamp() for dt in dts],
                        prepend=dts[0].timestamp())
        ixs = np.where(diffs > timediff_sec)[0]

        for split in np.split(query, ixs):
            # for Landsat, an additional check:
            # each split should consist of adjacent rows in the same path.
            if sensor in {'ETM', 'OLI_TIRS'}:
                path_rows = [_path_row(s) for s in split]
                paths, rows = zip(*path_rows)
                assert len(np.unique(paths)) == 1
                assert len(np.unique(rows)) == len(split)

            result.append(split)

    return result


def bands_landsat(entry):
    """
    Get only the band files from a Landsat RasterMetaEntry downloaded from RGD.

    Args:
        entry: RasterMetaEntry, the output type of rgdc.download_raster

    Returns:
        list of pathlib paths to band files
    """
    return [str(p) for p in entry.images if 'BQA' not in p.stem]


def bands_sentinel2(entry):
    """
    Get only the band files from a Sentinel-2 RasterMetaEntry downloaded from RGD.

    Args:
        entry: RasterMetaEntry, the output type of rgdc.download_raster

    Returns:
        list of pathlib paths to band files
    """
    return [str(p) for p in entry.images if p.match('*_B*.jp2')]
