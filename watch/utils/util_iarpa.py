import json
import os
import ubelt as ub
from watch.utils import util_path


def coerce_region_or_site_datas(arg, format='dataframe',
                                allow_raw=False,
                                workers=0, mode='thread', verbose=1,
                                desc=None):
    """
    Attempts to resolve an argument into site / region model data.

    The argument can be:

            1. A path to a geojson file (or a list of them)

            2. A glob string specifying multiple geojson files (or a list of them)

            3. A path to a json manifest file.

    Multiple threads / processes are used to load the specified information and
    the function generates dictionaries of information containing the file path
    and the loaded data as they become available.

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
        >>> # xdoctest +REQURIES(module:iarpa_smart_metrics)
        >>> from watch.utils.util_iarpa import *  # NOQA
        >>> from iarpa_smart_metrics.demo import generate_demodata
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
        >>> result = list(coerce_region_or_site_datas(arg))
        >>> assert len(result) == 5
        >>> #
        >>> # Glob for region files
        >>> arg = str(info1['true_region_dpath']) + '/*R*2*.geojson'
        >>> result = list(coerce_region_or_site_datas(arg))
        >>> assert len(result) == 3
        >>> #
        >>> # Manifest file
        >>> arg = manifest_fpath1
        >>> result = list(coerce_region_or_site_datas(arg))
        >>> assert len(result) == 2
        >>> #
        >>> # Manifest file glob
        >>> arg = str(manifest_fpath1.parent / '*.json')
        >>> result = list(coerce_region_or_site_datas(arg))
        >>> assert len(result) == 3
        >>> #
        >>> # Manifest file glob and a region path
        >>> arg = [manifest_fpath2, region_fpaths[0]]
        >>> result = list(coerce_region_or_site_datas(arg))
        >>> assert len(result) == 2
        >>> #
        >>> # Site glob and a manifest glob
        >>> arg = [str(info1['true_site_dpath']) + '/DR_R002_*.geojson',
        ...        str(manifest_fpath1 + '*')]
        >>> result = list(coerce_region_or_site_datas(arg))
        >>> assert len(result) == 9
        >>> #
        >>> # Site directory and manifest file.
        >>> arg = [str(info1['true_site_dpath']),
        ...        str(manifest_fpath1 + '*')]
        >>> result = list(coerce_region_or_site_datas(arg))
        >>> assert len(result) == 31

        >>> # Test raw loading and format swapping
        >>> from watch.utils import util_gis
        >>> arg = util_gis.demo_regions_geojson_text()
        >>> result1 = list(coerce_region_or_site_datas(arg, allow_raw=False))
        >>> assert len(result1) == 0
        >>> result2 = list(coerce_region_or_site_datas(arg, allow_raw=True))
        >>> assert len(result2) == 1
        >>> arg = result2
        >>> result3 = list(coerce_region_or_site_datas(arg, format='dataframe', allow_raw=True))
        >>> assert result3 == result2
        >>> result4 = list(coerce_region_or_site_datas(arg, format='json', allow_raw=True))
        >>> assert isinstance(result4[0]['data'], dict)
        >>> result5 = list(coerce_region_or_site_datas(
        >>>     result4, format='dataframe', allow_raw=True))
        >>> assert isinstance(result4[0]['data'], dict)

        >>> #
        >>> # Test nothing case
        >>> assert len(list(coerce_region_or_site_datas([], allow_raw=True))) == 0
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
            was_raw, item = _coerce_raw_geodata(item, format)
            if was_raw:
                raw_items.append(item)
            else:
                other_items.append(item)
        path_coercable = other_items
    else:
        path_coercable = arg

    # Handle the normal better-defined case of coercing arguments into paths
    paths = util_path.coerce_patterned_paths(path_coercable, '.geojson')
    geojson_fpaths = []
    for p in paths:
        resolved = None
        if isinstance(p, (str, os.PathLike)) and str(p).endswith('.json'):
            # Check to see if this is a manifest file
            peeked = json.loads(p.read_text())
            if isinstance(peeked, dict) and 'files' in peeked:
                resolved = peeked['files']
        if resolved is None:
            resolved = [p]
        geojson_fpaths.extend(resolved)

    if allow_raw:
        if verbose:
            print(f'Coerced {len(raw_items)} raw geojson item')
            if raw_items:
                if len(geojson_fpaths) == 0:
                    # Disable path verbosity if there were raw items, but no
                    # paths.
                    verbose = 0

    # Now all of resolved accumulator items should be geojson files.
    # Submit the data to be loaded.
    geojson_fpaths = list(ub.unique(geojson_fpaths))
    data_gen = load_site_or_region_datas(
        geojson_fpaths, workers=workers, mode=mode, desc=desc,
        verbose=verbose,
        yield_after_submit=True)

    # Start the background workers
    next(data_gen)

    if allow_raw:
        # yield the raw data before the generated data
        yield from raw_items

    # Finish the main generator
    yield from data_gen


def _coerce_raw_geodata(item, format):
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

    if isinstance(item, dict):
        # Allow the item to be parsed json
        was_raw = True
        assert item.get('type', None) == 'FeatureCollection'
        if format == 'dataframe':
            item = gpd.GeoDataFrame.from_features(item['features'])
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


def load_site_or_region_datas(geojson_fpaths, format='dataframe', workers=0,
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
        * coerce_region_or_site_datas -
            the coercable version of this function.
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
        loader = util_gis.read_geojson
    elif format == 'json':
        loader = json.load
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
