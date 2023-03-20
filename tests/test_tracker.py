def test_tracker():
    # test BAS and default (SC) modes
    # from watch.cli.kwcoco_to_geojson import *  # NOQA
    import datetime
    import itertools
    import json
    import os
    import sys
    from collections import defaultdict
    from typing import Dict, List, Tuple, Union

    import dateutil.parser
    import geojson
    import jsonschema
    import kwcoco
    import numpy as np
    import pandas as pd
    import shapely
    import shapely.ops
    import ubelt as ub
    import scriptconfig as scfg

    from watch.tasks.tracking import from_heatmap, from_polygon
    from watch.utils.kwcoco_extensions import sorted_annots

    from watch.cli.kwcoco_to_geojson import main
    from watch.demo import smart_kwcoco_demodata
    from watch.utils import util_gis
    import kwcoco
    import ubelt as ub
    # run BAS on demodata in a new place
    import watch
    coco_dset = watch.coerce_kwcoco('watch-msi', heatmap=True, geodata=True, dates=True)
    #coco_dset = smart_kwcoco_demodata.demo_smart_aligned_kwcoco()
    dpath = ub.Path.appdir('watch', 'test', 'tracking', 'unit_test1').ensuredir()
    coco_dset.reroot(absolute=True)
    coco_dset.fpath = dpath / 'bas_input.kwcoco.json'
    coco_dset.clear_annotations()
    coco_dset.dump(coco_dset.fpath, indent=2)
    region_id = 'dummy_region'
    regions_dir = dpath / 'regions/'
    bas_coco_fpath = dpath / 'bas_output.kwcoco.json'
    sc_coco_fpath = dpath / 'sc_output.kwcoco.json'
    sv_coco_fpath = dpath / 'sv_output.kwcoco.json'
    bas_fpath = dpath / 'bas_sites.json'
    sc_fpath = dpath / 'sc_sites.json'
    sv_fpath = dpath / 'sv_sites.json'
    # Run BAS
    args = bas_args = [
        '--in_file', coco_dset.fpath,
        '--out_site_summaries_dir', str(regions_dir),
        '--out_site_summaries_fpath',  str(bas_fpath),
        '--out_kwcoco', str(bas_coco_fpath),
        '--track_fn', 'saliency_heatmaps',
        '--track_kwargs', json.dumps({
           'thresh': 1e-9, 'min_area_square_meters': None,
           'max_area_square_meters': None,
           'polygon_simplify_tolerance': 1}),
    ]
    main(args)
    # Run SC on the same dset
    sites_dir = dpath / 'sites'
    args = sc_args = [
        '--in_file', str(bas_coco_fpath),
        '--out_sites_dir', str(sites_dir),
        '--out_sites_fpath', str(sc_fpath),
        '--out_kwcoco', str(sc_coco_fpath),
        '--track_fn', 'class_heatmaps',
        '--site_summary', str(bas_fpath),
        '--track_kwargs', json.dumps(
            {'thresh': 1e-9, 'min_area_square_meters': None, 'max_area_square_meters': None,
             'polygon_simplify_tolerance': 1, 'key': 'salient'}),
    ]
    main(args)
    # Run SV on the same dset
    sites_dir = dpath / 'sv_sites'
    args = sv_args = [
        '--in_file', str(sc_coco_fpath),
        '--out_sites_dir', str(sites_dir),
        '--out_sites_fpath', str(sv_fpath),
        '--out_kwcoco', str(sv_coco_fpath),
        '--track_fn', 'site_validation',
        '--site_summary', str(bas_fpath),
        '--track_kwargs', json.dumps(
            {'thresh': 1e-9, 'min_area_square_meters': None, 'max_area_square_meters': None,
             'polygon_simplify_tolerance': 1, 'key': 'salient', 'thresh': 0}),
    ]
    main(args)
    # Check expected results
    bas_coco_dset = kwcoco.CocoDataset(bas_coco_fpath)
    sc_coco_dset = kwcoco.CocoDataset(sc_coco_fpath)
    bas_trackids = bas_coco_dset.annots().lookup('track_id', None)
    sc_trackids = sc_coco_dset.annots().lookup('track_id', None)
    assert len(bas_trackids) and None not in bas_trackids
    assert len(sc_trackids) and None not in sc_trackids
    summaries = list(util_gis.coerce_geojson_datas(bas_fpath, format='dataframe'))
    sites = list(util_gis.coerce_geojson_datas(sc_fpath, format='dataframe'))
    sc_df = pd.concat([d['data'] for d in sites])
    bas_df = pd.concat([d['data'] for d in summaries])
    ssum_rows = bas_df[bas_df['type'] == 'site_summary']
    site_rows = sc_df[sc_df['type'] == 'site']
    obs_rows = sc_df[sc_df['type'] == 'observation']
    assert len(site_rows) > 0
    assert len(ssum_rows) > 0
    assert len(ssum_rows) == len(site_rows)
    assert len(ssum_rows) == len(site_rows)
    assert len(obs_rows) > len(site_rows)
    # Cleanup
    dpath.delete()