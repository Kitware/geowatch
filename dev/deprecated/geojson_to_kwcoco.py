r"""

This script is for converting the IARPA geojson to kwcoco. It relies on some
initial preprocessing, which is listed here:

See Also:
    $HOME/data/dvc-repos/smart_watch_dvc/dev/prep_drop0.sh

Notes:

    # --- STEP 0 ---
    # Download the RAW data
    # export GIRDER_API_KEY=<your secret API key>
    # export GIRDER_USERNAME=<your girder username>

    mkdir -p $HOME/data/dvc-repos/smart_watch_dvc/raw/drop0
    cd $HOME/data/dvc-repos/smart_watch_dvc/raw/drop0
    girder-client --api-url https://data.kitware.com/api/v1 download 602458192fa25629b95d17d7

    cd $HOME/data/dvc-repos/smart_watch_dvc/drop0

    # Setup kwcoco bundles for each site separately
    mkdir -p AE-Dubai-0001/_assets
    ln -s ../../../raw/drop0/AE-Dubai-0001 AE-Dubai-0001/_assets/images

    mkdir -p BR-Rio-0270/_assets
    ln -s ../../../raw/drop0/BR-Rio-0270 BR-Rio-0270/_assets/images

    mkdir -p BR-Rio-0277/_assets
    ln -s ../../../raw/drop0/BR-Rio-0277 BR-Rio-0277/_assets/images

    mkdir -p US-Waynesboro-0001/_assets
    ln -s ../../../raw/drop0/US-Waynesboro-0001 US-Waynesboro-0001/_assets/images

    # TODO: (I ran a different script to unzip, we might need to modify this
    # bash version for it to work correctly)
    cd $HOME/data/dvc-repos/smart_watch_dvc/drop0
    7z x "../raw/drop0/KR-Pyeongchang/Sentinel 2/*.zip" KR-Pyeongchang-S2/_assets
    7z x "../raw/drop0/KR-Pyeongchang/WV/*.tar.gz" KR-Pyeongchang-WV/_assets

    # --- STEP 1 ---
    # Given this setup, we run this script as follows

    cd $HOME/data/dvc-repos/smart_watch_dvc/drop0

    python ~/code/watch/scripts/geojson_to_kwcoco.py \
        --src ~/data/dvc-repos/smart_watch_dvc/drop0/210210_D0_manualKR.geojson.json \
        --bundle_dpath ~/data/dvc-repos/smart_watch_dvc/drop0/BR-Rio-0277 \
        --visualize=True --ignore_dem=True

    python ~/code/watch/scripts/geojson_to_kwcoco.py \
        --src ~/data/dvc-repos/smart_watch_dvc/drop0/210210_D0_manualKR.geojson.json \
        --bundle_dpath ~/data/dvc-repos/smart_watch_dvc/drop0/BR-Rio-0270 \
        --visualize=True --ignore_dem=True

    python ~/code/watch/scripts/geojson_to_kwcoco.py \
        --src ~/data/dvc-repos/smart_watch_dvc/drop0/210210_D0_manualKR.geojson.json \
        --bundle_dpath ~/data/dvc-repos/smart_watch_dvc/drop0/AE-Dubai-0001 \
        --visualize=True --ignore_dem=False

    python ~/code/watch/scripts/geojson_to_kwcoco.py \
        --src ~/data/dvc-repos/smart_watch_dvc/drop0/210210_D0_manualKR.geojson.json \
        --bundle_dpath ~/data/dvc-repos/smart_watch_dvc/drop0/US-Waynesboro-0001 \
        --visualize=True --ignore_dem=False

    python ~/code/watch/scripts/geojson_to_kwcoco.py \
        --src ~/data/dvc-repos/smart_watch_dvc/drop0/210210_D0_manualKR.geojson.json \
        --bundle_dpath ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-S2 \
        --visualize=True --ignore_dem=False

    python ~/code/watch/scripts/geojson_to_kwcoco.py \
        --src ~/data/dvc-repos/smart_watch_dvc/drop0/210210_D0_manualKR.geojson.json \
        --bundle_dpath ~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV \
        --visualize=True --ignore_dem=False

    # --- STEP 2 ---
    # Combine into a single dataset

    cd $HOME/data/dvc-repos/smart_watch_dvc/drop0

    kwcoco reroot --src BR-Rio-0270/data.kwcoco.json        --dst BR-Rio-0270/data.kwcoco.json.abs --absolute=True
    kwcoco reroot --src BR-Rio-0277/data.kwcoco.json        --dst BR-Rio-0277/data.kwcoco.json.abs --absolute=True
    kwcoco reroot --src AE-Dubai-0001/data.kwcoco.json      --dst AE-Dubai-0001/data.kwcoco.json.abs --absolute=True
    kwcoco reroot --src KR-Pyeongchang-S2/data.kwcoco.json  --dst KR-Pyeongchang-S2/data.kwcoco.json.abs --absolute=True
    kwcoco reroot --src KR-Pyeongchang-WV/data.kwcoco.json  --dst KR-Pyeongchang-WV/data.kwcoco.json.abs --absolute=True
    kwcoco reroot --src US-Waynesboro-0001/data.kwcoco.json --dst US-Waynesboro-0001/data.kwcoco.json.abs --absolute=True

    kwcoco union --src  \
        AE-Dubai-0001/data.kwcoco.json.abs \
        BR-Rio-0270/data.kwcoco.json.abs \
        BR-Rio-0277/data.kwcoco.json.abs \
        KR-Pyeongchang-S2/data.kwcoco.json.abs \
        KR-Pyeongchang-WV/data.kwcoco.json.abs \
        US-Waynesboro-0001/data.kwcoco.json.abs \
        --dst drop0.kwcoco.json.abs

    kwcoco reroot --src drop0.kwcoco.json.abs --dst drop0.kwcoco.json --old_prefix="$PWD/" --new_prefix="" --absolute=False
    kwcoco validate drop0.kwcoco.json

"""
import json
import kwcoco
import kwimage
import os
import scriptconfig as scfg
import ubelt as ub
import numpy as np
from os.path import join, basename, exists


class GeojsonToCocoConfig(scfg.Config):
    """
    """
    default = {
        'src': scfg.Value('in.geojson.json'),

        'bundle_dpath': scfg.Value(None, help=ub.paragraph(
            '''
            output directory where the coco bundle should be. NOTE: MUST
            ALREADY HAVE IMAGES IN ITS _assets directory. SEE MODULE LEVEL
            DOCSTRING.
            ''')),

        'ignore_dem': scfg.Value(False, help=ub.paragraph(
            '''
            if True, we ignore the digital elevation map
            ''')),

        'site_tag': scfg.Value(None, help=ub.paragraph(
            '''
            if given, inserts this site_tag into each image dict in the output
            coco file.
            '''
        )),

        'visualize': scfg.Value(True, help=ub.paragraph(
            '''
            if True, we also write visualizations of annotation ROIs in pixel
            space.
            '''
        ))
    }


class Skip(Exception):
    pass


def base_no_ext(p):
    return ub.augpath(p, dpath='', ext='', multidot=True)


def simple_mapping(regi, have):
    regi_base_to_fpath = ub.group_items(regi, base_no_ext)
    have_base_to_fpath = ub.group_items(have, base_no_ext)
    regi_dups = {k: v for k, v in regi_base_to_fpath.items() if len(v) > 1}
    have_dups = {k: v for k, v in have_base_to_fpath.items() if len(v) > 1}
    if regi_dups:
        print('regi_dups = {}'.format(ub.repr2(regi_dups, nl=1)))
        print('num regi dups {}'.format(len(regi_dups)))
    if have_dups:
        print('have_dups = {}'.format(ub.repr2(have_dups, nl=1)))

    regi_base = set(regi_base_to_fpath)
    have_base = set(have_base_to_fpath)

    try:
        import xdev
        overlaps = xdev.set_overlaps(regi_base, have_base, 'regi', 'have')
        print('overlaps = {}'.format(ub.repr2(overlaps, nl=1)))
    except ImportError:
        pass

    regi_missing = ub.dict_diff(regi_base_to_fpath, have_base_to_fpath)
    print('regi_missing = {}'.format(ub.repr2(regi_missing, nl=1)))

    for key, value in regi_missing.items():
        for regi_fpath in value:
            no_ext = os.path.splitext(regi_fpath)[0]
            for have_path in have:
                if no_ext in have_path:
                    print('no_ext = {!r}'.format(no_ext))
                    print('have_path = {!r}'.format(have_path))

    common = set(have_base_to_fpath) & set(regi_base_to_fpath)
    mapping = {}

    for c in common:
        regi_candidates = regi_base_to_fpath[c]
        have_candidates = have_base_to_fpath[c]

        if len(regi_candidates) == 1:
            regi_fpath = regi_candidates[0]
        else:
            if len(have_candidates) == 1:
                regi_fpath = regi_candidates[0]
            else:
                print('regi_candidates = {!r}'.format(regi_candidates))
                raise AssertionError

        if len(have_candidates) == 1:
            have_fpath = have_candidates[0]
        else:
            raise AssertionError
            if len(regi_candidates) == 1:
                have_fpath = have_candidates[0]
            else:
                print('have_candidates = {!r}'.format(have_candidates))
                raise AssertionError
        mapping[regi_fpath] = have_fpath
    return mapping


def _associate_images(geojson, asset_dpath):
    """
    In drop0 the geojson file paths are not perfectly aligned with paths on
    disk. Furthermore, we assume all datasets assets have been moved to their
    own kwcoco bundle. This function tries to find any associations between in
    those assets and whats in the geojson.
    """

    assert exists(asset_dpath), (
        'image assets must already exist')

    # TODO : check if a DG bundle, and then only return relevant images
    # instead of using a blocklist
    blocklist = {'HTML', '_viz_crops'}

    all_image_fpaths = []
    for r, ds, fs in os.walk(asset_dpath, followlinks=True):
        for d in list(ds):
            if d in blocklist:
                ds.remove(d)
        for f in fs:
            fpath = join(r, f)
            if fpath.lower().endswith(kwimage.im_io.IMAGE_EXTENSIONS):
                all_image_fpaths.append(fpath)
    if 1:
        fname_to_fpath = ub.group_items(all_image_fpaths, basename)
        disk_dups = {k: v for k, v in fname_to_fpath.items() if len(v) > 1}

        if disk_dups:
            print('Found {} potential dup groups'.format(len(disk_dups)))

        unambiguous_dups = {}
        ambiguous_dups = {}
        for k, v in ub.ProgIter(disk_dups.items(), desc='check if dups are the same'):
            sizes = np.array([os.stat(f).st_size for f in v])
            # print('sizes = {!r}'.format(sizes))
            # if np.any(sizes <= 100):
            #     print('At least one corrupted image : {}'.format(v))
            v = list(ub.compress(v, sizes > 100))

            hashes = [ub.hash_file(x, hasher='xxh64') for x in v]
            if not ub.allsame(hashes):
                print('k = {!r}'.format(k))
                print('v = {!r}'.format(v))
                ambiguous_dups[k] = v
            else:
                unambiguous_dups[k] = v
        for k, v in unambiguous_dups.items():
            fname_to_fpath[k] = v[0:1]

        if 0:
            print('ambiguous_dups = {}'.format(ub.repr2(ambiguous_dups, nl=2)))
            print('unambiguous_dups = {}'.format(ub.repr2(unambiguous_dups, nl=2)))

        all_image_fpaths = list(ub.flatten(fname_to_fpath.values()))

    registered_fpaths = []
    meta = geojson['metadata']
    for img in meta['images']:
        # height might be wrong
        img = img.copy()
        fnames = img['file_name']
        if isinstance(fnames, list):
            assert len(fnames) == 1
            fname = fnames[0]
        else:
            fname = fnames
        registered_fpaths.append(fname)

    assert not ub.find_duplicates(registered_fpaths)
    registered_fpaths = sorted(registered_fpaths)

    regi = registered_fpaths
    have = all_image_fpaths

    all_mappings = simple_mapping(regi, have, )
    return all_mappings


def hack_resolve_sensor_candidate(dset):
    """
    Make a sensor code that coarsely groups the different sensors

    set([tuple(sorted(g.get('sensor_candidates'))) for g in dset.imgs.values()])
    """
    known_mapping = {
        'GE01': 'WV',
        'WV01': 'WV',
        'WV02': 'WV',
        'WV03': 'WV',
        'WV03_VNIR': 'WV',
        'WV03_SWIR': 'WV',

        'LC08': 'LC',

        'S2A': 'S2',
        'S2B': 'S2',
        'S2-TrueColor': 'S2',
    }

    for img in dset.imgs.values():
        coarsend = {known_mapping[k] for k in img['sensor_candidates']}
        assert len(coarsend) == 1
        img['sensor_coarse'] = ub.peek(coarsend)


def main(**kw):
    """
    Ignore:
        cd ~/data/dvc-repos/smart_watch_dvc/raw/drop0

        import sys, ubelt
        sys.path.append(ubelt.expandpath('~/code/watch/scripts'))
        from geojson_to_kwcoco import *  # NOQA
        from geojson_to_kwcoco import _associate_images

        {'GE01', 'WV01', 'WV02', 'WV03', 'WV03_SWIR', 'WV03_VNIR'}

        kw = {
            'src': ub.expandpath('~/data/dvc-repos/smart_watch_dvc/raw/drop0/210210_D0_manualKR.geojson.json'),
            # 'bundle_dpath': ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/AE-Dubai-0001'),
            # 'bundle_dpath': ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-S2'),
            # Still has issues:
            'bundle_dpath': ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/BR-Rio-0277'),
            # 'bundle_dpath': ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/BR-Rio-0270'),
            # 'bundle_dpath': ub.expandpath('~/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV'),
        }
    """
    from ndsampler.utils.util_gdal import LazyGDalFrameFile
    from watch.gis.geotiff import geotiff_metadata
    import itertools as it

    # Parse commandline arguments
    config = GeojsonToCocoConfig(default=kw, cmdline=True)

    visualize = config['visualize']

    # Where we will create the COCO bundle
    bundle_dpath = config['bundle_dpath']

    # Read the source IARPA geojson
    src = config['src']
    with open(src, 'r') as file:
        geojson = json.load(file)

    # The provided geojson does not reference exact paths.
    # So we have to list out all of the available files and try to find an
    # association between the filenames registered in IARPA's geojson metadata
    # and what we actually have on disk.
    asset_dpath = join(bundle_dpath, '_assets')
    all_mappings = _associate_images(geojson, asset_dpath)
    print('all_mappings = {}'.format(ub.repr2(all_mappings, nl=1)))

    meta = geojson['metadata']

    # We will create a CocoDataset to store all of the information in a
    # well-formated data structure.
    dset = kwcoco.CocoDataset()

    # Copy the "info" from the geojson to the COCO
    dset.dataset['info'] = meta['info']

    # Copy the "categories" from the geojson to the COCO
    for cat in meta['categories']:
        dset.add_category(**cat)

    # Given all of the images in the geojson, register (in the CocoDataset) all
    # images that we were able to unamgiuously associate.
    gid_to_metadata = {}
    for img in ub.ProgIter(meta['images'], desc='register images'):
        img = img.copy()
        fnames = img.pop('file_name')
        if isinstance(fnames, list):
            assert len(fnames) == 1
            fname = fnames[0]
        else:
            fname = fnames

        if fname in all_mappings:
            fpath = all_mappings[fname]
            height = img.pop('height', 0)
            width = img.pop('width', 0)

            if os.stat(fpath).st_size < 10:
                continue

            # height, width = kwimage.load_image_shape(fpath)[0:2]

            if config['ignore_dem']:
                info = geotiff_metadata(fpath, elevation=False)
                img['dem_hint'] = 'ignore'
            else:
                info = geotiff_metadata(fpath)
                img['dem_hint'] = 'use'

            height, width = info['img_shape']

            gid = dset.add_image(file_name=fpath, height=height, width=width, **img)

            gid_to_metadata[gid] = info

            info['gpath'] = fpath
            img = dset.imgs[gid]
            img['approx_elevation'] = info['approx_elevation']
            img['approx_meter_gsd'] = info['approx_meter_gsd']
            img['sensor_candidates'] = sorted(set(info['sensor_candidates']))
            img['num_bands'] = info['num_bands']

            if config['site_tag']:
                img['site_tag'] = config['site_tag']
        else:
            continue

    # Use CocoDataset.reroot to change all paths to be relative to where will
    # will ultimately write the json file.
    dset.reroot(
        new_root=bundle_dpath, old_prefix=bundle_dpath, new_prefix='',
        absolute=False)

    if False:
        # Developer checks / info

        # Find annots that reference multiple images
        multi_gids = []
        for feat in geojson['features']:
            ann_meta = feat['metadata']
            gid_spec = ann_meta.get('image_id')
            if isinstance(gid_spec, list):
                multi_gids.append(tuple(sorted(gid_spec)))
        multi_gid_hist = ub.dict_hist(multi_gids)

        # Are the multi-gids disjoint?
        for key1, key2 in it.combinations(multi_gid_hist, 2):
            if len(set(key1) & set(key2)):
                print('key1 = {!r}'.format(key1))
                print('key2 = {!r}'.format(key2))
                raise Exception('multi-gids not disjoint')

    # Gather annotations that belong to this dataset
    # from kwimage.structs.polygon import _order_vertices
    # verts = _order_vertices(verts)

    hack_resolve_sensor_candidate(dset)

    if False:
        # For debugging
        import kwplot
        kwplot.autoplt()
        import geopandas as gpd
        wld_map_gdf = gpd.read_file(
            gpd.datasets.get_path('naturalearth_lowres')
        )
        ax = wld_map_gdf.plot()

    toconvert_anns = []
    bad_gids = []
    bad_aids = []
    feat_prog = ub.ProgIter(geojson['features'], desc='load anns')
    for feat in feat_prog:
        ann = {}
        ann_meta = feat['metadata'].copy()

        annotators = ann_meta.pop('annotators', [])
        annotator = ann_meta.pop('annotator', None)
        if annotator is not None:
            annotators.append(annotator)
        ann['annotators'] = annotators
        ann['segmentation_geos'] = feat['geometry']
        gid_spec = ann_meta.pop('image_id')
        ann.update(ann_meta)

        exterior = kwimage.Coords(np.array(ann['segmentation_geos']['coordinates'])[:, ::-1])
        kw_ann_poly = kwimage.Polygon(exterior=exterior.data[:, ::-1])
        sh_ann_poly = kw_ann_poly.to_shapely()

        if False:
            # For debugging
            import kwplot
            kwplot.autoplt()
            import geopandas as gpd
            wld_map_gdf = gpd.read_file(
                gpd.datasets.get_path('naturalearth_lowres')
            )
            ax = wld_map_gdf.plot()
            kw_ann_poly.draw(alpha=1.0, ax=ax, color='red')
            ax.set_ylim(37.6, 37.85)
            ax.set_xlim(128.6, 129)

            gid_list = gid_spec
            for gid in gid_list:
                if gid in dset.imgs:
                    info = gid_to_metadata[gid]
                    print(info['gpath'])
                    kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'].data[:, ::-1])
                    sh_img_poly = kw_img_poly.to_shapely()
                    print(sh_img_poly.intersects(sh_ann_poly))

                    # kw_gcp_poly = kwimage.Polygon(exterior=info['gcp_wld_coords'])
                    # kw_gcp_poly.draw(alpha=0.5, color='orange', ax=ax)
                    kw_img_poly.draw(alpha=0.5, color='green', ax=ax, border=True)

        orig_aid = ann['id']
        if isinstance(gid_spec, list):
            orig_aid = ann.pop('id')
            # print('gid_spec = {!r}'.format(gid_spec))
            # print('orig_aid = {!r}'.format(orig_aid))
            gid_list = gid_spec
        else:
            gid_list = [gid_spec]

        # maintain where we came from
        ann['orig_image_ids'] = gid_list
        ann['orig_aid'] = orig_aid

        flags = []
        for gid in gid_list:
            if gid in dset.imgs:
                info = gid_to_metadata[gid]
                kw_img_poly = kwimage.Polygon(exterior=info['wgs84_corners'].data[:, ::-1])
                sh_img_poly = kw_img_poly.to_shapely()
                if sh_img_poly.intersects(sh_ann_poly):
                    ann = ann.copy()
                    ann['image_id'] = gid
                    toconvert_anns.append(ann)
                    flags.append('does-intersect')
                else:
                    flags.append('does-not-intersect')
                    bad_gids.append(gid)
            else:
                flags.append('does-not-belong')

        if set(flags) == set(['does-not-intersect']):
            feat_prog.ensure_newline()
            print('OOB orig_aid = {}, orig_gids={}'.format(orig_aid, gid_list))
            for gid in gid_list:
                print('gpath = {}'.format(dset.imgs[gid]['file_name']))
            bad_aids.append(orig_aid)

    assert not ub.find_duplicates([ann['id'] for ann in toconvert_anns if 'id' in ann])

    gid_to_anns = ub.group_items(
        toconvert_anns, lambda ann: ann['image_id'])

    for gid, anns in gid_to_anns.items():
        for ann in anns:
            assert ann['image_id'] == gid

    max_aid = 0
    for ann in toconvert_anns:
        max_aid = max(max_aid, ann.get('id', 0))

    # Fix ids for duplicate anns
    for ann in toconvert_anns:
        if 'id' not in ann:
            max_aid = ann['id'] = max_aid + 1
            print('max_aid = {!r}'.format(max_aid))

    assert not ub.find_duplicates([ann['id'] for ann in toconvert_anns if 'id' in ann])

    total_polys = 0
    total_any_OOB = 0
    total_all_OOB = 0

    # Warp annotations from world space to pixel space
    valid_anns = []
    group_prog = ub.ProgIter(gid_to_anns.items(), desc='warp anns', verbose=1)
    for gid, anns in group_prog:
        gpath = dset.get_image_fpath(gid)
        if os.stat(gpath).st_size < 10:
            continue

        assert gid in dset.imgs
        img = dset.imgs[gid]
        anns = gid_to_anns[gid]
        gpath = dset.get_image_fpath(gid)

        # Parse out metadata about the Coordinate System of this GeoTiff
        info = gid_to_metadata[gid]

        height, width = info['img_shape']
        img['height'] = height
        img['width'] = width

        geo_poly_list = []
        for ann in anns:
            # Q: WHAT FORMAT ARE THESE COORDINATES IN?
            # A: I'm fairly sure these coordinates are all Traditional-WGS84-Lon-Lat
            # We convert them to authority compliant WGS84 (lat-lon)
            exterior = kwimage.Coords(np.array(ann['segmentation_geos']['coordinates'])[:, ::-1])
            geo_poly = kwimage.Polygon(exterior=exterior)
            geo_poly_list.append(geo_poly)

        geo_polys = kwimage.MultiPolygon(geo_poly_list)

        # Warp Auth-WGS84 to whatever the image world space is, and then from
        # there to pixel space.
        pxl_polys = geo_polys.warp(info['wgs84_to_wld']).warp(info['wld_to_pxl'])

        def _test_inbounds(pxl_poly):
            xs, ys = pxl_poly.data['exterior'].data.T
            flags_x1 = xs < 0
            flags_y1 = ys < 0
            flags_x2 = xs >= img['width']
            flags_y2 = ys >= img['height']
            flags = flags_x1 | flags_x2 | flags_y1 | flags_y2
            n_oob = flags.sum()
            is_any = n_oob > 0
            is_all = n_oob == len(flags)
            return is_any, is_all

        is_any_oob = []
        is_all_oob = []

        is_any_info = []
        is_all_info = []
        for ann, pxl_poly in zip(anns, pxl_polys.data):
            is_any, is_all = _test_inbounds(pxl_poly)
            is_any_oob.append(is_any)
            is_all_oob.append(is_all)

            if is_any:
                is_any_info.append({
                    'orig_aid': ann['orig_aid'],
                    'orig_gids': ann['orig_image_ids']})

            if is_all:
                is_all_info.append({
                    'orig_aid': ann['orig_aid'],
                    'orig_gids': ann['orig_image_ids']})

            ann['segmentation'] = pxl_poly.to_coco(style='new')
            pxl_box = pxl_poly.bounding_box().quantize().to_xywh()
            xywh = list(pxl_box.to_coco())[0]
            ann['bbox'] = xywh

        n_is_all = sum(is_all_oob)
        n_is_any = sum(is_any_oob)

        if n_is_all or n_is_any:
            group_prog.ensure_newline()
            # if n_is_any or n_is_all:
            print('gpath = {!r}'.format(gpath))
            print('gid = {!r}'.format(gid))
            print('{} / {} Any OOB Polys'.format(sum(is_any_oob), len(is_any_oob)))
            print('{} / {} All OOB Polys'.format(sum(is_all_oob), len(is_all_oob)))
            print('is_any_info = {}'.format(ub.repr2(is_any_info, nl=1)))
            print('is_all_info = {}'.format(ub.repr2(is_all_info, nl=1)))
            print('---')

        total_any_OOB += sum(is_any_oob)
        total_all_OOB += sum(is_all_oob)
        total_polys += len(is_all_oob)

        if visualize:
            dets = kwimage.Detections.from_coco_annots(anns, dset=dset)
            window = dets.boxes.bounding_box().quantize()
            window = window.to_cxywh().scale(3.0, about='center')

            new_dim = max(window.width.ravel()[0], window.height.ravel()[0])
            window.data[:, 2:4] = new_dim

            window = window.to_ltrb().quantize()
            min_x, min_y, max_x, max_y = [c.ravel()[0] for c in window.components]
            sl = tuple([slice(min_y, max_y), slice(min_x, max_x)])

            frame = LazyGDalFrameFile(gpath)
            imdata, crop_transform = kwimage.padded_slice(
                frame, sl, return_info=True)
            subdets = dets.translate((-min_x, -min_y))

            # TODO: use a more robust name that is gaurenteed to not conflict
            # should be fine for drop0 datasets.
            viz_dpath = ub.ensuredir((bundle_dpath, '_viz_crops'))
            base = basename(gpath)
            suffix = '_crop_xywh_{}_{}_{}_{}'.format(*window.to_xywh().data[0])
            viz_gpath = ub.augpath(base, suffix=suffix, dpath=viz_dpath, ext='.jpg')

            from watch.utils.util_norm import normalize_intensity
            canvas, _info = normalize_intensity(imdata, return_info=True)
            canvas = kwimage.atleast_3channels(canvas)

            debug_text = suffix
            canvas = kwimage.draw_text_on_image(
                canvas, debug_text, (5, 5), valign='top', color='red')

            canvas = subdets.draw_on(canvas, color='green')

            canvas = kwimage.ensure_uint255(canvas)
            kwimage.imwrite(viz_gpath, canvas)

            if 0:
                from watch.gis.geotiff import geotiff_crs_info
                info2 = geotiff_crs_info(info['gpath'], force_affine=True)

                rpc_tf = info2['rpc_transform']

                manual_subpolys = []
                for wgs_trad_poly in geo_polys.swap_axes().data:
                    pts_lonlat = wgs_trad_poly.data['exterior'].data
                    pts_out = rpc_tf.warp_pixel_from_world(pts_lonlat, return_elevation=False)
                    pts_xy = pts_out

                    zs = np.full((len(pts_lonlat), 1), fill_value=0)
                    pts_lonlatz = np.hstack([pts_lonlat, zs])
                    pts_out = rpc_tf.warp_pixel_from_world(pts_lonlatz, return_elevation=True)
                    pts_xy = pts_out[:, 0:2]

                    poly3 = kwimage.Polygon(exterior=pts_xy)
                    subpoly3 = poly3.translate((-min_x, -min_y))
                    manual_subpolys.append(subpoly3)

                subpolys4 = kwimage.MultiPolygon(manual_subpolys)
                # lats, lons = pts_in.T
                # rpc_tf.elevation.query(lon, lat)
                # pxl_polys2 = geo_polys.warp(info2['wgs84_to_wld']).warp(info2['wld_to_pxl'])
                # subpxl_polys2 = pxl_polys2.translate((-min_x, -min_y))
                canvas = subpolys4.draw_on(canvas, color='red', alpha=0.4)

            if 0:
                import kwplot
                kwplot.autompl()
                title = gpath.split('drop0')[1] + '\n' + str(window.to_xywh())
                kwplot.imshow(canvas, title=title)

        valid_anns.extend(anns)

    print('TOTAL Polys {!r}'.format(total_polys))
    print('TOTAL {} / {} Any OOB Polys'.format(total_any_OOB, total_polys))
    print('TOTAL {} / {} All OOB Polys'.format(total_all_OOB, total_polys))

    ub.find_duplicates([ann['id'] for ann in valid_anns if 'id' in ann])

    for ann in valid_anns:
        dset.add_annotation(**ann)

    if 0:
        dates = []
        for ann in dset.anns.values():
            gid = ann['image_id']
            img = dset.imgs[gid]
            import datetime
            date = datetime.datetime.strptime(img['date_captured'], '%Y/%m/%d')
            dates.append(date)
        min(dates), max(dates)
        # for date in dates:
        #     datetime.datetime.strftime('YY/mm/dd', date)

    dset.fpath = join(bundle_dpath, 'data.kwcoco.json')
    print('dset.fpath = {!r}'.format(dset.fpath))
    dset.dump(dset.fpath, newlines=True)

    """
        kwcoco stats drop0_partial_v2.kwcoco.json

        python ~/code/diyharn/dev/coco_tools/coco_bbox_analysis.py --src=drop0_partial_v2.kwcoco.json \
                --dpath=bbox_analysis_partial_v2 --attrs=rt_area,width,height,dlen

        python ~/code/diyharn/dev/coco_tools/coco_bbox_analysis.py --src=drop0_partial_v2.kwcoco.json \
                --dpath=bbox_analysis_partial_vr --attrs=rt_area, --log_scale=2 --with_kde=True

        python ~/code/video_caption/dev/coco_visual_summary.py \
                --src=drop0_partial_v2.kwcoco.json

    """


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/scripts/geojson_to_kwcoco.py
    """
    main()
