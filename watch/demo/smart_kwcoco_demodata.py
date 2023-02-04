"""
Extends kwcoco demodata to be more smart-like
"""
from dateutil.parser import isoparse
from os.path import dirname
from os.path import join
from watch.cli import coco_align_geotiffs
from watch.cli import geotiffs_to_kwcoco
from watch.demo import landsat_demodata
from watch.demo import sentinel2_demodata
import datetime
import geopandas as gpd
import kwarray
import kwcoco
import kwimage
import numpy as np
import ubelt as ub
import watch


def demo_smart_raw_kwcoco():
    """
    Creates a kwcoco dataset that attempts to exhibit common corner cases with
    special watch geospatial attributes. This is a raws dataset of
    tiles with annotations that are not organized into videos.

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> raw_coco_dset = demo_smart_raw_kwcoco()
        >>> print('raw_coco_dset = {!r}'.format(raw_coco_dset))
    """
    cache_dpath = ub.Path.appdir('watch', 'demo', 'kwcoco').ensuredir()
    raw_coco_fpath = join(cache_dpath, 'demo_smart_raw.kwcoco.json')
    stamp = ub.CacheStamp('raw_stamp', dpath=cache_dpath, depends=['v4'],
                          product=raw_coco_fpath)
    if stamp.expired():
        s2_demo_paths1 = sentinel2_demodata.grab_sentinel2_product(index=0)
        s2_demo_paths2 = sentinel2_demodata.grab_sentinel2_product(index=1)
        l8_demo_paths = landsat_demodata.grab_landsat_product()

        s2_demo_paths1.images
        s2_demo_paths2.images

        raw_coco_dset = kwcoco.CocoDataset()

        categories = [
            'Active Construction',
            'Site Preparation',
            'Post Construction',
            'Unknown'
        ]
        for catname in categories:
            raw_coco_dset.add_category(catname)

        rng = kwarray.ensure_rng(542370, api='python')

        # Add only the B01 for the first S2 image
        cands = [fname for fname in s2_demo_paths1.images
                 if fname.name.endswith('B01.jp2')]
        if len(cands) != 1:
            raise AssertionError(ub.paragraph(
                f'''
                Should only have 1 candidate. Got {len(cands)}.
                cands={cands}.
                '''))

        assert len(cands) == 1
        fname = cands[0]
        fpath = str(s2_demo_paths1.path / fname)
        img = geotiffs_to_kwcoco.make_coco_img_from_geotiff(fpath)
        baseinfo = watch.gis.geotiff.geotiff_filepath_info(fpath)
        capture_time = isoparse(baseinfo['filename_meta']['sense_start_time'])
        img['date_captured'] = datetime.datetime.isoformat(capture_time)
        img['sensor_coarse'] = 'S2'
        raw_coco_dset.add_image(**img)
        # Fix issue
        try:
            raw_coco_dset.imgs[1]['warp_pxl_to_wld'] = raw_coco_dset.imgs[1]['warp_pxl_to_wld'].concise()
        except Exception:
            pass

        # Add all bands for the seconds S2 image
        img = geotiffs_to_kwcoco.ingest_sentinel2_directory(str(s2_demo_paths2.path))
        raw_coco_dset.add_image(**img)

        # Add all bands of the L8 image
        lc_dpath = dirname(l8_demo_paths['bands'][0])
        img = geotiffs_to_kwcoco.ingest_landsat_directory(lc_dpath)
        raw_coco_dset.add_image(**img)

        # Add random annotations on each image
        expected_geos_crs_info = {
            'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
            'auth': ('EPSG', '4326')
        }
        for img in raw_coco_dset.imgs.values():
            geos_crs_info = img['geos_corners']['properties']['crs_info']
            assert geos_crs_info == expected_geos_crs_info
            goes_corners = kwimage.Polygon.coerce(img['geos_corners']).to_shapely()
            utm_gdf = gpd.GeoDataFrame(
                {'geometry': [goes_corners]},
                geometry='geometry', crs=geos_crs_info['auth'])
            crs_corners = utm_gdf.to_crs('crs84').geometry.iloc[0]
            corner_poly = kwimage.Polygon.from_shapely(crs_corners)

            # Create a dummy annotation which is a scaled down version of the corner points
            dummy_sseg_geos = corner_poly.scale(0.02, about='center')
            random_cat = rng.choice(raw_coco_dset.dataset['categories'])
            raw_coco_dset.add_annotation(
                image_id=img['id'],
                category_id=random_cat['id'],
                segmentation_geos=dummy_sseg_geos.to_geojson())

        raw_coco_dset.fpath = raw_coco_fpath
        raw_coco_dset.validate()
        raw_coco_dset.dump(raw_coco_dset.fpath, newlines=True)
        stamp.renew()

    raw_coco_dset = kwcoco.CocoDataset(raw_coco_fpath)
    return raw_coco_dset


def demo_smart_aligned_kwcoco():
    """
    This is an aligned dataset of videos

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> aligned_coco_dset = demo_smart_aligned_kwcoco()
        >>> print('aligned_coco_dset = {!r}'.format(aligned_coco_dset))
    """
    cache_dpath = ub.Path.appdir('watch/demo/kwcoco').ensuredir()
    aligned_kwcoco_dpath = join(cache_dpath, 'demo_aligned')
    aligned_coco_fpath = join(aligned_kwcoco_dpath, 'data.kwcoco.json')
    stamp = ub.CacheStamp('aligned_stamp', dpath=cache_dpath, depends=['v2'],
                          product=[aligned_coco_fpath])
    if stamp.expired():
        raw_coco_dset = demo_smart_raw_kwcoco()
        coco_align_geotiffs.main(
            src=raw_coco_dset,
            regions='annots',
            max_workers=0,
            context_factor=3.0,
            dst=aligned_kwcoco_dpath,
        )
        stamp.renew()

    aligned_coco_dset = kwcoco.CocoDataset(aligned_coco_fpath)
    return aligned_coco_dset


def demo_kwcoco_with_heatmaps(num_videos=1, num_frames=20, image_size=(512, 512)):
    """
    Return a dummy kwcoco file with special metdata

    DEPRECATED:
        Instead use watch.coerce_kwcoco('watch-msi-geodata-dates-heatmap-videos1-frames20-gsize512') or something similar

    TODO:
        rename

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> coco_dset = demo_kwcoco_with_heatmaps()

        key = 'salient'
        for vidid in coco_dset.videos():
            frames = []
            for gid in coco_dset.images(vidid=vidid):
                delayed = coco_dset._coco_image(gid).delay(channels=key, space='video')
                final = delayed.finalize()
                frames.append(final)
            vid_stack = kwimage.stack_images_grid(frames, axis=1, pad=5, bg_value=1)

            import kwplot
            kwplot.imshow(vid_stack)
    """
    import kwarray
    import kwcoco
    from kwarray.distributions import DiscreteUniform  # NOQA

    rng = 101893676  # random seed
    rng = kwarray.ensure_rng(rng)

    # img_w = DiscreteUniform(256, 512, rng=rng)
    # img_h = DiscreteUniform(256, 512, rng=rng)
    # image_size = (img_w, img_h)

    coco_dset = kwcoco.CocoDataset.demo(
        'vidshapes', num_videos=num_videos, num_frames=num_frames,
        multispectral=True, image_size=image_size)

    # from kwcoco.demo import perterb
    # perterb_config = {
    #     'box_noise': 0.5,
    #     'n_fp': 3,
    #     # 'with_probs': 1,
    # }
    # perterb.perterb_coco(coco_dset, **perterb_config)

    hack_in_heatmaps(coco_dset, rng=rng)
    hack_in_timedata(coco_dset)

    # Hack in geographic info
    hack_seed_geometadata_in_dset(coco_dset, force=True, rng=rng)

    from watch.utils import kwcoco_extensions
    # Do a consistent transfer of the hacked seeded geodat to the other images
    kwcoco_extensions.ensure_transfered_geo_data(coco_dset)

    kwcoco_extensions.warp_annot_segmentations_to_geos(coco_dset)
    return coco_dset


def hack_in_heatmaps(coco_dset, heatmap_dname='dummy_heatmaps', with_nan=False, rng=None):
    rng = kwarray.ensure_rng(rng)
    asset_dpath = ub.Path(coco_dset.assets_dpath)
    dummy_heatmap_dpath = asset_dpath / heatmap_dname
    dummy_heatmap_dpath.mkdir(exist_ok=1, parents=True)

    channels = 'notsalient|salient'
    channels = kwcoco.FusedChannelSpec.coerce(channels)
    chan_codes = channels.normalize().as_list()

    aux_width = 128
    aux_height = 128
    dims = (aux_width, aux_height)
    for img in coco_dset.index.imgs.values():

        warp_img_from_aux = kwimage.Affine.scale((
            img['width'] / aux_width, img['height'] / aux_height))
        warp_aux_from_img = warp_img_from_aux.inv()

        # Grab perterbed detections from this image
        img_dets = coco_dset.annots(gid=img['id']).detections

        # Transfom dets into aux space
        aux_dets = img_dets.warp(warp_aux_from_img)

        # Hack: use dets to draw some randomish heatmaps
        sseg = aux_dets.data['segmentations']
        chan_datas = []
        for _code in chan_codes:
            chan_data = np.zeros(dims, dtype=np.float32)
            for poly in sseg.data:
                poly.fill(chan_data, 1)

            # Add lots of noise to the data
            chan_data += (rng.randn(*dims) * 0.1)
            chan_data + chan_data.clip(0, 1)
            chan_data = kwimage.gaussian_blur(chan_data, sigma=1.2)
            chan_data = chan_data.clip(0, 1)
            mask = rng.randn(*dims)
            chan_data = chan_data * ((kwimage.fourier_mask(chan_data, mask)[..., 0]) + .5)
            chan_data += (rng.randn(*dims) * 0.1)
            chan_data = chan_data.clip(0, 1)
            chan_datas.append(chan_data)
        hwc_probs = np.stack(chan_datas, axis=2)

        if with_nan:
            invalid_mask = (rng.rand(*hwc_probs.shape) > 0.95)
            hwc_probs[invalid_mask] = np.nan

        # TODO do something with __WIP_add_auxiliary to make this clear and
        # concise
        heatmap_fpath = dummy_heatmap_dpath / 'dummy_heatmap_{}.tif'.format(img['id'])
        kwimage.imwrite(heatmap_fpath, hwc_probs, backend='gdal', compress='DEFLATE',
                        blocksize=128)
        aux_height, aux_width = hwc_probs.shape[0:2]

        auxlist = img['auxiliary']
        auxlist.append({
            'file_name': str(heatmap_fpath),
            'width': aux_width,
            'height': aux_height,
            'channels': channels.spec,
            'warp_aux_to_img': warp_img_from_aux.concise(),
        })


def hack_in_timedata(coco_dset):
    from kwarray.distributions import Uniform
    min_time = datetime.datetime(year=1970, month=1, day=1)
    max_time = datetime.datetime(year=2101, month=1, day=1)
    time_distri = Uniform(min_time.timestamp(), max_time.timestamp())

    # Hack in other metadata
    for vidid in coco_dset.videos():
        vid_gids = list(coco_dset.images(vidid=vidid))
        time_pool = sorted(time_distri.sample(len(vid_gids)))
        for gid, timestamp in zip(vid_gids, time_pool):
            ts = datetime.datetime.fromtimestamp(timestamp)
            img = coco_dset.index.imgs[gid]
            img['date_captured'] = ts.isoformat()


def hack_seed_geometadata_in_dset(coco_dset, force=False, rng=None):
    """
    Add random geo coordinates to one asset in each video

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes5-multispectral')
        >>> modified = hack_seed_geometadata_in_dset(coco_dset, force=True)
        >>> fpath = modified[0]
        >>> print(ub.cmd('gdalinfo ' + fpath)['out'])
    """
    import kwarray
    from watch.utils import kwcoco_extensions
    rng = kwarray.ensure_rng(rng)
    modified = []
    for vidid in coco_dset.videos():
        img = coco_dset.images(vidid=vidid).peek()
        coco_img = coco_dset._coco_image(img['id'])
        obj = coco_img.primary_asset()
        fpath = join(coco_dset.bundle_dpath, obj['file_name'])
        # print('fpath = {!r}'.format(fpath))

        format_info = kwcoco_extensions.geotiff_format_info(fpath)
        if force or not format_info['has_geotransform']:

            utm_box, utm_crs_info = _random_utm_box(rng=rng)
            auth = utm_crs_info['auth']
            assert auth[0] == 'EPSG'
            epsg_int = int(auth[1])

            ulx, uly, lrx, lry = utm_box.to_ltrb().data[0]

            command = f'gdal_edit.py -a_ullr {ulx} {uly} {lrx} {lry} -a_srs EPSG:{epsg_int} {fpath}'
            cmdinfo = ub.cmd(command, shell=True)
            if cmdinfo['ret'] != 0:
                print(cmdinfo['out'])
                print(cmdinfo['err'])
                assert cmdinfo['ret'] == 0
            modified.append(fpath)
    return modified


def _random_utm_box(rng=None):
    """
    rng = None

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import _random_utm_box
        >>> _random_utm_box()
    """
    import numpy as np
    from kwarray.distributions import Uniform
    import kwarray
    from watch.utils import util_gis
    from osgeo import osr
    import watch
    # stay away from edges and poles
    rng = kwarray.ensure_rng(rng)
    max_lat = 90 - 40
    max_lon = 180 - 80
    lat_distri = Uniform(-max_lat, max_lat, rng=rng)
    lon_distri = Uniform(-max_lon, max_lon, rng=rng)

    lon = lon_distri.sample()
    lat = lat_distri.sample()
    utm_epsg_int = util_gis.utm_epsg_from_latlon(lat, lon)

    wgs84_crs = osr.SpatialReference()
    wgs84_crs.ImportFromEPSG(4326)
    wgs84_crs.SetAxisMappingStrategy(osr.OAMS_AUTHORITY_COMPLIANT)

    utm_crs = osr.SpatialReference()
    utm_crs.ImportFromEPSG(utm_epsg_int)
    utm_from_wgs84 = osr.CoordinateTransformation(wgs84_crs, utm_crs)

    utm_crs_info = watch.gis.geotiff.make_crs_info_object(utm_crs)

    utm_x, utm_y, _ = utm_from_wgs84.TransformPoint(lat, lon, 1.0)
    # keep the aspect ratio more or less squareish
    w = rng.randint(10, 150)
    h = np.clip((rng.randn() + 1), 0.9, 1.1) * w

    """
    import sympy as sym
    radius, dist, lat1, lat2, lon1, lon2 = sym.symbols('radius, dist, lat1, lat2, lon1, lon2')
    haversine_expr = 2 * radius * sym.asin(sym.sqrt(
        sym.sin((lat2 - lat1) / 2) ** 2 + sym.cos(lat1) * sym.cos(lat2) * sym.sin((lon2 - lon1) / 2) ** 2
    ))
    sym.solve(sym.Eq(haversine_expr, dist), lon2)
    # sym.solve(sym.Eq(haversine_expr, dist), lat2)
    # import haversine
    # haversine.haversine((ulx, uly), (lrx, uly))
    # haversine.haversine((ulx, uly), (ulx, lry))
    # Inverse haversine
    from numpy import sqrt, cos, sin
    from numpy import arcsin as asin
    from numpy import pi
    ulx, uly, lrx, lry = kwimage.Boxes([[utm_x, utm_y, w, h]], 'cxywh').to_ltrb().data[0]
    lon1 = ulx
    lon2 = lrx
    lat1 = uly
    lat2 = lry
    # Make the box squareish
    radius = 6356.752
    dist = 2 * radius * asin(sqrt(sin(lat1 / 2 - lat2 / 2) ** 2 + sin(lon1 / 2 - lon2 / 2)**2 * cos(lat1) * cos(lat2)))
    possible_solutions = [
        lon1 - 2 * asin(sqrt(2) * sqrt((-cos(dist / radius) + cos(lat1 - lat2)) / (cos(lat1) * cos(lat2))) / 2),
        lon1 + 2 * asin(sqrt(2) * sqrt((-cos(dist / radius) + cos(lat1 - lat2)) / (cos(lat1) * cos(lat2))) / 2),
        lon1 + 2 * asin(sqrt(2) * sqrt((-cos(dist / radius) + cos(lat1 - lat2)) / (cos(lat1) * cos(lat2))) / 2) - 2 * pi,
        lon1 - 2 * asin(sqrt(2) * sqrt(-(cos(dist / radius) - cos(lat1 - lat2)) / (cos(lat1) * cos(lat2))) / 2) - 2 * pi]
    valid_solutions = [cand for cand in possible_solutions if cand > lon1]
    lrx = valid_solutions[0]
    """
    utm_box = kwimage.Boxes([[utm_x, utm_y, w, h]], 'cxywh')
    return utm_box, utm_crs_info


def demo_kwcoco_multisensor(num_videos=4, num_frames=10, heatmap=False,
                            dates=False, geodata=False, bad_nodata=False,
                            **kwargs):
    """
    Ignore:
        import watch
        coco_dset = watch.demo.demo_kwcoco_multisensor()
        coco_dset = watch.demo.demo_kwcoco_multisensor(max_speed=0.5)

        from watch.demo.smart_kwcoco_demodata import *  # NOQA
        import xdev
        globals().update(xdev.get_func_kwargs(demo_kwcoco_multisensor))
        kwargs = {}

    Args:

        num_videos (int): number of videos in the demo dataset

        num_frames (int): number of frames per video in the demo dataset

        heatmap (bool): if True adds dummy saliency heatmaps to the demodata.

        geodata (bool): if True adds dummy geographic referencing to
            the demodata.

        bad_nodata (bool):
            if True, zeros out some pixels which simulates bad nodata for
            testing.

        **kwargs : additional arguments passed to :func:`kwcoco.CocoDataset.demo`.

    Returns:
        kwcoco.CocoDataset

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> num_frames = 10
        >>> num_videos = 4
        >>> dates=True
        >>> geodata=True
        >>> heatmap=True
        >>> bad_nodata = True
        >>> kwargs = {}
        >>> coco_dset = demo_kwcoco_multisensor(dates=dates, geodata=geodata, heatmap=heatmap, bad_nodata=True)
    """
    dpath = ub.Path.appdir('watch', 'demo_kwcoco_bundles').ensuredir()

    demo_kwargs = {
        'num_frames': num_frames,
        'num_videos': num_videos,
        'rng': 9111665008,
        'multisensor': True,
        'multispectral': True,
        'image_size': 'random',
    }
    rng = kwarray.ensure_rng(demo_kwargs['rng'])
    demo_kwargs['rng' ] = rng
    demo_kwargs.update(kwargs)

    stamp_dpath = (dpath / '_stamps').ensuredir()

    depends = {
        'demo_kwargs': demo_kwargs,
        'dates': dates,
        'geodata': geodata,
        'heatmap': heatmap,
        'bad_nodata': bad_nodata,
    }

    bundle_name = 'watch_vidshapes_' + ub.hash_data(depends)[0:8]
    bundle_dpath = dpath / bundle_name
    coco_fpath = bundle_dpath / 'data.kwcoco.json'
    stamp = ub.CacheStamp('demo_kwcoco_multisensor', depends=depends,
                          dpath=stamp_dpath, product=[coco_fpath],
                          hasher='sha512')

    if not stamp.expired():
        coco_dset = kwcoco.CocoDataset(coco_fpath)
        return coco_dset

    coco_dset = kwcoco.CocoDataset.demo(
        'vidshapes', fpath=coco_fpath, **demo_kwargs)
    # Hack in sensor_coarse
    images = coco_dset.images()
    groups = ub.sorted_keys(ub.group_items(images.coco_images, lambda x: x.channels.spec))
    for idx, (k, g) in enumerate(groups.items()):
        for coco_img in g:
            coco_img.img['sensor_coarse'] = 'sensor{}'.format(idx)

    random_ignore_annots = True
    if random_ignore_annots:
        neg_cid = coco_dset.ensure_category('negative')
        ignore_cid = coco_dset.ensure_category('ignore')

        for coco_img in coco_dset.images().coco_images:
            dsize = np.array(coco_img.dsize)
            if rng.rand() > 0.8:
                n = min(rng.randint(0, 3), rng.randint(0, 3)) + 1
                for _ in range(n):
                    cid = ignore_cid if rng.rand() > 0.5 else neg_cid
                    poly = kwimage.Polygon.random(rng=rng).scale(dsize)
                    poly = poly.scale(1 / 4, about='centroid')
                    new_ann = {
                        'image_id': coco_img.img['id'],
                        'category_id': cid,
                        'bbox': list(poly.to_boxes().to_coco())[0],
                        'segmentation': poly.to_coco('new'),
                    }
                    coco_dset.add_annotation(**new_ann)

    if bad_nodata:
        # We can naively zero the image before adding georeferencing to
        for gid in coco_dset.images():
            coco_img = coco_dset.coco_image(gid)
            for obj in coco_img.iter_asset_objs():
                fpath = ub.Path(coco_dset.bundle_dpath) / obj['file_name']
                imdata = kwimage.imread(fpath)
                poly = kwimage.Polygon.random(rng=rng).scale(imdata.shape[0:2][::-1])
                imdata = poly.fill(imdata, value=0)
                kwimage.imwrite(fpath, imdata)

    if heatmap:
        hack_in_heatmaps(coco_dset, rng=rng)

    if dates:
        hack_in_timedata(coco_dset)

    if geodata:
        # Hack in geographic info
        hack_seed_geometadata_in_dset(coco_dset, force=True, rng=rng)
        from watch.utils import kwcoco_extensions
        # Do a consistent transfer of the hacked seeded geodata to the other images
        kwcoco_extensions.ensure_transfered_geo_data(coco_dset)
        kwcoco_extensions.warp_annot_segmentations_to_geos(coco_dset)

        # Also hack in an invalid region in the top left of some videos
        vidids = coco_dset.videos()
        for _idx, vidid in enumerate(vidids):
            gids = coco_dset.images(vidid=vidid)
            if _idx == 0:
                # For the first one make ALL frames invalid here
                pass
            elif _idx == 1:
                # For the second one make all but ONE frames invalid
                keep_idx = min(int(rng.rand() * len(gids)), len(gids) - 1)
                gids = list(gids[:keep_idx]) + list(gids[keep_idx + 1:])
            elif _idx == 2:
                # For the third, make nothing invalid
                gids = []
            else:
                # For the rest do a random subset
                gids = gids.compress(rng.rand(len(gids)) > 0.5)
            for gid in gids:
                coco_img = coco_dset.coco_image(gid)
                full_image_poly = kwimage.Boxes(
                    [(0, 0) + coco_img.dsize], 'xywh')
                demo_invalid_region = full_image_poly.scale(0.23)
                outer = full_image_poly.to_shapely()[0]
                inner = demo_invalid_region.to_shapely()[0]
                demo_valid_region = kwimage.Polygon.from_shapely(outer.difference(inner))
                coco_img.img['valid_region'] = demo_valid_region.to_coco('new')

        kwcoco_extensions.populate_watch_fields(
            coco_dset, target_gsd=0.3, enable_valid_region=True,
            overwrite=True)

    # Rewrite the file so it contains our changes.
    coco_dset.dump(coco_dset.fpath)
    stamp.renew()

    return coco_dset


def coerce_kwcoco(data='watch-msi', **kwargs):
    """
    coerce with watch special datasets

    Calls `kwcoco.CocoDataset.coerce` unless the code is `watch-msi`, and then
    we construct a special dataset with extra variables expected by the watch
    project.

    Args:
        data (str | Coercable[kwcoco.CocoDataset]):
            the special code to coerce

        **kwargs:
            modify how the demodata is created. For `watch-msi`, see
            :func:`demo_kwcoco_multisensor`, which has args like: `dates`,
            `geodata`, `heatmap`.

    Example:
        >>> import watch
        >>> dates=True
        >>> geodata=True
        >>> heatmap=True
        >>> kwargs = {}
        >>> coco_dset = watch.coerce_kwcoco(data='watch-msi', dates=dates, geodata=geodata, heatmap=heatmap)
        >>> coco_dset2 = watch.coerce_kwcoco(data='watch-msi-dates-geodata-gsize32')
        >>> assert 'date_captured' in coco_dset2.images().peek()
    """
    if isinstance(data, str) and 'watch' in data.split('-'):
        defaults = {
            'render': True,
            'num_videos': 4,
            'num_frames': 10,
            'num_tracks': 2,
            'anchors': None,
            'image_size': (600, 600),
            'aux': None,
            'multispectral': True,
            'multisensor': True,
            'max_speed': 0.01,
        }
        defaults.update(dict(
            num_videos=kwargs.get('num_videos', 4),
            num_frames=kwargs.get('num_frames', 10),
            heatmap=kwargs.get('heatmap', False),
            dates=kwargs.get('dates', False),
            geodata=kwargs.get('geodata', False),
            bad_nodata=kwargs.get('bad_nodata', False)
        ))
        vidkw_aliases = {
            'num_frames': {'frames'},
            'num_tracks': {'tracks'},
            'num_videos': {'videos'},
            'max_speed': {'speed'},
            'image_size': {'gsize'},
            'multispectral': {'msi'},
        }
        alias_to_key = {k: v for v, ks in vidkw_aliases.items() for k in ks}
        kwargs.update(_parse_demostr(data, defaults, alias_to_key)[0])
        kwargs.pop('sqlview', None)
        print('kwargs = {}'.format(ub.urepr(kwargs, nl=1)))
        return demo_kwcoco_multisensor(**kwargs)
    else:
        import os
        if isinstance(data, (str, os.PathLike)):
            expanded = ub.Path(data).expand()
            if str(expanded) != str(data):
                data = expanded
        return kwcoco.CocoDataset.coerce(data, **kwargs)


def _parse_demostr(data, defaults, alias_to_key=None):
    """
    Special suffixes can be added to generic demo names. Parse them out here.
    Arguments are `-` separated, only known defaulted values are parsed. Bare
    default names are interpreted as a value of True, otherwise the value
    should be numeric. TODO: generalize this and conslidate in the kwcoco
    demo method.

    Example:
        >>> from watch.demo.smart_kwcoco_demodata import _random_utm_box, _parse_demostr
        >>> data = 'foo-bar-baz1-biz2.3'
        >>> defaults = {}
        >>> alias_to_key = None
        >>> _parse_demostr(data, defaults)
        ({}, {'foo': True, 'bar': True, 'baz': 1, 'biz': 2.3})
    """
    if alias_to_key is None:
        alias_to_key = {}
    from watch.utils.util_codes import parse_delimited_argstr
    handled = defaults.copy()
    unhandled = parse_delimited_argstr(data)
    for key, value in list(unhandled.items()):
        key = alias_to_key.get(key, key)
        if key in handled:
            handled[key] = value
            unhandled.pop(key, None)
    return handled, unhandled
