"""
Extends kwcoco demodata to be more smart-like
"""
import datetime as datetime_mod
import kwarray
import kwcoco
import kwimage
import numpy as np
import ubelt as ub
import geowatch


def coerce_kwcoco(data='geowatch-msi', **kwargs):
    """
    coerce with geowatch special datasets

    Calls `kwcoco.CocoDataset.coerce` unless the code is `geowatch-msi`, and then
    we construct a special dataset with extra variables expected by the watch
    project.

    Args:
        data (str | Coercible[kwcoco.CocoDataset]):
            the special code to coerce

        **kwargs:
            modify how the demodata is created. For `geowatch-msi`, see
            :func:`demo_kwcoco_multisensor`, which has args like: `dates`,
            `geodata`, `heatmap`.

    Example:
        >>> import geowatch
        >>> dates=True
        >>> geodata=True
        >>> heatmap=True
        >>> kwargs = {}
        >>> coco_dset = geowatch.coerce_kwcoco(data='geowatch-msi', dates=dates, geodata=geodata, heatmap=heatmap)
        >>> coco_dset2 = geowatch.coerce_kwcoco(data='geowatch-msi-dates-geodata-gsize32')
        >>> assert 'date_captured' in coco_dset2.images().peek()
    """
    if isinstance(data, str) and ('geowatch' in data.split('-') or 'watch' in data.split('-')):
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
        defaults.update(ub.udict(kwargs) & ub.udict(defaults))
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
        # print('kwargs = {}'.format(ub.urepr(kwargs, nl=1)))
        return demo_kwcoco_multisensor(**kwargs)
    else:
        import os
        if isinstance(data, (str, os.PathLike)):
            expanded = ub.Path(data).expand()
            if str(expanded) != str(data):
                data = expanded
        return kwcoco.CocoDataset.coerce(data, **kwargs)


def demo_kwcoco_multisensor(num_videos=4, num_frames=10, heatmap=False,
                            dates=False, geodata=False, bad_nodata=False,
                            **kwargs):
    """
    Ignore:
        import geowatch
        coco_dset = geowatch.demo.demo_kwcoco_multisensor()
        coco_dset = geowatch.demo.demo_kwcoco_multisensor(max_speed=0.5)

        from geowatch.demo.smart_kwcoco_demodata import *  # NOQA
        import xdev
        globals().update(xdev.get_func_kwargs(demo_kwcoco_multisensor))
        kwargs = {}

    Args:

        num_videos (int): number of videos in the demo dataset

        num_frames (int): number of frames per video in the demo dataset

        heatmap (bool | ChannelSpec):
            if True adds dummy saliency heatmaps to the demodata.  Can also be
            given as data coercable to a channel spec, in which case those
            channels are generated.

        geodata (bool | dict): if True adds dummy geographic referencing to
            the demodata.
            If a dictionary can specify extra information.
            Available keys:
                region_geom

        bad_nodata (bool):
            if True, zeros out some pixels which simulates bad nodata for
            testing.

        dates (bool | dict):
            Include time data or not.
            If a dictionary can specify extra information.
            Available keys:
                start_time
                end_time

        **kwargs : additional arguments passed to :func:`kwcoco.CocoDataset.demo`.

    Returns:
        kwcoco.CocoDataset

    Example:
        >>> from geowatch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> num_frames = 2
        >>> num_videos = 1
        >>> image_size = (128, 128)
        >>> dates = True
        >>> geodata = True
        >>> heatmap = True
        >>> bad_nodata = True
        >>> kwargs = {}
        >>> coco_dset = demo_kwcoco_multisensor(dates=dates, geodata=geodata,
        >>>                                     heatmap=heatmap, bad_nodata=True,
        >>>                                     image_size=image_size)
    """
    dpath = kwargs.pop('dpath', None)
    if dpath is None:
        dpath = ub.Path.appdir('geowatch', 'demo_kwcoco_bundles').ensuredir()

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

    if geodata:
        renderkw = demo_kwargs.get('render', True)
        if renderkw:
            if not isinstance(renderkw, dict):
                renderkw = {}
            renderkw['main_ext'] = '.tif'
            renderkw['main_channels'] = 'red|green|blue'
        demo_kwargs['render'] = renderkw

    stamp_dpath = (dpath / '_stamps').ensuredir()

    depends = {
        'demo_kwargs': demo_kwargs,
        'dates': dates,
        'geodata': geodata,
        'heatmap': heatmap,
        'bad_nodata': bad_nodata,
        'version': 5,
    }

    _register_polygon_hash_data()

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
        channels = heatmap
        hack_in_heatmaps(coco_dset, channels, rng=rng)

    def coerce_bool_config_dict(data):
        if not isinstance(data, dict):
            if data:
                data = {'enabled': True}
            else:
                data = {'enabled': False}
        else:
            # Enable by default if given as a dictionary
            if 'enabled' not in data:
                data = data.copy()
                data['enabled'] = True
        return data

    geodata = coerce_bool_config_dict(geodata)
    dates = coerce_bool_config_dict(dates)

    if dates['enabled']:
        hack_in_timedata(coco_dset, dates, rng=rng)

    if geodata['enabled']:
        for ann in coco_dset.anns.values():

            # both of these errors occur in the call:
            # geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True, num_frames=16,
            #                     image_size=(8, 8))

            has_seg = (ann['segmentation'] is not None and
                       len(ann['segmentation']) > 0)
            if not has_seg:
                print('FIXME this should never print - empty segmentation generated')
                ann['segmentation'] = (kwimage.Boxes([ann['bbox']], 'xywh')
                                       .to_polygons()[0]
                                       .to_coco(style='new'))

            # why does coerce work here when
            # seg = kwimage.MultiPolygon.from_coco(ann['segmentation'])
            seg = kwimage.MultiPolygon.coerce(ann['segmentation'])
            try:
                seg.to_shapely()
            except ValueError:
                print('FIXME this should never print - invalid segmentation generated')
                import shapely
                import shapely.geometry  # NOQA
                from shapely.geometry import shape
                try:
                    shp = shape(seg.to_geojson())
                    seg = kwimage.MultiPolygon.from_shapely(shp.make_valid())
                    ann['segmentation'] = seg.to_coco(style='new')
                except ValueError:
                    ann['segmentation'] = (kwimage.Boxes([ann['bbox']], 'xywh')
                                           .to_polygons()[0]
                                           .to_coco(style='new'))

        # Hack in geographic info
        region_geom = geodata.get('region_geom', 'random')
        hack_seed_geometadata_in_dset(coco_dset, force=True, rng=rng,
                                      region_geom=region_geom)
        from geowatch.utils import kwcoco_extensions
        # Do a consistent transfer of the hacked seeded geodata to the other images
        kwcoco_extensions.ensure_transfered_geo_data(coco_dset)
        kwcoco_extensions.coco_populate_geo_heuristics(coco_dset)
        kwcoco_extensions.warp_annot_segmentations_to_geos(coco_dset)

        # Also hack in an invalid region in the top left of some videos
        vidids = coco_dset.videos()
        for _idx, vidid in enumerate(vidids):
            gids = coco_dset.images(video_id=vidid)
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


def demo_kwcoco_with_heatmaps(num_videos=1, num_frames=20, image_size=(512, 512)):
    """
    Return a dummy kwcoco file with special metdata

    DEPRECATED:
        Instead use geowatch.coerce_kwcoco('geowatch-msi-geodata-dates-heatmap-videos1-frames20-gsize512') or something similar

    Example:
        >>> from geowatch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> coco_dset = demo_kwcoco_with_heatmaps()

        key = 'salient'
        for vidid in coco_dset.videos():
            frames = []
            for gid in coco_dset.images(video_id=vidid):
                delayed = coco_dset.coco_image(gid).imdelay(channels=key, space='video')
                final = delayed.finalize()
                frames.append(final)
            vid_stack = kwimage.stack_images_grid(frames, axis=1, pad=5, bg_value=1)

            import kwplot
            kwplot.imshow(vid_stack)
    """
    assert image_size[0] == image_size[1]
    return coerce_kwcoco(
        f'geowatch-msi-geodata-dates-heatmap-videos{num_videos}-frames{num_frames}-gsize{image_size[0]}')


def hack_in_heatmaps(coco_dset, channels='auto', heatmap_dname='dummy_heatmaps', with_nan=False, rng=None):
    """
    Adds dummy heatmaps into a coco dataset.

    Args:
        channels (ChannelSpec): heatmap channels to generate

    CommandLine:
        xdoctest -m geowatch.demo.smart_kwcoco_demodata hack_in_heatmaps

    Example:
        >>> from geowatch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> with_nan = False
        >>> rng = None
        >>> heatmap_dname = 'dummy_heatmaps'
        >>> num_frames = 34
        >>> num_videos = 1
        >>> dates=True
        >>> geodata=True
        >>> bad_nodata = False
        >>> kwargs = {}
        >>> heatmap = channels = 'ac_salient,No Activity|Site Preparation|Active Construction|Post Construction'
        >>> # heatmap = channels = 'ac_salient'
        >>> coco_dset = demo_kwcoco_multisensor(dates=dates, geodata=geodata, heatmap=heatmap, bad_nodata=bad_nodata, num_frames=num_frames, num_videos=num_videos, multisensor=0, multispectral=0, max_speed=0)
        >>> # xdoctest: +SKIP
        >>> ub.cmd(f'geowatch visualize {coco_dset.fpath} --smart', system=1)

    Ignore:
        coco_dset.fpath
        ub.cmd(f'geowatch visualize {coco_dset.fpath} --smart', system=1)
        ub.cmd(f'geowatch stats {coco_dset.fpath}', system=1)
    """
    rng = kwarray.ensure_rng(rng)
    asset_dpath = ub.Path(coco_dset.assets_dpath)
    dummy_heatmap_dpath = asset_dpath / heatmap_dname
    dummy_heatmap_dpath.mkdir(exist_ok=1, parents=True)

    if channels == 'auto' or isinstance(channels, int):
        channels = 'notsalient|salient'
    from delayed_image import sensorchan_spec
    # channels = sensorchan_spec.SensorChanSpec.coerce(heatmap)
    # channels = heatmap if isinstance(heatmap, str) else 'auto'
    # channels = kwcoco.FusedChannelSpec.coerce(channels)
    sensorchan = sensorchan_spec.SensorChanSpec.coerce(channels)
    sensorchan = sensorchan.normalize()

    aux_width = 128
    aux_height = 128
    dims = (aux_width, aux_height)
    coco_images = coco_dset.images().coco_images

    # Precompute class osillation on a per-track basis
    if 1:
        chan_names = sensorchan.chans.fuse()
        aidchan_to_intensities = {}
        for tid, aids in coco_dset.index.trackid_to_aids.items():
            track_annots = coco_dset.annots(aids)
            num_frames = len(track_annots)
            if num_frames == 0:
                continue

            # loc = np.linspace(0, np.pi * 2, num_frames)

            for stream in sensorchan.chans.streams():
                stream_size = stream.numel()

                rng.rand(stream_size).argmax()

                stream_chan_names = stream.to_list()
                if len(stream_chan_names) == 1:
                    class_intensities = np.ones((num_frames, stream_size))
                else:
                    kwarray.shuffle(stream_chan_names, rng=rng)
                    transition_points = rng.randint(num_frames, size=stream_size - 1)
                    transition_points.sort()
                    transition_points = np.hstack([transition_points, [num_frames]])

                    data = np.zeros((num_frames, stream_size))
                    prev_rx = 0
                    for cx, rx in enumerate(transition_points):
                        data[prev_rx:rx, cx] = 1
                        prev_rx = rx
                    class_energy = data

                    # # Choose starting probability for each class
                    # start = (rng.rand(stream_size) * np.pi * 2)
                    # # kwarray.ArrayAPI.softmax(start)
                    # stream_loc = loc[:, None] + start[None, :]
                    # class_energy = np.sin(stream_loc)
                    # class_energy = (np.sin(stream_loc) / 2) + 0.5
                    # class_energy = kwarray.ArrayAPI.softmax(class_energy * 20, axis=1)
                    # class_energy = kwimage.gaussian_blur(class_energy)
                    class_intensities = kwarray.normalize(class_energy)

                if 0:
                    import kwplot
                    import pandas as pd
                    sns = kwplot.autosns()
                    df = pd.DataFrame(ub.dzip(stream_chan_names, class_intensities.T))
                    df1 = df.reset_index(names='frame_index')
                    df2 = df1.melt(**{'id_vars': ['frame_index'], 'value_name': 'intensity', 'var_name': 'catname'})
                    sns.lineplot(data=df2, x='frame_index', y='intensity', hue='catname')

                for aid, vals in zip(aids, class_intensities):
                    for c, v in zip(stream_chan_names, vals):
                        aidchan_to_intensities[(aid, c)] = v

    # _tasks = []
    for coco_img in coco_images:
        img = coco_img.img

        warp_img_from_aux = kwimage.Affine.scale((
            img['width'] / aux_width, img['height'] / aux_height))
        warp_aux_from_img = warp_img_from_aux.inv()

        # Grab perterbed detections from this image
        annots = coco_img.annots()
        # track_ids = annots.lookup('track_id', None)

        img_dets = annots.detections

        # Transfom dets into aux space
        aux_dets = img_dets.warp(warp_aux_from_img)

        # Hack: use dets to draw some randomish heatmaps
        sseg = aux_dets.data['segmentations']
        chan_to_intensities = {
            c: [aidchan_to_intensities[(aid, c)] for aid in annots]
            for c in chan_names
        }

        # new_assets = []
        for stream in sensorchan.streams():
            stream_hash = ub.hash_data(stream.spec)[0:16]
            if stream.sensor.spec != '*':
                raise NotImplementedError

            # for _code in stream.chans.to_list():
            chan_datas = []
            for fused_chan in stream.chans.to_list():
                sseg.meta['intensities'] = chan_to_intensities[fused_chan]
                chan_data = _random_chan_data(dims, sseg, rng)
                chan_datas.append(chan_data)

            hwc_probs = np.stack(chan_datas, axis=2)

            if with_nan:
                invalid_mask = (rng.rand(*hwc_probs.shape) > 0.95)
                hwc_probs[invalid_mask] = np.nan

            heatmap_fpath = dummy_heatmap_dpath / f'dummy_heatmap_{img["id"]}_{stream_hash}.tif'
            kwimage.imwrite(heatmap_fpath, hwc_probs, backend='gdal', compress='DEFLATE',
                            blocksize=128)
            aux_height, aux_width = hwc_probs.shape[0:2]

            new_asset = {
                'file_name': str(heatmap_fpath),
                'sensor_coarse': stream.sensor.spec,
                'width': aux_width,
                'height': aux_height,
                'sensor': stream.sensor.spec,
                'channels': stream.chans.concise().spec,
                'warp_aux_to_img': warp_img_from_aux.concise(),
            }
            # print(f'new_asset = {ub.urepr(new_asset, nl=1)}')
            coco_img.add_asset(**new_asset)

    #         new_assets.append(new_asset)
    #     _tasks.append([coco_img, new_assets])

    # for coco_img, new_assets in _tasks:
    #     for new_asset in new_assets:
    #         coco_img.add_asset(**new_asset)


def _random_chan_data(dims, sseg, rng):
    """
    Create a noisy random heatmap using sseg as a template

    Example:
        >>> from geowatch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> from geowatch.demo.smart_kwcoco_demodata import _random_chan_data, _random_utm_box, _parse_demostr, _register_polygon_hash_data
        >>> import kwarray
        >>> import kwimage
        >>> rng = kwarray.ensure_rng()
        >>> dims = (128, 128)
        >>> sseg1 = kwimage.Polygon.random().scale(dims).scale(0.5).translate((0, 32))
        >>> sseg2 = kwimage.Polygon.random().scale(dims).scale(0.5).translate((32, 0))
        >>> sseg3 = kwimage.Polygon.random().scale(dims).scale(0.5).translate((64, 0))
        >>> sseg4 = kwimage.Polygon.random().scale(dims).scale(0.5).translate((0, 64))
        >>> sseg = kwimage.PolygonList([sseg1, sseg2, sseg3, sseg4])
        >>> sseg.meta['intensities'] = [0.3, 0.8, 1.0, 0.1]
        >>> chan_data = _random_chan_data(dims, sseg, rng)
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.figure(doclf=1)
        >>> kwplot.imshow(chan_data)
        >>> sseg.draw(fill=0, border=1)
    """
    import numpy as np
    intensities = sseg.meta.get('intensities', None)

    # Generate a clean signal
    clean = np.zeros(dims, dtype=np.float32)
    if intensities is None:
        for poly in sseg.data:
            poly.fill(clean, 1)
    else:
        for v, poly in zip(intensities, sseg.data):
            poly.fill(clean, v)

    # Make a dirty copy of the signal
    dirty = clean.copy()
    # Add lots of noise to the data
    dirty += (rng.randn(*dims) * 0.1)
    dirty + dirty.clip(0, 1)
    dirty = kwimage.gaussian_blur(dirty, sigma=1.2)
    dirty = dirty.clip(1e-6, 1)
    mask = rng.randn(*dims)
    dirty = dirty * ((kwimage.fourier_mask(dirty, mask)[..., 0]) + .5)
    dirty += (rng.randn(*dims) * 0.1)
    dirty = dirty.clip(0, 1)

    # Blend between clean and dirty
    a1 = 0.5
    a2 = 1 - a1
    chan_data = (dirty * a1) + (clean * a2)
    return chan_data


def hack_in_timedata(coco_dset, dates=True, rng=None):
    """
    Adds date_captured fields to demo toydata
    """
    from kwarray.distributions import Uniform
    from kwutil import util_time
    datekw = ub.udict({
        'start_time': '1970-01-01',
        'end_time': '2101-01-01',
        'enabled': True,
    })
    if not isinstance(dates, dict):
        dates = {}

    if isinstance(dates, dict):
        extra = dates - datekw
        if extra:
            raise ValueError(f'Unexepcted date kwargs: {extra}')
        datekw.update(dates)

    rng = kwarray.ensure_rng(rng)
    min_time = util_time.coerce_datetime(datekw['start_time'])
    max_time = util_time.coerce_datetime(datekw['end_time'])
    print(f'min_time={min_time}')
    print(f'max_time={max_time}')
    time_distri = Uniform(min_time.timestamp(), max_time.timestamp(), rng=rng)

    # Hack in other metadata
    for vidid in coco_dset.videos():
        vid_gids = list(coco_dset.images(video_id=vidid))
        time_pool = sorted(time_distri.sample(len(vid_gids)))
        for gid, timestamp in zip(vid_gids, time_pool):
            ts = datetime_mod.datetime.fromtimestamp(timestamp)
            img = coco_dset.index.imgs[gid]
            img['date_captured'] = ts.isoformat()


def hack_seed_geometadata_in_dset(coco_dset, force=False, rng=None,
                                  region_geom=None):
    """
    Add random geo coordinates to one asset in each video

    Example:
        >>> from geowatch.demo.smart_kwcoco_demodata import *  # NOQA
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes5-multispectral')
        >>> modified = hack_seed_geometadata_in_dset(coco_dset, force=True)
        >>> fpath = modified[0]
        >>> print(ub.cmd('gdalinfo ' + fpath)['out'])
    """
    import kwarray
    import kwimage
    from geowatch.utils import kwcoco_extensions
    rng = kwarray.ensure_rng(rng)
    modified = []
    print('Hacking in seed geom data')

    override_geom_box = None

    if region_geom is None or isinstance(region_geom, str) and region_geom == 'random':
        ...
    else:
        assert len(list(coco_dset.videos())) == 1, 'only handle 1 video for now'
        override_geom_box = kwimage.Polygon.from_shapely(region_geom).box()
        override_geom_epsg = 4326

    for vidid in coco_dset.videos():
        img = coco_dset.images(video_id=vidid).peek()
        coco_img = coco_dset.coco_image(img['id'])
        obj = coco_img.primary_asset()
        fpath = str(ub.Path(coco_dset.bundle_dpath) / obj['file_name'])

        try:
            format_info = kwcoco_extensions.geotiff_format_info(fpath)
        except Exception:
            print(f'FAILED fpath={fpath}')
            raise
        if force or not format_info['has_geotransform']:

            if override_geom_box is None:
                utm_box, utm_crs_info = _random_utm_box(rng=rng)
                auth = utm_crs_info['auth']
                assert auth[0] == 'EPSG'
                epsg_int = int(auth[1])
                ulx, uly, lrx, lry = utm_box.to_ltrb().data[0]
            else:
                ulx, uly, lrx, lry = override_geom_box.to_ltrb().data
                epsg_int = int(override_geom_epsg)

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
    Create a random box in some UTM coordinate space.

    Args:
        rng (int | str | RandomState | None):

    Example:
        >>> from geowatch.demo.smart_kwcoco_demodata import _random_utm_box
        >>> _random_utm_box()
    """
    import numpy as np
    from kwarray.distributions import Uniform
    import kwarray
    from geowatch.utils import util_gis
    from osgeo import osr
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

    utm_crs_info = geowatch.gis.geotiff.make_crs_info_object(utm_crs)

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


def _parse_demostr(data, defaults, alias_to_key=None):
    """
    Special suffixes can be added to generic demo names. Parse them out here.
    Arguments are `-` separated, only known defaulted values are parsed. Bare
    default names are interpreted as a value of True, otherwise the value
    should be numeric. TODO: generalize this and conslidate in the kwcoco
    demo method.

    Example:
        >>> from geowatch.demo.smart_kwcoco_demodata import _random_utm_box, _parse_demostr
        >>> data = 'foo-bar-baz1-biz2.3'
        >>> defaults = {}
        >>> alias_to_key = None
        >>> _parse_demostr(data, defaults)
        ({}, {'foo': True, 'bar': True, 'baz': 1, 'biz': 2.3})
    """
    if alias_to_key is None:
        alias_to_key = {}
    from geowatch.utils.util_codes import parse_delimited_argstr
    handled = defaults.copy()
    unhandled = parse_delimited_argstr(data)
    for key, value in list(unhandled.items()):
        key = alias_to_key.get(key, key)
        if key in handled:
            handled[key] = value
            unhandled.pop(key, None)
    return handled, unhandled


def random_inscribed_polygon(bounding_polygon, rng=None):
    """
    Ignore:
        if 1:
            import kwplot
            kwplot.plt.ion()
            bounding_box.draw(facecolor='blue', alpha=0.8, setlim=1, fill=True, edgecolor='darkblue')
            utm_poly.draw(facecolor='orange', alpha=0.8, setlim=1, fill=True, edgecolor='darkorange')
            rando_utm.draw(facecolor='green', alpha=0.8, setlim=1, fill=True, edgecolor='darkgreen')
            inscribed_utm.draw(facecolor='red', alpha=0.8, setlim=1, fill=True, edgecolor='darkred')
    """
    import kwimage
    # Make a random polygon inscribed in the utm region
    bounding_box = kwimage.Box(bounding_polygon.bounding_box())
    rano_01 = kwimage.Polygon.random(tight=1, rng=rng)
    # Move to the origin, scale to match the box, and then move to the center
    # of the polygon of interest.
    rando = rano_01.translate((-.5, -.5)).scale((
        bounding_box.width, bounding_box.height)).translate(
            bounding_polygon.centroid)
    # Take the intersection ito inscribe
    inscribed = rando.intersection(bounding_polygon)
    return inscribed


def demo_dataset_with_regions_and_sites(dpath=None):
    """
    Get a demo coco dataset with region and site models.
    """
    import ubelt as ub
    import geowatch
    from geowatch.geoannots import geomodels
    from geowatch.geoannots.geococo_objects import CocoGeoVideo
    from geowatch.utils import util_gis
    import kwimage
    import kwarray

    coco_dset = geowatch.coerce_kwcoco(
        'geowatch-msi', heatmap=True, geodata=True,
        dates=True, image_size=(96, 96)
    )
    rng = kwarray.ensure_rng(4329423)

    # Make some region models for this dataset
    import geopandas as gpd
    region_and_sites = []
    crs84 = util_gis.get_crs84()

    video_name_to_crs84_bounds = {}

    for video in coco_dset.videos().objs:
        coco_video = CocoGeoVideo(video=video, dset=coco_dset)

        dates = coco_video.images.lookup('date_captured')
        start_time = min(dates)
        end_time = max(dates)

        utm_part = coco_video.wld_corners_gdf
        utm_poly = kwimage.Polygon.coerce(utm_part.iloc[0]['geometry'])
        # Make a random inscribed polygon to use as the test region
        utm_region_poly = random_inscribed_polygon(utm_poly, rng=rng)

        # Shrink it so we are more likely to remove annotations
        utm_region_poly = utm_region_poly.scale(0.5, about='centroid')

        crs84_region_poly = kwimage.Polygon.coerce(gpd.GeoDataFrame(
            geometry=[utm_region_poly],
            crs=utm_part.crs).to_crs(crs84).iloc[0]['geometry'])

        video_name_to_crs84_bounds[coco_video['name']] = crs84_region_poly

        region_model, sites = geomodels.RegionModel.random(
            region_id=coco_video['name'], region_poly=crs84_region_poly,
            rng=rng, start_time=start_time, end_time=end_time, with_sites=True)
        region_and_sites.append((region_model, sites))

    video0 = coco_dset.videos().objs[0]
    video0_images = coco_dset.images(video_id=video0['id'])
    assert len(video0_images) >= 10, 'should have a several frames'
    assert len(set(video0_images.lookup('date_captured'))) > 5, (
        'should be on different dates')

    if dpath is None:
        dpath = ub.Path.appdir('geowatch', 'demo', 'regions_and_sites').ensuredir()
    dpath.delete().ensuredir()

    # Write region models to disk
    region_dpath = (dpath / 'region_models').ensuredir()
    site_dpath = (dpath / 'site_models').ensuredir()
    for region_model, sites in region_and_sites:
        region_fpath = region_dpath / (region_model.region_id + '.geojson')
        region_fpath.write_text(region_model.dumps(indent=4))

        for site in sites:
            site_fpath = site_dpath / (site.site_id + '.geojson')
            site_fpath.write_text(site.dumps(indent=' ' * 4))

    return coco_dset, region_dpath, site_dpath


@ub.memoize
def _register_polygon_hash_data():
    """
    Allows ub.hash_data hash shapely geometry
    """
    import shapely.geometry.base

    @ub.hash_data.extensions.register(shapely.geometry.base.BaseGeometry)
    def hash_my_type(data):
        return b'Geometry', data.wkb
