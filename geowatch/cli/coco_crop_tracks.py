r"""
CommandLine:

    DVC_DPATH=$(geowatch_dvc --hardware="hdd")
    echo $DVC_DPATH
    python -m geowatch.cli.coco_crop_tracks \
        --src="$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json" \
        --dst="$DVC_DPATH/Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json" \
        --mode=process --workers=8


    # Small test of KR only
    DVC_DPATH=$(geowatch_dvc --hardware="hdd")
    echo $DVC_DPATH
    python -m geowatch.cli.coco_crop_tracks \
        --src="$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json" \
        --dst="$DVC_DPATH/Cropped-Drop2-TA1-test/data.kwcoco.json" \
        --mode=process --workers=8 --channels="red|green|blue" \
        --include_sensors="WV,S2,L8" --select_videos '.name | startswith("KR_R001")' \
        --target_gsd=3


    TODO:
        - [ ] option to merge overlapping regions?
"""
import scriptconfig as scfg
import ubelt as ub


class CocoCropTrackConfig(scfg.DataConfig):
    """
    Create a dataset of aligned temporal sequences around objects of interest
    in an unstructured collection of annotated geotiffs.

    High Level Steps:
        * Find a set of geospatial AOIs
        * For each AOI find all images that overlap
        * Orthorectify (or warp) the selected spatial region and its
          annotations to a cannonical space.
    """
    __default__ = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory or kwcoco json file for the output'),

        'workers': scfg.Value(1, type=str, help='number of parallel procs. This can also be an expression accepted by coerce_num_workers.'),

        'mode': scfg.Value('process', type=str, help='process, thread, or serial'),

        'context_factor': scfg.Value(1.8, help=ub.paragraph('scale factor')),

        'sqlmode': scfg.Value(0, type=str, help='if True use sqlmode'),

        'keep': scfg.Value('img', help='set to None to recompute'),

        'target_gsd': scfg.Value(1, help='GSD of new kwcoco videospace'),

        'channels': scfg.Value(None, help='only crop these channels if specified'),

        'select_images': scfg.Value(
            None, type=str, help=ub.paragraph(
                '''
                A jq query to specify images. See kwcoco subset --help for more details.
                ''')),

        'select_videos': scfg.Value(
            None, help=ub.paragraph(
                '''
                A jq query to specify videos. See kwcoco subset --help for more details.
                ''')),

        'include_sensors': scfg.Value(None, help='if specified can be comma separated valid sensors'),

        'exclude_sensors': scfg.Value(None, help='if specified can be comma separated invalid sensors'),

    }


def main(cmdline=0, **kwargs):
    """
    Simple CLI for cropping to tracks

    Ignore:
        from geowatch.cli.coco_crop_tracks import *  # NOQA
        import geowatch
        dvc_dpath = geowatch.find_dvc_dpath(hardware='hdd')
        # dvc_dpath = geowatch.find_dvc_dpath(hardware='ssd')
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
        src = base_fpath
        dst = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json'
        cmdline = 0
        include_sensors = None
        kwargs = dict(
            workers=21,
            src=src, dst=dst, include_sensors=include_sensors)

    Ignore:
        from geowatch.cli.coco_crop_tracks import *  # NOQA
        import geowatch
        dvc_dpath = geowatch.find_dvc_dpath(hardware='hdd')
        base_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json'
        src = base_fpath
        dst = dvc_dpath / 'Cropped-Drop2-TA1-2022-02-15/data.kwcoco.json'
        cmdline = 0
        include_sensors = ['WV']
        kwargs = dict(src=src, dst=dst, include_sensors=include_sensors)
        kwargs['workers'] = 8
    """
    import kwcoco
    config = CocoCropTrackConfig.cli(cmdline=cmdline, data=kwargs, strict=True)
    print('config = {}'.format(ub.urepr(config, nl=1)))
    src = config['src']
    dst = ub.Path(config['dst'])
    dst_bundle_dpath = dst.parent

    print('load = {}'.format(ub.urepr(src, nl=1)))
    if config['sqlmode']:
        coco_dset = kwcoco.CocoSqlDatabase.coerce(src)
    else:
        coco_dset = kwcoco.CocoDataset.coerce(src)
    # sql_coco_dset = coco_dset.view_sql()
    # coco_dset = sql_coco_dset

    from geowatch.utils import kwcoco_extensions
    valid_gids = kwcoco_extensions.filter_image_ids(
        coco_dset,
        include_sensors=config['include_sensors'],
        exclude_sensors=config['exclude_sensors'],
        select_images=config['select_images'],
        select_videos=config['select_videos'],
    )
    if len(valid_gids) != coco_dset.n_images:
        coco_dset = coco_dset.subset(valid_gids)

    context_factor = config['context_factor']
    channels = config['channels']
    if channels is not None:
        channels = kwcoco.FusedChannelSpec.coerce(channels)

    print('Generate jobs')
    crop_job_gen = generate_crop_jobs(coco_dset, dst_bundle_dpath,
                                      context_factor=context_factor,
                                      channels=channels)
    crop_job_iter = iter(crop_job_gen)

    keep = config['keep']
    from kwutil import util_parallel
    workers = util_parallel.coerce_num_workers(config['workers'])
    jobs = ub.JobPool(mode=config['mode'], max_workers=workers)

    prog = ub.ProgIter(desc='submit crop jobs')
    last_key = None
    with prog:
        try:
            while True:
                crop_asset_task = next(crop_job_iter)
                # prog.ensure_newline()
                jobs.submit(run_crop_asset_task, crop_asset_task, keep)
                key = (crop_asset_task['region'], crop_asset_task['tid'])
                if last_key != key:
                    last_key = key
                    prog.set_postfix_str(key)
                prog.update()
        except StopIteration:
            pass

    results = []
    failed = []
    for job in jobs.as_completed(desc='collect jobs'):
        try:
            result = job.result()
        except Exception as ex:
            # import traceback
            # stack = traceback.extract_stack()
            # stack_lines = traceback.format_list(stack)
            # tbtext = ''.join(stack_lines)
            # print(ub.highlight_code(tbtext, 'pytb'))
            print('ex = {}'.format(ub.urepr(ex, nl=1)))
            print('Failed crop asset task ex = {!r}'.format(ex))
            raise
            failed.append(job)
        else:
            results.append(result)

    # Group assets by the track they belong to
    tid_to_assets = ub.group_items(results, lambda x: x['tid'])
    tid_to_size = ub.map_vals(len, tid_to_assets)
    import kwarray
    import numpy as np
    print(f'{len(tid_to_size)=}')
    arr = np.array(list(tid_to_size.values()))
    stats = kwarray.stats_dict(arr)
    if len(arr):
        quantile = [0.25, 0.50, 0.75]
        quant_values = np.quantile(arr, quantile)
        quant_keys = ['q_{:0.2f}'.format(q) for q in quantile]
        for k, v in zip(quant_keys, quant_values):
            stats[k] = v
    print(f'track length stats {ub.urepr(stats, nl=1)!s}')

    # Rebuild the manifest
    target_gsd = config['target_gsd']
    new_dset = make_track_kwcoco_manifest(dst, dst_bundle_dpath, tid_to_assets,
                                          target_gsd=target_gsd)
    import safer
    with safer.open(dst, 'w', temp_file=not ub.WIN32) as file:
        new_dset.dump(file, newlines=True, indent='    ')

    r"""
    geowatch visualize \
        /home/joncrall/data/dvc-repos/smart_watch_dvc-hdd/Cropped-Drop2-TA1-2022-02-15/data.kwcoco.json \
        --channels="red|green|blue" \
        --animate=True \
        --workers=8

    DVC_DPATH=$(geowatch_dvc --hardware="hdd")
    echo $DVC_DPATH
    cp $DVC_DPATH/Cropped-Drop2-TA1-2022-02-15/data.kwcoco.json \
       $DVC_DPATH/Cropped-Drop2-TA1-2022-02-15/imgonly.kwcoco.json


    python -m geowatch reproject_annotations \
        --src "$DVC_DPATH/Cropped-Drop2-TA1-2022-02-15/imgonly.kwcoco.json" \
        --dst "$DVC_DPATH/Cropped-Drop2-TA1-2022-02-15/projected.kwcoco.json" \
        --site_models="$DVC_DPATH/annotations/site_models/*.geojson" \
        --region_models="$DVC_DPATH/annotations/region_models/*.geojson"

    # {viz_part}

    """


def make_track_kwcoco_manifest(dst, dst_bundle_dpath, tid_to_assets,
                               target_gsd=1):
    """
    Rebundle in a a new kwcoco file

    TODO:
        - [ ] populate auxiliary is taking a long time, speed it up.
    """
    from geowatch.utils import kwcoco_extensions
    import kwimage
    import kwcoco
    # Make the new kwcoco file where 1 track is mostly 1 video
    # TODO: we could crop the kwcoco annotations here too, but
    # we can punt on that for now and just reproject them.
    new_dset = kwcoco.CocoDataset()

    new_dset.fpath = dst
    for tid, track_assets in tid_to_assets.items():

        new_video = {
            'name': tid,
        }
        new_vidid = new_dset.add_video(**new_video)
        gid_to_assets = ub.group_items(track_assets, lambda x: x['gid'])

        new_images = []
        for gid, img_assets in gid_to_assets.items():
            auxiliary = []
            region = None
            datetime_ = None
            sensor_coarse = None
            for aux in img_assets:
                fname = aux['file_name']
                aux = aux.copy()
                aux.pop('gid', None)
                aux.pop('tid', None)
                region = aux.pop('region', None)
                sensor_coarse = aux['sensor_coarse']
                datetime_ = aux.pop('datetime')
                aux['file_name'] = str(fname)
                auxiliary.append(aux)

            auxiliary = sorted(auxiliary, key=lambda obj: obj['channels'])

            new_img = {
                'name': fname.parent.name,
                'file_name': None,
                'auxiliary': auxiliary,
                'date_captured': datetime_.isoformat(),
                'sensor_coarse': sensor_coarse,
                'parent_region': region,
                'timestamp': datetime_.timestamp(),
            }
            new_images.append(new_img)
            # new_gid = new_dset.add_image(**new_img)
        new_images = sorted(new_images, key=lambda x: x['date_captured'])
        for idx, new_img in enumerate(new_images):
            new_img['frame_index'] = idx
            new_img['video_id'] = new_vidid
            new_dset.add_image(**new_img)

    for new_img in ub.ProgIter(new_dset.dataset['images'], desc='populate auxiliary'):
        for obj in new_img['auxiliary']:
            kwcoco_extensions._populate_canvas_obj(
                dst_bundle_dpath, obj, overwrite={'warp'}, with_wgs=True)

    for new_img in new_dset.dataset['images']:
        kwcoco_extensions._recompute_auxiliary_transforms(new_img)

    for new_img in new_dset.dataset['images']:
        new_coco_img = kwcoco.CocoImage(new_img)
        new_coco_img._bundle_dpath = dst_bundle_dpath
        new_coco_img._video = {}
        kwcoco_extensions._populate_valid_region(new_coco_img)

    from kwcoco.util.util_json import ensure_json_serializable
    for new_img in ub.ProgIter(new_dset.dataset['images'], desc='cleanup imgs'):
        for obj in ub.flatten([[new_img], new_img['auxiliary']]):
            if 'warp_to_wld' in obj:
                obj['warp_to_wld'] = kwimage.Affine.coerce(obj['warp_to_wld']).concise()
            if 'wld_to_pxl' in obj:
                obj['wld_to_pxl'] = kwimage.Affine.coerce(obj['wld_to_pxl']).concise()
            # obj.pop('wgs84_to_wld', None)
            # obj.pop('valid_region_utm', None)
            # obj.pop('utm_corners', None)
            # obj.pop('wgs84_corners', None)
            # obj.pop('utm_crs_info', None)
        new_img.update(ensure_json_serializable(new_img))

    # Make the asset order consistent by channel
    for img in new_dset.dataset['images']:
        if 'auxiliary' in img:
            img['auxiliary'] = sorted(img['auxiliary'], key=lambda aux: aux['channels'])

    for video_id in ub.ProgIter(new_dset.videos(), desc='populate videos'):
        kwcoco_extensions.coco_populate_geo_video_stats(
            new_dset, target_gsd=target_gsd, video_id=video_id
        )

    return new_dset


# @xdev.profile
def generate_crop_jobs(coco_dset, dst_bundle_dpath, channels=None, context_factor=1.0):
    """
    Generator that yields parameters to be used to call gdal_translate

    Benchmark:
        # Test kwimage versus shapely warp

        kw_poly = kwimage.Polygon.random()
        kw_poly = kwimage.Boxes.random(1).to_polygons()[0]

        sh_poly = kw_poly.to_shapely()
        transform = kwimage.Affine.random()

        import timerit
        ti = timerit.Timerit(100, bestof=10, verbose=2)
        for timer in ti.reset('shapely'):
            with timer:
                # This is faster than fancy indexing
                a, b, x, d, e, y = transform.matrix.ravel()[0:6]
                sh_transform = (a, b, d, e, x, y)
                sh_warp_poly = affine_transform(sh_poly, sh_transform)

        for timer in ti.reset('kwimage'):
            with timer:
                kw_warp_poly = kw_poly.warp(transform)

        kw_warp_poly2 = kwimage.Polygon.from_shapely(sh_warp_poly)
        print(kw_warp_poly2)
        print(kw_warp_poly)
    """
    import shapely
    from kwutil import util_time
    import kwcoco
    from shapely.affinity import affine_transform
    import kwimage

    # Build to_extract-like objects so this script can eventually be combined
    # with coco-align-geotiffs to make something that's ultimately better.

    # HACK: only take tracks with specific categories (todo: glob-parameterize)
    valid_categories = {
        'No Activity',
        'Site Preparation',
        'Active Construction',
        'Post Construction',
        'negative',
    }
    print('\n\n')

    # [aids[0] tid, aids in coco_dset.index.trackid_to_aids]

    try:
        tid_to_aids = coco_dset.index.trackid_to_aids
    except Exception:
        annots = coco_dset.index.annots()
        tid_to_aids = ub.group_items(annots.lookup('id'), annots.lookup('track_id'))

    bundle_dpath = ub.Path(coco_dset.bundle_dpath)
    _lut_coco_image = ub.memoize(coco_dset.coco_image)

    # for video_id in coco_dset.videos():
    #     video = coco_dset.index.videos[video_id]
    #     # video_images = coco_dset.images(video_id=video_id)
    #     # Get all annotations for this video in video space
    #     # video_aids = list(ub.flatten(video_images.annots.lookup('id')))
    #     # video_annots = coco_dset.annots(video_aids)
    #     # tid_to_anns = ub.group_items(video_annots.objs, video_annots.lookup('track_id'))
    #     # coco_dset.index.trackid_to_aids
    #     # track_group_dname = dst_bundle_dpath /

    print(f'{len(tid_to_aids)=}')
    # if 0:
    #     tids = tid_to_aids.keys()
    # else:
    vidid_and_tid_list = []
    for tid, aids in ub.ProgIter(tid_to_aids.items(), total=len(tid_to_aids), desc='one loop over tracks'):
        aid0 = ub.peek(aids)
        ann0 = coco_dset.index.anns[aid0]
        cid = ann0['category_id']
        category = coco_dset.index.cats[cid]
        catname = category['name']
        flag = catname in valid_categories
        if flag:
            img0 = coco_dset.index.imgs[ann0['image_id']]
            vidid = img0['video_id']
            vidid_and_tid_list.append((vidid, tid))
    vidid_and_tid_list = sorted(vidid_and_tid_list)

    # print(f'{len(tids)=}')
    for vidid, tid in vidid_and_tid_list:
        aids = tid_to_aids[tid]
        # print(f'{len(aids)=}')

        video = coco_dset.index.videos[vidid]
        region = video['name']

        # One iteration over the track to gather the segmentation region.
        annots = coco_dset.annots(list(aids))
        annot_dates = annots.images.lookup('date_captured')
        sortx = ub.argsort(annot_dates)
        annots = annots.take(sortx)

        # Hack only take some of the frames
        if 1:
            # Pad with some number of no activity frames
            cname_to_idxs = ub.group_items(list(range(len(annots))), annots.cnames)
            # Take 5
            k = 'Post Construction'
            if k in cname_to_idxs:
                cname_to_idxs[k] = sorted(cname_to_idxs[k])[0:5]
            k = 'No Activity'
            if k in cname_to_idxs:
                cname_to_idxs[k] = sorted(cname_to_idxs[k])[-5:]

            final_idxs = sorted(ub.flatten(cname_to_idxs.values()))
            annots = annots.take(final_idxs)

        track_polys = []
        anns = annots.objs
        if len(anns) == 0:
            continue
        # print(f'{len(anns)=}')
        for ann in anns:
            gid = ann['image_id']
            coco_img = _lut_coco_image(gid)
            vid_sseg = coco_img._annot_segmentation(ann, space='video')
            track_polys.append(vid_sseg.to_shapely())
        # Might need to pad out this region
        vid_track_poly_sh = shapely.ops.unary_union(track_polys).convex_hull

        SQUARE_BOUNDS = 1
        SHAPELY_TRANFORM = 1
        if SQUARE_BOUNDS:
            vid_track_poly = kwimage.MultiPolygon.from_shapely(vid_track_poly_sh).data[0]
            cx, cy, w, h = vid_track_poly.to_boxes().to_cxywh().data[0]
            w = h = max(w, h)
            vid_track_square = kwimage.Boxes([[cx, cy, w, h]], 'cxywh')
            vid_track_square = vid_track_square.scale(context_factor, about='center')
            vid_track_poly_sh = vid_track_square.to_shapely()[0]
            # Ensure we dont crop past the edges
            video_bounds_sh = kwimage.Boxes([
                [0, 0, video['width'], video['height']]
            ], 'xywh').to_shapely()[0]
            vid_track_poly_sh = vid_track_poly_sh.intersection(video_bounds_sh)

        # vid_track_poly = vid_track_poly.scale(1.1, about='center')

        # Given the segmentation region, generate track tasks
        track_name = tid if isinstance(tid, str) else 'track_{}'.format(tid)
        track_dname = ub.Path(region) / track_name
        # String representing the spatial crop
        space_str = ub.hash_data(vid_track_poly_sh.wkt, base='abc')[0:8]

        crop_track_task = {
            'tid': tid,
        }
        # crop_img_tasks = []
        # Assume each annotation corresponds to exactly one image
        for ann in anns:
            gid = ann['image_id']
            coco_img = _lut_coco_image(gid)

            # Skip this image if it corresponds to an invalid region
            try:
                valid_region = coco_img.valid_region(space='video')
                if valid_region is not None:
                    valid_region_sh = valid_region.to_shapely()
                    if not valid_region_sh.intersects(vid_track_poly_sh):
                        continue
            except AttributeError as ex:
                print('warning (might need kwcoco > 0.2.27) ex = {!r}'.format(ex))

            sensor_coarse = coco_img.get('sensor_coarse', 'unknown')
            datetime_ = util_time.coerce_datetime(coco_img['date_captured'])
            iso_time = util_time.isoformat(datetime_, sep='T', timespec='seconds')

            crop_img_task = {
                'gid': gid,
                'sensor_coarse': sensor_coarse,
                'datetime': datetime_,
                **crop_track_task,
            }
            # crop_asset_tasks = []
            track_img_dname = track_dname / sensor_coarse

            warp_img_from_vid = coco_img.warp_img_from_vid

            for obj in coco_img.iter_asset_objs():
                chan_code = obj['channels']
                obj_channels = kwcoco.FusedChannelSpec.coerce(chan_code)
                if channels is not None:
                    if obj_channels.intersection(channels).numel() == 0:
                        # Skip this channel
                        continue

                warp_img_from_aux = kwimage.Affine.coerce(obj['warp_aux_to_img'])
                warp_aux_from_img = warp_img_from_aux.inv()
                warp_aux_from_vid = warp_aux_from_img @ warp_img_from_vid

                if SHAPELY_TRANFORM:
                    a, b, x, d, e, y = warp_aux_from_vid.matrix.ravel()[0:6]
                    sh_transform = (a, b, d, e, x, y)
                    aux_track_poly_sh = affine_transform(vid_track_poly_sh, sh_transform)
                    aux_track_poly = kwimage.Polygon.from_shapely(aux_track_poly_sh)
                # else:
                #     aux_track_poly = vid_track_poly.warp(warp_aux_from_vid)
                crop_box_asset_space = aux_track_poly.bounding_box().quantize().to_xywh().data[0]

                # Construct a name for the subregion to extract.
                num = 0
                # to prevent long names for docker (limit is 242 chars)
                chan_pname = obj_channels.path_sanitize(maxlen=10)
                name = 'crop_{}_{}_{}_{}_{}'.format(iso_time, track_name, space_str, sensor_coarse, num)
                dst_fname = track_img_dname / name / f'{name}_{chan_pname}.tif'

                crop_asset_task = {**crop_img_task}
                crop_asset_task['parent_file_name'] = obj['file_name']
                crop_asset_task['file_name'] = dst_fname
                crop_asset_task['region'] = region
                crop_asset_task['src'] = bundle_dpath / obj['file_name']
                crop_asset_task['dst'] = dst_bundle_dpath / dst_fname
                crop_asset_task['crop_box_asset_space'] = crop_box_asset_space
                crop_asset_task['channels'] = chan_code
                crop_asset_task['num_bands'] = obj['num_bands']
                yield crop_asset_task


def run_crop_asset_task(crop_asset_task, keep):
    from osgeo import osr
    osr.GetPROJSearchPaths()
    from geowatch.utils import util_gdal
    import kwimage
    _crop_task = crop_asset_task.copy()
    src = _crop_task.pop('src')
    dst = _crop_task.pop('dst')
    crop_box_asset_space = _crop_task.pop('crop_box_asset_space')
    cache_hit = keep in {'img'} and dst.exists()
    if not cache_hit:
        pixel_box = kwimage.Boxes([crop_box_asset_space], 'xywh')
        dst.parent.ensuredir()
        util_gdal.gdal_single_translate(src, dst, pixel_box, tries=10)
    return _crop_task


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/geowatch/cli/coco_crop_tracks.py
    """
    main(cmdline=1)
