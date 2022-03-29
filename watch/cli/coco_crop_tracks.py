import scriptconfig as scfg
import kwcoco
import kwimage
import ubelt as ub


class CocoCropTrackConfig(scfg.Config):
    """
    Create a dataset of aligned temporal sequences around objects of interest
    in an unstructured collection of annotated geotiffs.

    High Level Steps:
        * Find a set of geospatial AOIs
        * For each AOI find all images that overlap
        * Orthorectify (or warp) the selected spatial region and its
          annotations to a cannonical space.
    """
    default = {
        'src': scfg.Value('in.geojson.json', help='input dataset to chip'),

        'dst': scfg.Value(None, help='bundle directory or kwcoco json file for the output'),

        'workers': scfg.Value(0, type=str, help='number of parallel procs. This can also be an expression accepted by coerce_num_workers.'),

        'sqlmode': scfg.Value(0, type=str, help='if True use sqlmode'),
    }


def main(cmdline=0, **kwargs):
    """
    Simple CLI for cropping to tracks

    Ignore:
        from watch.cli.coco_crop_tracks import *  # NOQA
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
        dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
        src = base_fpath
        dst = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json'
        cmdline = 0
        kwargs = dict(src=src, dst=dst)
    """
    config = CocoCropTrackConfig(cmdline=cmdline, data=kwargs)
    src = config['src']
    dst = ub.Path(config['dst'])
    dst_bundle_dpath = dst.parent

    if config['sqlmode']:
        coco_dset = kwcoco.CocoSqlDatabase.coerce(src)
    else:
        coco_dset = kwcoco.CocoDataset.coerce(src)
        # sql_coco_dset = coco_dset.view_sql()
        # coco_dset = sql_coco_dset

    crop_job_gen = generate_crop_jobs(coco_dset, dst_bundle_dpath)
    crop_job_iter = iter(crop_job_gen)

    keep = 'img'
    # crop_asset_task = next(crop_job_iter)
    # _run_crop_asset_task(crop_asset_task, keep)
    jobs = ub.JobPool(mode='process', max_workers=8)
    prog = ub.ProgIter(desc='submit crop jobs')
    with prog:
        try:
            while True:
                crop_asset_task = next(crop_job_iter)
                prog.ensure_newline()
                jobs.submit(_run_crop_asset_task, crop_asset_task, keep)
                prog.update()
        except StopIteration:
            pass

    results = []
    for job in jobs.as_completed(desc='collect jobs'):
        result = job.result()
        results.append(result)


def generate_crop_jobs(coco_dset, dst_bundle_dpath):
    import shapely
    from watch.utils import util_time

    # Build to_extract-like objects so this script can eventually be combined
    # with coco-align-geotiffs to make something that's ultimately better.
    valid_categories = {
        'No Activity', 'Site Preparation', 'Active Construction',
        'Post Construction',
    }

    # [aids[0] tid, aids in coco_dset.index.trackid_to_aids]

    try:
        tid_to_aids = coco_dset.index.trackid_to_aids
    except Exception:
        annots = coco_dset.index.annots()
        tid_to_aids = ub.group_items(annots.lookup('id'), annots.lookup('track_id'))

    bundle_dpath = ub.Path(coco_dset.bundle_dpath)

    # for vidid in coco_dset.videos():
    #     video = coco_dset.index.videos[vidid]
    #     # video_images = coco_dset.images(vidid=vidid)
    #     # Get all annotations for this video in video space
    #     # video_aids = list(ub.flatten(video_images.annots.lookup('id')))
    #     # video_annots = coco_dset.annots(video_aids)
    #     # tid_to_anns = ub.group_items(video_annots.objs, video_annots.lookup('track_id'))
    #     # coco_dset.index.trackid_to_aids
    #     # track_group_dname = dst_bundle_dpath /

    for tid, aids in tid_to_aids.items():
        aid0 = ub.peek(aids)
        ann0 = coco_dset.index.anns[aid0]
        cid = ann0['category_id']
        category = coco_dset.index.cats[cid]
        catname = category['name']
        if catname in valid_categories:
            break
            raise Exception

        img0 = coco_dset.index.imgs[ann0['image_id']]
        video = coco_dset.index.videos[img0['video_id']]

        # One iteration over the track to gather the segmentation region.
        anns = coco_dset.annots(aids).objs
        track_polys = []
        for ann in anns:
            gid = ann['image_id']
            coco_img = coco_dset.coco_image(gid)
            vid_sseg = coco_img._annot_segmentation(ann, space='video')
            track_polys.append(vid_sseg.to_shapely())
        sh_vid_track_poly = shapely.ops.unary_union(track_polys).convex_hull
        # Might need to pad out this region
        vid_track_poly = kwimage.MultiPolygon.from_shapely(sh_vid_track_poly)

        # Given the segmentation region, generate track tasks
        track_name = tid if isinstance(tid, str) else 'track_{}'.format(tid)
        track_dname = ub.Path(video['name']) / track_name
        # String representing the spatial crop
        # space_str = ub.hash_data(vid_track_poly.to_geojson(), base='abc')[0:8]
        space_str = track_name

        crop_track_task = {
            'tid': tid,
        }
        # crop_img_tasks = []
        # Assume each annotation corresponds to exactly one image
        for ann in anns:
            gid = ann['image_id']
            coco_img = coco_dset.coco_image(gid)
            sensor_coarse = coco_img.get('sensor_coarse', 'unknown')
            warp_img_from_vid = coco_img.warp_img_from_vid
            datetime_ = util_time.coerce_datetime(coco_img['date_captured'])
            iso_time = util_time.isoformat(datetime_, sep='T', timespec='seconds')
            # img_crop = vid_track_poly.warp(warp_img_from_vid)
            crop_img_task = {
                'gid': gid,
                'sensor_coarse': sensor_coarse,
                'datetime': datetime_,
                **crop_track_task,
            }
            # crop_asset_tasks = []
            track_img_dname = track_dname / sensor_coarse
            for obj in coco_img.iter_asset_objs():
                warp_img_from_aux = kwimage.Affine.coerce(obj['warp_aux_to_img'])
                warp_aux_from_img = warp_img_from_aux.inv()
                warp_aux_from_vid = warp_aux_from_img @ warp_img_from_vid
                aux_track_poly = vid_track_poly.warp(warp_aux_from_vid)
                chan_code = obj['channels']

                if 1:
                    chan_pname = kwcoco.FusedChannelSpec.coerce(chan_code).path_sanitize()
                    if len(chan_pname) > 10:
                        # Hack to prevent long names for docker (limit is 242 chars)
                        num_bands = kwcoco.FusedChannelSpec.coerce(chan_code).numel()
                        chan_pname = '{}_{}'.format(ub.hash_data(chan_pname, base='abc')[0:8], num_bands)
                    else:
                        num_bands = obj['num_bands']

                # Construct a name for the subregion to extract.
                num = 0
                name = 'crop_{}_{}_{}_{}'.format(iso_time, space_str, sensor_coarse, num)
                dst_fname = track_img_dname / name / f'{name}_{chan_pname}.tif'

                crop_pixel_box = list(aux_track_poly.bounding_box().quantize().to_xywh().to_coco())[0]
                crop_asset_task = {**crop_img_task}
                crop_asset_task['parent_file_name'] = obj['file_name']
                crop_asset_task['file_name'] = dst_fname
                crop_asset_task['src'] = bundle_dpath / obj['file_name']
                crop_asset_task['dst'] = dst_bundle_dpath / dst_fname
                crop_asset_task['crop_pixel_box'] = crop_pixel_box
                print('crop_pixel_box = {!r}'.format(crop_pixel_box))
                crop_asset_task['channels'] = chan_code
                crop_asset_task['num_bands'] = num_bands

                # aux_track_poly
                yield crop_asset_task
        #         crop_asset_tasks.append(crop_asset_task)
        #     crop_img_task['crop_asset_tasks'] = crop_asset_tasks
        #     crop_img_tasks.append(crop_img_task)
        # crop_track_task['crop_img_tasks'] = crop_img_tasks


def _run_crop_asset_task(crop_asset_task, keep):
    from watch.utils import util_gdal
    _crop_task = crop_asset_task.copy()
    src = _crop_task.pop('src')
    dst = _crop_task.pop('dst')
    crop_pixel_box = _crop_task.pop('crop_pixel_box')
    cache_hit = keep in {'img'} and dst.exists()
    if not cache_hit:
        pixel_box = kwimage.Boxes([crop_pixel_box], 'xywh')
        print('src = {!r}'.format(src))
        print('dst = {!r}'.format(dst))
        print('crop_pixel_box = {!r}'.format(crop_pixel_box))
        dst.parent.ensuredir()
        util_gdal.gdal_single_translate(src, dst, pixel_box)
    return _crop_task
