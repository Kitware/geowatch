r"""
CommandLine:

    DVC_DPATH=$(python -m watch.cli.find_dvc --hardware="hdd")
    echo $DVC_DPATH
    XDEV_PROFILE=1 python -m watch.cli.coco_crop_tracks \
        --src="$DVC_DPATH/Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json" \
        --dst="$DVC_DPATH/Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json" \
        --mode=process --workers=0
"""
import scriptconfig as scfg
import kwcoco
import kwimage
import ubelt as ub

import xdev


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

        'workers': scfg.Value(1, type=str, help='number of parallel procs. This can also be an expression accepted by coerce_num_workers.'),

        'mode': scfg.Value('process', type=str, help='process, thread, or serial'),

        'sqlmode': scfg.Value(0, type=str, help='if True use sqlmode'),
    }


def main(cmdline=0, **kwargs):
    """
    Simple CLI for cropping to tracks

    Ignore:
        from watch.cli.coco_crop_tracks import *  # NOQA
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')
        # dvc_dpath = watch.find_smart_dvc_dpath(hardware='ssd')
        base_fpath = dvc_dpath / 'Aligned-Drop3-TA1-2022-03-10/data.kwcoco.json'
        src = base_fpath
        dst = dvc_dpath / 'Cropped-Drop3-TA1-2022-03-10/data.kwcoco.json'
        cmdline = 0
        kwargs = dict(src=src, dst=dst)
    """
    config = CocoCropTrackConfig(cmdline=cmdline, data=kwargs)
    print('config = {}'.format(ub.repr2(dict(config), nl=1)))
    src = config['src']
    dst = ub.Path(config['dst'])
    dst_bundle_dpath = dst.parent

    print('load = {}'.format(ub.repr2(src, nl=1)))
    if config['sqlmode']:
        coco_dset = kwcoco.CocoSqlDatabase.coerce(src)
    else:
        coco_dset = kwcoco.CocoDataset.coerce(src)
    # sql_coco_dset = coco_dset.view_sql()
    # coco_dset = sql_coco_dset

    print('Generate jobs')
    crop_job_gen = generate_crop_jobs(coco_dset, dst_bundle_dpath)
    crop_job_iter = iter(crop_job_gen)

    # keep = 'img'
    keep = None
    # crop_asset_task = next(crop_job_iter)
    # run_crop_asset_task(crop_asset_task, keep)
    from watch.utils.lightning_ext import util_globals
    workers = util_globals.coerce_num_workers(config['workers'])
    jobs = ub.JobPool(mode=config['mode'], max_workers=workers)
    # prog = ub.ProgIter(desc='submit crop jobs', freq=1000)
    prog = ub.ProgIter(desc='submit crop jobs')
    last_key = None
    with prog:
        try:
            while True:
                crop_asset_task = next(crop_job_iter)
                prog.ensure_newline()
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
            print('ex = {!r}'.format(ex))
            failed.append(job)
        results.append(result)


@xdev.profile
def generate_crop_jobs(coco_dset, dst_bundle_dpath):
    import shapely
    from watch.utils import util_time

    # Build to_extract-like objects so this script can eventually be combined
    # with coco-align-geotiffs to make something that's ultimately better.

    # HACK: only take tracks with specific categories (todo: glob-parameterize)
    valid_categories = {
        'No Activity',
        'Site Preparation',
        'Active Construction',
        'Post Construction',
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

    # for vidid in coco_dset.videos():
    #     video = coco_dset.index.videos[vidid]
    #     # video_images = coco_dset.images(vidid=vidid)
    #     # Get all annotations for this video in video space
    #     # video_aids = list(ub.flatten(video_images.annots.lookup('id')))
    #     # video_annots = coco_dset.annots(video_aids)
    #     # tid_to_anns = ub.group_items(video_annots.objs, video_annots.lookup('track_id'))
    #     # coco_dset.index.trackid_to_aids
    #     # track_group_dname = dst_bundle_dpath /

    print(f'{len(tid_to_aids)=}')
    if 0:
        tids = tid_to_aids.keys()
    else:
        tids = []
        vidids = []
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
                tids.append(tid)
                vidids.append(vidid)

    print(f'{len(tids)=}')
    for tid, vidid in zip(tids, vidids):
        aids = tid_to_aids[tid]
        print(f'{len(aids)=}')

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
        print(f'{len(anns)=}')
        for ann in anns:
            gid = ann['image_id']
            coco_img = _lut_coco_image(gid)
            vid_sseg = coco_img._annot_segmentation(ann, space='video')
            track_polys.append(vid_sseg.to_shapely())
        # Might need to pad out this region
        vid_track_poly_sh = shapely.ops.unary_union(track_polys).convex_hull
        vid_track_poly = kwimage.MultiPolygon.from_shapely(vid_track_poly_sh).data[0]
        vid_track_poly = vid_track_poly.scale(1.1, about='center')

        # Given the segmentation region, generate track tasks
        track_name = tid if isinstance(tid, str) else 'track_{}'.format(tid)
        track_dname = ub.Path(region) / track_name
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
            coco_img = _lut_coco_image(gid)

            # Skip this image if it corresponds to an invalid region
            valid_region = coco_img.valid_region(space='video').to_shapely()
            if not valid_region.intersects(vid_track_poly_sh):
                continue

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
                warp_img_from_aux = kwimage.Affine.coerce(obj['warp_aux_to_img'])
                warp_aux_from_img = warp_img_from_aux.inv()
                warp_aux_from_vid = warp_aux_from_img @ warp_img_from_vid
                aux_track_poly = vid_track_poly.warp(warp_aux_from_vid)
                crop_box_asset_space = list(aux_track_poly.bounding_box().quantize().to_xywh().to_coco())[0]

                chan_code = obj['channels']
                # to prevent long names for docker (limit is 242 chars)
                chan_pname = kwcoco.FusedChannelSpec.coerce(chan_code).path_sanitize(maxlen=10)

                # Construct a name for the subregion to extract.
                num = 0
                name = 'crop_{}_{}_{}_{}'.format(iso_time, space_str, sensor_coarse, num)
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
    from watch.utils import util_gdal
    _crop_task = crop_asset_task.copy()
    src = _crop_task.pop('src')
    dst = _crop_task.pop('dst')
    crop_box_asset_space = _crop_task.pop('crop_box_asset_space')
    cache_hit = keep in {'img'} and dst.exists()
    if not cache_hit:
        pixel_box = kwimage.Boxes([crop_box_asset_space], 'xywh')
        # print('src = {!r}'.format(src))
        # print('dst = {!r}'.format(dst))
        # print('crop_box_asset_space = {!r}'.format(crop_box_asset_space))
        dst.parent.ensuredir()
        util_gdal.gdal_single_translate(src, dst, pixel_box)
    return _crop_task


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/coco_crop_tracks.py
    """
    main(cmdline=1)
