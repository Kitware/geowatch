"""
DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
"""
# xdoctest: +REQUIRES(env:DVC_DPATH)
# Run the following tests on real watch data if DVC is available
import os
from os.path import join
import kwcoco
import ubelt as ub
import kwimage
import numpy as np
from typing import List, Dict


def demo():
    # display with matplotlib
    DRAW = 1
    if DRAW:
        import kwplot
        kwplot.autompl()

    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
    dset = kwcoco.CocoDataset(coco_fpath)

    # Given an image
    img1_id = ub.peek(dset.imgs.keys())

    # Insetad of doing this
    annot_ids = dset.index.gid_to_aids[img1_id]
    img1 = dset.delayed_load(img1_id, channels='red|green|blue', space='video').finalize()
    dets1 = kwimage.Detections.from_coco_annots(
                    dset.annots(annot_ids).objs, dset=dset)

    if DRAW:
        canvas1 = kwimage.normalize_intensity(img1)
        canvas2 = dets1.draw_on(canvas1.copy())
        canvas3 = dets1.draw_on(np.zeros_like(canvas1))
        kwplot.imshow(canvas1, fnum=1, pnum=(3, 3, 1), title='kwcoco, orig, img only')
        kwplot.imshow(canvas2, fnum=1, pnum=(3, 3, 2), title='kwcoco, orig, img + annot')
        kwplot.imshow(canvas3, fnum=1, pnum=(3, 3, 3), title='kwcoco, orig, annot only')

    # This is the fix
    warp_vid_from_img = kwimage.Affine.coerce(dset.index.imgs[img1_id]['warp_img_to_vid'])
    dets1_fixed = dets1.warp(warp_vid_from_img)

    if DRAW:
        canvas1 = kwimage.normalize_intensity(img1)
        canvas2 = dets1_fixed.draw_on(canvas1.copy())
        canvas3 = dets1_fixed.draw_on(np.zeros_like(canvas1))
        kwplot.imshow(canvas1, fnum=1, pnum=(3, 3, 4), title='kwcoco, fix, img only')
        kwplot.imshow(canvas2, fnum=1, pnum=(3, 3, 5), title='kwcoco, fix, img + annot')
        kwplot.imshow(canvas3, fnum=1, pnum=(3, 3, 6), title='kwcoco, fix, annot only')

    # You may just want to do this
    import ndsampler
    sampler = ndsampler.CocoSampler(dset)

    # The sampler can load a sample
    sample       : Dict[str, object] = sampler.load_sample({'gids': [img1_id], 'channels': 'red|green|blue'})

    # Here are useful items in the sample, it contains more metadata than this
    frame_imdata : np.ndarray               = sample['im']
    frame_dets   : List[kwimage.Detections] = sample['annots']['frame_dets']

    # We loaded a sequence of images, so just take the first
    dets1 = frame_dets[0]
    img1 = frame_imdata[0]
    if DRAW:
        # display with matplotlib
        import kwplot
        kwplot.autompl()
        canvas1 = kwimage.normalize_intensity(img1)
        canvas2 = dets1.draw_on(canvas1.copy(), labels=False)
        canvas3 = dets1.draw_on(np.zeros_like(canvas1), labels=False)
        kwplot.imshow(canvas1, fnum=1, pnum=(3, 3, 7), title='ndsampler, img only')
        kwplot.imshow(canvas2, fnum=1, pnum=(3, 3, 8), title='ndsampler, img + annot')
        kwplot.imshow(canvas3, fnum=1, pnum=(3, 3, 9), title='ndsampler, annot only')


def demo2():
    # display with matplotlib
    DRAW = 1
    if DRAW:
        import kwplot
        kwplot.autompl()

    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
    dset = kwcoco.CocoDataset(coco_fpath)

    # Given an image
    video_id = ub.peek(dset.index.videos.keys())

    video_info = dset.index.videos[video_id]
    video_gids = dset.index.vidid_to_gids[video_id]

    vid_w = video_info['width']
    vid_h = video_info['height']

    rng = np.random.RandomState(0)
    chosen_gids = sorted(rng.choice(video_gids, size=3, replace=False))

    target = {
        # Right half of the video
        'space_slice': (slice(0, vid_h), slice(vid_w // 2, vid_w)),
        'gids': chosen_gids,
        'channels': 'red|green|blue'
    }

    import ndsampler
    sampler = ndsampler.CocoSampler(dset)

    # The sampler can load a sample
    sample       : Dict[str, object] = sampler.load_sample(target)

    # Here are useful items in the sample, it contains more metadata than this
    frame_imdata : np.ndarray               = sample['im']
    frame_dets   : List[kwimage.Detections] = sample['annots']['frame_dets']

    # We loaded a sequence of images, so just take the first

    if DRAW:
        vizlist = []
        for frame_idx in range(len(chosen_gids)):
            imdata = frame_imdata[frame_idx]
            dets = frame_dets[frame_idx]

            canvas = kwimage.normalize_intensity(imdata[:, :, 0:3])

            try:
                canvas = dets.draw_on(canvas, color='classes')
            except Exception:
                # Seems bugged, disable labels
                canvas = dets.draw_on(canvas, labels=False)

            vizlist.append(canvas)

        final_canvas = kwimage.stack_images(vizlist, axis=1, overlap=-10)  # negative overlap is a weird parameter for a pad I know. I don't know what I was thinking.
        kwplot.imshow(final_canvas)


def demo3():
    import torch
    # display with matplotlib
    DRAW = 1
    if DRAW:
        import kwplot
        kwplot.autompl()

    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/combo_data.kwcoco.json')
    dset = kwcoco.CocoDataset(coco_fpath)

    # Given a video
    video_id = ub.peek(dset.index.videos.keys())
    video_gids = dset.index.vidid_to_gids[video_id]

    # Choose an image, lets build a pandas dataframe to make an informed choice

    df_rows = []
    for gid in video_gids:
        _imginfo = dset.index.imgs[gid]
        df_rows.append({
            'gid': gid,
            'sensor': _imginfo['sensor_coarse'],
            'num_annots': len(dset.index.gid_to_aids[gid])
        })
    import pandas as pd
    frames_df = pd.DataFrame(df_rows)
    candidates = frames_df[frames_df.sensor == 'S2'].sort_values('num_annots')
    image_id = candidates.gid.iloc[13]

    # Gether the annotations inside that image
    annot_ids = dset.index.gid_to_aids[image_id]

    # Create information about video space and image space
    # (Reminder: annotations are stored in image space)
    img_info = dset.index.imgs[image_id]
    img_dims = (img_info['height'], img_info['width'])

    video_info = dset.index.videos[video_id]
    vid_dims = (video_info['height'], video_info['width'])

    warp_vid_from_img = kwimage.Affine.coerce(dset.index.imgs[image_id]['warp_img_to_vid'])

    # Construct variants of the image and annotations
    imdata_in_imgspace = dset.delayed_load(image_id, channels='red|green|blue', space='image').finalize()
    imdata_in_vidspace = dset.delayed_load(image_id, channels='red|green|blue', space='video').finalize()

    dets_in_imgspace = kwimage.Detections.from_coco_annots(
        dset.annots(annot_ids).objs, dset=dset)
    dets_in_vidspace = dets_in_imgspace.warp(warp_vid_from_img)

    # Mask construction in image space
    segmentations = dets_in_imgspace.data['segmentations'].data
    category_ids = [dets_in_imgspace.classes.idx_to_id[cidx]
                    for cidx in dets_in_imgspace.data['class_idxs']]
    combined = []
    for sseg, cid in zip(segmentations, category_ids):
        assert cid > 0
        np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
        mask1 = torch.from_numpy(np_mask)
        combined.append(mask1.unsqueeze(0))
    overall_mask = torch.max(torch.cat(combined, dim=0), dim=0)[0]
    mask_in_imgspace = ((overall_mask > 1) * 255).numpy().astype(np.uint8)

    # Mask construction in video space
    segmentations = dets_in_vidspace.data['segmentations'].data
    category_ids = [dets_in_vidspace.classes.idx_to_id[cidx]
                    for cidx in dets_in_vidspace.data['class_idxs']]
    combined = []
    for sseg, cid in zip(segmentations, category_ids):
        assert cid > 0
        np_mask = sseg.to_mask(dims=vid_dims).data.astype(float) * cid
        mask1 = torch.from_numpy(np_mask)
        combined.append(mask1.unsqueeze(0))
    overall_mask = torch.max(torch.cat(combined, dim=0), dim=0)[0]
    mask_in_vidspace = ((overall_mask > 1) * 255).numpy().astype(np.uint8)

    if DRAW:
        vidspace_canvas_raw = kwimage.normalize_intensity(imdata_in_vidspace)
        imgspace_canvas_raw = kwimage.normalize_intensity(imdata_in_imgspace)
        print('imgspace_canvas_raw.shape = {!r}'.format(imgspace_canvas_raw.shape))
        print('vidspace_canvas_raw.shape = {!r}'.format(vidspace_canvas_raw.shape))

        vidspace_canvas = dets_in_imgspace.draw_on(vidspace_canvas_raw.copy(), color='orange', labels=['img-space'] * len(annot_ids))
        vidspace_canvas = dets_in_vidspace.draw_on(vidspace_canvas, color='dodgerblue', labels=['vid-space'] * len(annot_ids))

        imgspace_canvas = dets_in_imgspace.draw_on(imgspace_canvas_raw.copy(), color='orange', labels=['img-space'] * len(annot_ids))
        imgspace_canvas = dets_in_vidspace.draw_on(imgspace_canvas, color='dodgerblue', labels=['vid-space'] * len(annot_ids))

        heatmask_in_vidspace = kwimage.ensure_alpha_channel(mask_in_vidspace, alpha=0.4)
        heatmask_in_imgspace = kwimage.ensure_alpha_channel(mask_in_imgspace, alpha=0.4)
        overlay_in_vidspace = kwimage.overlay_alpha_layers([heatmask_in_vidspace, vidspace_canvas_raw])
        overlay_in_imgspace = kwimage.overlay_alpha_layers([heatmask_in_imgspace, imgspace_canvas_raw])

        pnum_ = kwplot.PlotNums(nRows=3, nCols=2)
        kwplot.imshow(imgspace_canvas, fnum=1, pnum=pnum_(), title='img-space-canvas', show_ticks=1)
        kwplot.imshow(vidspace_canvas, fnum=1, pnum=pnum_(), title='vid-space-canvas', show_ticks=1)

        kwplot.imshow(mask_in_imgspace, fnum=1, pnum=pnum_(), title='img-space-mask', show_ticks=1)
        kwplot.imshow(mask_in_vidspace, fnum=1, pnum=pnum_(), title='vid-space-mask', show_ticks=1)

        kwplot.imshow(overlay_in_imgspace, fnum=1, pnum=pnum_(), title='img-space-overlay', show_ticks=1)
        kwplot.imshow(overlay_in_vidspace, fnum=1, pnum=pnum_(), title='vid-space-overlay', show_ticks=1)


def demo4():
    import torch
    # display with matplotlib
    DRAW = 1
    if DRAW:
        import kwplot
        kwplot.autompl()

    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/combo_data.kwcoco.json')
    dset = kwcoco.CocoDataset(coco_fpath)

    # Given a video
    video_id = ub.peek(dset.index.videos.keys())
    video_gids = dset.index.vidid_to_gids[video_id]

    # Choose an image, lets build a pandas dataframe to make an informed choice

    df_rows = []
    for gid in video_gids:
        _imginfo = dset.index.imgs[gid]
        df_rows.append({
            'gid': gid,
            'sensor': _imginfo['sensor_coarse'],
            'num_annots': len(dset.index.gid_to_aids[gid])
        })
    import pandas as pd
    frames_df = pd.DataFrame(df_rows)
    candidates = frames_df[frames_df.sensor == 'S2'].sort_values('num_annots')
    image_id = candidates.gid.iloc[13]

    # Gether the annotations inside that image
    annot_ids = dset.index.gid_to_aids[image_id]

    # Create information about video space and image space
    # (Reminder: annotations are stored in image space)
    img_info = dset.index.imgs[image_id]
    img_dims = (img_info['height'], img_info['width'])

    video_info = dset.index.videos[video_id]
    vid_dims = (video_info['height'], video_info['width'])

    warp_vid_from_img = kwimage.Affine.coerce(dset.index.imgs[image_id]['warp_img_to_vid'])

    # Construct variants of the image and annotations
    imdata_in_imgspace = dset.delayed_load(image_id, channels='red|green|blue', space='image').finalize()
    imdata_in_vidspace = dset.delayed_load(image_id, channels='red|green|blue', space='video').finalize()

    dets_in_imgspace = kwimage.Detections.from_coco_annots(
        dset.annots(annot_ids).objs, dset=dset)
    dets_in_vidspace = dets_in_imgspace.warp(warp_vid_from_img)

    # Mask construction in image space
    segmentations = dets_in_imgspace.data['segmentations'].data
    category_ids = [dets_in_imgspace.classes.idx_to_id[cidx]
                    for cidx in dets_in_imgspace.data['class_idxs']]
    combined = []
    for sseg, cid in zip(segmentations, category_ids):
        assert cid > 0
        np_mask = sseg.to_mask(dims=img_dims).data.astype(float) * cid
        mask1 = torch.from_numpy(np_mask)
        combined.append(mask1.unsqueeze(0))
    overall_mask = torch.max(torch.cat(combined, dim=0), dim=0)[0]
    mask_in_imgspace = ((overall_mask > 1) * 255).numpy().astype(np.uint8)

    # Mask construction in video space
    segmentations = dets_in_vidspace.data['segmentations'].data
    category_ids = [dets_in_vidspace.classes.idx_to_id[cidx]
                    for cidx in dets_in_vidspace.data['class_idxs']]
    combined = []
    for sseg, cid in zip(segmentations, category_ids):
        assert cid > 0
        np_mask = sseg.to_mask(dims=vid_dims).data.astype(float) * cid
        mask1 = torch.from_numpy(np_mask)
        combined.append(mask1.unsqueeze(0))
    overall_mask = torch.max(torch.cat(combined, dim=0), dim=0)[0]
    mask_in_vidspace = ((overall_mask > 1) * 255).numpy().astype(np.uint8)

    if DRAW:
        vidspace_canvas_raw = kwimage.normalize_intensity(imdata_in_vidspace)
        imgspace_canvas_raw = kwimage.normalize_intensity(imdata_in_imgspace)
        print('imgspace_canvas_raw.shape = {!r}'.format(imgspace_canvas_raw.shape))
        print('vidspace_canvas_raw.shape = {!r}'.format(vidspace_canvas_raw.shape))

        vidspace_canvas = dets_in_imgspace.draw_on(vidspace_canvas_raw.copy(), color='orange', labels=['img-space'] * len(annot_ids))
        vidspace_canvas = dets_in_vidspace.draw_on(vidspace_canvas, color='dodgerblue', labels=['vid-space'] * len(annot_ids))

        imgspace_canvas = dets_in_imgspace.draw_on(imgspace_canvas_raw.copy(), color='orange', labels=['img-space'] * len(annot_ids))
        imgspace_canvas = dets_in_vidspace.draw_on(imgspace_canvas, color='dodgerblue', labels=['vid-space'] * len(annot_ids))

        heatmask_in_vidspace = kwimage.ensure_alpha_channel(mask_in_vidspace, alpha=0.4)
        heatmask_in_imgspace = kwimage.ensure_alpha_channel(mask_in_imgspace, alpha=0.4)
        overlay_in_vidspace = kwimage.overlay_alpha_layers([heatmask_in_vidspace, vidspace_canvas_raw])
        overlay_in_imgspace = kwimage.overlay_alpha_layers([heatmask_in_imgspace, imgspace_canvas_raw])

        pnum_ = kwplot.PlotNums(nRows=3, nCols=2)
        kwplot.imshow(imgspace_canvas, fnum=1, pnum=pnum_(), title='img-space-canvas', show_ticks=1)
        kwplot.imshow(vidspace_canvas, fnum=1, pnum=pnum_(), title='vid-space-canvas', show_ticks=1)

        kwplot.imshow(mask_in_imgspace, fnum=1, pnum=pnum_(), title='img-space-mask', show_ticks=1)
        kwplot.imshow(mask_in_vidspace, fnum=1, pnum=pnum_(), title='vid-space-mask', show_ticks=1)

        kwplot.imshow(overlay_in_imgspace, fnum=1, pnum=pnum_(), title='img-space-overlay', show_ticks=1)
        kwplot.imshow(overlay_in_vidspace, fnum=1, pnum=pnum_(), title='vid-space-overlay', show_ticks=1)


def demo_change_pairs():
    # display with matplotlib
    DRAW = 1
    if DRAW:
        import kwplot
        kwplot.autompl()

    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    # coco_fpath = join(dvc_dpath, 'drop1-S2-L8-aligned/data.kwcoco.json')
    bundle_dpath = join(dvc_dpath, 'drop1-S2-L8-aligned')
    coco_fpath = join(bundle_dpath, 'combo_data.kwcoco.json')
    dset = kwcoco.CocoDataset(coco_fpath)

    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    import kwarray
    window_overlap = 0.5
    window_dims = (96, 96)
    keepbound = False
    vidid_to_space_slider = {}
    for vidid, video in dset.index.videos.items():
        full_dims = [video['height'], video['width']]
        window_dims_ = full_dims if window_dims == 'full' else window_dims
        slider = kwarray.SlidingWindow(full_dims, window_dims_,
                                       overlap=window_overlap,
                                       keepbound=keepbound,
                                       allow_overshoot=True)
        vidid_to_space_slider[vidid] = slider

    # from ndsampler import isect_indexer
    # _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(dset)

    all_regions = []
    positive_idxs = []
    negative_idxs = []

    print('dset.cats = {}'.format(ub.urepr(dset.cats, nl=1)))

    # Given an video
    # video_id = ub.peek()
    for video_id in dset.index.videos.keys():
        slider = vidid_to_space_slider[video_id]

        video_info = dset.index.videos[video_id]
        video_gids = dset.index.vidid_to_gids[video_id]

        import pyqtree
        qtree = pyqtree.Index((0, 0, video_info['width'], video_info['height']))
        qtree.aid_to_ltrb = {}

        tid_to_info = ub.ddict(list)
        video_aids = dset.images(video_gids).annots.lookup('id')
        for aids, gid in zip(video_aids, video_gids):

            warp_vid_from_img = kwimage.Affine.coerce(dset.index.imgs[gid]['warp_img_to_vid'])
            # warp_img_from_vid = warp_vid_from_img.inv()

            img_info = dset.index.imgs[gid]
            frame_index = img_info['frame_index']
            tids = dset.annots(aids).lookup('track_id')
            cids = dset.annots(aids).lookup('category_id')
            for tid, aid, cid in zip(tids, aids, cids):

                imgspace_box = kwimage.Boxes([dset.index.anns[aid]['bbox']], 'xywh')
                vidspace_box = imgspace_box.warp(warp_vid_from_img)

                tlbr_box = vidspace_box.to_tlbr().data[0]
                qtree.insert(aid, tlbr_box)
                qtree.aid_to_ltrb[aid] = tlbr_box

                dset.index.anns[aid]['bbox']

                tid_to_info[tid].append({
                    'gid': gid,
                    'cid': cid,
                    'frame_index': frame_index,
                    # 'box': box,
                    'cname': dset._resolve_to_cat(cid)['name'],
                    'aid': aid,
                })

        # print('tid_to_info = {}'.format(ub.urepr(tid_to_info, nl=2, sort=0)))
        for space_region in list(slider):
            y_sl, x_sl = space_region

            # Find all annotations that pass through this spatial region
            vid_box = kwimage.Boxes.from_slice((y_sl, x_sl))
            query = vid_box.to_tlbr().data[0]
            isect_aids = sorted(set(qtree.intersect(query)))

            isect_annots = dset.annots(isect_aids)
            unique_tracks = set(isect_annots.lookup('track_id'))
            if len(unique_tracks) > 1:
                break

            frame_idxs = isect_annots.images.lookup('frame_index')
            isect_annots = isect_annots.take(ub.argsort(frame_idxs))

            frame_idxs = isect_annots.images.lookup('frame_index')
            track_ids = isect_annots.lookup('track_id')

            region_tid_to_aids = ub.group_items(isect_annots._ids, track_ids)
            region_tid_to_info = {}
            # For each track gather, start and stop info within this region
            for tid, aids in region_tid_to_aids.items():
                cids = dset.annots(aids).lookup('category_id')

                # phase_boundaries = np.diff(cids)

                frame_idxs = dset.annots(aids).images.lookup('frame_index')
                region_tid_to_info[tid] = {
                    'tid': tid,
                    'frame_idxs': frame_idxs,
                    'cids': cids,
                    'aids': aids,
                }
                pass

            pre_cid  = dset.index.name_to_cat['No Activity']['id']
            post_cid = dset.index.name_to_cat['Post Construction']['id']
            ignore_cid = dset.index.name_to_cat['Unknown']['id']

            import itertools as it
            video_frame_idxs = list(range(len(video_gids)))

            for frame_idxs in list(it.combinations(video_frame_idxs, 2)):

                gids = list(ub.take(video_gids, frame_idxs))

                has_change = False
                region_tracks = []
                for tid, track_info in region_tid_to_info.items():
                    sample_cids = []
                    for fx in frame_idxs:
                        idx = np.searchsorted(track_info['frame_idxs'], fx, 'right') - 1
                        if idx < 0:
                            cid = pre_cid
                        else:
                            cid = track_info['cids'][idx]
                        sample_cids.append(cid)

                    if len(sample_cids) and sample_cids[0] == post_cid:
                        # any leading "change completed" that transition into
                        # no-change labels should be marked as negative.
                        sample_cids
                        pass

                    if len(set(sample_cids) - {ignore_cid}) > 1:
                        # This frame sampling has a change in it
                        has_change = True

                    region_tracks.append({
                        'tid': tid,
                        'sample_cids': sample_cids,
                    })

                if has_change:
                    positive_idxs.append(len(all_regions))
                else:
                    negative_idxs.append(len(all_regions))

                all_regions.append({
                    'gids': gids,
                    'space_slice': space_region,
                    'has_change': has_change,
                    'region_tracks': region_tracks,
                })

    print(len(positive_idxs))
    print(len(negative_idxs))

    for idx in positive_idxs:
        print(all_regions[idx])
