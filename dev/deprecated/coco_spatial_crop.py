"""
TODO:
    - [ ] Can generalize this into a parametarized script later, but for now
          we will just split on the images in half

    - [ ] Could use gdalwarp to preserve geodata

"""
import scriptconfig as scfg
import kwimage
import ubelt as ub


class CocoSpatialCropConfig(scfg.Config):
    default = {
        'src': scfg.Value(help='input dataset'),
        'dst': scfg.Value(help='output dataset (ideally with a new bundle path)'),

        'suffix': '_left',
        'max_workers': 6,
        # ''
    }


def main(cmdline=True, **kwargs):
    r"""

    Example:
        >>> from watch.cli.coco_spatial_crop import *  # NOQA
        >>> from watch.cli import coco_spatial_crop
        >>> import kwcoco
        >>> src = kwcoco.CocoDataset.demo('vidshapes8-multispectral')
        >>> dst = src.bundle_dpath + '_left/data.kwcoco.json'
        >>> cmdline = False
        >>> kwargs = dict(src=src, dst=dst, suffix='_left')
        >>> dst_dset = coco_spatial_crop.main(**kwargs)

    Ignore:
        from watch.cli.coco_visualize_videos import *  # NOQA
        from watch.cli import coco_visualize_videos
        coco_visualize_videos.main(src=dst, viz_dpath=str(pathlib.Path(dst_dset.bundle_dpath) / '_viz'))
        coco_visualize_videos.main(src=src, viz_dpath=str(pathlib.Path(dst_dset.bundle_dpath) / '_viz_orig'))

    Ignore:
        >>> from watch.cli.coco_spatial_crop import *  # NOQA
        >>> from watch.cli import coco_spatial_crop
        >>> import os
        >>> DVC_DPATH = ub.expandpath(os.environ.get('DVC_DPATH', '$HOME/data/dvc-repos/smart_watch_dvc'))
        >>> DVC_DPATH = pathlib.Path(DVC_DPATH)
        >>> src = DVC_DPATH / 'drop1-S2-L8-aligned/LT_Kaunas_R01.kwcoco.json'
        >>> dst = DVC_DPATH / 'drop1-S2-L8-aligned_left/LT_Kaunas_R01_left.kwcoco.json'
        >>> cmdline = False
        >>> kwargs = dict(src=src, dst=dst, suffix='_left')
        >>> dst_dset = coco_spatial_crop.main(**kwargs)


    kwcoco subset \
        --src $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/data.kwcoco.json \
        --dst $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/LT_Kaunas_R01.kwcoco.json \
       --select_videos '.name | startswith("LT_Kaunas_R01")'

    python -m watch.cli.coco_visualize_videos \
        --src $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/LT_Kaunas_R01.kwcoco.json \
        --viz_dpath $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/_viz_LT_Kaunas \
        --space video

    python -m watch.cli.coco_spatial_crop \
        --src $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/LT_Kaunas_R01.kwcoco.json \
        --dst $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_left/LT_Kaunas_R01_left.kwcoco.json \
        --suffix="_left"

    python -m watch.cli.coco_spatial_crop \
        --src $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned/LT_Kaunas_R01.kwcoco.json \
        --dst $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_right/LT_Kaunas_R01_right.kwcoco.json \
        --suffix="_right"

    python -m watch.cli.coco_visualize_videos \
        --src $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_left/LT_Kaunas_R01_left.kwcoco.json \
        --viz_dpath $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_left/_viz_LT_Kaunas_left \
        --space video

    python -m watch.cli.coco_visualize_videos \
        --src $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_right/LT_Kaunas_R01_right.kwcoco.json \
        --viz_dpath $HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_right/_viz_LT_Kaunas_right \
        --space video

    python -m watch.cli.gifify -i "$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_right/_viz_LT_Kaunas_right/LT_Kaunas_R01_right/_anns/r|g|b" \
        -o LT_Kaunas_right.gif --frames_per_second=1

    python -m watch.cli.gifify -i "$HOME/data/dvc-repos/smart_watch_dvc/drop1-S2-L8-aligned_left/_viz_LT_Kaunas_left/LT_Kaunas_R01_left/_anns/r|g|b" \
        -o LT_Kaunas_left.gif --frames_per_second=1


    """
    import kwcoco
    # from watch.utils import kwcoco_extensions
    from os.path import join
    import pathlib
    import kwimage
    config = CocoSpatialCropConfig(data=kwargs, cmdline=cmdline)
    src_dset = kwcoco.CocoDataset.coerce(config['src'])
    suffix = config['suffix']

    # HACK:
    if suffix == '_left':
        x_frac_min = 0.0
        x_frac_max = 0.5
        y_frac_min = 0.0
        y_frac_max = 1.0
    elif suffix == '_right':
        x_frac_min = 0.50
        x_frac_max = 1.0
        y_frac_min = 0.0
        y_frac_max = 1.0
    elif suffix == '_hacktest':
        x_frac_min = 0.53
        x_frac_max = 0.95
        y_frac_min = 1 / 3
        y_frac_max = 1.0
    else:
        # Crop region is not parametarized correctly yet
        raise NotImplementedError(suffix)

    dst_dset = kwcoco.CocoDataset()
    dst_dset.fpath = config['dst']

    dest_dpath = pathlib.Path(dst_dset.bundle_dpath) / ('crop' + suffix)
    dest_rel_dpath = dest_dpath.relative_to(dst_dset.bundle_dpath)

    for cat in src_dset.cats.values():
        dst_dset.ensure_category(**cat)

    prog = ub.ProgIter(total=len(src_dset.index.imgs), desc='warp dataset')
    prog.begin()

    # Create job pool to do crops in the background
    pool = ub.JobPool(mode='thread', max_workers=config['max_workers'])

    for vidid, video in src_dset.index.videos.items():
        new_video = video.copy()
        new_video['name'] = video['name'] + suffix
        prog.set_postfix_str(f'process vidid={video["id"]} - {video["name"]}')

        video_box = kwimage.Boxes([
            [0, 0, video['width'], video['height']]], 'xywh')

        x_vid_min = int(video['width'] * x_frac_min)
        x_vid_max = int(video['width'] * x_frac_max)
        y_vid_min = int(video['height'] * y_frac_min)
        y_vid_max = int(video['height'] * y_frac_max)

        video_crop_box = kwimage.Boxes([[
            x_vid_min, y_vid_min, x_vid_max, y_vid_max]], 'ltrb')
        new_vid_width = video_crop_box.width[0].item()
        new_vid_height = video_crop_box.height[0].item()

        newvid_from_vid = kwimage.Affine.coerce(offset=(
            -video_crop_box.tl_x.item(),
            -video_crop_box.tl_y.item(),
        ))

        newvid_box = video_box.warp(newvid_from_vid).clip(
            0, 0, new_vid_width, new_vid_height)

        new_video['width']  = int(newvid_box.width.item())
        new_video['height'] = int(newvid_box.height.item())
        new_vidid = dst_dset.add_video(**new_video)

        src_gids = src_dset.index.vidid_to_gids[vidid]

        # src_dset.images(src_gids).lookup('sensor_coarse')
        # src_dset.images(src_gids).lookup('warp_img_to_vid')

        if 0:
            print('video_box = {!r}'.format(video_box))
            print('video_crop_box = {}'.format(ub.repr2(video_crop_box, nl=1)))
            print('newvid_from_vid = {!r}'.format(newvid_from_vid.concise()))

        for gid in src_gids:

            img = src_dset.index.imgs[gid]
            new_img = ub.dict_diff(img, {
                'warp_to_wld',
                'wld_to_pxl',
                'wld_crs_info',
                'utm_crs_info',
                'utm_corners',
                'auxiliary',
                'width',
                'height',
                'warp_img_to_vid',
            })

            vid_from_img = kwimage.Affine.coerce(img.get('warp_img_to_vid', None))
            img_from_vid = vid_from_img.inv()

            img_box = kwimage.Boxes([[0, 0, img['width'], img['height']]], 'xywh')
            img_crop_box = video_crop_box.warp(img_from_vid)

            img_crop_box_quant = img_crop_box.quantize().clip(
                0, 0, img['width'], img['height'])
            img_from_newcropimg = kwimage.Affine.coerce(offset=(
                img_crop_box_quant.tl_x.item(),
                img_crop_box_quant.tl_y.item(),
            ))
            newcropimg_from_img = img_from_newcropimg.inv()
            newcropimg_box = img_box.warp(newcropimg_from_img).clip(
                0, 0, img_crop_box_quant.width.item(),
                img_crop_box_quant.height.item())

            newvid_from_newcropimg = (
                newvid_from_vid @ vid_from_img @ img_from_newcropimg
            )
            newcropimg_from_newvid = newvid_from_newcropimg.inv()

            new_img['width']  = int(newcropimg_box.width.item())
            new_img['height'] = int(newcropimg_box.height.item())
            new_img['video_id'] = new_vidid
            new_img['auxiliary'] = []
            new_img['warp_img_to_vid'] = newvid_from_newcropimg.concise()

            # img_crop_box = video_crop_box.warp(img_from_vid)
            for aux in img.get('auxiliary', []):

                """
                There are 3 types of auxiliary iamges here:

                    * aux - the original auxilliary image

                    * newaux - the aux image that *exactly* corresponds to the
                        video crop box, this is related directly to the new
                        image space.

                    * newcropaux - the aux image quantized so it can be sliced
                        it is related to newaux by a offset factor. This is the
                        space on disk.
                """
                img_from_aux = kwimage.Affine.coerce(aux.get('warp_aux_to_img', None))
                aux_from_img = img_from_aux.inv()
                aux_from_vid = aux_from_img @ img_from_vid
                vid_from_aux = aux_from_vid.inv()

                aux_crop_box = video_crop_box.warp(aux_from_vid)
                aux_crop_box_quant = aux_crop_box.quantize().clip(
                    0, 0, aux['width'], aux['height'])

                aux_from_newcropaux = kwimage.Affine.coerce(offset=(
                    aux_crop_box_quant.tl_x.item(),
                    aux_crop_box_quant.tl_y.item(),
                ))

                # Probably overly complex, but the units should cancel
                newcropimg_from_newcropaux = (
                    newcropimg_from_newvid @ newvid_from_vid @
                    vid_from_aux @ aux_from_newcropaux)

                aux_fpath = join(src_dset.bundle_dpath, aux.get('file_name'))

                aux_suffix = pathlib.Path(aux_fpath).relative_to(src_dset.bundle_dpath)
                new_fname = ub.augpath(dest_rel_dpath / aux_suffix, suffix=suffix)
                new_fpath = join(dst_dset.bundle_dpath, new_fname)

                pathlib.Path(new_fpath).parent.mkdir(exist_ok=1, parents=True)

                #  remove keys that are no longer valid
                newcropaux = ub.dict_diff(aux, {
                    'warp_to_wld',
                    'wld_to_pxl',
                    'wld_crs_info',
                    'utm_crs_info',
                    'utm_corners',
                    'width',
                    'height',
                    'file_name',
                    'warp_aux_to_img',
                })
                newcropaux['warp_aux_to_img'] = newcropimg_from_newcropaux.concise()
                newcropaux['width'] = int(aux_crop_box_quant.width.item())
                newcropaux['height'] = int(aux_crop_box_quant.height.item())
                newcropaux['file_name'] = new_fname

                aux_slices = aux_crop_box_quant.to_slices()[0]
                cropjob_kw = dict(
                    aux_fpath=aux_fpath,
                    new_fpath=new_fpath,
                    aux_slices=aux_slices,
                )
                pool.submit(crop_worker, **cropjob_kw)

                new_img['auxiliary'].append(newcropaux)

            new_gid = dst_dset.add_image(**new_img)

            # Warp the annotations into the crop space
            anns = src_dset.annots(gid=gid)
            dets = anns.detections
            overlap = dets.boxes.isect_area(img_crop_box)
            flags = (overlap > 0)[:, 0]

            # Port over annotations that fall in the box
            port_anns = list(ub.compress(anns.objs, flags))
            port_dets = dets.compress(flags)

            new_dets = port_dets.warp(newcropimg_from_img)
            new_anns = list(new_dets.to_coco(dset=src_dset, style='new'))
            for new_ann, ann in zip(new_anns, port_anns):
                other_info = ub.dict_diff(ann, new_ann)
                new_ann.update(other_info)
                new_ann['image_id'] = new_gid
                dst_dset.add_annotation(**new_ann)

            prog.update(1)
    prog.end()

    # for cropjob_kw in crop_jobs:
    for job in ub.ProgIter(pool.as_completed(), total=len(pool), desc='finish crop jobs'):
        job.result()

    print('write dst_dset.fpath = {!r}'.format(dst_dset.fpath))
    dst_dset.dump(dst_dset.fpath, newlines=True)
    return dst_dset


def crop_worker(aux_fpath, new_fpath, aux_slices):
    aux_src_imdata = kwimage.imread(aux_fpath)
    aux_dst_imdata = aux_src_imdata[aux_slices]
    kwimage.imwrite(new_fpath, aux_dst_imdata, backend='gdal',
                    compress='DEFLATE')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/cli/coco_spatial_crop.py
    """
    main()
