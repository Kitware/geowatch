#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub

#TODO: Either use coco_add_ignore_buffer here or use run_all_pointgen to calculate it


class CocoAddIgnoreBufferConfig(scfg.DataConfig):
    """
    Given a kwcoco file and buffer size update annotations to include ignore buffer regions around polygons
    """

    src = scfg.Value(
        None,
        help="Specify input kwcoco filepath",
    )

    ignore_buffer_size = scfg.Value(
        "10@10GSD",
        help="give a resolved unit (e.g. 10@10mGSD) for a buffer size around each other polygon.",
    )

    dst = scfg.Value(None, help="Specify output kwcoco filepath")


def main(cmdline=1, **kwargs):
    r"""
    IGNORE:
        python $HOME/Desktop/geowatch/geowatch/cli/coco_add_ignore_buffer.py \
            --src $HOME/Desktop/dvc_repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip \
            --dst $HOME/Desktop/dvc_repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002_modified-rawbands.kwcoco.zip

    CommandLine:
        xdoctest -m geowatch.cli.coco_add_ignore_buffer
        xdoctest $HOME/Desktop/geowatch/geowatch/cli/coco_add_ignore_buffer.py

    Example:
        >>> from geowatch.cli.coco_add_ignore_buffer import *
        >>> import geowatch
        >>> import ubelt as ub
        >>> import kwcoco
        >>> dpath = ub.Path.appdir('geowatch/tests/ignore_buffer')
        >>> dpath.ensuredir()
        >>> ignore_buffer_size = '10@10GSD'
        >>> dst = dpath / 'out.kwcoco.zip'
        >>> src = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True)
        >>> kwargs = dict(src=src.data_fpath, dst=dst)
        >>> main(cmdline=0, **kwargs)
        >>> result_dest = kwcoco.CocoDataset(dst)
        >>> # Check non of the ignore polygons are overlaping non-ignore
        >>> from shapely.ops import unary_union
        >>> for video_id in result_dest.videos():
        >>>     images = result_dest.images(video_id=video_id)
        >>>     for image_id in images:
        >>>         src_coco_img = src.coco_image(image_id)
        >>>         dst_coco_img = result_dest.coco_image(image_id)
        >>>         src_annots = src_coco_img.annots()
        >>>         dst_annots = dst_coco_img.annots()
        >>>         new_aids = ub.oset(dst_annots) - ub.oset(src_annots)
        >>>         old_aids = list(src_annots)
        >>>         # The old polygons (new polys should not intersect these)
        >>>         old_polys = [p.to_shapely() for p in result_dest.annots(old_aids).detections.data['segmentations']]
        >>>         # The new polygons that should be the ignored regions
        >>>         new_polys = [p.to_shapely() for p in result_dest.annots(new_aids).detections.data['segmentations']]
        >>>         new_poly = unary_union(new_polys)
        >>>         old_poly = unary_union(old_polys)
        >>>         isect_poly = new_poly.intersection(old_poly)
        >>>         union_poly = new_poly.union(old_poly)
        >>>         iou = isect_poly.area / union_poly.area
        >>>         print(f'image_id={image_id}, iou = {ub.urepr(iou, nl=1)}')
        >>>         # The iou should be nearly zero (up to float errors)
        >>>         assert iou < 1e-5
    """
    import rich
    from rich.markup import escape

    config = CocoAddIgnoreBufferConfig.cli(
        cmdline=cmdline, data=kwargs,
        # special_options=False  # requires recent scriptconfig
    )
    rich.print("config = " + escape(ub.urepr(config)))

    import kwimage
    from geowatch.utils import util_resolution
    from shapely.ops import unary_union
    import numpy as np
    import kwcoco
    import kwutil

    if config.src is None:
        raise ValueError('must specify src kwcoco')

    dset = kwcoco.CocoDataset(config.src)

    dset = dset.reroot(absolute=True)

    # TODO: if unit is not specified, work in videospace instead of world space
    utm_gsd = util_resolution.ResolvedUnit.coerce("1GSD")
    ignore_buffer = util_resolution.ResolvedScalar.coerce(config.ignore_buffer_size)

    ignore_buffer_gsd = ignore_buffer.at_resolution(utm_gsd).scalar
    videos = dset.videos()

    pman = kwutil.util_progress.ProgressManager()
    with pman:
        # Ignore eff
        for video_id in pman.progiter(videos, desc="looping over videos..."):
            images = dset.images(video_id=video_id)

            for image_id in pman.progiter(images, desc="looping over images..."):
                coco_img = dset.coco_image(image_id)
                _imgspace_resolution = coco_img.resolution(space="image")
                image_pxl_per_meter = 1 / np.array(_imgspace_resolution["mag"])
                ignore_buffer_pixel = ignore_buffer_gsd * image_pxl_per_meter
                # TODO: buffer utilzing shapely method currently only accounts
                # for a singular float distance need to account for several
                # distances in the future, but for now the average of the
                # buffer region suggested is used.
                ignore_buffer_pixel = ignore_buffer_pixel.mean()
                annots = coco_img.annots()
                # annot_cat_ids = annots.lookup("category_id")
                annot_segmenations = annots.lookup("segmentation")
                # print(annot_segmenations)
                # annot_cat_names = dset.categories(annot_cat_ids).lookup("name")

                annot_polys = [
                    kwimage.MultiPolygon.coerce(s).to_shapely()
                    for s in annot_segmenations
                ]
                # We do not want to ignore any existing annotation region.
                do_not_ignore_poly = unary_union(annot_polys)

                # For each existing annotation
                new_ignore_polys = []
                SANITY_CHECK = 0
                if SANITY_CHECK == 1:
                    for poly in annot_polys:
                        expanded_poly = poly.buffer(ignore_buffer_pixel)
                        # SANITY_CHECK = 0
                        # TODO: dont use big whiles
                        # SANITY_CHECK
                        iou = 1
                        while iou > 0.0001 and not expanded_poly.is_empty:
                            # Expand the region around it
                            expanded_poly = poly.buffer(ignore_buffer_pixel)
                            # Remove any regions touching existing annotation
                            new_ignore_geom = expanded_poly - do_not_ignore_poly
                            for nonignore_poly in annot_polys:
                                isect_poly = nonignore_poly.intersection(
                                    new_ignore_geom
                                )
                                union_poly = nonignore_poly.union(new_ignore_geom)
                            iou = isect_poly.area / union_poly.area
                            # print(iou)
                            expanded_poly = new_ignore_geom
                        if not new_ignore_geom.is_empty:
                            new_ignore_polys.append(expanded_poly)
                else:
                    for poly in annot_polys:
                        expanded_poly = poly.buffer(ignore_buffer_pixel)
                        # Expand the region around it
                        # Remove any regions touching existing annotation
                        new_ignore_geom = expanded_poly - do_not_ignore_poly
                        if not new_ignore_geom.is_empty:
                            new_ignore_polys.append(new_ignore_geom)

                if 0:
                    # kwimage.MultiPolygon.coerce(do_not_ignore_poly).draw(setlim=1,color='kitware_red')
                    kwimage.MultiPolygon.coerce(annot_polys[0]).draw(
                        color="kitware_green"
                    )
                    for poly in new_ignore_polys:
                        kwimage.MultiPolygon.coerce(poly).draw(color="kitware_blue")
                    kwimage.MultiPolygon.coerce(do_not_ignore_poly).draw(
                        setlim=1, color="kitware_red"
                    )

                for poly in new_ignore_polys:
                    _poly = kwimage.MultiPolygon.from_shapely(poly)
                    dset.add_annotation(
                        image_id=image_id,
                        category_id=dset.ensure_category("ignore"),
                        bbox=_poly.bounding_box().to_xywh(),
                        segmentation=_poly,
                    )
                # g.plot() #Shows base polys
                # kwplot.kwplot.plt.show()
                # f.plot() #Shows the aftermath of do_not_ignore_poly
                # kwplot.plt.show()
                # p.plot() # Shows all polys with their buffer
                # kwplot.plt.show()
    out_path = config.dst
    # Write to the compressed path
    dset.dump(out_path)
    rich.print(f"Wrote modified kwcoco to: [link={out_path}]{out_path}[/link]")

__config__ = CocoAddIgnoreBufferConfig


if __name__ == "__main__":
    """
    python -m geowatch.cli.coco_add_ignore_buffer
    """
    main()
