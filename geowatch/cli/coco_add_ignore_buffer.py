import scriptconfig as scfg
import ubelt as ub


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
        >>>         coco_img = result_dest.coco_image(image_id)
        >>>         annots = coco_img.annots()
        >>>         dets = annots.detections
        >>>         catnames = [dets.classes[idx] for idx in dets.class_idxs]
        >>>         polys = [p.to_shapely() for p in annots.detections.data['segmentations']]
        >>>         catname_to_polys = ub.group_items(polys, catnames)
        >>>         ignore_poly = unary_union(catname_to_polys.get('ignore', []))
        >>>         nonignore_polys = ub.udict(catname_to_polys) - {'ignore'}
        >>>         nonignore_poly = unary_union(list(ub.flatten(nonignore_polys.values())))
        >>>         isect_poly = nonignore_poly.intersection(ignore_poly)
        >>>         union_poly = nonignore_poly.union(ignore_poly)
        >>>         iou = isect_poly.area / union_poly.area
        >>>         print(f'image_id={image_id}, iou = {ub.urepr(iou, nl=1)}')
        >>>         # FIXME: this iou should be zero but it appears not to be!
        >>>         assert iou < 0.1
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

    dset = kwcoco.CocoDataset(config.src)

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

            for image_id in pman.progiter(images, desc='looping over images...'):
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
                #annot_cat_ids = annots.lookup("category_id")
                annot_segmenations = annots.lookup("segmentation")
                # print(annot_segmenations)
                #annot_cat_names = dset.categories(annot_cat_ids).lookup("name")

                annot_polys = [
                    kwimage.MultiPolygon.coerce(s).to_shapely()
                    for s in annot_segmenations
                ]
                # We do not want to ignore any existing annotation region.
                do_not_ignore_poly = unary_union(annot_polys)

                # For each existing annotation
                new_ignore_polys = []
                for poly in annot_polys:
                    # Expand the region around it
                    expanded_poly = poly.buffer(ignore_buffer_pixel)
                    # Remove any regions touching existing annotation
                    new_ignore_geom = expanded_poly - do_not_ignore_poly
                    if not new_ignore_geom.is_empty:
                        new_ignore_polys.append(new_ignore_geom)

                if 0:
                    # kwimage.MultiPolygon.coerce(do_not_ignore_poly).draw(setlim=1,color='kitware_red')
                    kwimage.MultiPolygon.coerce(annot_polys[0]).draw(color="kitware_green")
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


if __name__ == "__main__":
    """
    python -m geowatch.cli.coco_addignore_buffer
    """
    main()