import scriptconfig as scfg


class CocoAddIgnoreBufferConfig(scfg.DataConfig):
    """
    Given a kwcoco file and buffer size update annotations to include ignore buffer regions around polygons
    """

    src = scfg.Value(
        "geowatch-msi",
        help="Specify input kwcoco filepath",
    )

    ignore_buffer_size = scfg.Value(
        "10@10GSD",
        help="give a resolved unit (e.g. 10@10mGSD) for a buffer size around each other polygon.",
    )

    dst = scfg.Value(None, help="Specify output kwcoco filepath")


def main(cmdline=1, **kwargs):
    """
    IGNORE:
        python /home/local/KHQ/vincenzo.dimatteo/Desktop/geowatch/geowatch/cli/coco_add_ignore_buffer.py --src /home/local/KHQ/vincenzo.dimatteo/Desktop/dvc_repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip --dst /home/local/KHQ/vincenzo.dimatteo/Desktop/dvc_repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002_modified-rawbands.kwcoco.zip


    CommandLine:
        xdoctest -m geowatch.cli.coco_add_ignore_buffer
        xdoctest -m geowatch.cli.coco_add_ignore_buffer.py
        xdoctest /home/local/KHQ/vincenzo.dimatteo/Desktop/geowatch/geowatch/cli/coco_add_ignore_buffer.py


    Example:
        >>> from geowatch.cli.coco_add_ignore_buffer import *
        >>> import geowatch
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('geowatch/tests/ignore_buffer')
        >>> dpath.ensuredir()
        >>> ignore_buffer_size = '10@10GSD'
        >>> dst = dpath / 'out.kwcoco.zip'
        >>> src = geowatch.coerce_kwcoco('geowatch-msi', geodata=True, dates=True)
        >>> main(cmdline=0,src=src.data_fpath, dst=dst)

    """

    import kwimage
    from geowatch.utils import util_resolution
    from shapely.ops import unary_union
    import ubelt as ub
    import numpy as np
    import kwcoco
    import rich
    from rich.markup import escape

    config = CocoAddIgnoreBufferConfig.cli(
        cmdline=cmdline, data=kwargs, special_options=False
    )
    rich.print("config = " + escape(ub.urepr(config)))

    dset = kwcoco.CocoDataset(config.src)
    utm_gsd = util_resolution.ResolvedUnit.coerce("1GSD")
    ignore_buffer = util_resolution.ResolvedScalar.coerce(config.ignore_buffer_size)

    ignore_buffer_gsd = ignore_buffer.at_resolution(utm_gsd).scalar
    videos = dset.videos()
    # Ignore eff
    for video_id in ub.ProgIter(videos, desc="looping over videos..."):
        images = dset.images(video_id=video_id)

        for image_id in images:
            coco_img = dset.coco_image(image_id)
            coco_img.resolution(space="image")
            _imgspace_resolution = coco_img.resolution(space="image")
            image_pxl_per_meter = 1 / np.array(_imgspace_resolution["mag"])
            ignore_buffer_pixel = ignore_buffer_gsd * image_pxl_per_meter
            # TODO:  buffer utilzing shapely metod currently only accounts for a singular float distance
            #       need to account for several distances in the future, but for now the average of the
            #       buffer region suggested is used.
            ignore_buffer_pixel = ignore_buffer_pixel.mean()
            annots = coco_img.annots()
            #annot_cat_ids = annots.lookup("category_id")
            annot_segmenations = annots.lookup("segmentation")
            # print(annot_segmenations)
            #annot_cat_names = dset.categories(annot_cat_ids).lookup("name")

            annot_polys = [
                kwimage.MultiPolygon.coerce(s).to_shapely() for s in annot_segmenations
            ]
            annot_polys_buffer = [
                poly.buffer(ignore_buffer_pixel).difference(poly)
                for poly in annot_polys
            ]
            do_not_ignore_poly = unary_union(annot_polys)
            annot_polys_buffer = [
                poly.difference(do_not_ignore_poly) for poly in annot_polys_buffer
            ]
            annot_polys_buffer = [
                poly for poly in annot_polys_buffer if not poly.is_empty
            ]

            if 0:
                # kwimage.MultiPolygon.coerce(do_not_ignore_poly).draw(setlim=1,color='kitware_red')
                kwimage.MultiPolygon.coerce(annot_polys[0]).draw(color="kitware_green")
                for poly in annot_polys_buffer:
                    kwimage.MultiPolygon.coerce(poly).draw(color="kitware_blue")
                kwimage.MultiPolygon.coerce(do_not_ignore_poly).draw(
                    setlim=1, color="kitware_red"
                )

            for poly in annot_polys_buffer:
                x = kwimage.MultiPolygon.from_shapely(poly)
                dset.add_annotation(
                    image_id=image_id,
                    category_id=dset.ensure_category("ignore"),
                    bbox=x.bounding_box().to_xywh(),
                    segmentation=x,
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
