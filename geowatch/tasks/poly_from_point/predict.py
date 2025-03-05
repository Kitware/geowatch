"""
Primary entrypoint to convert polygons to points.


CommandLine:
    python -m geowatch.tasks.poly_from_point.predict --help

CommandLine:
    xdoctest -m geowatch.tasks.poly_from_point.predict __doc__:0


Example:
    >>> from geowatch.tasks.poly_from_point.predict import *  # NOQA
    >>> import geowatch
    >>> from geowatch.geoannots import geomodels
    >>> import ubelt as ub
    >>> dpath = ub.Path.appdir('geowatch/poly_from_point/doc').ensuredir()
    >>> region_model, site_models = geomodels.RegionModel.random(with_sites=True)
    >>> region_models = [region_model]
    >>> point_model = site_models.to_point_model()
    >>> print(f'Number of region models: {len(region_models)}')
    >>> print(f'Number of site models: {len(site_models)}')
    >>> # It is also easy to convert these models to geopandas
    >>> region_model = region_models[0]
    >>> gdf = region_model.pandas()
    >>> print(gdf)
    >>> filepath_to_points = dpath / 'points.geojson'
    >>> filepath_to_points.write_text(point_model.dumps())
    >>> filepath_to_region = dpath / 'region.geojson'
    >>> filepath_to_region.write_text(region_model.dumps())
    >>> filepath_output = dpath / 'output_region.geojson'
    >>> gpd.read_file(filepath_to_points) # check
    >>> kwargs = dict(
    >>>     filepath_to_points=filepath_to_points,
    >>>     filepath_to_region=filepath_to_region,
    >>>     filepath_output=filepath_output,
    >>> )
    >>> cmdline = 0
    >>> PolyFromPointCLI.main(cmdline=cmdline, **kwargs)

    # To Viz
    import xdev
    viz_fpath = dpath / 'viz.png'
    ub.cmd(f'geowatch draw_region {filepath_output} --fpath {viz_fpath}', verbose=3)
    xdev.startfile(viz_fpath)

"""
import scriptconfig as scfg
import numpy as np
import geopandas as gpd
import kwcoco
import ubelt as ub
from geowatch.geoannots.geomodels import RegionModel, SiteSummary
import kwutil


class PolyFromPointCLI(scfg.DataConfig):
    r"""
    Convert points to polygons based on trimaping or SAM (trimap seems to work
    better, SAM could be improved).

    Example
    -------

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=hdd)
    DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
    echo "$DVC_DATA_DPATH"
    echo "$DVC_EXPT_DPATH"

    DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
    python -m geowatch.tasks.poly_from_point.predict \
        --method 'ellipse' \
        --filepath_output KR_R001-genpoints.geojson \
        --region_id KR_R001 \
        --size_prior "20x20@10mGSD" \
        --ignore_buffer "10@10mGSD" \
        --filepath_to_images None \
        --filepath_to_points "$DVC_DATA_DPATH/submodules/annotations/supplemental_data/point_based_annotations.geojson" \
        --filepath_to_region "$DVC_DATA_DPATH/annotations/drop8/region_models/KR_R001.geojson" \

    geowatch draw_region KR_R001-genpoints.geojson --fpath KR_R001-genpoints.png

    """
    filepath_to_points = scfg.Value(
        None,
        # "/mnt/ssd2/data/dvc-repos/smart_phase3_data/submodules/annotations/point_based_annotations.zip",
        help="Filepath to point-based annotations in geojson format.",
    )
    filepath_to_region = scfg.Value(
        None,
        # "/mnt/ssd2/data/dvc-repos/smart_phase3_data/annotations/drop8/region_models/KR_R002.geojson",
        help="Filepath to geojson regions file.",
    )
    filepath_output = scfg.Value(
        "output_region.geojson",
        help="Output region model with the polygons inferred from the points",
    )

    # --- TODO: improve api clarity

    region_id = scfg.Value(
        None,
        help="if the kwcoco file is unspecified, the region_id to extract points from must be given.",
    )
    filepath_to_images = scfg.Value(
        None,
        # "/mnt/ssd2/data/dvc-repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip",
        help="Filepath to the kwcoco corresponding to a region",
    )

    filepath_to_sam = scfg.Value(
        None,
        # "/mnt/ssd3/segment-anything/demo/model/sam_vit_h_4b8939.pth",
        help="If the methos id SAM, specify the filepath to the SAM weights",
    )

    size_prior = scfg.Value(
        "20.06063 x 20.0141229 @ 10mGSD",
        help=ub.paragraph(
            """
            The expected size of the objects in world coorindates.
            Must be specified as
            ``<w> x <h> @ <magnitude> <resolution>``. E.g.  ``20x25@10mGSD``
            will assume objects 200 by 250 meters.
            """
        ),
        alias=["box_size"],
    )

    ignore_buffer = scfg.Value(
        None,
        help=ub.paragraph(
            """
            If specified, give a resolved unit (e.g. 10@10mGSD) for
            a buffer size around each other polygon.
            """
        ),
    )

    method = scfg.Value(
        "ellipse",
        choices=["sam", "box", "ellipse"],
        help="Method for extracting polygons from points",
    )

    threshold = scfg.Value(
        0.45,
        help="If Sam is used specify a threshold for the frequencey polygons appear across images",
    )

    time_prior = scfg.Value(
        "1 year", help="time prior before and after", alias=["time_pad"]
    )

    @classmethod
    def main(cls, cmdline=0, **kwargs):
        r"""
        Ignore:
            DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=hdd)
            DVC_EXPT_DPATH=$(geowatch_dvc --tags='phase3_expt' --hardware=auto)
            echo "$DVC_DATA_DPATH"
            echo "$DVC_EXPT_DPATH"

            python -m geowatch.tasks.poly_from_point.predict \
                --method 'sam' \
                --filepath_to_images "$DVC_DATA_DPATH/Aligned-Drop8-ARA/KR_R001/imganns-KR_R001-rawbands.kwcoco.zip" \
                --filepath_to_points "$DVC_DATA_DPATH/annotations/point_based_annotations.zip" \
                --filepath_to_region "$DVC_DATA_DPATH/annotations/drop8/region_models/KR_R001.geojson" \
                --filepath_to_sam "$DVC_EXPT_DPATH/models/sam/sam_vit_h_4b8939.pth"

            DVC_DATA_DPATH=$(geowatch_dvc --tags='phase3_data' --hardware=ssd)
            python -m geowatch.tasks.poly_from_point.predict \
                --method 'ellipse' \
                --filepath_output KR_R001-genpoints.geojson \
                --region_id KR_R001 \
                --size_prior "20x20@10mGSD" \
                --ignore_buffer "10@10mGSD" \
                --filepath_to_images None \
                --filepath_to_points "$DVC_DATA_DPATH/submodules/annotations/supplemental_data/point_based_annotations.geojson" \
                --filepath_to_region "$DVC_DATA_DPATH/annotations/drop8/region_models/KR_R001.geojson" \

            geowatch draw_region KR_R001-genpoints.geojson --fpath KR_R001-genpoints.png
        """
        config = PolyFromPointCLI.cli(cmdline=cmdline, data=kwargs)
        import rich
        from rich.markup import escape
        from shapely import geometry

        rich.print(f"config = {escape(ub.urepr(config, nl=1))}")

        from geowatch.utils import util_resolution
        import kwimage

        filepath_to_points = ub.Path(config.filepath_to_points)
        filepath_output = ub.Path(config.filepath_output)
        config.time_prior = kwutil.util_time.timedelta.coerce(config.time_prior)
        config.size_prior = util_resolution.ResolvedWindow.coerce(config.size_prior)

        if config.ignore_buffer is not None:
            config.ignore_buffer = util_resolution.ResolvedScalar.coerce(
                config.ignore_buffer
            )

        if not filepath_to_points.exists():
            raise FileNotFoundError(f"filepath_to_points={filepath_to_points}")

        main_region = RegionModel.coerce(config.filepath_to_region)
        main_region_header = main_region.header

        main_region_header["properties"]["originator"] = "Rutgers"
        main_region_header["properties"]["comments"] = f"poly_from_points: {config.method}"

        if config.region_id is None:
            config.region_id = main_region.region_id

        if config.filepath_to_images is not None:
            filepath_to_images = ub.Path(config.filepath_to_images)
            if not filepath_to_images.exists():
                raise FileNotFoundError(f"filepath_to_points={filepath_to_points}")

            # Load the kwcoco file and use its metadata extract information about
            # video space.  We convert the size priors to videospace in this case.
            # This is only necessary if we need to reference the images based on
            # the "method".
            dset = kwcoco.CocoDataset(filepath_to_images)
            assert dset.n_videos == 1, "only handling 1 video for now"
            video_obj = list(dset.videos().objs)[0]
            if config.region_id is None:
                config.region_id = video_obj["name"]

            assert config.region_id == video_obj["name"], "inconsistent name"

            utm_crs, warp_vid_from_utm = get_vidspace_info(video_obj)
        else:
            dset = None
            utm_crs = None
            if config.region_id is None:
                raise ValueError("region_id is required when a kwcoco path is not given")
            if config.method == "sam":
                raise ValueError("SAM requires a kwcoco file for video space")

        # Convert the size prior to meters
        utm_gsd = util_resolution.ResolvedUnit.coerce("1mGSD")
        size_prior_utm = config.size_prior.at_resolution(utm_gsd)

        points_gdf_crs84 = load_point_annots(filepath_to_points, config.region_id)

        main_region_header_points = [
            geometry.Point(point)
            for point in main_region_header["geometry"]["coordinates"][0]
        ]
        poly = geometry.Polygon(main_region_header_points)
        indexs = points_gdf_crs84.index
        for idx, point in enumerate(points_gdf_crs84["geometry"]):
            if poly.contains(point):
                print("hit")
            else:
                points_gdf_crs84.drop(indexs[idx], axis=0, inplace=True)
        print('points_gdf_crs84:')
        print("\n", points_gdf_crs84)

        # Transform the points into a UTM CRS. If we didn't determine a which UTM
        # crs to work with from the kwcoco file, then we need to infer a good one.
        if utm_crs is None:
            from kwgis.utils import util_gis

            points_gdf_utm = util_gis.project_gdf_to_local_utm(
                points_gdf_crs84, max_utm_zones=10, mode=1
            )
            utm_crs = points_gdf_utm.crs
        else:
            points_gdf_utm = points_gdf_crs84.to_crs(utm_crs)

        points_utm = points_gdf_utm["geometry"]

        if config.method == "box":
            prior_width, prior_height = size_prior_utm.window
            utm_polygons = kwimage.Boxes(
                [[p.x, p.y, prior_width, prior_height] for p in points_utm],
                "cxywh",
            ).to_polygons()
        elif config.method == "ellipse":
            prior_width, prior_height = size_prior_utm.window
            utm_polygons = [
                kwimage.Polygon.circle(xy=(p.x, p.y), r=(prior_width, prior_height))
                for p in points_utm
            ]
        elif config.method == "sam":
            utm_polygons = convert_points_to_poly_with_sam_method(
                dset, video_obj, points_gdf_utm, config
            )
        else:
            raise KeyError(f"Unknown Method: {config.method}")

        new_region_model = convert_polygons_to_region_model(
            utm_polygons,
            main_region_header,
            points_gdf_utm,
            points_gdf_crs84,
            config,
        )
        print(f"Writing output to {filepath_output}")
        filepath_output.write_text(new_region_model.dumps())


"""
NOTES:

Transforms / spaces to be aware of

+---------------------+
+ CRS84 lat/lon space |  * for generaly global non-polar coordinates
+---------------------+
          |
          | (nonlinear warp)
          |
+-----------------------+
+ UTM local meter space |  * specific to one of 120 local zones
+-----------------------+
          |
          | (usually linear warp)
          |
+--------------------+
+ KWCoco Video Space |  * pixel space for aligned images in a sequence
+--------------------+
          |
          | (usually linear warp)
          |
+--------------------+
+ KWCoco Image Space |  * pixel space for an image on disk
+--------------------+
          |
          | (usually linear warp)
          |
+--------------------+
+ KWCoco Asset Space | * This example doesn't cover asset space, which is rarely used.
+--------------------+ * If images are composed of multiple assets, each asset has its own space.


SeeAlso:
    :mod:`geowatch.cli.reproject_annotations` - for logic we use to reproject
    polygon annotations on to pixel datasets.
"""
# REGION: SiteSummary and site
# Region, SiteSummary ->
# SiteModel: Site Type, Observation Type -> S
# SITE: Just contain overiew geometry. Geometry can change at timestamp
# import kwplot

"""
    >>> # xdoctest: +REQUIRES(env:HAS_DVC)
    >>> import geowatch
    #>>> dvc_data_dpath = geowatch.find_dvc_dpath(tags='phase2_data', hardware='auto')
    >>> region_models_dpath = '/mnt/ssd2/data/dvc-repos/smart_phase3_data/annotations/drop8/region_models'
    >>> site_models_dpath = '/mnt/ssd2/data/dvc-repos/smart_phase3_data/annotations/drop8/site_models'
    >>> from geowatch.geoannots import geomodels
    >>> region_models = list(geomodels.RegionModel.coerce_multiple(region_models_dpath))
    >>> site_models = list(geomodels.SiteModel.coerce_multiple(site_models_dpath, workers=8))
    >>> print(f'Number of region models: {len(region_models)}')
    >>> print(f'Number of site models: {len(site_models)}')
    >>> # Quick demo of associating sites to regions
    >>> region_id_to_sites = ub.group_items(site_models, key=lambda s: s.header['properties']['region_id'])
    >>> region_id_to_num_sites = ub.udict(region_id_to_sites).map_values(len)
    >>> print('region_id_to_num_sites = {}'.format(ub.urepr(region_id_to_num_sites, nl=1)))
    >>> # It is also easy to convert these models to geopandas
    >>> region_model = region_models[0]
    >>> gdf = region_model.pandas()
    >>> print(gdf)
    """


# TODO: Update predictor for smaller region around each point, track mask point itself generated
# TODO : kwutil.util_time.timedelta.coerce


def convert_polygons_to_region_model(
    utm_polygons, main_region_header, points_gdf_utm, points_gdf_crs84, config
):
    """
    Given polygons in a CRS, convert them to CRS84 polygon-based RegionModels.
    """
    from geowatch.utils import util_resolution
    from kwgis.utils import util_gis

    utm_gsd = util_resolution.ResolvedUnit.coerce("1mGSD")

    if config.ignore_buffer is not None:
        ignore_buffer_size = config.ignore_buffer.at_resolution(utm_gsd).scalar
    else:
        ignore_buffer_size = None

    assert len(points_gdf_utm) == len(utm_polygons)
    time_prior = config.time_prior

    config.ignore_buffer

    new_properties = []
    new_utm_geometries = []
    for idx, polygon in enumerate(utm_polygons):
        point_row_utm = points_gdf_utm.iloc[idx]
        point_row_crs84 = points_gdf_crs84.iloc[idx]
        assert point_row_crs84["site_id"] == point_row_utm["site_id"]
        mid_date = kwutil.util_time.datetime.coerce(point_row_utm["date"])
        start_date = mid_date - time_prior
        end_date = mid_date + time_prior

        status = point_row_crs84.status
        if status == "positive":
            # Translate to a valid T&E positive name
            status = "positive_pending"

        properties = {
            "type": "site_summary",
            "status": status,
            "version": "2.0.1",
            "site_id": point_row_utm["site_id"],
            "mgrs": main_region_header["properties"]["mgrs"],
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "score": 1,
            "originator": "poly_from_point"
            + config.method,  # TODO: add some config info here
            "model_content": "annotation",
            "validated": "False",
            "cache": {
                "orig_point_utm": str(point_row_utm.geometry),
                "orig_point_crs84": str(point_row_crs84.geometry),
            },
        }
        try:
            if polygon is None:
                raise Exception
            polygon_video_space = polygon.convex_hull
            polygon_video_space.to_shapely()

        except Exception:
            continue
            raise NotImplementedError("fixme define default polygon")
            default_polygon = NotImplemented
            polygon_video_space = default_polygon
            new_properties.append(properties)
            new_utm_geometries.append(polygon_video_space)

        else:
            new_properties.append(properties)
            new_utm_geometries.append(polygon_video_space)

    new_utm_gdf = gpd.GeoDataFrame(
        {"geometry": new_utm_geometries}, crs=points_gdf_utm.crs
    )

    region_id = main_region_header["properties"]["region_id"]

    if ignore_buffer_size is not None:
        # Hack to find a good new site id starting point for the dummy ignore polys
        # we will add.
        max_site_id = max(
            points_gdf_crs84["site_id"].apply(lambda x: int(x.split("_")[-1]))
        )
        next_site_num = max_site_id + 1

        # Add ignore buffer regions
        buffered_shapes = new_utm_gdf["geometry"].buffer(ignore_buffer_size)

        # new_utm_gdf[new_utm_gdf['status'] != 'ignore'].unary_union
        # new_union_geom = new_utm_gdf.unary_union
        # new_ignore_geoms = buffered_shapes.difference(new_union_geom)
        new_ignore_geoms = buffered_shapes.difference(new_utm_gdf)

        new_ignore_polys = []
        new_ignore_props = []
        for geom, props in zip(new_ignore_geoms, new_properties):
            start_date = props["start_date"]
            end_date = props["end_date"]
            # Breakup multipolygons into multiple new "sites"
            polys = []
            if geom.geom_type == "MultiPolygon":
                polys = [p for p in geom.geoms if p.is_valid]
                for poly in geom.geoms:
                    if poly.is_valid:
                        new_ignore_polys.append(poly)
            elif geom.is_valid:
                polys = [geom]

            for poly in polys:
                new_site_id = f"{region_id}_{next_site_num:04d}"
                ignore_prop = {
                    "type": "site_summary",
                    "status": "ignore",
                    "version": "2.0.1",
                    "site_id": new_site_id,
                    "mgrs": main_region_header["properties"]["mgrs"],
                    "start_date": start_date,
                    "end_date": end_date,
                    "score": 1,
                    "originator": "poly_from_point_"
                    + config.method,  # TODO: right now just added method not sure what else from config might be useful
                    "model_content": "annotation",
                    "validated": "False",
                    "cache": {},
                }
                new_ignore_polys.append(poly)
                new_ignore_props.append(ignore_prop)
                next_site_num += 1

        new_utm_geometries = new_utm_geometries + new_ignore_polys
        new_properties = new_properties + new_ignore_props
        new_utm_gdf = gpd.GeoDataFrame(
            {"geometry": new_utm_geometries}, crs=points_gdf_utm.crs
        )

    # Convert the new UTM geometrices back into CRS84
    crs84 = util_gis.get_crs84()
    new_crs84_gdf = new_utm_gdf.to_crs(crs84)

    site_sums = []
    for props, geom in zip(new_properties, new_crs84_gdf.geometry):
        res = SiteSummary(**{"properties": props, "geometry": geom})
        site_sums.append(res)
    new_region_model = RegionModel([main_region_header] + site_sums)
    new_region_model.fixup()
    new_region_model.validate(strict=False)
    return new_region_model


def extract_polygons(im):
    import kwimage

    data = im > 0.5
    mask = kwimage.Mask(data, "c_mask")
    polygon = mask.to_multi_polygon()
    return polygon


def image_predicted(im, geo_polygons_image, filename):
    import kwimage

    im = np.zeros((len(im), len(im[0])))
    for mask in geo_polygons_image:
        np.logical_or(im, mask, out=im)
    kwimage.imwrite(filename, im * 255)


def show_mask(mask, ax, random_color=False):
    color = np.array([255 / 255, 255 / 255, 255 / 255, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)


def load_sam(filepath_to_sam):
    # from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    import geowatch_tpl
    import torch

    segment_anything = geowatch_tpl.import_submodule("segment_anything")
    # SamAutomaticMaskGenerator = segment_anything.SamAutomaticMaskGenerator
    SamPredictor = segment_anything.SamPredictor
    sam_model_registry = segment_anything.sam_model_registry

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_TYPE = "vit_h"
    # TODO wrap root ubelt
    root = ub.Path(filepath_to_sam)

    sam = sam_model_registry[MODEL_TYPE](checkpoint=root)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    return predictor


def comput_average_boxes(dset):
    import kwimage
    from sklearn.cluster import KMeans

    total = []
    for gid in range(len(dset.images().coco_images)):
        coco_img = dset.images().coco_images[gid]
        # imgspace_dets = coco_img.annots().detections
        # vidspace_dets = imgspace_dets.warp(coco_img.warp_vid_from_img)
        # total_iou = 0
        # count = 0
        det_obj = coco_img._detections_for_resolution(space="video")
        det_class_names = [det_obj.classes[idx] for idx in det_obj.class_idxs]
        class_of_intrest = {
            "positive",
            "Site Preparation",
            "Active Construction",
            "Post Construction",
            "No Activity",
            "transient",
        }
        flags = [c in class_of_intrest for c in det_class_names]
        det_keep = det_obj.compress(flags)
        total.append(det_keep.boxes)
    all_boxes = kwimage.Boxes.concatenate(total)
    all_w_h = all_boxes.to_xywh().data[:, 2:4]
    kmeans = KMeans(n_clusters=1, random_state=0, n_init="auto").fit(all_w_h)
    Bounding_Boxes = kmeans.cluster_centers_
    print(Bounding_Boxes)


def get_vidspace_info(video_obj):
    """
    Extract information about the videospace of a kwcoco video object
    """
    from kwgis.utils import util_gis
    import kwimage

    # Find the world-space CRS for this region
    video_crs = util_gis.coerce_crs(video_obj["wld_crs_info"])
    # Build the affine transform from world space to video space.
    # This is typically a UTM world space, and always a pixel video space.
    warp_vid_from_wld = kwimage.Affine.coerce(video_obj["warp_wld_to_vid"])
    utm_crs = video_crs
    warp_vid_from_utm = warp_vid_from_wld
    return utm_crs, warp_vid_from_utm


def load_point_annots(filepath_to_points, region_id):
    """
    filepath_to_points = '/home/joncrall/.cache/geowatch/poly_from_point/doc/points.geojson'
    filepath_to_points = '/media/joncrall/flash1/smart_phase3_data/annotations/point_based_annotations.geojson'
    """
    fpath = filepath_to_points
    # Read json text directly from the zipfile
    if fpath.endswith(".zip"):
        # HACK
        file = ub.zopen(fpath + "/" + "point_based_annotations.geojson", "r")
    else:
        file = open(fpath, "r")
    with file:
        points_gdf_crs84 = gpd.read_file(file)
    # Simplified logic to grab only the rows corresponding to this video based on
    # assumptions about site-id patterns. Not robust.

    if 1:
        # hack to handle xxx region ids
        import kwutil

        site_region_ids = points_gdf_crs84["site_id"].apply(
            lambda s: "_".join(s.split("_")[0:2])
        )
        original_regionid_to_flag = {}
        for orig_site_region_id in set(site_region_ids):
            site_region_id = orig_site_region_id
            if site_region_id.endswith("xxx"):
                site_region_id = site_region_id.replace("xxx", "*")
            site_region_id_pat = kwutil.util_pattern.Pattern.coerce(site_region_id)
            flag = site_region_id_pat.match(region_id)
            original_regionid_to_flag[orig_site_region_id] = flag

    flags = [original_regionid_to_flag[rid] for rid in site_region_ids]
    points_gdf_crs84 = points_gdf_crs84[flags]
    return points_gdf_crs84


def convert_points_to_poly_with_sam_method(dset, video_obj, points_gdf_utm, config):
    import kwimage
    from geowatch.utils import util_resolution

    utm_crs, warp_vid_from_utm = get_vidspace_info(video_obj)

    video_image_ids = dset.images(video_id=video_obj["id"])

    # Now we can warp all the points into video space.
    points_vidspace = points_gdf_utm["geometry"].affine_transform(
        warp_vid_from_utm.to_shapely()
    )

    warp_utm_from_vid = warp_vid_from_utm.inv()

    video_space_gsd = util_resolution.ResolvedUnit.coerce(
        video_obj["target_gsd"], default_unit="mGSD"
    )
    size_prior_vidspace = config.size_prior.at_resolution(video_space_gsd)
    prior_width, prior_height = size_prior_vidspace.window

    num_points = len(points_vidspace)
    num_frames = len(video_image_ids)

    # Allocate a buffer for a "soft mask" for each point.
    # TODO: Currently each point gets a buffer the size of the entire video
    # which is way too big. If we use smaller context window we can save a
    # lot of memory, but that will require bookkeeping of translations from
    # the "cropped" regions back into "videospace".
    all_predicted_regions = np.zeros(
        (num_points, video_obj["height"], video_obj["width"]),
        dtype=np.float32,
    )
    predictor = load_sam(config.filepath_to_sam)

    for image_id in ub.ProgIter(video_image_ids, desc="Looping Over Videos..."):
        # geo_polygons_image = []
        # im = np.zeros((video_obj["height"], video_obj["width"]), dtype=np.uint8)
        coco_image = dset.coco_image(image_id)

        # Get the transform from video space to image space
        # Note that each point is associated with a date, so not all the
        # points warped here are actually associated with this image.
        # warp_img_from_vid = coco_image.warp_img_from_vid
        # region_points_gdf_imgspace = points_vidspace.affine_transform(
        #   warp_img_from_vid.to_shapely()
        # )
        delayed_img = coco_image.imdelay("red|green|blue", space="video")
        imdata = delayed_img.finalize()

        # Depending on the sensor intensity might be out of standard ranges,
        # we can use kwimage to robustly normalize for this. This lets
        # us visualize data with false color.
        canvas = kwimage.normalize_intensity(imdata, axis=2)
        img = np.ascontiguousarray(canvas)

        # On Jon's machine SAM wants uint8 for some reason
        img = kwimage.ensure_uint255(img)
        predictor.set_image(img, "RGB")
        regions = kwimage.Boxes(
            [[p.x, p.y, prior_width, prior_height] for p in points_vidspace],
            "cxywh",
        )
        regions = regions.to_ltrb()

        for count_individual_mask, (point, box) in enumerate(
            zip(points_vidspace, regions)
        ):
            # if count_individual_mask >10:
            #   break
            masks, scores, logits = predictor.predict(
                point_coords=np.array([[point.x, point.y]]),
                point_labels=np.array([1]),
                # box= gt_BB,
                box=box.data,
                multimask_output=True,
            )

            binarized_mask = masks[0] > 0.5
            all_predicted_regions[count_individual_mask] += binarized_mask

    all_predicted_regions /= num_frames

    threshold = config.threshold
    for idx, mask in enumerate(all_predicted_regions):
        try:
            res = mask > threshold
            mask = kwimage.Mask(res, "c_mask")
            vidspace_polygon = mask.to_multi_polygon()
            utm_polygon = vidspace_polygon.warp(warp_utm_from_vid)
            utm_polygon = utm_polygon.convex_hull
            utm_polygon.to_shapely()
            yield utm_polygon
        except Exception:
            # TODO: add default polygon
            yield None

__cli__ = PolyFromPointCLI

if __name__ == "__main__":
    __cli__.main()
