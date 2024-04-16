"""

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

import geowatch_tpl  # NOQA
import scriptconfig as scfg
import numpy as np
import torch

# from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import geopandas as gpd
import kwcoco
import kwimage
import ubelt as ub
from geowatch.utils import util_gis
from sklearn.cluster import KMeans
from geowatch.geoannots.geomodels import RegionModel, SiteSummary
import kwutil

# import matplotlib.pyplot as plt

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


class HeatMapConfig(scfg.DataConfig):
    filepath_to_images = scfg.Value(
        "/mnt/ssd2/data/dvc-repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip",
        help="Filepath containning images",
    )
    filepath_to_points = scfg.Value(
        "/mnt/ssd2/data/dvc-repos/smart_phase3_data/submodules/annotations/point_based_annotations.zip",
        help="Filepath to Geopoints",
    )
    filepath_to_region = scfg.Value(
        "/mnt/ssd2/data/dvc-repos/smart_phase3_data/annotations/drop8/region_models/KR_R002.geojson",
        help="Filepath to Regions",
    )
    filepath_to_sam = scfg.Value(
        "/mnt/ssd3//segment-anything/demo/model/sam_vit_h_4b8939.pth",
        help="Filepath to SAM model",
    )
    file_output = scfg.Value("KR_R002-SAM.geojson", help="Output dest")
    box_size = scfg.Value(
        [20.06063, 20.0141229],
        help="Specify Bounding Box for SAM to use during prediction",
    )

    method = scfg.Value(
        "sam", choices=["sam", "box"], help="Method for extracting polygons from points"
    )

    time_pad = scfg.Value('1 year', help='time prior before and after')


def extract_sam_polygons(
    image_id,
    all_predicted_regions,
    main_region_header,
    region_points_gdf_imgspace,
    warp_vid_from_wld,
    region_gdf_utm,
):
    # default = np.zeros(polygons[0].shape)
    # default = default > 1
    # default[0:9, 0:9] = True
    # mask = kwimage.Mask(default, "c_mask")
    # default_polygon = mask.to_multi_polygon()
    # default_polygon = default_polygon.convex_hull

    for idx, mask in enumerate(all_predicted_regions):
        try:
            res = mask > (45 * (image_id + 1)) / 100
            # point_row = region_gdf_utm.iloc[idx]
            mask = kwimage.Mask(res, "c_mask")
            polygon = mask.to_multi_polygon()
            # polygon_video_space = polygon.convex_hull
            polygon.to_shapely()
            yield polygon
        except Exception:
            # TODO: add default polygon
            ...


def convert_polygons_to_region_model(
    polygons,
    main_region_header,
    warp_vid_from_wld,
    region_gdf_utm,
    region_gdf_crs84,
    time_pad,
):
    print(f"{len(polygons)=}")
    """
    default = np.zeros(polygons[0].shape)
    default = default > 1
    default[0:9, 0:9] = True
    mask = kwimage.Mask(default, "c_mask")
    default_polygon = mask.to_multi_polygon()
    default_polygon = default_polygon.convex_hull
    """

    assert len(region_gdf_utm) == len(polygons)

    vid_space_summaries = []
    vid_space_geometries = []
    for idx, polygon in enumerate(polygons):
        point_row_utm = region_gdf_utm.iloc[idx]
        point_row_crs84 = region_gdf_crs84.iloc[idx]
        assert point_row_crs84['site_id'] == point_row_utm['site_id']
        mid_date = kwutil.util_time.datetime.coerce(point_row_utm["date"])
        start_date = mid_date - time_pad
        end_date = mid_date + time_pad
        polygon_video_space = polygon.convex_hull
        properties = {
            "type": "site_summary",
            "status": "positive_annotated",
            "version": "2.0.1",
            "site_id": point_row_utm["site_id"],
            "mgrs": main_region_header["properties"]["mgrs"],
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat(),
            "score": 1,
            "originator": "Rutgers_SAM",
            "model_content": "annotation",
            "validated": "False",
            "cache": {
                "orig_point_utm": str(point_row_utm.geometry),
                "orig_point_crs84": str(point_row_crs84.geometry),
            },
        }
        try:
            polygon_video_space.to_shapely()
        except Exception:
            vid_space_summaries.append(properties)
            raise NotImplementedError('fixme define default polygon')
            default_polygon = NotImplemented
            polygon_video_space = default_polygon

        vid_space_summaries.append(properties)
        vid_space_geometries.append(polygon_video_space)

    # Warp the videospace polygons back into UTM world space.
    warp_world_from_vid = warp_vid_from_wld.inv()

    utm_space_geometries = [
        p.warp(warp_world_from_vid).to_shapely() for p in vid_space_geometries
    ]
    wld_gdf = gpd.GeoDataFrame(
        dict(geometry=utm_space_geometries), crs=region_gdf_utm.crs
    )
    # Finally convert back to CRS84 to create site summaries
    crs84_gdf = wld_gdf.to_crs("crs84")
    site_sums = []
    for props, geom in zip(vid_space_summaries, crs84_gdf.geometry):
        res = SiteSummary(**{"properties": props, "geometry": geom})
        site_sums.append(res)
    new_region_model = RegionModel([main_region_header] + site_sums)
    new_region_model.fixup()
    new_region_model.validate(strict=False)
    return new_region_model


def extract_polygons(im):
    data = im > 0.5
    mask = kwimage.Mask(data, "c_mask")
    polygon = mask.to_multi_polygon()

    return polygon


def image_predicted(
    im,
    geo_polygons_image,
    filename,
):
    im = np.zeros((len(im), len(im[0])))
    for mask in geo_polygons_image:
        np.logical_or(im, mask, out=im)
    kwimage.imwrite(filename, im * 255)


def show_mask(mask, ax, random_color=False):

    color = np.array([255 / 255, 255 / 255, 255 / 255, 1.0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)


def read_points_anns(filepath_to_points):

    fpath = filepath_to_points
    # Read json text directly from the zipfile
    file = ub.zopen(fpath + "/" + "point_based_annotations.geojson", "r")
    with file:
        gdf_crs84 = gpd.read_file(file)
    return gdf_crs84


def load_sam(filepath_to_sam):
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


# TODO: GENERALIZE FOR ALL REGIONS
# Pick one region coco file
def get_points(video_obj, filepath_to_points):

    # Find the world-space CRS for this region
    video_crs = util_gis.coerce_crs(video_obj["wld_crs_info"])

    # Build the affine transform from world space to video space.
    # This is typically a UTM world space, and always a pixel video space.
    warp_vid_from_wld = kwimage.Affine.coerce(video_obj["warp_wld_to_vid"])
    gdf_crs84 = read_points_anns(filepath_to_points)
    # Simplified logic to grab only the rows corresponding to this video based on
    # assumptions about site-id patterns. Not robust.
    flags = gdf_crs84.site_id.str.startswith(video_obj["name"])
    region_gdf_crs84 = gdf_crs84[flags]

    # Warp the points (in CRS84 - i.e. lat/lon to UTM space for this region)
    region_gdf_utm = region_gdf_crs84.to_crs(video_crs)
    region_points_utm = region_gdf_utm["geometry"]

    # Now we can warp all the points into video space.
    region_points_gdf_vidspace = region_points_utm.affine_transform(
        warp_vid_from_wld.to_shapely()
    )

    return region_points_gdf_vidspace, warp_vid_from_wld, region_gdf_utm, region_gdf_crs84


# TODO: add hard coded to config
def main():
    r"""
    IGNORE:
        black /mnt/ssd2/data/test/geowatch/geowatch/tasks/poly_from_point/predict.py
        pyenv shell 3.10.5
        source $(pyenv prefix)/envs/pyenv-geowatch/bin/activate
        python -m geowatch.tasks.poly_from_point.predict --method 'box'
        from geowatch.tasks.poly_from_point.predict import *


    Ignore:
        python -m geowatch.tasks.poly_from_point.predict --method 'box' \
            --filepath_to_images "$HOME/data/dvc-repos/smart_phase3_data/Aligned-Drop8-ARA/KR_R002/imganns-KR_R002-rawbands.kwcoco.zip" \
            --filepath_to_points "$HOME/data/dvc-repos/smart_phase3_data/annotations/point_based_annotations.zip" \
            --filepath_to_region "$HOME/data/dvc-repos/smart_phase3_data/annotations/drop8/region_models/KR_R002.geojson"

    """
    config = HeatMapConfig.cli(cmdline=1)

    box_width = config.box_size[0]
    box_height = config.box_size[1]
    filepath_to_images = config.filepath_to_images
    filepath_to_sam = config.filepath_to_sam
    filepath_to_points = config.filepath_to_points
    filepath_to_region = config.filepath_to_region
    method = config.method
    output = config.file_output
    main_region = RegionModel.coerce(filepath_to_region)
    main_region_header = main_region.header
    time_pad = kwutil.util_time.timedelta.coerce(config.time_pad)

    main_region_header["properties"]["originator"] = "Rutgers"
    main_region_header["properties"]["comments"] = "SAM Points"

    # Now for each image we can project the points into its on-disk coordinates
    # (Note: if we have already sampled image data in video space, you can use the
    #  above points directly, but for completeness this example demonstrates
    #  how to warp them all the way down to the image level.)
    dset = kwcoco.CocoDataset(filepath_to_images)
    video_obj = list(dset.videos().objs)[0]
    video_image_ids = dset.images(video_id=video_obj["id"])

    region_points_gdf_vidspace, warp_vid_from_wld, region_gdf_utm, region_gdf_crs84 = get_points(
        video_obj, filepath_to_points
    )
    if method == "box":
        regions = kwimage.Boxes(
            [[p.x, p.y, box_width, box_height] for p in region_points_gdf_vidspace],
            "cxywh",
        )
        polygons = regions.to_polygons()

        result = convert_polygons_to_region_model(
            polygons,
            main_region_header,
            warp_vid_from_wld,
            region_gdf_utm,
            region_gdf_crs84,
            time_pad,
        )
        output = ub.Path(output)
        output.write_text(result.dumps())

        ...
    if method == "sam":
        count = 0
        geo_polygons_total = []
        count_individual_mask = 0
        # average_image = kwarray.Stitcher((video_obj["height"], video_obj["width"]))
        all_predicted_regions = np.zeros(
            (len(region_points_gdf_vidspace), video_obj["height"], video_obj["width"])
        )
        predictor = load_sam(filepath_to_sam)

        for image_id in ub.ProgIter(video_image_ids, desc="Looping Over Videos..."):
            # geo_polygons_image = []
            # im = np.zeros((video_obj["height"], video_obj["width"]), dtype=np.uint8)
            coco_image = dset.coco_image(image_id)

            # Get the transform from video space to image space
            # Note that each point is associated with a date, so not all the
            # points warped here are actually associated with this image.
            warp_img_from_vid = coco_image.warp_img_from_vid
            region_points_gdf_imgspace = region_points_gdf_vidspace.affine_transform(
                warp_img_from_vid.to_shapely()
            )

            delayed_img = coco_image.imdelay("red|green|blue", space="video")
            imdata = delayed_img.finalize()

            # Depending on the sensor intensity might be out of standard ranges,
            # we can use kwimage to robustly normalize for this. This lets
            # us visualize data with false color.
            canvas = kwimage.normalize_intensity(imdata, axis=2)
            img = np.ascontiguousarray(canvas)
            predictor.set_image(img, "RGB")
            regions = kwimage.Boxes(
                [[p.x, p.y, box_width, box_height] for p in region_points_gdf_imgspace],
                "cxywh",
            )
            regions = regions.to_ltrb()

            for count_individual_mask, (point, box) in enumerate(
                zip(region_points_gdf_imgspace, regions)
            ):
                masks, scores, logits = predictor.predict(
                    point_coords=np.array([[point.x, point.y]]),
                    point_labels=np.array([1]),
                    # box= gt_BB,
                    box=box.data,
                    multimask_output=True,
                )

                binarized_mask = masks[0] > 0.5
                all_predicted_regions[count_individual_mask] += binarized_mask

                mask = masks[0]
                data = extract_polygons(mask)
                geo_polygons_total.append(data)
                # geo_masks_image.append(mask)
                # geo_masks_total.append(mask)

            # filename = f"{count}_sam_image_mask.png"
            # data = image_predicted(im,geo_polygons_image,filename)
            count = count + 1
        # geo_masks_total=np.reshape(geo_masks_total,(image_id+1,len(region_points_gdf_imgspace)))

        polygons = list(
            extract_sam_polygons(
                image_id,
                all_predicted_regions,
                main_region_header,
                region_points_gdf_imgspace,
                warp_vid_from_wld,
                region_gdf_utm,
            )
        )

        result = convert_polygons_to_region_model(
            polygons,
            main_region_header,
            region_points_gdf_imgspace,
            warp_vid_from_wld,
            region_gdf_utm,
            region_gdf_crs84,
            time_pad,
        )
        print(result.dumps())
        output.write_text(result.dumps())


if __name__ == "__main__":
    main()
