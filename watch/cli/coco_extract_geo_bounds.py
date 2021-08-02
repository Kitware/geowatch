"""
Given annotated raw data in kwcoco format (with extra geo information) extract
the geojson space-time bounds.

TODO:
    - [ ] If annotations are given in pixel coordinates only, but there is a
          introspectable transformation from pixel to world coordinates, try to
          apply that to put data into world coordinates.
"""
import datetime
import dateutil.parser
import kwcoco
import kwimage
import numpy as np
import ubelt as ub
import json
import itertools as it
import networkx as nx
import scriptconfig as scfg
from shapely import ops


class CocoExtractBoundsConfig(scfg.Config):
    """
    Extract bounds of geojson tiffs (in a kwcoco file) into a regions file
    """
    default = {
        'src': scfg.Value('in.kwcoco.json', help='input dataset'),
        'dst': scfg.Value(None, help='output dataset, if none writes to stdout'),
        'breakup_times': scfg.Value(False, help='if true, breaks out regions into times that only intersect with existing images.'),
        'mode': scfg.Value('annots', help='can either be annots or images'),
    }


def main(**kwargs):
    """
    Ignore:
        kwargs = {
            'src': ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json'),
        }
    """
    config = CocoExtractBoundsConfig(default=kwargs, cmdline=True)
    src_fpath = config['src']
    dset = kwcoco.CocoDataset(src_fpath)

    breakup_times = config['breakup_times']
    mode = config['mode']

    # Find the clustered ROI regions
    geojson = find_spacetime_cluster_regions(dset, mode, breakup_times)

    dst_fpath = config['dst']

    if dst_fpath is None:
        print(json.dumps(geojson, indent=' ' * 4))
    else:
        print(f'Writing to dst_fpath={dst_fpath}')
        with open(dst_fpath, 'w') as file:
            json.dump(geojson, file, indent=' ' * 4)
    return dst_fpath
    # json.dumps(regions)
    # print('regions = {}'.format(ub.repr2(regions, nl=2)))


def find_spacetime_cluster_regions(dset, mode='annots', breakup_times=False):
    """
    Given a dataset find spatial regions of interest that contain annotations
    """
    gid_to_rois = {}

    if mode == 'images':
        import watch
        for gid, img in ub.ProgIter(dset.imgs.items(), total=len(dset.imgs)):

            if img.get('sensor_coarse') != 'S2':
                continue

            fpath = dset.get_image_fpath(img)
            info = watch.gis.geotiff.geotiff_metadata(fpath)
            print(info['wgs84_crs_info'])
            lonlat = info['wgs84_corners'].reorder_axes((1, 0))
            kw_poly = kwimage.structs.Polygon(exterior=lonlat.data[::-1])
            sh_poly = kw_poly._ensure_vertex_order().to_shapely()
            time = dateutil.parser.parse(img['date_captured'])
            gid_to_rois[gid] = ([sh_poly], time)

            break

    elif mode == 'annots':
        aid_to_poly = {}
        for aid, ann in dset.anns.items():
            # TODO: handle the case where this special field doesn't exist
            geo = _fix_geojson_poly(ann['segmentation_geos'])
            lonlat = np.array(geo['coordinates'][0])
            kw_poly = kwimage.structs.Polygon(exterior=lonlat)
            # kw_poly = kwimage.structs.Polygon.from_geojson(geo)
            aid_to_poly[aid] = kw_poly.to_shapely()

        # TODO: if there are only midly overlapping regions, we should likely split
        # them up. We can also group by UTM coordinates to reduce computation.

        # Combine all polygons from each "image" into a single shapely shape
        for gid, aids in dset.index.gid_to_aids.items():
            if len(aids):
                img = dset.index.imgs[gid]

                time = dateutil.parser.parse(img['date_captured'])

                sh_annot_polys = ub.dict_subset(aid_to_poly, aids)
                sh_annot_polys_ = [p.buffer(0) for p in sh_annot_polys.values()]
                sh_annot_polys_ = [p.buffer(0.000001) for p in sh_annot_polys_]

                # What CRS should we be doing this in? Is WGS84 OK?
                # Should we switch to UTM?
                img_rois_ = ops.unary_union(sh_annot_polys_)
                try:
                    img_rois = list(img_rois_)
                except Exception:
                    img_rois = [img_rois_]

                kw_img_rois = [
                    kwimage.Polygon.from_shapely(p.convex_hull).bounding_box().to_polygons()[0]
                    for p in img_rois]
                sh_img_rois = [p.to_shapely() for p in kw_img_rois]
                gid_to_rois[gid] = (sh_img_rois, time)
    else:
        raise KeyError(mode)

    # Find which groups of images are spatially connected
    graph = nx.Graph()
    next_roi_idx = it.count(0)

    for gid, (rois, time) in gid_to_rois.items():
        for roi in rois:
            roi = roi.buffer(0)
            new_idx = next(next_roi_idx)

            new_edges = []
            for roi_idx2 in graph.nodes:
                if roi.intersects(graph.nodes[roi_idx2]['roi']):
                    new_edges.append((new_idx, roi_idx2))

            graph.add_node(new_idx, roi=roi, time=time, gid=gid)
            graph.add_edges_from(new_edges)

    regions = []

    ccs = list(nx.connected_components(graph))
    for cc in ccs:
        rois = [graph.nodes[idx]['roi'] for idx in cc]
        times = [graph.nodes[idx]['time'] for idx in cc]
        min_time = min(times)
        max_time = max(times)
        combo_rois_ = ops.unary_union(rois)
        try:
            combo_rois = list(combo_rois_)
        except Exception:
            combo_rois = [combo_rois_]
        # hack to ensure convexity
        combo_rois_ = ops.unary_union([roi.convex_hull for roi in combo_rois])
        try:
            combo_rois = list(combo_rois_)
        except Exception:
            combo_rois = [combo_rois_]

        for roi in combo_rois:
            poly = kwimage.Polygon.from_shapely(roi)._ensure_vertex_order()
            if breakup_times:
                for time in sorted(set(times)):
                    min_time_ = time
                    max_time_ = time + datetime.timedelta(days=1.0)
                    feat = {
                        'type': 'Feature',
                        'geometry': poly.to_geojson(),
                        'properties': {
                            'max_time': datetime.datetime.isoformat(max_time_),
                            'min_time': datetime.datetime.isoformat(min_time_),
                            'crs': {
                                # TODO: ensure this codeing is right
                                # 'auth': ('EPSG', '4326'),
                                # 'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
                                'note': 'geojson is lon-lat',
                            }
                        }
                    }
                    regions.append(feat)
            else:
                feat = {
                    'type': 'Feature',
                    'geometry': poly.to_geojson(),
                    'properties': {
                        'max_time': datetime.datetime.isoformat(max_time),
                        'min_time': datetime.datetime.isoformat(min_time),
                        'crs': {
                            # 'auth': ('EPSG', '4326'),
                            # 'axis_mapping': 'OAMS_TRADITIONAL_GIS_ORDER',
                            'note': 'geojson is lon-lat',
                        }
                    }
                }
                regions.append(feat)

    geojson = {
        'type': 'FeatureCollection',
        'features': regions,
    }
    return geojson


def _fix_geojson_poly(geo):
    """
    We were given geojson polygons with one fewer layers of nesting than
    the spec allows for. Fix this.

    Example:
        >>> geo1 = kwimage.Polygon.random().to_geojson()
        >>> fixed1 = _fix_geojson_poly(geo1)
        >>> #
        >>> geo2 = {'type': 'Polygon', 'coordinates': geo1['coordinates'][0]}
        >>> fixed2 = _fix_geojson_poly(geo2)
        >>> assert fixed1 == fixed2
        >>> assert fixed1 == geo1
        >>> assert fixed2 != geo2
    """
    def check_leftmost_depth(data):
        # quick check leftmost depth of a nested struct
        item = data
        depth = 0
        while isinstance(item, list):
            if len(item) == 0:
                raise Exception('no child node')
            item = item[0]
            depth += 1
        return depth
    if geo['type'] == 'Polygon':
        data = geo['coordinates']
        depth = check_leftmost_depth(data)
        if depth == 2:
            # correctly format by adding the outer nesting
            fixed = geo.copy()
            fixed['coordinates'] = [geo['coordinates']]
        elif depth == 3:
            # already correct
            fixed = geo
        else:
            raise Exception(depth)
    else:
        fixed = geo
    return fixed


_SubConfig = CocoExtractBoundsConfig


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.cli.coco_extract_geo_bounds \
          --src $HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
          --dst $HOME/data/grab_tiles_out/regions2.geojson.json \
          --breakup_times=True \
          --mode=images

        python ~/code/watch/scripts/grab_tiles_demo.py \
            --regions $HOME/data/grab_tiles_out/regions2.geojson.json \
            --out_dpath $HOME/data/grab_tiles_out \
            --backend fels

        python -m watch.cli.coco_extract_geo_bounds \
          --src $HOME/data/dvc-repos/smart_watch_dvc/drop0/drop0.kwcoco.json \
          --breakup_times=True \
          --dst $HOME/data/grab_tiles_out/regions.geojson.json

        source ~/internal/safe/secrets
        python ~/code/watch/scripts/grab_tiles_demo.py \
            --regions $HOME/data/grab_tiles_out/regions.geojson.json \
            --out_dpath $HOME/data/grab_tiles_out \
            --rgdc_username=$WATCH_RGD_USERNAME \
            --rgdc_password=$WATCH_RGD_PASSWORD \
            --backend rgdc

        python ~/code/watch/scripts/grab_tiles_demo.py \
            --regions $HOME/data/grab_tiles_out/regions.geojson.json \
            --out_dpath $HOME/data/grab_tiles_out \
            --backend fels

        python -m watch.cli.geotiffs_to_kwcoco.py \
            --geotiff_dpath ~/data/grab_tiles_out/fels \
            --dst $HOME/data/grab_tiles_out/fels/data.kwcoco.json

    """
    main()
