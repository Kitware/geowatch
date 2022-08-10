
def main():
    """
    Make a set of non-overlapping regions of roughly site-sizes

    DVC_DPATH=$(smartwatch_dvc --hardware="hdd")
    SENSORS=TA1-S2-L8-ACC
    #SENSORS=TA1-S2-ACC
    #SENSORS=L2-S2-L8
    DATASET_SUFFIX=Drop4-2022-07-25-c30-$SENSORS
    REGION_GLOBSTR="$DVC_DPATH/annotations/region_models/*.geojson"
    SITE_GLOBSTR="$DVC_DPATH/annotations/site_models/*.geojson"

    """
    import watch
    dvc_dpath = watch.find_smart_dvc_dpath(hardware='hdd')

    site_fpaths = list((dvc_dpath / 'annotations/site_models/').glob('*.geojson'))
    region_fpaths = list((dvc_dpath / 'annotations/region_models/').glob('*.geojson'))

    from watch.utils import util_gis
    import ubelt as ub
    jobs = ub.JobPool('process', max_workers=8)
    site_results = list(ub.ProgIter(jobs.executor.map(util_gis.read_geojson, site_fpaths), total=len(site_fpaths)))

    jobs = ub.JobPool('process', max_workers=8)
    region_results = list(ub.ProgIter(jobs.executor.map(util_gis.read_geojson, region_fpaths), total=len(region_fpaths)))

    region_to_prototype = {}
    for result in region_results:
        region_rows = result[result['type'] == 'region']
        region_id = region_rows.iloc[0]['region_id']
        region_to_prototype[region_id] = region_rows
        pass

    import kwimage

    region_id_to_geoms = ub.ddict(list)
    for result in ub.ProgIter(site_results):
        site_models = result[result['type'] == 'site']
        if len(site_models):
            sm = site_models.iloc[0]
            region_id = sm['region_id']
            region_id_to_geoms[region_id].append(site_models)

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    # import numpy as np
    import xdev

    # for region_id, geoms in list(region_id_to_geoms.items()):
    items = [(k, v) for k, v in list(region_id_to_geoms.items()) if not k.endswith('xxx')]

    # site_meters_min = 2048
    site_meters_min = 384
    total_area = {}
    # kwplot.multi_plot(xdata=list(total_area.keys()), ydata=list(total_area.values()), fnum=2)
    # for site_meters_min in np.arange(1, 8) * 384:
    total_area[site_meters_min] = 0

    new_region_gdfs = []

    for region_id, geoms in xdev.InteractiveIter(items):
    # for region_id, geoms in items:
        region_sites = pd.concat(geoms).reset_index()
        region_sites = region_sites[region_sites['status'] != 'ignore']
        region_sites = region_sites[region_sites['status'] != 'positive_partial']
        region_sites = region_sites[region_sites['status'] != 'positive_pending']
        region_sites = region_sites[region_sites['status'] != 'positive_unbounded']
        region_sites = region_sites[region_sites['status'] != 'positive_excluded']

        region_sites_utm = util_gis.project_gdf_to_local_utm(region_sites, max_utm_zones=2)
        # centroids = np.array([pt.xy for pt in region_sites_utm.geometry.centroid]).reshape(-1, 2)

        # Context for each site in meters

        # site_meters_min = 1024
        # site_meters_min = 1024 + 512
        polylist = kwimage.PolygonList([kwimage.Polygon.from_shapely(s) for s in region_sites_utm.geometry])
        polybbs = kwimage.Boxes.concatenate([p.to_boxes() for p in polylist])
        candidates_bbs = polybbs.scale(1.7, about='center')
        candidates_bbs = candidates_bbs.to_cxywh()
        candidates_bbs.data[..., 2] = np.maximum(candidates_bbs.data[..., 2], site_meters_min)
        candidates_bbs.data[..., 3] = np.maximum(candidates_bbs.data[..., 3], site_meters_min)

        # Add some translated boxes to the mix to see if they do any better
        extras = [
            candidates_bbs.translate((-site_meters_min / 10, 0)),
            candidates_bbs.translate((+site_meters_min / 10, 0)),
            candidates_bbs.translate((0, -site_meters_min / 10)),
            candidates_bbs.translate((0, +site_meters_min / 10)),
            candidates_bbs.translate((-site_meters_min / 3, 0)),
            candidates_bbs.translate((+site_meters_min / 3, 0)),
            candidates_bbs.translate((0, -site_meters_min / 3)),
            candidates_bbs.translate((0, +site_meters_min / 3)),
        ]
        candidates_bbs = kwimage.Boxes.concatenate([candidates_bbs] + extras)

        # Find the minimum boxes that cover all of the regions
        # xs, ys = centroids.T
        # ws = hs = np.full(len(xs), fill_value=site_meters)
        # utm_boxes = kwimage.Boxes(np.stack([xs, ys, ws, hs], axis=1), 'cxywh').to_xywh()

        boxes_df = gpd.GeoDataFrame(geometry=candidates_bbs.to_shapley(), crs=region_sites_utm.crs)
        box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_df, region_sites_utm, predicate='contains')
        import kwarray
        cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
        keep_bbs = candidates_bbs.take(cover_idxs)
        removed = len(candidates_bbs) - len(keep_bbs)
        if removed:
            print(f'removed={removed}')
        total_area[site_meters_min] += keep_bbs.area.sum()

        keep_boxes_df = gpd.GeoDataFrame(geometry=keep_bbs.to_shapley(), crs=region_sites_utm.crs)
        keep_boxes_df['keep_idx'] = cover_idxs
        keep_boxes_crs84_df = keep_boxes_df.to_crs(util_gis._get_crs84())

        region_id = region_sites['region_id'].unique()[0]
        region_proto = region_to_prototype[region_id]

        for _, row in keep_boxes_crs84_df.iterrows():
            site_idxs = box_poly_overlap[row.keep_idx]
            subregion_row = region_proto.copy()

            # Give the new region a name based on the site
            new_id = sorted(region_sites.iloc[site_idxs]['site_id'])[0] + '_box'
            subregion_row = subregion_row.set_geometry([row.geometry])
            subregion_row.loc[:, 'region_id'] = new_id
            new_region_gdfs.append(subregion_row)

        if 1:
            import kwplot
            kwplot.autompl()
            kwplot.figure(fnum=1, doclf=1)
            polylist.draw(color='pink', setlim=1)
            candidates_bbs.draw(color='blue', setlim=1)
            keep_bbs.draw(color='orange')
            xdev.InteractiveIter.draw()
            # utm_boxes.draw(setlim=1, color='orange')
                # keep_boxes.draw(color='blue')


        subregion_dpath = (dvc_dpath / 'subregions').ensuredir()

        import json
        for new_gdf in ub.ProgIter(new_region_gdfs):
            region_id = new_gdf['region_id'].iloc[0]
            fpath = subregion_dpath / f'{region_id}.geojson'
            text = new_gdf.to_json()
            fpath.write_text(json.dumps(json.loads(text), indent='  '))
        # len(new_region_gdfs)

        """
        rsync -avprPR ~/data/dvc-repos/smart_watch_dvc-hdd/./subregions horologic:data/dvc-repos/smart_data_dvc
        """

        # If there is a way to pack annotations, we should do that.

        # # kwimage.Boxes(
        # est_pxl_size_at_3gsd = region_sites_utm.geometry.area.apply(np.sqrt) / 3

        # import networkx as nx
        # graph = nx.Graph()
        # for idxs in overlaps.values():
        #     graph.add_nodes_from(idxs)
        #     for u, v in ub.iter_window(idxs, 2):
        #         graph.add_edge(u, v)

        # for cc_idxs in nx.connected_components(graph):
        #     print(list(cc_idxs))
