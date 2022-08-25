
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

    # for region_id, geoms in items:
    for region_id, geoms in xdev.InteractiveIter(items):
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
        polygons = kwimage.PolygonList([kwimage.Polygon.from_shapely(s) for s in region_sites_utm.geometry])

        scale = 1.7
        min_box_dim = 384
        max_box_dim = 384 * 4

        candidate_bbs, _ = find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim, max_iters=0)
        old_keep_bbs, old_overlap_idxs = find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim, max_iters=1)
        keep_bbs, overlap_idxs = find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim, max_iters=100)

        # polybbs = kwimage.Boxes.concatenate([p.to_boxes() for p in polylist])
        # candidate_bbs = polybbs.scale(1.7, about='center')
        # candidate_bbs = candidate_bbs.to_cxywh()
        # candidate_bbs.data[..., 2] = np.maximum(candidate_bbs.data[..., 2], site_meters_min)
        # candidate_bbs.data[..., 3] = np.maximum(candidate_bbs.data[..., 3], site_meters_min)

        # # Add some translated boxes to the mix to see if they do any better
        # extras = [
        #     candidate_bbs.translate((-site_meters_min / 10, 0)),
        #     candidate_bbs.translate((+site_meters_min / 10, 0)),
        #     candidate_bbs.translate((0, -site_meters_min / 10)),
        #     candidate_bbs.translate((0, +site_meters_min / 10)),
        #     candidate_bbs.translate((-site_meters_min / 3, 0)),
        #     candidate_bbs.translate((+site_meters_min / 3, 0)),
        #     candidate_bbs.translate((0, -site_meters_min / 3)),
        #     candidate_bbs.translate((0, +site_meters_min / 3)),
        # ]
        # candidate_bbs = kwimage.Boxes.concatenate([candidate_bbs] + extras)

        # # Find the minimum boxes that cover all of the regions
        # # xs, ys = centroids.T
        # # ws = hs = np.full(len(xs), fill_value=site_meters)
        # # utm_boxes = kwimage.Boxes(np.stack([xs, ys, ws, hs], axis=1), 'cxywh').to_xywh()

        # boxes_df = gpd.GeoDataFrame(geometry=candidate_bbs.to_shapley(), crs=region_sites_utm.crs)
        # box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_df, region_sites_utm, predicate='contains')
        # import kwarray
        # cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
        # keep_bbs = candidate_bbs.take(cover_idxs)
        # removed = len(candidate_bbs) - len(keep_bbs)
        # if removed:
        #     print(f'removed={removed}')
        # total_area[site_meters_min] += keep_bbs.area.sum()

        keep_boxes_df = gpd.GeoDataFrame(geometry=keep_bbs.to_shapley(), crs=region_sites_utm.crs)
        keep_boxes_df['site_idxs'] = overlap_idxs
        keep_boxes_crs84_df = keep_boxes_df.to_crs(util_gis._get_crs84())

        region_id = region_sites['region_id'].unique()[0]
        region_proto = region_to_prototype[region_id]

        for _, row in keep_boxes_crs84_df.iterrows():
            site_idxs = row.site_idxs
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
            polygons.draw(color='kw_blue', setlim=0)
            # candidate_bbs.draw(color='black', setlim=0, lw=10)
            old_keep_bbs.draw(color='kw_darkgreen', setlim=0, lw=6, alpha=0.8)
            keep_bbs.draw(color='kw_green', setlim=1, lw=3, alpha=0.8)
            xdev.InteractiveIter.draw()

            size1 = np.sqrt(candidate_bbs.area.sum())
            size2 = np.sqrt(old_keep_bbs.area.sum())
            size3 = np.sqrt(keep_bbs.area.sum())
            print(f'size1={size1}')
            print(f'size2={size2}')
            print(f'size3={size3}')

            loss1 = size1 / len(candidate_bbs)
            loss2 = size2 / len(candidate_bbs)
            loss3 = size3 / len(candidate_bbs)
            print(f'loss1={loss1}')
            print(f'loss2={loss2}')
            print(f'loss3={loss3}')

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


def find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim, merge_thresh=0.001, max_iters=100):
    """
    Given a set of polygons we want to find a small set of boxes that
    completely cover all of those polygons.

    We are going to do some set-cover shenanigans by making a bunch of
    candidate boxes based on some hueristics and find a set cover of those.

    Then we will search for small boxes that can be merged, and iterate.

    References:
        https://aip.scitation.org/doi/pdf/10.1063/1.5090003?cookieSet=1
        Mercantile - https://pypi.org/project/mercantile/0.4/
        BingMapsTiling - XYZ Tiling for webmap services
        https://mercantile.readthedocs.io/en/stable/api/mercantile.html#mercantile.bounding_tile
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.713.6709&rep=rep1&type=pdf

    Ignore:
        >>> # Create random polygons as test data
        >>> import kwimage
        >>> import kwarray
        >>> from kwarray import distributions
        >>> rng = kwarray.ensure_rng(934602708841)
        >>> num = 200
        >>> #
        >>> canvas_width = 2000
        >>> offset_distri = distributions.Uniform(canvas_width, rng=rng)
        >>> scale_distri = distributions.Uniform(10, 150, rng=rng)
        >>> #
        >>> polygons = []
        >>> for _ in range(num):
        >>>     poly = kwimage.Polygon.random(rng=rng)
        >>>     poly = poly.scale(scale_distri.sample())
        >>>     poly = poly.translate(offset_distri.sample(2))
        >>>     polygons.append(poly)
        >>> polygons = kwimage.PolygonList(polygons)
        >>> #
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(doclf=1)
        >>> plt.gca().set_xlim(0, canvas_width)
        >>> plt.gca().set_ylim(0, canvas_width)
        >>> _ = polygons.draw(fill=0, border=1, color='pink')
        >>> #
        >>> scale = 1.0
        >>> min_box_dim = 240
        >>> max_box_dim = 500
        >>> #
        >>> keep_bbs, overlap_idxs = find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim)
        >>> # xdoctest: +REQUIRES(--show)
        >>> #
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=1)
        >>> polygons.draw(color='pink')
        >>> # candidate_bbs.draw(color='blue', setlim=1)
        >>> keep_bbs.draw(color='orange', setlim=1)
        >>> plt.gca().set_title('find_low_overlap_covering_boxes')
    """
    import kwimage
    import kwarray
    import numpy as np
    import geopandas as gpd
    import ubelt as ub
    from watch.utils import util_gis
    import networkx as nx

    polygons_sh = [p.to_shapely() for p in polygons]
    polygons_gdf = gpd.GeoDataFrame(geometry=polygons_sh)

    polybbs = kwimage.Boxes.concatenate([p.to_boxes() for p in polygons])
    initial_candiate_bbs = polybbs.scale(scale, about='center')
    initial_candiate_bbs = initial_candiate_bbs.to_cxywh()
    initial_candiate_bbs.data[..., 2] = np.maximum(initial_candiate_bbs.data[..., 2], min_box_dim)
    initial_candiate_bbs.data[..., 3] = np.maximum(initial_candiate_bbs.data[..., 3], min_box_dim)

    candidate_bbs = initial_candiate_bbs

    def refine_candidates(candidate_bbs, iter_idx):
        # Add some translated boxes to the mix to see if they do any better
        extras = [
            candidate_bbs.translate((-min_box_dim / 10, 0)),
            candidate_bbs.translate((+min_box_dim / 10, 0)),
            candidate_bbs.translate((0, -min_box_dim / 10)),
            candidate_bbs.translate((0, +min_box_dim / 10)),
            candidate_bbs.translate((-min_box_dim / 3, 0)),
            candidate_bbs.translate((+min_box_dim / 3, 0)),
            candidate_bbs.translate((0, -min_box_dim / 3)),
            candidate_bbs.translate((0, +min_box_dim / 3)),
        ]
        candidate_bbs = kwimage.Boxes.concatenate([candidate_bbs] + extras)

        # Find the minimum boxes that cover all of the regions
        # xs, ys = centroids.T
        # ws = hs = np.full(len(xs), fill_value=site_meters)
        # utm_boxes = kwimage.Boxes(np.stack([xs, ys, ws, hs], axis=1), 'cxywh').to_xywh()

        boxes_gdf = gpd.GeoDataFrame(geometry=candidate_bbs.to_shapley(), crs=polygons_gdf.crs)
        box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
        cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
        keep_bbs = candidate_bbs.take(cover_idxs)
        box_ious = keep_bbs.ious(keep_bbs)

        if iter_idx > 0:
            # Dont do it on the first iter to compare to old algo
            laplace = box_ious - np.diag(np.diag(box_ious))
            mergable = laplace > merge_thresh
            g = nx.Graph()
            g.add_edges_from(list(zip(*np.where(mergable))))
            cliques = sorted(nx.find_cliques(g), key=len)[::-1]

            used = set()
            merged_boxes = []
            for clique in cliques:
                if used & set(clique):
                    continue

                new_box = keep_bbs.take(clique).bounding_box()
                w = new_box.width.ravel()[0]
                h = new_box.height.ravel()[0]
                if w < max_box_dim and h < max_box_dim:
                    merged_boxes.append(new_box)
                    used.update(clique)

            unused = sorted(set(range(len(keep_bbs))) - used)
            post_merge_bbs = kwimage.Boxes.concatenate([keep_bbs.take(unused)] + merged_boxes)

            boxes_gdf = gpd.GeoDataFrame(geometry=post_merge_bbs.to_shapley(), crs=polygons_gdf.crs)
            box_poly_overlap = util_gis.geopandas_pairwise_overlaps(boxes_gdf, polygons_gdf, predicate='contains')
            cover_idxs = list(kwarray.setcover(box_poly_overlap).keys())
            new_cand_bbs = post_merge_bbs.take(cover_idxs)
        else:
            new_cand_bbs = keep_bbs

        new_cand_overlaps = list(ub.take(box_poly_overlap, cover_idxs))
        return new_cand_bbs, new_cand_overlaps

    new_cand_overlaps = None

    for iter_idx in range(max_iters):
        old_candidate_bbs = candidate_bbs
        candidate_bbs, new_cand_overlaps = refine_candidates(candidate_bbs, iter_idx)
        num_old = len(old_candidate_bbs)
        num_new = len(candidate_bbs)
        if num_old == num_new:
            residual = (old_candidate_bbs.data - candidate_bbs.data).max()
            if residual > 0:
                print('improving residual = {}'.format(ub.repr2(residual, nl=1)))
            else:
                print('converged')
                break
        else:
            print(f'improving: {num_old} -> {num_new}')
    else:
        print('did not converge')
    keep_bbs = candidate_bbs
    overlap_idxs = new_cand_overlaps

    if 0:
        import kwplot
        kwplot.autoplt()
        kwplot.figure(fnum=1, doclf=1)
        polygons.draw(color='pink')
        # candidate_bbs.draw(color='blue', setlim=1)
        keep_bbs.draw(color='orange', setlim=1)

    return keep_bbs, overlap_idxs
