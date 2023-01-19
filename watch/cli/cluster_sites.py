"""
UNFINISHED
"""
import scriptconfig as scfg
import ubelt as ub
import kwimage


class ClusterSiteConfig(scfg.DataConfig):
    """
    Creates a new region file that groups nearby sites.
    """
    src = scfg.Value(None, help='input site summary file')
    dst = scfg.Value(None, help='output region file')


def main(cmdline=0, **kwargs):
    """
    kwargs = dict(
        src='/home/joncrall/remote/toothbrush/data/dvc-repos/smart_data_dvc-ssd/_testdag/pred/flat/bas_poly/bas_poly_id_d3d1d348/site_summaries_manifest.json'
    )
    """
    from watch.utils import util_gis
    config = ClusterSiteConfig.cli(data=kwargs)

    site_results = util_gis.coerce_geojson_datas(
        config['src'], workers=0,
        desc='load geojson site-models')

    region_id_to_geoms = ub.ddict(list)
    for result in ub.ProgIter(site_results):
        ss_df = result['data']
        site_summaries = ss_df[ss_df['type'] == 'site_summary']
        if len(site_summaries):
            sm = site_summaries.iloc[0]
            region_id = sm['region_id']
            region_id_to_geoms[region_id].append(site_summaries)

    # site_meters_min = 2048
    site_meters_min = 384
    total_area = {}
    total_area[site_meters_min] = 0

    scale = 1.7
    min_box_dim = 384
    max_box_dim = 384 * 4

    import pandas as pd
    from watch.utils import util_kwimage

    for region_id, geoms in region_id_to_geoms.items():
        region_sites = pd.concat(geoms).reset_index()

        #
        # region_sites['status'] == 'system_confirmed'

        region_sites_utm = util_gis.project_gdf_to_local_utm(region_sites, max_utm_zones=2)
        polygons = kwimage.PolygonList([kwimage.Polygon.from_shapely(s) for s in region_sites_utm.geometry])

        keep_bbs, overlap_idxs = util_kwimage.find_low_overlap_covering_boxes(polygons, scale, min_box_dim, max_box_dim, max_iters=100)
