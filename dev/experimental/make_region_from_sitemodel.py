import scriptconfig as scfg


class MakeRegionFromSiteModelConfig(scfg.DataConfig):
    src = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testpipe/pred/flat/bas_poly/bas_poly_id_52890263/sites/KR_R002_0002.geojson'
    dst = '/home/joncrall/remote/toothbrush/data/dvc-repos/smart_expt_dvc/_testsc_oneoff/small_KR_R002_0002.geojson'


def main():
    from watch.utils import util_gis
    from watch.utils import util_time
    import copy
    config = MakeRegionFromSiteModelConfig.legacy()
    config['src']
    site_infos = list(util_gis.coerce_geojson_datas(config.src, format='json'))
    import kwimage

    site_geoms = []
    site_summaries = []
    for site_info in site_infos:
        site_data = site_info['data']
        for feat in site_data['features']:
            if feat['properties']['type'] == 'site':
                site_summary = copy.deepcopy(feat)
                site_summary['properties']['type'] = 'site_summary'
                site_summaries.append(site_summary)
                site_geoms.append(kwimage.MultiPolygon.coerce(feat['geometry']).to_shapely())

    start_times = [util_time.coerce_datetime(f['properties']['start_date'])
                   for f in site_summaries]
    end_times = [util_time.coerce_datetime(f['properties']['end_date'])
                 for f in site_summaries]

    end_time = max(end_times)
    start_times = max(start_times)

    from shapely.ops import unary_union
    region_geom_inner = unary_union(site_geoms)

    import geopandas as gpd
    df = gpd.GeoDataFrame({'geometry': [region_geom_inner]}, crs='crs84')
    utm_df = util_gis.project_gdf_to_local_utm(df)
    outer_geom_utf = utm_df.convex_hull.scale(1.5, 1.5, origin='centroid')
    outer_geom_crs84 = outer_geom_utf.to_crs('crs84')

    region_geom = outer_geom_crs84.iloc[0]

    region_id = 'test_KR_region'

    import geojson
    region_feature = geojson.Feature(
        properties={
            "type": "region",
            "region_id": region_id,
            "version": "2.4.3",
            "mgrs": site_summary['properties']['mgrs'],
            "start_date": start_times.date().isoformat(),
            "end_date": end_time.date().isoformat(),
            "originator": "kit-demo",
            "model_content": "annotation",
            "comments": 'region-from-sitemodel',
        },
        geometry=region_geom
    )
    region = geojson.FeatureCollection([region_feature] + site_summaries)
    import json
    text = json.dumps(region)
    import ubelt as ub

    dst_fpath = ub.Path(config.dst)
    dst_fpath.write_text(text)
