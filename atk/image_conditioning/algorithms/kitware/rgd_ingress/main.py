import os
import json
from datetime import datetime

from algorithm_toolkit import Algorithm, AlgorithmChain
from rgd_client import Rgdc
import pystac
import shapely as shp
import shapely.ops
import shapely.geometry
from watch.tools.stac_to_kwcoco import convert


class Main(Algorithm):

    def date(self, arg):
        dt = [int(d) for d in arg.split('-')]
        return datetime(dt[0], dt[1], dt[2])

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        top, left, bottom, right = params['aoi_bounds']
        geojson_bbox = {
            "type":
            "Polygon",
            "coordinates": [[[top, left], [top, right], [bottom, right],
                             [bottom, left], [top, left]]]
        }
        dt_min = self.date(params['date_range'][0])
        dt_max = self.date(params['date_range'][1])

        client = Rgdc(username=params['username'],
                      password=params['password'],
                      api_url="https://watch.resonantgeodata.com/api")
        kwargs = {
            'query': json.dumps(geojson_bbox),
            'predicate': 'intersects',
            'acquired': (dt_min, dt_max)
        }
        query_s2 = (client.search(**kwargs, instrumentation='S2A') +
                    client.search(**kwargs, instrumentation='S2B'))
        query_l7 = client.search(**kwargs, instrumentation='ETM')
        query_l8 = client.search(**kwargs, instrumentation='OLI_TIRS')
        if not params['dry_run']:
            os.makedirs(params['output_dir'], exist_ok=True)
        catalog = pystac.Catalog('RGD ingress catalog',
                                 'STAC catalog of RGD search results',
                                 href=os.path.join(params['output_dir'],
                                                   'catalog.json'))
        catalog.set_root(catalog)
        for search_result in query_s2 + query_l7 + query_l8:
            stac_item = client.get_raster(search_result, stac=True)
            stac_item['id'] = search_result['subentry_name']
            item = pystac.Item.from_dict(stac_item)
            item.set_self_href(os.path.join(params['output_dir'],
                                            stac_item['id'],
                                            stac_item['id']+'.json'))

            # TODO: Refactor these filtering steps out to separate ATK
            # algorithms

            # Completely skip ingress of STAC item if 'eo:cloud_cover'
            # is present and not <= the max value
            if('max_cloud_cover' in params
               and stac_item['properties'].get('eo:cloud_cover', 0) >
               params['max_cloud_cover']):
                continue

            # Completely skip ingress of STAC item when a minimum AOI
            # overlap is specified and the item's geometry doesn't
            # meet that threshold
            if 'min_aoi_overlap' in params:
                polys = shp.geometry.shape(stac_item['geometry']).buffer(0)
                union_poly = shp.ops.cascaded_union(polys)
                aoi_poly = shp.geometry.shape(geojson_bbox)
                overlap =\
                    union_poly.intersection(aoi_poly).area / aoi_poly.area

                if overlap < params['min_aoi_overlap']:
                    continue

            if not params['dry_run']:
                paths = client.download_raster(search_result,
                                               params['output_dir'],
                                               nest_with_name=True,
                                               keep_existing=True)

            for asset in item.get_assets():
                dic = item.assets[asset].to_dict()
                dic['href'] = os.path.join(params['output_dir'],
                                           stac_item['id'],
                                           dic['title'])
                item.assets[asset] = pystac.Asset.from_dict(dic)
            catalog.add_item(item)

        catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

        if params['kwcoco']:
            convert(params['kwcoco'], 
                    os.path.join(params['output_dir'], 'catalog.json'),
                    params['ignore_dem'])

        cl.add_to_metadata('stac_catalog', catalog.to_dict())
        cl.add_to_metadata('output_dir', params['output_dir'])

        # Do not edit below this line
        return cl
