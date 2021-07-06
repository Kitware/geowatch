from algorithm_toolkit import Algorithm, AlgorithmChain
import os
import json
from datetime import datetime, timedelta
from rgd_client import Rgdc
import pystac
import requests


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
                                 href=os.path.join(params['output_dir'], 'catalog.json'))
        catalog.set_root(catalog)
        for search_result in query_s2 + query_l7 + query_l8:
            if not params['dry_run']:
                paths = client.download_raster(search_result, 
                                               params['output_dir'], 
                                               nest_with_name=True, 
                                               keep_existing=True)
            stac_item = client.get_raster(search_result, stac=True)
            stac_item['id'] = search_result['subentry_name']
            item = pystac.Item.from_dict(stac_item)
            item.set_self_href(os.path.join(params['output_dir'], 
                                            stac_item['id'], 
                                            stac_item['id']+'.json'))
            for asset in item.get_assets():
                dic = item.assets[asset].to_dict()
                dic['href'] = os.path.join(params['output_dir'], 
                                           stac_item['id'], 
                                           dic['title'])
                item.assets[asset] = pystac.Asset.from_dict(dic)
            catalog.add_item(item)

        catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

        cl.add_to_metadata('stac_catalog', catalog.to_dict())
        cl.add_to_metadata('output_dir', params['output_dir'])

        # Do not edit below this line
        return cl