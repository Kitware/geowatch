import os
import subprocess

from algorithm_toolkit import Algorithm, AlgorithmChain
from pystac_client import Client
import pystac
import shapely as shp
import shapely.ops
import shapely.geometry


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        catalog = Client.open(params['stac_api_url'],
                              headers={"x-api-key": params['stac_api_key']})

        search_results = catalog.search(collections=params['collections'],
                                        bbox=params['aoi_bounds'],
                                        datetime=params['date_range'])

        search_results_catalog = search_results.items_as_collection().to_dict()

        # Should probably move this after filtering to making this
        # more like "max filtered results"
        max_results = params.get('max_results', 0)
        if max_results > 0:
            search_results_catalog['features'] =\
                search_results_catalog['features'][:max_results]

        os.makedirs(params['output_dir'], exist_ok=True)
        catalog = pystac.Catalog('STAC ingress catalog',
                                 'STAC catalog of SMART search results',
                                 href=os.path.join(params['output_dir'],
                                                   'catalog.json'))
        catalog.set_root(catalog)

        # TODO: Parallelize this download step?
        for feature in search_results_catalog.get('features', ()):
            # TODO: Refactor these filtering steps out to separate ATK
            # algorithms

            # Completely skip ingress of STAC item if 'eo:cloud_cover'
            # is present and not <= the max value
            if('max_cloud_cover' in params
               and feature['properties'].get('eo:cloud_cover', 0) >
               params['max_cloud_cover']):
                continue

            # Completely skip ingress of STAC item when a minimum AOI
            # overlap is specified and the item's geometry doesn't
            # meet that threshold
            if 'min_aoi_overlap' in params:
                polys = shp.geometry.shape(feature['geometry']).buffer(0)
                union_poly = shp.ops.cascaded_union(polys)
                aoi_poly = shp.geometry.box(*params['aoi_bounds'])
                overlap =\
                    union_poly.intersection(aoi_poly).area / aoi_poly.area

                if overlap < params['min_aoi_overlap']:
                    continue

            # TODO: Wrap in a try catch for KeyError?
            asset_href = feature['assets']['data']['href']

            asset_basename = os.path.basename(asset_href)

            feature_output_dir = os.path.join(
                params['output_dir'], feature['id'])
            asset_outpath = os.path.join(
                feature_output_dir, asset_basename)

            command = ['aws', 's3', '--profile', 'iarpa', 'cp']
            if params['dry_run'] == 1:
                command.append('--dryrun')
            else:
                os.makedirs(feature_output_dir, exist_ok=True)

            command.extend([asset_href, asset_outpath])

            # TODO: Manually check return code / output
            self.logger.info("Running: {}".format(' '.join(command)))
            subprocess.run(command, check=True)

            # Update feature asset href to point to local outpath
            feature['assets']['data']['href'] = asset_outpath

            item = pystac.Item.from_dict(feature)
            item.set_collection(None)  # Clear the collection if present
            item.set_self_href(os.path.join(params['output_dir'],
                                            feature['id'],
                                            feature['id'] + '.json'))
            catalog.add_item(item)

        catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)

        cl.add_to_metadata('stac_catalog', catalog.to_dict())
        cl.add_to_metadata('output_dir', params['output_dir'])

        # Do not edit below this line
        return cl
