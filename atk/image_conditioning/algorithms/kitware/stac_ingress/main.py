import os
import subprocess

from algorithm_toolkit import Algorithm, AlgorithmChain
from pystac_client import Client


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

        max_results = params.get('max_results', 0)
        if max_results > 0:
            search_results_catalog['features'] =\
                search_results_catalog['features'][:max_results]

        if params['dry_run'] != 1:
            os.makedirs(params['output_dir'], exist_ok=True)

        # TODO: Parallelize this download step?
        for feature in search_results_catalog.get('features', ()):
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

        cl.add_to_metadata('stac_catalog', search_results_catalog)
        cl.add_to_metadata('output_dir', params['output_dir'])

        # Do not edit below this line
        return cl
