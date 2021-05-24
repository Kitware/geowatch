import os
import json
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

        if params['dry_run'] != 1:
            os.makedirs(params['output_dir'], exist_ok=True)

        # TODO: Parallelize this download step?
        for feature in search_results_catalog.get('features', ()):
            # TODO: Wrap in a try catch for KeyError?
            asset_href = feature['assets']['data']['href']

            _, asset_ext = os.path.splitext(asset_href)
            asset_outpath = os.path.join(
                params['output_dir'],
                '{}{}'.format(feature['id'], asset_ext))

            command = ['aws', 's3', '--profile', 'iarpa', 'cp']
            if params['dry_run'] == 1:
                command.append('--dryrun')

            command.extend([asset_href, asset_outpath])

            # TODO: Manually check return code / output
            self.logger.info("Running: {}".format(' '.join(command)))
            subprocess.run(command, check=True)

            # Update feature asset href to point to local outpath
            feature['assets']['data']['href'] = asset_outpath

        stac_catalog_output = {
            'output_type': 'text',
            'output_value': json.dumps(
                search_results_catalog,
                indent=2, sort_keys=True)}

        output_dir_output = {
            'output_type': 'text',
            'output_value': params['output_dir']}

        cl.add_to_metadata('stac_catalog', stac_catalog_output)
        cl.add_to_metadata('output_dir', output_dir_output)

        # Do not edit below this line
        return cl
