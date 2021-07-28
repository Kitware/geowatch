import copy

import pystac
from algorithm_toolkit import Algorithm, AlgorithmChain

from watch.cli.stac_egress import stac_egress


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        stac_catalog_dict = copy.deepcopy(params['stac_catalog'])
        stac_catalog = pystac.Catalog.from_dict(stac_catalog_dict)

        dry_run = 'dry_run' in params and params['dry_run'] == 1

        output_stac_catalog = stac_egress(
            stac_catalog,
            params['s3_bucket'],
            dry_run,
            outdir=params.get('output_dir'))

        cl.add_to_metadata('stac_catalog', output_stac_catalog)
        cl.add_to_metadata('s3_bucket', params['s3_bucket'])

        # Do not edit below this line
        return cl
