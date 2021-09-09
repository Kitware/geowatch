import copy

import pystac
from algorithm_toolkit import Algorithm, AlgorithmChain

from watch.cli.add_angle_bands import add_angle_bands


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict

        stac_catalog_dict = copy.deepcopy(params['stac_catalog'])

        stac_catalog = pystac.Catalog.from_dict(stac_catalog_dict)

        output_stac_catalog = add_angle_bands(stac_catalog,
                                              params['output_dir'])

        cl.add_to_metadata('stac_catalog', output_stac_catalog.to_dict())
        cl.add_to_metadata('output_dir', params['output_dir'])

        # Do not edit below this line
        return cl
