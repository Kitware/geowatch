from algorithm_toolkit import Algorithm, AlgorithmChain

from watch.cli.stac_to_kwcoco import convert

class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here
        dset = convert(params['kwcoco'], params['catalog'])

        cl.add_to_metadata('output_path', params['kwcoco'])
        cl.add_to_metadata('dataset', dset)

        # Do not edit below this line
        return cl
