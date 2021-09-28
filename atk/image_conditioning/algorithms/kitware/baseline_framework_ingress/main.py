from algorithm_toolkit import Algorithm, AlgorithmChain

from watch.cli.baseline_framework_ingress import baseline_framework_ingress


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        catalog = baseline_framework_ingress(params['input_path'],
                                             params['output_dir'],
                                             params.get('aws_profile'))

        cl.add_to_metadata('stac_catalog', catalog.to_dict())

        # Do not edit below this line
        return cl
