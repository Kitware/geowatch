from algorithm_toolkit import Algorithm, AlgorithmChain

from watch.cli.baseline_framework_egress import baseline_framework_egress


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        te_output = baseline_framework_egress(
            params['stac_catalog'],
            params['output_path'],
            params['output_bucket'],
            params.get('aws_profile'),
            params.get('dryrun', 0) == 1)

        cl.add_to_metadata('te_output', te_output)
        cl.add_to_metadata('output_path', params['output_path'])
        cl.add_to_metadata('output_bucket', params['output_bucket'])

        # Do not edit below this line
        return cl
