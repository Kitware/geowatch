import os

from algorithm_toolkit import Algorithm, AlgorithmChain

from watch.tasks.template.predict import predict_on_dataset


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        predict_on_dataset(dataset=params['kwcoco_path'],
                           out_dpath=params['output_dir'])

        # Would be nice to pass the full output path to the prediction
        # function above so that we don't have to assume we know where
        # it will end up
        kwcoco_output_path =\
            os.path.join(params['output_dir'], "predictions.kwcoco.json")

        cl.add_to_metadata('kwcoco_output_path', kwcoco_output_path)

        # Do not edit below this line
        return cl
