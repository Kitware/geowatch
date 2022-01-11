from algorithm_toolkit import Algorithm, AlgorithmChain
from watch.cli import coco_align_geotiffs


class Main(Algorithm):

    def run(self):
        cl = self.cl  # type: AlgorithmChain.ChainLedger
        params = self.params  # type: dict
        # Add your algorithm code here

        new_dset = coco_align_geotiffs.main(
            src=params['kwcoco_path'],
            dst=params['output_dir'],
            regions=params['regions_path'],
            max_workers=4,
            aux_workers=4,
            context_factor=1.0,
            write_subsets=False,
            visualize=False,

        )

        kwcoco_output_path = new_dset.fpath

        cl.add_to_metadata('kwcoco_output_path', kwcoco_output_path)

        # Do not edit below this line
        return cl
