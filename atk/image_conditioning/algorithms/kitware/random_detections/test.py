from algorithm_toolkit import AlgorithmTestCase
import ubelt as ub
import kwcoco

from main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        output_dir = ub.ensure_app_cache_dir(
            'watch/atk/test_random_detections')
        ub.delete(output_dir)  # remove the dir and contents if it exists
        ub.ensuredir(output_dir)  # create the empty directory.

        # configure params for your algorithm
        self.params = {'kwcoco_path': "special:vidshapes8",
                       'output_dir': output_dir}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        # Add tests and assertions below
        kwcoco_output = kwcoco.CocoDataset.coerce(
            self.cl.get_from_metadata('kwcoco_output_path'))

        assert(kwcoco_output.n_annots != 0)
