from algorithm_toolkit import AlgorithmTestCase
import os
import ubelt as ub
import kwcoco

from main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        output_dir = ub.ensure_app_cache_dir(
            'watch/atk/test_kwcoco_align_geotiffs')
        ub.delete(output_dir)  # remove the dir and contents if it exists
        ub.ensuredir(output_dir)  # create the empty directory.

        from watch.demo import smart_kwcoco_demodata
        raw_coco_dset = smart_kwcoco_demodata.demo_smart_raw_kwcoco()

        # Create an dummy input regions file for testing
        from watch.cli import coco_extract_geo_bounds
        regions_fpath = os.path.join(
            output_dir, 'demo_regions.geojson')
        coco_extract_geo_bounds.main(
            src=raw_coco_dset.fpath, mode='images', dst=regions_fpath,
        )

        # configure params for your algorithm
        self.params = {'kwcoco_path': raw_coco_dset.fpath,
                       'regions_path': regions_fpath,
                       'output_dir': output_dir}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        # Add tests and assertions below
        kwcoco_output = kwcoco.CocoDataset.coerce(
            self.cl.get_from_metadata('kwcoco_output_path'))

        assert(kwcoco_output.n_annots != 0)
