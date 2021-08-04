import os
import pystac
import tempfile

from algorithm_toolkit import AlgorithmTestCase

from .main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        # configure params for your algorithm
        with tempfile.TemporaryDirectory() as output_dir:
            dry_run = 1
            self.params = {
                'stac_api_key': os.environ['STAC_API_KEY'],
                'aoi_bounds': [128.662489, 37.659517, 128.676673, 37.664560],
                'date_range': ["2017", "2018"],
                'stac_api_url': "https://api.smart-stac.com/",
                'output_dir': output_dir,
                'collections': ["worldview-nitf"],
                'dry_run': dry_run,
                'max_results': 5,
                'min_aoi_overlap': 0.25}

            self.alg = Main(cl=self.cl, params=self.params)
            self.alg.run()

            stac_catalog = pystac.Catalog.from_dict(self.cl.get_from_metadata('stac_catalog'))

            self.assertTrue(output_dir == self.cl.get_from_metadata('output_dir'))
            self.assertTrue(stac_catalog.get_self_href().startswith(output_dir))
            if not dry_run:
                self.assertTrue(os.path.isfile(stac_catalog.get_self_href()))
            for item in stac_catalog.get_items():
                self.assertTrue(item.get_self_href().startswith(output_dir))
                if not dry_run:
                   self.assertTrue(os.path.isfile(item.get_self_href()))
                for asset in item.get_assets():
                    self.assertTrue(item.assets[asset].get_absolute_href().startswith(output_dir))
                    if not dry_run:
                        self.assertTrue(os.path.isfile(item.assets[asset].get_absolute_href()))
