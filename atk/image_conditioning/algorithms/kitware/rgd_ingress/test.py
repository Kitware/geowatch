from algorithm_toolkit import AlgorithmTestCase

from .main import Main

import os
import json
import pystac
import tempfile


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        # configure params for your algorithm
        with tempfile.TemporaryDirectory() as output_dir:
            dry_run = 0
            kwcoco = os.path.join(output_dir, 'rgd_results.kwcoco.json')
            ignore_dem = 1
            self.params = {
                'username' : os.environ['WATCH_RGD_USER'],
                'password' : os.environ['WATCH_RGD_PW'],
                'date_range' : ['2018-11-01', '2018-11-08'],
                'output_dir' : output_dir,
                'aoi_bounds' : [128.662489, 37.659517, 128.676673, 37.664560],
                'dry_run' : dry_run,
                'max_cloud_cover' : 0.5,
                'min_aoi_overlap': 0.25,
                'kwcoco' : kwcoco,
                'ignore_dem' : ignore_dem
            }

            self.alg = Main(cl=self.cl, params=self.params)
            self.alg.run()

            # Add tests and assertions below
            stac_catalog = pystac.Catalog.from_dict(self.cl.get_from_metadata('stac_catalog'))

            self.assertTrue(output_dir == self.cl.get_from_metadata('output_dir'))
            self.assertTrue(stac_catalog.get_self_href().startswith(output_dir))
            if kwcoco:
                self.assertTrue(os.path.isfile(kwcoco))
                self.assertTrue(kwcoco.startswith(output_dir))
                with open(kwcoco, 'r') as f:
                    kwfile = json.load(f)
                for img in kwfile['images']:
                    if img['file_name']:
                        paths = [img['file_name']]
                    else:
                        paths = [f['file_name'] for f in img['auxiliary']]
                    for path in paths:
                        if path.startswith(output_dir):
                            self.assertTrue(os.path.isfile(path))
                        else:
                            self.assertTrue(os.path.isfile(os.path.join(output_dir, path)))
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
