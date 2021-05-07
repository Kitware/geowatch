import os
import json

from algorithm_toolkit import AlgorithmTestCase

from .main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        # configure params for your algorithm
        output_dir = "/data/outdir"
        dry_run = 1
        self.params = {
            'stac-api-key': os.environ['STAC_API_KEY'],
            'aoi-bounds': [128.662489, 37.659517, 128.676673, 37.664560],
            'date-range': ["2017", "2018"],
            'stac-api-url': "https://api.smart-stac.com/",
            'output-dir': output_dir,
            'collections': ["worldview-nitf"],
            'dry-run': dry_run}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        stac_catalog = json.loads(
            self.cl.get_from_metadata('stac-catalog')['output_value'])

        for feature in stac_catalog.get('features', ()):
            # Ensure that the asset paths have been updated to point
            # to the local copy
            self.assertTrue(
                feature['assets']['data']['href'].startswith(output_dir))

            if dry_run != 1:
                # Ensure that the local copy exists
                self.assertTrue(
                    os.path.isfile(feature['assets']['data']['href']))
