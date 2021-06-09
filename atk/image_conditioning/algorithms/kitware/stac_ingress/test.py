import os

from algorithm_toolkit import AlgorithmTestCase

from .main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        # configure params for your algorithm
        output_dir = "/data/outdir"
        dry_run = 1
        self.params = {
            'stac_api_key': os.environ['STAC_API_KEY'],
            'aoi_bounds': [128.662489, 37.659517, 128.676673, 37.664560],
            'date_range': ["2017", "2018"],
            'stac_api_url': "https://api.smart-stac.com/",
            'output_dir': output_dir,
            'collections': ["worldview-nitf"],
            'dry_run': dry_run}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        stac_catalog = self.cl.get_from_metadata('stac_catalog')

        for feature in stac_catalog.get('features', ()):
            # Ensure that the asset paths have been updated to point
            # to the local copy
            self.assertTrue(
                feature['assets']['data']['href'].startswith(output_dir))

            if dry_run != 1:
                # Ensure that the local copy exists
                self.assertTrue(
                    os.path.isfile(feature['assets']['data']['href']))
