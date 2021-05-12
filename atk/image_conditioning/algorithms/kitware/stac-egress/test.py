import json

from algorithm_toolkit import AlgorithmTestCase

from .main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        test_catalog = '''
{
  "features": [
    {
      "assets": {
        "data": {
          "href": "/data/test/testfile1.NTF"
        }
      },
      "id": "1"
    },
    {
      "assets": {
        "data": {
          "href": "/data/test/testfile2.NTF"
        }
      },
      "id": "2"
    }
  ]
}
'''

        s3_outpath = "s3://dry-run-bucket"
        dry_run = 1
        self.params = {
            'stac-catalog': test_catalog,
            's3-bucket': s3_outpath,
            'dry-run': dry_run}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        stac_catalog = json.loads(
            self.cl.get_from_metadata('stac-catalog')['output_value'])

        for feature in stac_catalog.get('features', ()):
            # Ensure that the asset paths have been updated to point
            # to the local copy
            self.assertTrue(
                feature['assets']['data']['href'].startswith(s3_outpath))
