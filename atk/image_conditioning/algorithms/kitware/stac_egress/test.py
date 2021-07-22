from datetime import datetime
import os

from algorithm_toolkit import AlgorithmTestCase
import pystac
import ubelt as ub

from main import Main


class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        # Working directory
        working_dir = ub.ensure_app_cache_dir(
            'watch/atk_test/stac_egress')
        ub.delete(working_dir)  # remove the dir and contents if it exists
        ub.ensuredir(working_dir)  # create the empty directory.

        # TODO: Establish a test catalog somewhere that we can use for
        # these kinds of tests
        test_catalog = pystac.Catalog(
            'test_catalog',
            'Just a test catalog',
            href=os.path.join(working_dir, 'catalog.json'))

        test_item_1_path = os.path.join(working_dir, '1', '1.json')
        test_item_1 = pystac.Item(
            '1', href=test_item_1_path,
            geometry=None, bbox=None, datetime=datetime.now(), properties={})
        test_asset_1_path = os.path.join(working_dir, "testfile1.NTF")
        os.mknod(test_asset_1_path)
        test_item_1.add_asset(
            'data',
            pystac.Asset.from_dict(
                {'href': test_asset_1_path,
                 'title': "asset_1",
                 'roles': ['data']}))
        pystac.write_file(test_item_1, test_item_1_path)

        test_item_2_path = os.path.join(working_dir, '2', '2.json')
        test_item_2 = pystac.Item(
            '2', href=test_item_2_path,
            geometry=None, bbox=None, datetime=datetime.now(), properties={})
        test_asset_2_path = os.path.join(working_dir, "testfile2.NTF")
        os.mknod(test_asset_2_path)
        test_item_2.add_asset(
            'data',
            pystac.Asset.from_dict(
                {'href': test_asset_2_path,
                 'title': "asset_2",
                 'roles': ['data']}))
        pystac.write_file(test_item_2, test_item_2_path)

        test_catalog.add_item(test_item_1)
        test_catalog.add_item(test_item_2)

        s3_outpath = "s3://dry-run-bucket"
        dry_run = 1
        self.params = {
            'stac_catalog': test_catalog.to_dict(),
            's3_bucket': s3_outpath,
            'dry_run': dry_run}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        stac_catalog = self.cl.get_from_metadata('stac_catalog')

        for item in stac_catalog.get_all_items():
            # Ensure that the asset paths have been updated to point
            # to the local copy
            for asset_name, asset in item.assets.items():
                self.assertTrue(asset.href.startswith(s3_outpath))
