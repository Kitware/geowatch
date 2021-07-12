from algorithm_toolkit import AlgorithmTestCase

from .main import Main
import pystac
import json

class MainTestCase(AlgorithmTestCase):

    def runTest(self):
        # configure params for your algorithm

        #will have to change depending on where test stac catalog is located
        stac_catalog = '/home/jacob/Documents/atktestdir/catalog.json'
        kwcoco_outpath = '/home/jacob/Documents/test.kwcoco.json'
        self.params = {'kwcoco':kwcoco_outpath,
                       'catalog':stac_catalog}

        self.alg = Main(cl=self.cl, params=self.params)
        self.alg.run()

        # Add tests and assertions below
        self.assertTrue(kwcoco_outpath==self.cl.get_from_metadata('output_path'))
        dataset = json.loads(self.cl.get_from_metadata('dataset'))
        catalog = pystac.Catalog.from_file(stac_catalog)
        items = []
        images = []
        for item in catalog.get_items():
            items.append(item.id)
            for asset in item.get_assets():
                if asset!='data' and 'data' not in item.assets[asset].roles:
                    continue
                images.append(item.assets[asset].get_absolute_href())
        for item in dataset['images']:
            self.assertTrue(item['name'] in items)
            if item['file_name']:
                self.assertTrue(item['file_name'] in images)
            else:
                for aux in item['auxiliary']:
                    self.assertTrue(aux['file_name'] in images)
