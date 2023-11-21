r"""
Helpers for reading data downloaded from digital globe


Notes:
    The data in the Core3D dataset is public and can be rehosted.

    https://spacenet.ai/core3d/

    AWS_PROFILE=iarpa aws s3 ls s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Satellite-Images/
    AWS_PROFILE=iarpa aws s3 ls s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Tiled-Examples-for-Urban-3D-Challenge-Comparisons/02_Master/challenge-inputs/

    AWS_PROFILE=iarpa aws s3 cp \
            s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Tiled-Examples-for-Urban-3D-Challenge-Comparisons/02_Master/challenge-inputs/RIC_Tile_000_DSM.tif RIC_Tile_000_DSM.tif
    AWS_PROFILE=iarpa aws s3 cp \
            s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Tiled-Examples-for-Urban-3D-Challenge-Comparisons/02_Master/challenge-inputs/RIC_Tile_000_DTM.tif RIC_Tile_000_DTM.tif
    AWS_PROFILE=iarpa aws s3 cp \
            s3://spacenet-dataset/Hosted-Datasets/CORE3D-Public-Data/Tiled-Examples-for-Urban-3D-Challenge-Comparisons/02_Master/challenge-inputs/RIC_Tile_000_RGB.tif RIC_Tile_000_RGB.tif

Requirements:
    pip install xmltodict
    pip install pyshp
    pip install cogeotiff
"""
from os.path import exists
from os.path import join
import ubelt as ub
import xmltodict
from os.path import dirname, abspath


class DigitalGlobeBundle(ub.NiceRepr):
    """
    Data structure to organize information in digital globe bundles

    TODO: need public digital globe demodata for a doctest

    Maybe we can grab them from here?
    https://www.maxar.com/product-samples
    https://ard.maxar.com/samples#v5/

    https://spacenet.ai/core3d/

    Requirements:
        pip isntall pyshp

    Ignore:
        # This has a different format than our stuff... bleh..
        sample_zip_fpath = ub.grabdata('https://maxar-marketing.s3.amazonaws.com/product-samples/Rome_Colosseum_2022-03-22_WV03_HD.zip', hash_prefix='2a99cea2b37bed9b5867fa21a1bd')
        from kwcoco.util import util_archive
        archive = util_archive.Archive(sample_zip_fpath)
        dpath = (ub.Path(sample_zip_fpath).parent / 'MaxarSample').ensuredir()
        metadata_fpath = list(dpath.glob('*.MAN'))[0]
        archive.extractall(dpath)
        delivery_metadata_fpath = dpath / '050012575010_01/050012575010_01_README.XML'
        self = DigitalGlobeBundle(delivery_metadata_fpath)
    """

    def __init__(self, delivery_metadata_fpath, pointer=None, autobuild=True):
        self.data = {
            'delivery_metadata_fpath': delivery_metadata_fpath,
            'product_metas': None,
            'pointer': pointer,
        }
        if autobuild:
            self.parse_delivery_metadata()

    def __nice__(self):
        return self.data['delivery_metadata_fpath']

    def parse_delivery_metadata(self):
        import shapefile
        delivery_metadata_fpath = self.data['delivery_metadata_fpath']
        dpath = dirname(delivery_metadata_fpath)

        with open(delivery_metadata_fpath, 'r') as file:
            delivery_metadata = xmltodict.parse(file.read())

        self.data['other'] = ub.dict_diff(delivery_metadata, {'DeliveryMetadata'})
        self.data['non_product'] = ub.dict_diff(delivery_metadata, {'DeliveryMetadata'})

        product_list = delivery_metadata['DeliveryMetadata']['product']

        pointer = self.data['pointer']

        product_metas = []
        for product in product_list:
            product_meta = product.copy()
            prod_files = product_meta.pop('productFile')

            import kwimage
            # Find the files associated with the order AOI
            aoi_fpaths = {
                'shp': None,
                'dbf': None,
                'shx': None,
                'prj': None,
            }
            misc_exts = {
                '_LAYOUT.JPG',
                'NEXTVIEW.TXT',
                '_README.TXT',
                '_README.XML',
                '-BROWSE.JPG',
            }
            prod_types = []
            for v in prod_files:
                product_type = None

                for ext in misc_exts:
                    if v['filename'].lower().endswith(ext.lower()):
                        product_type = 'misc'

                if product_type is None:
                    for ext in aoi_fpaths.keys():
                        fname = 'ORDER_SHAPE.' + ext
                        if v['filename'].lower().endswith(fname.lower()):
                            fpath = join(dpath, v['relativeDirectory'], v['filename'])
                            assert aoi_fpaths[ext] is None
                            aoi_fpaths[ext] = fpath
                            product_type = 'aoi'

                if product_type is None:
                    if v['relativeDirectory'].lower().endswith('GIS_FILES'.lower()):
                        product_type = 'misc-gis'

                if product_type is None:
                    for ext in kwimage.im_io.IMAGE_EXTENSIONS:
                        if v['filename'].lower().endswith(ext.lower()):
                            product_type = 'image'

                if product_type is None:
                    product_type = 'other'

                prod_types.append(product_type)

            type_to_prods = ub.group_items(prod_files, prod_types)
            type_to_prods['image']
            type_to_prods['misc']
            type_to_prods['gis']
            type_to_prods['aoi']
            type_to_prods['other']
            # prod_type_hist = ub.map_vals(len, type_to_prods)
            # print('prod_type_hist = {}'.format(ub.urepr(prod_type_hist, nl=1)))

            aoi_files = {key: open(val, 'rb') for key, val in aoi_fpaths.items()}
            try:
                shp_wkt = ub.ensure_unicode(aoi_files['prj'].read())
                shp_reader = shapefile.Reader(
                    shp=aoi_files['shp'],
                    dbf=aoi_files['dbf'],
                    shx=aoi_files['shx']
                )
                aoi_geojson = shp_reader.shape().__geo_interface__
                product_meta['aoi_geojson'] = aoi_geojson
                product_meta['shp_wkt'] = shp_wkt
            finally:
                for val in aoi_files.values():
                    val.close()
                shp_reader.close()

            if pointer is not None:
                for v in type_to_prods['image']:
                    prod_fname = v['filename']
                    flag = pointer.endswith(prod_fname)
                    v['is_pointer'] = flag

            product_meta['images'] = type_to_prods['image']
            product_metas.append(product_meta)

        self.data['product_metas'] = product_metas

    @classmethod
    def from_pointer(cls, pointer, **kw):
        """
        Args:
            pointer (str): a path to any file inside a digital globe
                bundle. We will search for the DeliveryMetadata.xml data.

        Ignore:
            pointer = '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/_assets/20170907_a_KRP_011777481_10_0/011777481010_01_003/011777481010_01/011777481010_01_P001_MUL/17SEP07021826-M1BS-011777481010_01_P001.TIF'

            pointer = '/home/joncrall/data/dvc-repos/smart_watch_dvc/drop0/KR-Pyeongchang-WV/_assets/20170907_a_KRP_011777481_10_0/011777481010_01_003/011777481010_01/011777481010_01_P001_PAN/17SEP07021826-P1BS-011777481010_01_P001.TIF'

            cls = DigitalGlobeBundle
            self = DigitalGlobeBundle.from_pointer(pointer)

            for meta in self.data['product_metas']:
                meta['sensorVehicle']

            dict_list = self.data['product_metas']
            from netharn.data.collate import default_collate
            default_collate(dict_list)
            print(varried['sensorVehicle'])
        """
        dpath = abspath(pointer)
        delivery_fpath = search_path_ancestors(
            path=dpath, fname='DeliveryMetadata.xml')
        if delivery_fpath is None:
            raise Exception('cannot find DG DeliveryMetadata.xml')

        self = cls(delivery_fpath, pointer=pointer, **kw)
        return self

    @classmethod
    def coerce(cls, key, **kw):
        try:
            self = cls.pointer(key, **kw)
        except Exception:
            self = None
        return self


def search_path_ancestors(path, fname, stop_fname=None, max_steps=1000):
    """
    Search path and all of its containing folders for a file name ``fname``.

    Args:
        path (str): directory to start the search
        fname (str): path to search for
        stop_fname (str): stop if we find a file with this name.
    """
    import itertools as it
    dpath = path
    found = None
    for idx in it.count():
        fpath = join(dpath, fname)
        if exists(fpath):
            found = fpath
            break
        if stop_fname is not None:
            stop_fpath = join(dpath, stop_fname)
            if exists(stop_fpath):
                raise Exception('found stop fname, cannot find {}'.format(fname))
        dpath_next = dirname(dpath)
        if idx > max_steps:
            raise Exception('too many steps, cannot find {}'.format(fname))
        if dpath_next == dpath:
            raise Exception('reached the root, cannot find {}'.format(fname))
        dpath = dpath_next
    return found
