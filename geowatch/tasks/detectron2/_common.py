import ubelt as ub
import os


class Detectron2WrapperBase:

    def __init__(self, config):
        """
        Args:
            config (DetectronFitCLI): the fit CLI config
        """
        self.config = config
        self.dataset_infos = None
        self.cfg = None  # the detectron level config
        self.output_dpath = None  # Where we will actually write to
        self.trainer = None

    def register_datasets(self):
        dataset_paths = {
            'vali': self.config.vali_fpath,
            'train': self.config.train_fpath,
        }
        self.dataset_infos = {}
        with ub.Timer('Registering COCO files with detectron'):
            for key, fpath in dataset_paths.items():
                if fpath is not None:
                    row = register_detectron_dataset(fpath)
                    self.dataset_infos[key] = row
                else:
                    self.dataset_infos[key] = None

    def resolve_cfg(self):
        raise NotImplementedError('abstract')


def register_detectron_dataset(fpath):
    from detectron2.data.datasets import register_coco_instances
    fpath = ub.Path(fpath)
    assert fpath.exists()
    row = {'path': fpath}
    # row['name'] = fpath.name.split('.', 1)[0]
    row['name'] = os.fspath(fpath)
    row['categories'] = parse_json_section(fpath, 'categories')
    register_coco_instances(row['name'], {}, row['path'], row['path'].parent)
    return row


def parse_json_section(fpath, tablename):
    """
    Given a path to a json or compressed json file, attempt to read just one
    specific section. If this section is at the start of the file then this
    operation will be very fast.

    Ignore:
        from geowatch.tasks.detectron2.fit import *  # NOQA
        import kwcoco
        dset = kwcoco.CocoDataset.demo('vidshapes8')
        fpath = dset.fpath
        tablename = 'categories'
        table = parse_json_section(fpath, tablename)
        print(f'table = {ub.urepr(table, nl=1)}')
    """
    from kwcoco.util import ijson_ext
    import zipfile
    if zipfile.is_zipfile(fpath):
        # We have a compressed json file, but we can still read the header
        # fairly quickly.
        zfile = zipfile.ZipFile(fpath)
        names = zfile.namelist()
        assert len(names) == 1
        member = names[0]
        # Stream the header directly from the zipfile.
        file = zfile.open(member, 'r')
    else:
        # Normal json file
        file = open(fpath, 'r')

    with file:
        # import ijson
        # We only expect there to be one info section
        # try:
        #     # Try our extension if the main library fails (due to NaN)
        #     table_iter = ijson.items(file, prefix=tablename)
        #     table = next(info_section_iter)
        # except ijson.IncompleteJSONError:
        # Try our extension if the main library fails (due to NaN)
        # file.seek(0)
        # Nans are too frequent, only use our extension
        table_iter = ijson_ext.items(file, prefix=tablename)
        table = next(table_iter)
    return table
