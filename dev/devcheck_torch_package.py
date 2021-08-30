# millnames = ['',' K',' M',' B',' T']
# return '{:.2f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


def poc():
    from watch.tasks.fusion import methods
    import torch
    import numpy as np
    import math
    from torch import package

    model = methods.MultimodalTransformer("smt_it_stm_p8")

    package_path = 'torch_package.zip'
    module_name = 'watch_tasks_fusion'

    verbose = True
    exp = package.PackageExporter(package_path, verbose=verbose)

    exp.extern("**", exclude=["watch.tasks.fusion.**"])
    exp.intern("watch.tasks.fusion.**")
    package_name = 'my_module_name'
    resource_name = 'my_resource_name'
    exp.save_pickle(package_name, resource_name, model)
    exp.close()

    # TODO: this is not a problem yet, but some package types will (mainly
    # binaries) will need to be excluded also and added as mocks

    from zipfile import ZipFile
    myzip = ZipFile(package_path)

    importer = package.PackageImporter(package_path)
    recon = importer.load_pickle(package_name, resource_name)



def load_model_from_package(package_path, module_name="watch_tasks_fusion", model_name="model.pkl"):
    imp = package.PackageImporter(package_path)
    return imp.load_pickle(module_name, model_name)

class Lambda(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
