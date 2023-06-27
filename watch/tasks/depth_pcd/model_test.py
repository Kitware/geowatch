"""
CommandLine:
    HAS_DVC=1 xdoctest -m watch.tasks.depth_pcd.model_test __doc__

Example:
    >>> # xdoctest: +REQUIRES(env:HAS_DVC)
    >>> import numpy as np
    >>> import watch
    >>> import ubelt as ub
    >>> from watch.tasks.depth_pcd.model import getModel
    >>> model = getModel()
    >>> expt_dvc_dpath = watch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    >>> model.load_weights(expt_dvc_dpath + '/models/depth_pcd/basicModel2.h5')
    >>> out = model.predict(np.zeros((1,400,400,3)))
    >>> shapes = [o.shape for o in out]
    >>> print('shapes = {}'.format(ub.urepr(shapes, nl=1)))
"""
