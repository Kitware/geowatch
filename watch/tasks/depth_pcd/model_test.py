"""
Example:
    >>> # xdoctest: +REQUIRES(env:HAS_DVC)
    >>> import numpy as np
    >>> import watch
    >>> from watch.tasks.depth_pcd.model import getModel
    >>> #expt_dvc_dpath = watch.find_smart_dvc_dpath(tags='phase2_expt', hardware='auto')
    >>> model = getModel()
    >>> #model.load_weights(expt_dvc_dpath + '/models/depth_pcd/basicModel.h5')
    >>> model.predict(np.zeros((1,400,400,3)))
"""