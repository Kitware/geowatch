def patch_numpy_dtypes():
    import numpy as np
    np.bool = bool


def patch_numpy_2x():
    """
    Help compatability with 2.x

    This is mainly for handling dependencies that haven't officially ported to
    numpy 2.x yet. When they catch up this can be removed.

    Main culprits are tensorboard and lightning

    Ignore:
        from geowatch.monkey.monkey_numpy import patch_numpy_2x
        patch_numpy_2x()
    """
    import numpy as np
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
        np.string_ = np.bytes_
        np.unicode_ = np.str_
        np.Inf = np.inf
