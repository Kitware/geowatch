"""
CommandLine:
    HAS_DVC=1 xdoctest geowatch/tasks/depth_pcd/model_test.py __doc__


Example:
    >>> # xdoctest: +REQUIRES(env:HAS_DVC)
    >>> import numpy as np
    >>> import geowatch
    >>> import ubelt as ub
    >>> from geowatch.tasks.depth_pcd.model import getModel
    >>> model = getModel()
    >>> expt_dvc_dpath = geowatch.find_dvc_dpath(tags='phase2_expt', hardware='auto')
    >>> model.load_weights(expt_dvc_dpath + '/models/depth_pcd/basicModel2.h5')
    >>> out = model.predict(np.zeros((1,400,400,3)))
    >>> shapes = [o.shape for o in out]
    >>> print('shapes = {}'.format(ub.urepr(shapes, nl=1)))
"""


def mwe_tensorflow():
    r"""
    Small example that tests if tensorflow will raise a DNN error in this env
    or not.

    References:
        https://www.tensorflow.org/install/pip

    Check CuDNN version

        !apt-cache policy libcudnn8


    Debugging:
        # Try running this example in the minimum pyenv311 env before
        # installing geowatch
        docker run \
            --gpus all  \
            --volume "$HOME"/.cache/pip:/root/.cache/pip \
            -it pyenv:311 \
            bash

        # pip install tensorflow ipython nvidia-cudnn-cu11
        pip install tensorflow=="2.12.0" nvidia-cudnn-cu11==8.6.0.163

        python -c "if 1:
            import tensorflow as tf
            print(tf.config.list_physical_devices())
            from tensorflow.keras.models import Model
            conv = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            ])
            i = tf.keras.Input([28, 28, 1], batch_size=1)
            out = conv(i)
            model = Model(inputs=i, outputs=[out])
            import numpy as np
            model.predict(np.zeros((1, 28, 28, 1)))
        "

    """

    import tensorflow as tf
    from tensorflow.keras.models import Model
    conv = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    ])

    i = tf.keras.Input([28, 28, 1], batch_size=1)
    out = conv(i)

    model = Model(inputs=i, outputs=[out])
    import numpy as np
    out = model.predict(np.zeros((1, 28, 28, 1)))
