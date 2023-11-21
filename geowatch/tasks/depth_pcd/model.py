if 1:
    # Put imports in an if 1 to avoid linting errors
    import ubelt as ub
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Monkeypatch deeplab_v2 into the namespace
    import geowatch_tpl
    TPL_DPATH = ub.Path(geowatch_tpl.MODULE_DPATH)
    import tensorflow as tf

    HACK_CPU_ONLY = 1
    if HACK_CPU_ONLY:
        # Hide GPU from visible devices
        tf.config.set_visible_devices([], 'GPU')
    else:
        for d in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(d, True)

    # print(tf.config.list_physical_devices('GPU'))

    from deeplab2 import config_pb2
    from deeplab2.model.deeplab import DeepLab

    from google.protobuf import text_format
    from tensorflow.keras.layers import Layer, Dense, concatenate, Conv2D, \
        BatchNormalization, \
        GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model, Sequential
    import cv2
    import numpy as np
    # import sys
    # sys.path.append('geowatch/tasks/depth_pcd')
    # sys.path.append('geowatch/tasks/depth_pcd/deeplab2')


img_size = 400


def normalize(im, q=5, r=128):
    imr = cv2.resize(im, (r, r))
    minV = np.percentile(imr[imr > 0], q)
    maxV = np.percentile(imr[imr > 0], 100 - q)
    den = max(1, maxV - minV)
    #    b = 0.5
    im = (im - minV) / den  # +np.random.uniform(-b,b,1)
    im[im > 1] = 1
    im[im < 0] = 0
    im = im * 255
    im = cv2.resize(im, (img_size, img_size))
    return im


def getModel(proto=TPL_DPATH / 'deeplab2/max_deeplab_s_backbone_os16.textproto', use_ln=False):
    options = config_pb2.ExperimentOptions()

    with tf.io.gfile.GFile(proto) as f:
        text_format.Parse(f.read(), options)

    model = DeepLab(options, use_ln)
    i = tf.keras.Input([img_size, img_size, 3], batch_size=1)

    backbone = model.layers[0]
    backout = backbone(i, training=True)['backbone_output']
    outs = model(i, training=True)
    ll = concatenate([tf.expand_dims(outs['center_heatmap'], axis=-1), outs['offset_map']], axis=-1)
    lo = outs['semantic_logits']

    l5 = Layer(name='imapx')(img_size * Dense(1, name='imapxDense')(ll)[..., 0])
    l6 = Layer(name='imapy')(img_size * Dense(1, name='imapyDense')(ll)[..., 0])

    l3 = Layer(name='mapx')(img_size * Dense(1, name='mapxDense')(ll)[..., 0])
    l4 = Layer(name='mapy')(img_size * Dense(1, name='mapyDense')(ll)[..., 0])

    m1 = concatenate([tf.expand_dims(l4, axis=-1), tf.expand_dims(l3, axis=-1)], axis=-1)
    m2 = concatenate([tf.expand_dims(l6, axis=-1), tf.expand_dims(l5, axis=-1)], axis=-1)

    # lo=tf.sigmoid(outs['semantic_logits'])
    l2 = Layer(name='agl')(Dense(1, name='agl1Dense')(lo)[..., 0])  # (ll)[...,0])
    l2b = Layer(name='agl2')(Dense(1, name='agl2Dense')(lo)[..., 0])  # (ll)[...,0])

    l2_warp = Layer(name='agl_warp')(
        bilinear_warp(tf.expand_dims(l2, axis=-1), m2))
    l2b_warp = Layer(name='agl2_warp')(
        bilinear_warp(tf.expand_dims(l2b, axis=-1), m1))

    diffL = Layer(name='diff')(tf.expand_dims(tf.abs(l2 - l2b_warp), axis=-1)
                               + tf.expand_dims(tf.abs(l2b - l2_warp), axis=-1))
    #    ds = 4
    fs = 16
    mask = Layer(name='mask')(
        # UpSampling2D((ds,ds),name='maskUp')(\
        Dense(1, activation='sigmoid', name='maskDense')(
            # BatchNormalization(name='maskBN')(\
            # MaxPooling2D((ds,ds),name='maskPool')(
            Conv2D(64, fs, padding='same', name='maskConv', activation='gelu')(
                BatchNormalization(name='maskBN')(diffL)))[..., 0])

    bhead = Sequential(name='bhead')
    bhead.add(Conv2D(64, 16, padding='same', name='bheadC', activation='gelu'))
    bhead.add(BatchNormalization(name='bheadBN2'))
    bhead.add(Dense(1, activation='sigmoid', name='bheadD'))

    # mask=Layer(name='mask')(bhead(diffL)[...,0])
    bdmask = Layer(name='bdmask')(bhead(tf.expand_dims(l2, axis=-1))[..., 0])
    bdmask2 = Layer(name='bdmask2')(bhead(tf.expand_dims(l2b, axis=-1))[..., 0])
    # classification
    classification = Layer(name='class')(Dense(1, activation='sigmoid', name='classD')(
        Dropout(0.99, name='dropClass')(
            GlobalAveragePooling2D()(backout))))
    # BatchNormalization(name='classBN')(backout))))

    model = Model(inputs=i, outputs=[l2, l2b, l3, l4, l5, l6,
                                     bdmask, bdmask2,
                                     classification,
                                     mask,
                                     l2b_warp,
                                     l2_warp])
    return model


def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing='ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg  # return collectively for elementwise processing


def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fy, fx = tf.unstack(flow, axis=-1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0 + 1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0 + 1

    # warping indices
    h_lim = tf.cast(h - 1, tf.float32)
    w_lim = tf.cast(w - 1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)

    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis=3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis=3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis=3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis=3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy) * (fx_1 - fx), axis=3)
    c_01 = tf.expand_dims((fy_1 - fy) * (fx - fx_0), axis=3)
    c_10 = tf.expand_dims((fy - fy_0) * (fx_1 - fx), axis=3)
    c_11 = tf.expand_dims((fy - fy_0) * (fx - fx_0), axis=3)

    warp = c_00 * x_00 + c_01 * x_01 + c_10 * x_10 + c_11 * x_11
    return warp[..., 0]
