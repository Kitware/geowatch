

def test_multimodel_with_null_sensor():
    """
    Test that unspecified sensors (i.e. None Values) work well enough and are
    translated to "*".
    """
    import numpy as np
    kwargs = {
        'name': 'unnamed_model',
        'weight_decay': 0.0,
        'lr_scheduler': 'CosineAnnealingLR',
        'class_weights': 'auto',
        'saliency_weights': 'auto',
        'stream_channels': 16,
        'tokenizer': 'rearrange',
        'token_norm': 'none',
        'arch_name': 'smt_it_stm_p2',
        'decoder': 'mlp',
        'dropout': 0.1,
        'window_size': 4,
    }
    dataset_stats = {
        'unique_sensor_modes': {
            ('*', 'gray'),
        },
        'sensor_mode_hist': {
            (None, 'gray'): 10000,
        },
        'input_stats': {
            (None, 'gray'): {
                'mean': np.array([[[0.5]]], dtype=np.float64),
                'std': np.array([[[0.5]]], dtype=np.float64),
                'min': np.array([[[0.]]], dtype=np.float64),
                'max': np.array([[[1.]]], dtype=np.float64),
                'n': np.array([[[100]]], dtype=np.float64)
            }
        },
        'class_freq': {
            'nevative': 100000,
            'positive': 10000,
        },
        'modality_input_stats': {
            (None, 'gray', None): {
                'mean': np.array([0.5], dtype=np.float64),
                'std': np.array([0.5], dtype=np.float64),
                'min': np.array([0.], dtype=np.float64),
                'max': np.array([1.], dtype=np.float64),
                'n': np.array([100], dtype=np.float64)
            },
        },
        'annot_class_freq': {},
        'track_class_freq': {},
    }
    input_sensorchan = None
    input_channels = None
    classes = ['nevative', 'positive']

    from geowatch.tasks.fusion.methods.channelwise_transformer import MultimodalTransformer
    self = MultimodalTransformer(classes=classes, dataset_stats=dataset_stats,
                                 input_sensorchan=input_sensorchan,
                                 input_channels=input_channels, **kwargs)
    assert list(self.input_norms.keys())  == ['*']
    assert list(self.input_norms['*'].keys())  == ['gray']
