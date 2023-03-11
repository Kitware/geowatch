def model_hparams(model):
    _ref = model
    for _try in range(5):
        if hasattr(_ref, 'hparams'):
            return _ref.hparams
        # might be in a distributed data parallel module.
        elif hasattr(_ref, 'module'):
            _ref = _ref.module
        elif hasattr(_ref, '_modules') and '_forward_module' in _ref._modules:
            _ref = _ref._modules['_forward_module']
        else:
            return None
