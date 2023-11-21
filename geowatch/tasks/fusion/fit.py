#!/usr/bin/env python3
"""
THIS IS THE OLD FIT SCRIPT. MOST OF IT HAS BEEN REMOVED.
USE fit_lightning.py INSTEAD.

Trains a fusion machine learning model on target dataset.

SeeAlso:
    README.md
    fit.py
    predict.py
    evaluate.py
    experiments/crall/onera_experiments.sh
    experiments/crall/drop1_experiments.sh
    experiments/crall/toy_experiments.sh
"""


def coerce_initializer(init):
    import os
    import ubelt as ub
    from geowatch.monkey import monkey_torchmetrics
    from geowatch.monkey import monkey_torch
    monkey_torchmetrics.fix_torchmetrics_compatability()

    initializer = None

    maybe_packaged_model = False
    if isinstance(init, (str, os.PathLike)):
        if ub.Path(init).exists():
            maybe_packaged_model = True

    if maybe_packaged_model:
        try:
            from geowatch.tasks.fusion import utils
            other_model = utils.load_model_from_package(init)
            monkey_torch.fix_gelu_issue(other_model)
        except Exception:
            print('Not a packaged model')
        else:
            from torch_liberator.initializer import Pretrained
            import torch
            import tempfile
            tfile = tempfile.NamedTemporaryFile(prefix='pretrained_state', suffix='.pt')
            # state_dict = other_model.state_dict()
            try:
                state_dict = other_model.state_dict()
            except Exception:
                if hasattr(other_model, 'head_metrics'):
                    other_model.head_metrics.clear()
                    state_dict = other_model.state_dict()
                else:
                    raise

            # HACK:
            # Remove the normalization keys, we don't want to transfer them
            # in this step. They will be set correctly depending on if
            # normalize_inputs=transfer or not.
            HACK_IGNORE_INPUT_NORMS = True
            if HACK_IGNORE_INPUT_NORMS:
                ignore_keys = [key for key in state_dict if 'input_norms' in key]
                for k in ignore_keys:
                    state_dict.pop(k)
                print('Hacking a packaged model for init')

            # print(ub.urepr(sorted(state_dict.keys())))
            weights_fpath = tfile.name
            torch.save(state_dict, weights_fpath)
            init_cls = Pretrained
            init_kw = {'fpath': tfile.name}
            initializer = init_cls(**init_kw)
            # keep the temporary file alive as long as the initializer is
            initializer._tfile = tfile
            initializer.other_model = other_model

    if initializer is None:
        # Try a netharn method (todo: port to geowatch to remove netharn deps)
        from geowatch.utils import util_netharn
        init_cls, init_kw = util_netharn.Initializer.coerce(init=init)
        initializer = init_cls(**init_kw)

    return initializer


"""
Ignore:
"""
