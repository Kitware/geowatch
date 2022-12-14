from watch.tasks.rutgers_material_seg_v2.models.material_mlp import MaterialMLP

MODELS = {'material_mlp': MaterialMLP}


def build_model(model_name, n_in_channels, n_out_channels, **kwargs):

    try:
        model_func = MODELS[model_name]
    except KeyError:
        raise NotImplementedError(f'Model "{model_name}" not implimented in build_model function.')

    model = model_func(n_in_channels, n_out_channels, **kwargs)

    return model
