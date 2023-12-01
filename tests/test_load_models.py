import ubelt as ub
import os


def tryget_dvc_dpath():
    _default = ub.expandpath('$HOME/data/dvc-repos/smart_watch_dvc')
    dvc_dpath = os.environ.get('DVC_DPATH', _default)
    dvc_dpath = ub.Path(dvc_dpath)
    if not dvc_dpath.exists():
        import pytest
        pytest.skip('this test depends on data in DVC_DPATH')
    return dvc_dpath


def test_load_uky_models():
    import torch
    dvc_dpath = tryget_dvc_dpath()
    uky_dpath = (dvc_dpath / 'models/uky_invariants/sort_augment_overlap')
    checkpoint_fpaths = list(uky_dpath.glob('*.ckpt'))
    states = []
    for fpath in checkpoint_fpaths:
        loaded = torch.load(fpath)
        states.append(loaded)

    for state in states:
        info = ub.dict_isect(state, {'epoch', 'hyper_parameters', 'global_step'})
        print('info = {}'.format(ub.urepr(info, nl=2)))
        print(state['epoch'])
        print(state['hyper_parameters'])
        print(state['hparams_name'])
        print(state['global_step'])

    # inv_sort, inv_overlap, inv_shared, or inv_augment?

    r"""
        /models/uky_invariants/sort_augment_overlap/S2_drop1-S2-L8-aligned-old.0.ckpt
        /models/uky_invariants/sort_augment_overlap/L8_drop1-S2-L8-aligned-old.0.ckpt
        /models/uky_invariants/sort_augment_overlap/LS_drop1-S2-L8-aligned-old.0.ckpt
        /models/uky_invariants/sort_augment_overlap/S2_drop1-S2-aligned-old.0.ckpt


    Notes on prediction:
        DVC_DPATH=/home/joncrall/data/dvc-repos/smart_watch_dvc
        CHECKPOINT_FPATH=$DVC_DPATH/models/uky_invariants/sort_augment_overlap/S2_drop1-S2-L8-aligned-old.0.ckpt
        CHECKPOINT_FPATH=$DVC_DPATH/models/uky_invariants/sort_augment_overlap/L8_drop1-S2-L8-aligned-old.0.ckpt
        CHECKPOINT_FPATH=$DVC_DPATH/models/uky_invariants/sort_augment_overlap/LS_drop1-S2-L8-aligned-old.0.ckpt
        CHECKPOINT_FPATH=$DVC_DPATH/models/uky_invariants/sort_augment_overlap/S2_drop1-S2-aligned-old.0.ckpt

        python -m geowatch stats $DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json

        python -m geowatch.tasks.invariants.predict \
            --sensor S2 \
            --input_kwcoco $DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json \
            --output_kwcoco $DVC_DPATH/drop1-S2-L8-aligned/_partial_uky_pred_S2.kwcoco.json \
            --ckpt_path $DVC_DPATH/models/uky_invariants/sort_augment_overlap/S2_drop1-S2-L8-aligned-old.0.ckpt

        python -m geowatch.tasks.invariants.predict \
            --sensor L8 \
            --input_kwcoco $DVC_DPATH/drop1-S2-L8-aligned/data.kwcoco.json \
            --output_kwcoco $DVC_DPATH/drop1-S2-L8-aligned/_partial_uky_pred_L8.kwcoco.json \
            --ckpt_path $DVC_DPATH/models/uky_invariants/sort_augment_overlap/L8_drop1-S2-L8-aligned-old.0.ckpt

        cd $DVC_DPATH/drop1-S2-L8-aligned
        kwcoco union --src _partial_uky_pred_S2.kwcoco.json _partial_uky_pred_L8.kwcoco.json --dst uky_invariants.kwcoco.json

        python ~/code/watch/geowatch/cli/coco_combine_features.py --src \
                _partial_uky_pred_S2.kwcoco.json \
                _partial_uky_pred_L8.kwcoco.json \
                --dst uky_invariants.kwcoco.json

        python -m geowatch stats _partial_uky_pred_S2.kwcoco.json
        python -m geowatch stats _partial_uky_pred_L8.kwcoco.json
        python -m geowatch stats uky_invariants.kwcoco.json

        cd $DVC_DPATH/drop1-S2-L8-aligned
        cd $DVC_DPATH/drop1-S2-L8-aligned
        python ~/code/watch/geowatch/cli/coco_combine_features.py --src \
                data.kwcoco.json \
                landcover.kwcoco.json \
                uky_invariants.kwcoco.json \
                --dst ./combo_data.kwcoco.json

        rice_field|cropland|water|inland_water|river_or_stream|sebkha|snow_or_ice_field|bare_ground|sand_dune|built_up|grassland|brush|forest|wetland|road

        python ~/data/dvc-repos/smart_watch_dvc/dev/coco_show_auxiliary.py combo_data.kwcoco.json --channel2 bare_ground
        python ~/data/dvc-repos/smart_watch_dvc/dev/coco_show_auxiliary.py combo_data.kwcoco.json --channel2 water
        python ~/data/dvc-repos/smart_watch_dvc/dev/coco_show_auxiliary.py combo_data.kwcoco.json --channel2 forest
        python ~/data/dvc-repos/smart_watch_dvc/dev/coco_show_auxiliary.py combo_data.kwcoco.json --channel2 inland_water
        python ~/data/dvc-repos/smart_watch_dvc/dev/coco_show_auxiliary.py combo_data.kwcoco.json --channel2 river_or_stream
    """
