"""
Templates for script that will be written to disk for use by a user or another
program.
"""
import ubelt as ub


def train_time_helper_scripts(dpath, train_coco_fpath, vali_coco_fpath):
    """
    Example:
        >>> from geowatch.tasks.fusion.helper_scripts import *
        >>> dpath = ub.Path('.')
        >>> train_coco_fpath = ub.Path('train.kwcoco.zip')
        >>> vali_coco_fpath = ub.Path('vali.kwcoco.zip')
        >>> scripts = train_time_helper_scripts(dpath, train_coco_fpath, vali_coco_fpath)
        >>> print(f'scripts = {ub.urepr(scripts, nl=3, sv=1)}')
    """
    scripts = {}

    key = 'start_tensorboard'
    script = scripts[key] = {}
    script['fpath'] = dpath / f'{key}.sh'
    script['text'] = ub.codeblock(
        f'''
        #!/usr/bin/env bash
        tensorboard --logdir {dpath}
        ''')

    try:
        generating_fpath = __file__
    except NameError:
        generating_fpath = 'geowatch/tasks/fusion/helper_scripts.py'
    autogen_comment = f'# Generated by {generating_fpath}'

    key = 'draw_tensorboard'
    script = scripts[key] = {}
    script['fpath'] = dpath / f'{key}.sh'
    script['text'] = (ub.codeblock(
        fr'''
        #!/usr/bin/env bash
        {autogen_comment}

        # Update the main plots and stack them into a nice figure
        GEOWATCH_PREIMPORT=0 python -m geowatch.utils.lightning_ext.callbacks.tensorboard_plotter \
            {dpath}
        '''
    ))

    checkpoint_header_part = ub.codeblock(
        fr'''
        #!/usr/bin/env bash
        {autogen_comment}

        # Device defaults to CPU, but the user can pass a GPU in
        # as the first argument.
        DEVICE=${{1:-"cpu"}}

        TRAIN_DPATH="{dpath}"
        echo $TRAIN_DPATH

        ### --- Choose Checkpoint --- ###

        # Find a checkpoint to evaluate
        # TODO: should add a geowatch helper for this
        CHECKPOINT_FPATH=$(python -c "if 1:
            import pathlib
            train_dpath = pathlib.Path('$TRAIN_DPATH')
            found = sorted((train_dpath / 'checkpoints').glob('*.ckpt'))
            found = [f for f in found if 'last.ckpt' not in str(f)]
            print(found[-1])
            ")
        echo "$CHECKPOINT_FPATH"

        ### --- Repackage Checkpoint --- ###

        # Convert it into a package, then get the name of that
        geowatch repackage "$CHECKPOINT_FPATH"

        PACKAGE_FPATH=$(python -c "if 1:
            import pathlib
            p = pathlib.Path('$CHECKPOINT_FPATH')
            found = list(p.parent.glob(p.stem + '*.pt'))
            print(found[-1])
        ")
        echo "$PACKAGE_FPATH"

        PACKAGE_NAME=$(python -c "if 1:
            import pathlib
            p = pathlib.Path('$PACKAGE_FPATH')
            print(p.stem.replace('.ckpt', ''))
        ")
        echo "$PACKAGE_NAME"
        ''')

    key = 'draw_train_batches'
    script = scripts[key] = {}
    script['fpath'] = dpath / f'{key}.sh'
    script['text'] = chr(10).join([
        checkpoint_header_part,
        ub.codeblock(
            fr'''
            ### --- Train Batch Prediction --- ###

            # Predict on the validation set
            export IGNORE_OFF_BY_ONE_STITCHING=1  # hack
            python -m geowatch.tasks.fusion.predict \
                --package_fpath "$PACKAGE_FPATH" \
                --test_dataset "{train_coco_fpath}" \
                --pred_dataset "$TRAIN_DPATH/monitor/train/preds/$PACKAGE_NAME/pred-$PACKAGE_NAME.kwcoco.zip" \
                --window_overlap 0 \
                --clear_annots=False \
                --test_with_annot_info=True \
                --use_centered_positives=True \
                --use_grid_positives=False \
                --use_grid_negatives=False \
                --draw_batches=True \
                --devices "$DEVICE"
            ''')
    ])

    key = 'draw_vali_batches'
    script = scripts[key] = {}
    script['fpath'] = dpath / f'{key}.sh'
    script['text'] = chr(10).join([
        checkpoint_header_part,
        ub.codeblock(
            fr'''
            ### --- Validation Batch Prediction --- ###

            # Predict on the validation set
            export IGNORE_OFF_BY_ONE_STITCHING=1  # hack
            python -m geowatch.tasks.fusion.predict \
                --package_fpath "$PACKAGE_FPATH" \
                --test_dataset "{vali_coco_fpath}" \
                --pred_dataset "$TRAIN_DPATH/monitor/vali/preds/$PACKAGE_NAME/pred-$PACKAGE_NAME.kwcoco.zip" \
                --window_overlap 0 \
                --clear_annots=False \
                --test_with_annot_info=True \
                --use_centered_positives=True \
                --use_grid_positives=False \
                --use_grid_negatives=False \
                --draw_batches=True \
                --devices "$DEVICE"
            ''')
    ])

    key = 'draw_train_dataset'
    script = scripts[key] = {}
    script['fpath'] = dpath / f'{key}.sh'
    script['text'] = chr(10).join([
        checkpoint_header_part,
        ub.codeblock(
            fr'''
            ### --- Train Full-Image Prediction (best run on a GPU) --- ###

            # Predict on the training set
            export IGNORE_OFF_BY_ONE_STITCHING=1  # hack
            python -m geowatch.tasks.fusion.predict \
                --package_fpath "$PACKAGE_FPATH" \
                --window_overlap 0 \
                --test_dataset "{train_coco_fpath}" \
                --pred_dataset "$TRAIN_DPATH/monitor/train/preds/$PACKAGE_NAME/pred-$PACKAGE_NAME.kwcoco.zip" \
                --clear_annots False \
                --devices "$DEVICE"

            # Visualize train predictions
            geowatch visualize "$TRAIN_DPATH/monitor/train/preds/$PACKAGE_NAME/pred-$PACKAGE_NAME.kwcoco.zip" --smart
            ''')
    ])

    key = 'draw_vali_dataset'
    script = scripts[key] = {}
    script['fpath'] = dpath / f'{key}.sh'
    script['text'] = chr(10).join([
        checkpoint_header_part,
        ub.codeblock(
            fr'''
            ### --- Validation Full-Image Prediction (best run on a GPU) --- ###

            # Predict on the validation set
            export IGNORE_OFF_BY_ONE_STITCHING=1  # hack
            python -m geowatch.tasks.fusion.predict \
                --package_fpath "$PACKAGE_FPATH" \
                --test_dataset "{vali_coco_fpath}" \
                --pred_dataset "$TRAIN_DPATH/monitor/vali/preds/$PACKAGE_NAME/pred-$PACKAGE_NAME.kwcoco.zip" \
                --window_overlap 0 \
                --clear_annots=False \
                --devices "$DEVICE"

            # Visualize vali predictions
            geowatch visualize $TRAIN_DPATH/monitor/vali/preds/$PACKAGE_NAME/pred-$PACKAGE_NAME.kwcoco.zip --smart
            ''')
    ])
    return scripts
