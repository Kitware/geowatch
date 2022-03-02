import scriptconfig as scfg
import ubelt as ub


class PrepareSplitsConfig(scfg.Config):
    """
    This generates the bash commands necessary to split a base kwcoco file into
    the standard train / validation splits.

    """
    default = {
        'base_fpath': scfg.Value(None, help='base coco file to split'),
        'virtualenv_cmd': scfg.Value(None, type=str, help=ub.paragraph(
            '''
            Command to start the appropriate virtual environment if your bashrc
            does not start it by default.''')),
        'keep_sessions': scfg.Value(False, help='if True does not close tmux sessions'),
        'run': scfg.Value(True, help='if True execute the pipeline'),
        'cache': scfg.Value(True, help='if True skip completed results'),
        'serial': scfg.Value(False, help='if True use serial mode')
    }


def main(cmdline=False, **kwargs):
    """
    The idea is that we should have a lightweight scheduler.  I think something
    fairly minimal can be implemented with tmux, but it would be nice to have a
    more robust slurm extension.

    TODO:
        - [ ] Option to just dump the serial bash script that does everything.
    """
    from watch.utils import tmux_queue

    config = PrepareSplitsConfig(cmdline=cmdline)
    config.update(kwargs)

    if config['base_fpath'] == 'auto':
        # Auto hack.
        import watch
        dvc_dpath = watch.find_smart_dvc_dpath()
        base_fpath = dvc_dpath / 'Drop2-Aligned-TA1-2022-01/data.kwcoco.json'
    else:
        base_fpath = ub.Path(config['base_fpath'])

    splits = {
        'nowv': base_fpath.augment(suffix='_nowv', multidot=True),

        'train': base_fpath.augment(suffix='_train', multidot=True),
        'nowv_train': base_fpath.augment(suffix='_nowv_train', multidot=True),
        'wv_train': base_fpath.augment(suffix='_wv_train', multidot=True),

        'vali': base_fpath.augment(suffix='_vali', multidot=True),
        'nowv_vali': base_fpath.augment(suffix='_nowv_vali', multidot=True),
        'wv_vali': base_fpath.augment(suffix='_wv_vali', multidot=True),
    }
    print('splits = {!r}'.format(splits))

    tq = tmux_queue.TMUXMultiQueue(name='watch-splits', size=2)
    if config['virtualenv_cmd']:
        tq.add_header_command(config['virtualenv_cmd'])

    # Perform train/validation splits with and without worldview
    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {base_fpath} \
            --dst {splits['train']} \
            --select_videos '.name | startswith("KR_") | not'
        ''')
    tq.submit(command, index=0)

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['train']} \
            --dst {splits['nowv_train']} \
            --select_images '.sensor_coarse != "WV"'
        ''')
    tq.submit(command, index=0)

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['train']} \
            --dst {splits['wv_train']} \
            --select_images '.sensor_coarse == "WV"'
        ''')
    tq.submit(command, index=0)

    # Perform vali/validation splits with and without worldview
    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {base_fpath} \
            --dst {splits['vali']} \
            --select_videos '.name | startswith("KR_")'
        ''')
    tq.submit(command, index=1)

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['vali']} \
            --dst {splits['nowv_vali']} \
            --select_images '.sensor_coarse != "WV"'
        ''')
    tq.submit(command, index=1)

    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {splits['vali']} \
            --dst {splits['wv_vali']} \
            --select_images '.sensor_coarse == "WV"'
        ''')
    tq.submit(command, index=1)

    # Add in additional no-worldview full dataset
    command = ub.codeblock(
        fr'''
        python -m kwcoco subset \
            --src {base_fpath} \
            --dst {splits['nowv']} \
            --select_images '.sensor_coarse != "WV"'
        ''')
    tq.submit(command, index=1)

    tq.rprint()

    if config['run']:
        if config['serial']:
            tq.serial_run()
        else:
            tq.run()
        agg_state = tq.monitor()
        if not config['keep_sessions']:
            if not agg_state['errored']:
                tq.kill()


if __name__ == '__main__':
    """
    CommandLine:
        DVC_DPATH=$(python -m watch.cli.find_dvc)
        python -m watch.cli.prepare_splits \
            --base_fpath=$DVC_DPATH/Drop2-Aligned-TA1-2022-02-15/data.kwcoco.json \
            --run=0 --serial=True
    """
    main(cmdline=True)
