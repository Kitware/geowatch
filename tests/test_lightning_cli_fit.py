r"""
python -m watch.tasks.fusion fit --help

Testing ddp

# Fit
export CUDA_VISIBLE_DEVICES=1
PHASE2_DATA_DPATH=$(smartwatch_dvc --tags="phase2_data" --hardware="ssd")
PHASE2_EXPT_DPATH=$(smartwatch_dvc --tags="phase2_expt")
DATASET_CODE=Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC
TRAIN_FNAME=combo_train_I.kwcoco.json
VALI_FNAME=combo_vali_I.kwcoco.json
TEST_FNAME=combo_vali_I.kwcoco.json
WORKDIR=$PHASE2_EXPT_DPATH/training/$HOSTNAME/$USER
KWCOCO_BUNDLE_DPATH=$PHASE2_DATA_DPATH/$DATASET_CODE
TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/$TRAIN_FNAME
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/$VALI_FNAME
TEST_FPATH=$KWCOCO_BUNDLE_DPATH/$TEST_FNAME
INITIAL_STATE=noop
EXPERIMENT_NAME=Drop4_BAS_invariants_30GSD_V016
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME
python -m watch.tasks.fusion fit \
    --trainer.default_root_dir="$DEFAULT_ROOT_DIR" \
    --data.train_dataset="$TRAIN_FPATH" \
    --data.vali_dataset="$VALI_FPATH" \
    --data.time_steps=3 \
    --data.chip_size=96 \
    --data.batch_size=2 \
    --data.input_space_scale=30GSD \
    --data.window_space_scale=30GSD \
    --data.output_space_scale=30GSD \
    --model=watch.tasks.fusion.methods.HeterogeneousModel \
    --model.name="$EXPERIMENT_NAME" \
    --optimizer=torch.optim.AdamW \
    --optimizer.lr=1e-3 \
    --trainer.accelerator="gpu" \
    --trainer.devices="0," \
    --data.channels="red|green|blue,invariants:17"

"""


def test_noop_model_training():
    from watch.tasks.fusion import fit_lightning
    import ubelt as ub
    config = ub.codeblock(
        '''
        subcommand: fit
        fit:
            model:
              class_path: NoopModel
            data:
              train_dataset: watch-msi-dates-geodata-gsize128-videos5-frames10
              vali_dataset: watch-msi-dates-geodata-gsize128-videos2-frames10
              num_workers: 0
              batch_size: 1
              time_steps: 2
              channels: B1,B10,B11
            trainer:
              max_steps: 20
        ''')
    fit_lightning.main(config=config)


def test_fit_cli_training():
    """
    smartwatch stats watch-msi-dates-geodata-gsize128-videos5-frames10
    smartwatch stats watch-msi-dates-geodata-gsize128-videos2-frames10

    python -m watch.tasks.fusion fit --model.help
    """
    import pytest
    pytest.skip()
    # Train a real model
    real_test_train()


def real_test_train():
    """
    xdoctest ~/code/watch/tests/test_lightning_cli_fit.py real_test_train
    python -c "from test_lightning_cli_fit import real_test_train; real_test_train()"
    from test_lightning_cli_fit import real_test_train
    real_test_train()
    """
    from watch.tasks.fusion import fit_lightning
    import ubelt as ub
    dpath = ub.Path.appdir('watch/tests/test_fusion_fit/demo_real').ensuredir()
    config = ub.codeblock(
        f'''
        subcommand: fit
        fit:
            data:
              train_dataset: watch-msi-dates-geodata-gsize128-videos5-frames10
              vali_dataset: watch-msi-dates-geodata-gsize128-videos2-frames10
              num_workers: 8
              batch_size: 16
              time_steps: 2
              channels: B1,B10,B11
              chip_overlap: 0.2
              chip_size: 64
              dist_weights: 0
              temporal_dropout: 0.0
              time_sampling: contiguous
              time_span: 2y
              verbose: 1
            model:
              class_path: watch.tasks.fusion.methods.HeterogeneousModel
              init_args:
                token_width: 8
                token_dim: 64
                position_encoder:
                  class_path: watch.tasks.fusion.methods.heterogeneous.MipNerfPositionalEncoder
                  init_args:
                    in_dims: 3
                    max_freq: 3
                    num_freqs: 16
                backbone:
                  class_path: watch.tasks.fusion.architectures.transformer.TransformerEncoderDecoder
                  init_args:
                    encoder_depth: 2
                    decoder_depth: 2
                    dim: 160
                    queries_dim: 96
                    logits_dim: 64
                    latent_dim_head: 256
                spatial_scale_base: 1.0
                temporal_scale_base: 1.0
                global_change_weight: 0.0
                global_class_weight: 1.0
                global_saliency_weight: 0.5
                class_loss: dicefocal
                saliency_loss: focal
                decoder: simple_conv
            optimizer:
              class_path: torch.optim.AdamW
              init_args:
                lr: 1e-4
                weight_decay: 0
            profile: false
            seed_everything: true
            trainer:
              default_root_dir: {dpath}
              # max_steps: 2000
              logger: true
              # accelerator: gpu
              # devices: 1
              # callbacks:
              #   - class_path: pytorch_lightning.callbacks.ModelCheckpoint
              #     init_args:
              #       monitor: val_class_f1_macro
              #       save_top_k: 10
              #       mode: max
              # check_val_every_n_epoch: 5
              # enable_checkpointing: true
              # enable_model_summary: true
              # enable_progress_bar: true
              # gradient_clip_algorithm: null
              # gradient_clip_val: null
              # log_every_n_steps: 50
              # num_sanity_val_steps: 2
              # replace_sampler_ddp: true
              # track_grad_norm: 2

        ''')

    fit_lightning.main(config)


def test_partial_init_callback():
    from watch.tasks.fusion import fit_lightning
    import ubelt as ub
    dpath1 = ub.Path.appdir('watch/tests/test_fusion_fit/partial_init/base_model').ensuredir()
    # Get the package we just trained and init from it
    # avail_package_fpaths = (sorted((dpath1 / 'lightning_logs/').glob('*'))

    # a = (sorted((dpath1 / 'lightning_logs/').glob('*')))
    # if len(a):
    #     avail_package_fpaths = (sorted(a[-1] / 'packages').glob('*.pt'))
    # else:
    #     avail_package_fpaths = []
    # if len(avail_package_fpaths) == 0:
    # Make an initializer package only if we need to.
    config = ub.codeblock(
        f'''
        subcommand: fit
        fit:
            model:
              class_path: watch.tasks.fusion.methods.HeterogeneousModel
            data:
              train_dataset: watch-msi-dates-geodata-gsize128-videos5-frames10
              vali_dataset: watch-msi-dates-geodata-gsize128-videos2-frames10
              num_workers: 0
              batch_size: 1
              time_steps: 2
              channels: B1,B10,B11
            optimizer:
              class_path: torch.optim.SGD
              init_args:
                lr: 1e-3
            trainer:
              default_root_dir: {dpath1}
              max_steps: 2
        ''')
    fit_lightning.main(config=config)

    avail_package_fpaths = sorted((sorted((dpath1 / 'lightning_logs/').glob('*'))[-1] / 'packages').glob('*.pt'))
    package_fpath = avail_package_fpaths[-1]

    dpath2 = ub.Path.appdir('watch/tests/test_fusion_fit/partial_init/preinit_model').ensuredir()
    config = ub.codeblock(
        f'''
        subcommand: fit
        fit:
            model:
              class_path: watch.tasks.fusion.methods.HeterogeneousModel
            data:
              train_dataset: watch-msi-dates-geodata-gsize128-videos5-frames10
              vali_dataset: watch-msi-dates-geodata-gsize128-videos2-frames10
              num_workers: 0
              batch_size: 1
              time_steps: 2
              channels: B1,B11,r|g|b
            optimizer:
              class_path: torch.optim.SGD
              init_args:
                lr: 1e-3
            trainer:
              default_root_dir: {dpath2}
              max_steps: 2
            initializer:
              init: {package_fpath}
        ''')
    cli = fit_lightning.main(config=config)
    print(f'cli={cli}')


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/tests/test_lightning_cli_fit.py
    """
    test_partial_init_callback()
