import scriptconfig as scfg
import ubelt as ub


class TrainingSummaryConfig(scfg.DataConfig):
    train_dpath = scfg.Value(None, help='path to the training directory', position=1)


def main(cmdline=1, **kwargs):
    """
    kwargs = {}
    kwargs['train_dpath'] = '/home/joncrall/remote/horologic/smart_expt_dvc/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-L8-ACC/runs/Drop4_BAS_BGR_15GSD_multihead_perceiver_V008/lightning_logs/version_0/'


    ls /home/joncrall/remote/horologic/smart_expt_dvc/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_frominvar30_V001/lightning_logs/version_1

    ls /home/joncrall/remote/horologic/smart_expt_dvc/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_scratch_V002/lightning_logs/version_0

    cmdline = 0
    """
    config = TrainingSummaryConfig.legacy(cmdline=cmdline, data=kwargs)
    train_dpath = ub.Path(config['train_dpath']).expand()

    train_batch_dpath = (train_dpath / 'monitor' / 'train' / 'batch')
    vali_batch_dpath = (train_dpath / 'monitor' / 'validate' / 'batch')

    tensorboard_dpath = (train_dpath / 'monitor' / 'tensorboard')
    train_loss_fpath = tensorboard_dpath / 'train_loss.png'
    vali_loss_fpath = tensorboard_dpath / 'val_loss.png'

    import kwimage
    print('Reading train loss image')
    train_loss_plot = kwimage.imread(train_loss_fpath)

    print('Reading vali loss image')
    vali_loss_plot = kwimage.imread(vali_loss_fpath)

    print('Stacking')
    loss_plots = kwimage.stack_images([train_loss_plot, vali_loss_plot], axis=1, pad=10)

    num = 4
    print('Read train batches')
    train_batch_fpaths = sorted(train_batch_dpath.glob('*.jpg'))[-num:]
    print('Read vali batches')
    vali_batch_fpaths = sorted(vali_batch_dpath.glob('*.jpg'))[-num:]

    train_batch_imgs = [kwimage.imread(g) for g in train_batch_fpaths]
    vali_batch_imgs = [kwimage.imread(g) for g in vali_batch_fpaths]

    train_stack_ = kwimage.stack_images(train_batch_imgs, axis=0, pad=10)
    vali_stack_ = kwimage.stack_images(vali_batch_imgs, axis=0, pad=10)

    train_stack = kwimage.draw_header_text(train_stack_, 'Train Batches', fit=True)
    vali_stack = kwimage.draw_header_text(vali_stack_, 'Validation Batches', fit=True)

    train_name = train_dpath.parent.parent.name

    outsmall_fpath = train_dpath / 'summary-small.jpg'
    print(f'Building outsmall_fpath={outsmall_fpath}')
    final_canvas_small = kwimage.stack_images([loss_plots, train_stack, vali_stack], axis=0, pad=10, resize='larger')
    final_canvas_small = kwimage.draw_header_text(final_canvas_small, train_name, fit=True)
    kwimage.imwrite(outsmall_fpath, final_canvas_small)

    outlarge_fpath = train_dpath / 'summary-large.jpg'
    print(f'Building outlarge_fpath={outlarge_fpath}')
    final_canvas_large = kwimage.stack_images([loss_plots, train_stack, vali_stack], axis=0, pad=10, resize='smaller')
    final_canvas_large = kwimage.draw_header_text(final_canvas_large, train_name, fit=True)
    kwimage.imwrite(outlarge_fpath, final_canvas_large)

    print('Finished')


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.dvc.make_training_summary_image /home/joncrall/remote/horologic/smart_expt_dvc/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_frominvar30_V001/lightning_logs/version_1

    python -m watch.dvc.make_training_summary_image /home/joncrall/remote/horologic/smart_expt_dvc/training/horologic/jon.crall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_scratch_V002/lightning_logs/version_0

    python -m watch.dvc.make_training_summary_image /home/joncrall/data/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Aligned-Drop4-2022-08-08-TA1-S2-WV-PD-ACC/runs/Drop4_SC_RGB_from_sc006_V003_cont/lightning_logs/version_2/
    """
    main()
