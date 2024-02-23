#!/usr/bin/env python3
r"""
Parses an existing tensorboard event file and draws the plots as pngs on disk
in the monitor/tensorboard directory.

Derived from netharn/mixins.py for dumping tensorboard plots to disk

CommandLine:
    # cd into training directory
    WATCH_PREIMPORT=0 python -m geowatch.utils.lightning_ext.callbacks.tensorboard_plotter .

    python -m geowatch.utils.lightning_ext.callbacks.tensorboard_plotter \
        /data/joncrall/dvc-repos/smart_expt_dvc/training/toothbrush/joncrall/Drop6/runs/Drop6_BAS_scratch_landcover_10GSD_split2_V4/lightning_logs/version_4/

"""
import scriptconfig as scfg
import os
import ubelt as ub
from pytorch_lightning.callbacks import Callback


__all__ = ['TensorboardPlotter']


# TODO: can move the callback to its own file and have the CLI variant with
# core logic live separately for faster response times when using the CLI (i.e.
# avoid lightning import overhead).
class TensorboardPlotter(Callback):
    """
    Asynchronously dumps PNGs to disk visualize tensorboard scalars.
    exit

    CommandLine:
        xdoctest -m geowatch.utils.lightning_ext.callbacks.tensorboard_plotter TensorboardPlotter

    Example:
        >>> # xdoctest: +REQUIRES(module:tensorboard)
        >>> from geowatch.utils.lightning_ext import demo
        >>> from geowatch.monkey import monkey_lightning
        >>> import pytorch_lightning as pl
        >>> import pandas as pd
        >>> monkey_lightning.disable_lightning_hardware_warnings()
        >>> self = demo.LightningToyNet2d(num_train=55)
        >>> default_root_dir = ub.Path.appdir('lightning_ext/tests/TensorboardPlotter').ensuredir()
        >>> #
        >>> trainer = pl.Trainer(callbacks=[TensorboardPlotter()],
        >>>                      default_root_dir=default_root_dir,
        >>>                      max_epochs=3, accelerator='cpu', devices=1)
        >>> trainer.fit(self)
        >>> train_dpath = trainer.logger.log_dir
        >>> print('trainer.logger.log_dir = {!r}'.format(train_dpath))
        >>> data = read_tensorboard_scalars(train_dpath)
        >>> for key in data.keys():
        >>>     d = data[key]
        >>>     df = pd.DataFrame({key: d['ydata'], 'step': d['xdata'], 'wall': d['wall']})
        >>>     print(df)
    """

    def _on_epoch_end(self, trainer, logs=None, serial=False):
        # The following function draws the tensorboard result. This might take
        # a some non-trivial amount of time so we attempt to run in a separate
        # process.
        from kwutil import util_environ
        if util_environ.envflag('DISABLE_TENSORBOARD_PLOTTER'):
            return

        if trainer.global_rank != 0:
            return

        # train_dpath = trainer.logger.log_dir
        train_dpath = trainer.log_dir
        if train_dpath is None:
            import warnings
            warnings.warn('The trainer logdir is not set. Cannot dump a batch plot')
            return

        func = _dump_measures

        model = trainer.model
        # TODO: get step number
        if hasattr(model, 'get_cfgstr'):
            model_cfgstr = model.get_cfgstr()
        else:
            from geowatch.utils.lightning_ext import util_model
            from kwutil.slugify_ext import smart_truncate
            hparams = util_model.model_hparams(model)
            model_config = {
                'type': str(model.__class__),
                'hp': smart_truncate(ub.urepr(hparams, compact=1, nl=0), max_length=8),
            }
            model_cfgstr = smart_truncate(ub.urepr(
                model_config, compact=1, nl=0), max_length=64)

        args = (train_dpath, model_cfgstr)

        proc_name = 'dump_tensorboard'

        if not serial:
            # This causes thread-unsafe warning messages in the inner loop
            # Likely because we are forking while a thread is alive
            if not hasattr(trainer, '_internal_procs'):
                trainer._internal_procs = ub.ddict(dict)

            # Clear finished processes from the pool
            for pid in list(trainer._internal_procs[proc_name].keys()):
                proc = trainer._internal_procs[proc_name][pid]
                if not proc.is_alive():
                    trainer._internal_procs[proc_name].pop(pid)

            # only start a new process if there is room in the pool
            if len(trainer._internal_procs[proc_name]) < 1:
                import multiprocessing
                proc = multiprocessing.Process(target=func, args=args)
                proc.daemon = True
                proc.start()
                trainer._internal_procs[proc_name][proc.pid] = proc
            else:
                # Draw is already in progress
                pass
        else:
            func(*args)

    def on_train_epoch_end(self, trainer, logs=None):
        return self._on_epoch_end(trainer, logs=logs)

    def on_validation_epoch_end(self, trainer, logs=None):
        return self._on_epoch_end(trainer, logs=logs)

    def on_test_epoch_end(self, trainer, logs=None):
        return self._on_epoch_end(trainer, logs=logs)


def read_tensorboard_scalars(train_dpath, verbose=1, cache=1):
    """
    Reads all tensorboard scalar events in a directory.
    Caches them because reading events of interest from protobuf can be slow.

    Ignore:
        train_dpath = '/home/joncrall/.cache/lightning_ext/tests/TensorboardPlotter/lightning_logs/version_2'
        tb_data = read_tensorboard_scalars(train_dpath)
    """
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError('tensorboard/tensorflow is not installed')
    train_dpath = ub.Path(train_dpath)
    event_paths = sorted(train_dpath.glob('events.out.tfevents*'))
    # make a hash so we will re-read of we need to
    cfgstr = ub.hash_data(list(map(ub.hash_file, event_paths))) if cache else ''
    cacher = ub.Cacher('tb_scalars', depends=cfgstr, enabled=cache,
                       dpath=train_dpath / '_cache')
    datas = cacher.tryload()
    if datas is None:
        datas = {}
        for p in ub.ProgIter(list(reversed(event_paths)), desc='read tensorboard',
                             enabled=verbose, verbose=verbose * 3):
            p = os.fspath(p)
            if verbose:
                print('reading tensorboard scalars')
            ea = event_accumulator.EventAccumulator(p)
            if verbose:
                print('loading tensorboard scalars')
            ea.Reload()
            if verbose:
                print('iterate over scalars')
            for key in ea.scalars.Keys():
                if key not in datas:
                    datas[key] = {'xdata': [], 'ydata': [], 'wall': []}
                subdatas = datas[key]
                events = ea.scalars.Items(key)
                for e in events:
                    subdatas['xdata'].append(int(e.step))
                    subdatas['ydata'].append(float(e.value))
                    subdatas['wall'].append(float(e.wall_time))

        # Order all information by its wall time
        for _key, subdatas in datas.items():
            sortx = ub.argsort(subdatas['wall'])
            for d, vals in subdatas.items():
                subdatas[d] = list(ub.take(vals, sortx))
        cacher.save(datas)
    return datas


def _write_helper_scripts(out_dpath, train_dpath):
    """
    Writes scripts to let the user refresh data on the fly
    """
    from geowatch.utils import util_chmod

    train_dpath_ = train_dpath.resolve().shrinkuser()

    # TODO: make this a nicer python script that aranges figures nicely.
    stack_fpath = (out_dpath / 'stack.sh')
    stack_fpath.write_text(ub.codeblock(
        fr'''
        #!/usr/bin/env bash
        kwimage stack_images --out "{train_dpath_}/monitor/tensorboard-stack.png" -- {train_dpath_}/monitor/tensorboard/*.png
        '''))
    util_chmod.new_chmod(stack_fpath, 'ug+x')

    refresh_fpath = (out_dpath / 'redraw.sh')
    refresh_fpath.write_text(ub.codeblock(
        fr'''
        #!/usr/bin/env bash
        WATCH_PREIMPORT=0 python -m geowatch.utils.lightning_ext.callbacks.tensorboard_plotter \
            {train_dpath_}
        '''))
    util_chmod.new_chmod(refresh_fpath, 'ug+x')
    # import stat
    # refresh_fpath.chmod(refresh_fpath.stat().st_mode | stat.S_IEXEC)


def _dump_measures(train_dpath, title='?name?', smoothing='auto', ignore_outliers=True, verbose=0):
    """
    This is its own function in case we need to modify formatting
    """
    import kwplot
    from kwplot.auto_backends import BackendContext
    import pandas as pd
    import numpy as np

    train_dpath = ub.Path(train_dpath).resolve()
    if not train_dpath.name.startswith('version_'):
        # hack
        if not (train_dpath / 'monitor').exists():
            if (train_dpath / '../monitor').exists():
                train_dpath = (train_dpath / '..')
            elif (train_dpath / '../../monitor').exists():
                train_dpath = (train_dpath / '../..')

    tb_data = read_tensorboard_scalars(train_dpath, cache=0, verbose=verbose)

    out_dpath = ub.Path(train_dpath, 'monitor', 'tensorboard').ensuredir()
    _write_helper_scripts(out_dpath, train_dpath)

    if isinstance(smoothing, str) and smoothing == 'auto':
        smoothing_values = [0.6, 0.95]
    elif isinstance(smoothing, list):
        smoothing_values = [smoothing]
    else:
        smoothing_values = [smoothing]

    plot_keys = [k for k in tb_data.keys() if '/' not in k]
    # print(f'plot_keys={plot_keys}')
    y01_measures = [
        '_acc', '_ap', '_mAP', '_auc', '_mcc', '_brier', '_mauc',
        '_f1', '_iou',
    ]
    y0_measures = ['error', 'loss']

    keys = set(tb_data.keys()).intersection(set(plot_keys))
    # print(f'keys={keys}')
    # unused = set(tb_data.keys()) - set(keys)

    # no idea what hp metric is, but it doesn't seem important
    keys = keys - {'hp_metric'}

    HACK_NO_SMOOTH = {'lr', 'momentum', 'epoch'}

    if len(keys) == 0:
        print('warning: no known keys to plot')
        print(f'available keys: {list(tb_data.keys())}')

    with BackendContext('agg'):
        import seaborn as sns
        sns.set()
        # meta = tb_data.get('meta', {})
        # nice = meta.get('name', '?name?')
        nice = title
        fig = kwplot.figure(fnum=1)
        fig.clf()
        ax = fig.gca()

        # import kwimage
        # color1 = kwimage.Color('kw_green').as01()
        # color2 = kwimage.Color('kw_green').as01()
        prog = ub.ProgIter(keys, desc='dump plots', verbose=verbose * 3)
        for key in prog:
            prog.set_extra(key)
            snskw = {
                'y': key,
                'x': 'step',
            }

            d = tb_data[key]
            df_orig = pd.DataFrame({key: d['ydata'], 'step': d['xdata']})
            num_non_nan = (~df_orig[key].isnull()).sum()
            num_nan = (df_orig[key].isnull()).sum()
            df_orig['smoothing'] = 0.0
            variants = [df_orig]
            if key not in HACK_NO_SMOOTH and smoothing_values:
                for _smoothing_value in smoothing_values:
                    if 0:
                        # TODO: can we get a hueristic for how much smoothing
                        # we might want? Look at the entropy of the derivative
                        # curve?
                        import scipy.stats
                        deriv = np.diff(df_orig[key])
                        counts1, bins1 = np.histogram(deriv[deriv < 0], bins=25)
                        counts2, bins2 = np.histogram(deriv[deriv >= 0], bins=25)
                        counts = np.hstack([counts1, counts2])
                        # bins = np.hstack([bins1, bins2])
                        # dict(zip(bins, counts))
                        entropy = scipy.stats.entropy(counts)
                        print(f'entropy={entropy}')

                    if _smoothing_value > 0:
                        df_smooth = df_orig.copy()
                        beta = _smoothing_value
                        ydata = df_orig[key]
                        df_smooth[key] = smooth_curve(ydata, beta)
                        df_smooth['smoothing'] = _smoothing_value
                        variants.append(df_smooth)

            if len(variants) == 1:
                df = variants[0]
            else:
                if verbose:
                    print('Combine smoothed variants')
                df = pd.concat(variants).reset_index()
                snskw['hue'] = 'smoothing'

            kw = {}
            if any(m.lower() in key.lower() for m in y01_measures):
                kw['ymin'] = 0.0
                kw['ymax'] = 1.0
            elif any(m.lower() in key.lower() for m in y0_measures):
                ydata = df[key]
                kw['ymin'] = min(0.0, ydata.min())
                if ignore_outliers and num_non_nan > 3:
                    if verbose:
                        print('Finding outliers')
                    low, kw['ymax'] = tensorboard_inlier_ylim(ydata)

            if verbose:
                print('Begin plot')
            # NOTE: this is actually pretty slow
            ax.cla()
            try:
                if num_non_nan <= 1:
                    sns.scatterplot(data=df, **snskw)
                else:
                    sns.lineplot(data=df, **snskw)
            except Exception as ex:
                title = nice + '\n' + key + str(ex)
            else:
                title = nice + '\n' + key
                initial_ylim = ax.get_ylim()
                if kw.get('ymax', None) is None:
                    kw['ymax'] = initial_ylim[1]
                if kw.get('ymin', None) is None:
                    kw['ymin'] = initial_ylim[0]
                try:
                    ax.set_ylim(kw['ymin'], kw['ymax'])
                except Exception:
                    ...
            if num_nan > 0:
                title += '(num_nan={})'.format(num_nan)

            ax.set_title(title)

            # png is smaller than jpg for this kind of plot
            fpath = out_dpath / (key + '.png')
            if verbose:
                print('Save plot: ' + str(fpath))
            ax.figure.savefig(fpath)
            ax.figure.subplots_adjust(top=0.8)


def smooth_curve(ydata, beta):
    """
    Curve smoothing algorithm used by tensorboard
    """
    import pandas as pd
    alpha = 1.0 - beta
    if alpha <= 0:
        return ydata
    ydata_smooth = pd.Series(ydata).ewm(alpha=alpha).mean().values
    return ydata_smooth


# def inlier_ylim(ydata):
#     """
#     outlier removal used by tensorboard
#     """
#     import kwarray
#     normalizer = kwarray.find_robust_normalizers(ydata, {
#         'low': 0.05,
#         'high': 0.95,
#     })
#     low = normalizer['min_val']
#     high = normalizer['max_val']
#     return (low, high)


def tensorboard_inlier_ylim(ydata):
    """
    outlier removal used by tensorboard
    """
    import numpy as np
    q1 = 0.05
    q2 = 0.95
    low_, high_ = np.quantile(ydata, [q1, q2])

    # Extrapolate how big the entire span should be based on inliers
    inner_q = q2 - q1
    inner_extent = high_ - low_
    extrap_total_extent = inner_extent  / inner_q

    # amount of padding to add to either side
    missing_p1 = q1
    missing_p2 = 1 - q2
    frac1 = missing_p1 / (missing_p2 + missing_p1)
    frac2 = missing_p2 / (missing_p2 + missing_p1)
    missing_extent = extrap_total_extent - inner_extent

    pad1 = missing_extent * frac1
    pad2 = missing_extent * frac2

    low = low_ - pad1
    high = high_ + pad2
    return (low, high)


def redraw_cli(train_dpath):
    """
    Create png plots for the tensorboard data in a training directory.
    """
    from kwutil.util_yaml import Yaml
    train_dpath = ub.Path(train_dpath)

    expt_name = train_dpath.parent.parent.name

    hparams_fpath = train_dpath / 'hparams.yaml'
    if hparams_fpath.exists():
        print('Found hparams')
        hparams = Yaml.load(hparams_fpath)
        if 'name' in hparams:
            title = hparams['name']
        else:
            from kwutil.slugify_ext import smart_truncate
            model_config = {
                # 'type': str(model.__class__),
                'hp': smart_truncate(ub.urepr(hparams, compact=1, nl=0), max_length=8),
            }
            model_cfgstr = smart_truncate(ub.urepr(
                model_config, compact=1, nl=0), max_length=64)
            title = model_cfgstr
        title = expt_name + '\n' + title
    else:
        print('Did not find hparams')
        title = expt_name

    if 1:
        # Add in other relevant data
        # ...
        config_fpath = train_dpath / 'config.yaml'
        if config_fpath.exists():

            config = Yaml.load(config_fpath)
            trainer_config = config.get('trainer', {})
            optimizer_config = config.get('optimizer', {})
            data_config = config.get('data', {})
            optimizer_args = optimizer_config.get('init_args', {})

            devices = trainer_config.get('devices', None)

            batch_size = data_config.get('batch_size', None)
            accum_batches = trainer_config.get('accumulate_grad_batches', None)
            optim_lr = optimizer_args.get('lr', None)
            decay = optimizer_args.get('weight_decay', None)
            # optim_name = optimizer_config.get('class_path', '?').split('.')[-1]
            learn_dynamics_str = ub.codeblock(
                f'''
                BS=({batch_size} x {accum_batches}), LR={optim_lr}, decay={decay}, devs={devices}
                '''
            )
            title = title + '\n' + learn_dynamics_str
            # print(learn_dynamics_str)

    print(f'train_dpath={train_dpath}')
    print(f'title={title}')
    _dump_measures(train_dpath, title, verbose=1)
    import rich
    tensorboard_dpath = train_dpath / 'monitor/tensorboard'
    rich.print(f'[link={tensorboard_dpath}]{tensorboard_dpath}[/link]')


class TensorboardPlotterCLI(scfg.DataConfig):
    """
    Helper CLI executable to redraw on demand.
    """
    train_dpath = scfg.Value('.', help='train_dpath', position=1)

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        import rich
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1))
        redraw_cli(config.train_dpath)


if __name__ == '__main__':
    """
    CommandLine:
        WATCH_PREIMPORT=0 python -X importtime -m geowatch.utils.lightning_ext.callbacks.tensorboard_plotter .
    """
    TensorboardPlotterCLI.main()
    # import fire
    # fire.Fire(redraw_cli)
