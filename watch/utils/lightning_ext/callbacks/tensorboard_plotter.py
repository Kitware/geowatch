"""
Derived from netharn/mixins.py for dumping tensorboard plots to disk
"""
# from distutils.version import LooseVersion
import ubelt as ub
import numpy as np
from os.path import join
import pandas as pd
import pytorch_lightning as pl
from packaging.version import parse as Version
PL_VERSION = Version(pl.__version__)


__all__ = ['TensorboardPlotter']


class TensorboardPlotter(pl.callbacks.Callback):
    """
    Asynchronously dumps PNGs to disk visualize tensorboard scalars.

    CommandLine:
        xdoctest -m watch.utils.lightning_ext.callbacks.tensorboard_plotter TensorboardPlotter

    Example:
        >>> #
        >>> from watch.utils.lightning_ext import demo
        >>> self = demo.LightningToyNet2d(num_train=55)
        >>> default_root_dir = ub.ensure_app_cache_dir('lightning_ext/tests/TensorboardPlotter')
        >>> #
        >>> trainer = pl.Trainer(callbacks=[TensorboardPlotter()],
        >>>                      default_root_dir=default_root_dir,
        >>>                      max_epochs=3)
        >>> trainer.fit(self)
        >>> train_dpath = trainer.logger.log_dir
        >>> print('trainer.logger.log_dir = {!r}'.format(train_dpath))
        >>> data = read_tensorboard_scalars(train_dpath)
        >>> for key in data.keys():
        >>>     d = data[key]
        >>>     df = pd.DataFrame({key: d['ydata'], 'step': d['xdata'], 'wall': d['wall']})
        >>>     print(df)
    """

    def _on_epoch_end(self, trainer, logs=None):
        # The following function draws the tensorboard result. This might take
        # a some non-trivial amount of time so we attempt to run in a separate
        # process.
        serial = False

        train_dpath = trainer.logger.log_dir

        func = _dump_measures

        model = trainer.model
        # TODO: get step number
        if hasattr(model, 'get_cfgstr'):
            model_cfgstr = model.get_cfgstr()
        else:
            from watch.utils.slugify_ext import smart_truncate
            model_config = {
                'type': str(model.__class__),
                'hp': smart_truncate(ub.repr2(model.hparams, compact=1, nl=0), max_length=8),
            }
            model_cfgstr = smart_truncate(ub.repr2(
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
                # if 0:
                #     harn.warn('NOT DOING MPL DRAW')
        else:
            func(*args)

    if PL_VERSION < Version('1.6'):
        def on_epoch_end(self, trainer, logs=None):
            return self._on_epoch_end(trainer, logs=logs)
    else:
        def on_train_epoch_end(self, trainer, logs=None):
            return self._on_epoch_end(trainer, logs=logs)

        def on_validation_epoch_end(self, trainer, logs=None):
            return self._on_epoch_end(trainer, logs=logs)

        def on_test_epoch_end(self, trainer, logs=None):
            return self._on_epoch_end(trainer, logs=logs)


def read_tensorboard_scalars(train_dpath, verbose=1, cache=1):
    """
    Reads all tensorboard scalar events in a directory.
    Caches them becuase reading events of interest from protobuf can be slow.

    Ignore:
        train_dpath = '/home/joncrall/.cache/lightning_ext/tests/TensorboardPlotter/lightning_logs/version_2'
        tb_data = read_tensorboard_scalars(train_dpath)
    """
    import glob
    from os.path import join
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except ImportError:
        raise ImportError('tensorboard/tensorflow is not installed')
    event_paths = sorted(glob.glob(join(train_dpath, 'events.out.tfevents*')))
    # make a hash so we will re-read of we need to
    cfgstr = ub.hash_data(list(map(ub.hash_file, event_paths))) if cache else ''
    cacher = ub.Cacher('tb_scalars', depends=cfgstr, enabled=cache,
                       dpath=join(train_dpath, '_cache'))
    datas = cacher.tryload()
    if datas is None:
        datas = {}
        for p in ub.ProgIter(list(reversed(event_paths)), desc='read tensorboard', enabled=verbose):
            ea = event_accumulator.EventAccumulator(p)
            ea.Reload()
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


def _dump_measures(train_dpath, title='?name?', smoothing=0.0, ignore_outliers=True):
    """
    This is its own function in case we need to modify formatting
    """
    import kwplot
    from kwplot.auto_backends import BackendContext

    out_dpath = ub.ensuredir((train_dpath, 'monitor', 'tensorboard'))
    tb_data = read_tensorboard_scalars(train_dpath, cache=0, verbose=0)

    with BackendContext('agg'):
        import seaborn as sns
        sns.set()
        # meta = tb_data.get('meta', {})
        # nice = meta.get('name', '?name?')
        nice = title
        fig = kwplot.figure(fnum=1)
        fig.clf()
        ax = fig.gca()

        mode = ''
        plot_keys = [k for k in tb_data.keys() if '/' not in k]
        # plot_keys = [key for key in tb_data if
        #              ('train_' + mode in key or
        #               'val_' + mode in key or
        #               'test_' + mode in key)]
        y01_measures = [
            '_acc', '_ap', '_mAP', '_auc', '_mcc', '_brier', '_mauc',
            '_f1', '_iou',
        ]
        y0_measures = ['error', 'loss']

        keys = set(tb_data.keys()).intersection(set(plot_keys))

        def tag_grouper(k):
            # parts = ['train_epoch', 'vali_epoch', 'test_epoch']
            # parts = [p.replace('epoch', 'mode') for p in parts]
            parts = [p + mode for p in ['train_', 'vali_', 'test_']]
            for p in parts:
                if p in k:
                    return p.split('_')[0]
            return 'unknown'

        INDIVIDUAL_PLOTS = True

        HACK_NO_SMOOTH = {'lr', 'momentum', 'epoch'}

        if INDIVIDUAL_PLOTS:
            # print('keys = {!r}'.format(keys))
            for key in keys:
                d = tb_data[key]
                df = pd.DataFrame({key: d['ydata'], 'step': d['xdata']})
                if key not in HACK_NO_SMOOTH:
                    df[key] = smooth_curve(df[key], smoothing)

                kw = {}
                if any(m.lower() in key.lower() for m in y01_measures):
                    kw['ymin'] = 0.0
                    kw['ymax'] = 1.0
                elif any(m.lower() in key.lower() for m in y0_measures):
                    kw['ymin'] = min(0.0, df[key].min())
                    if ignore_outliers:
                        low, kw['ymax'] = inlier_ylim([df[key]])

                # NOTE: this is actually pretty slow
                ax.cla()

                sns.lineplot(data=df, x='step', y=key)
                title = nice + '\n' + key
                ax.set_title(title)

                # png is smaller than jpg for this kind of plot
                fpath = join(out_dpath, key + '.png')
                ax.figure.savefig(fpath)


def smooth_curve(ydata, beta):
    """
    Curve smoothing algorithm used by tensorboard
    """
    alpha = 1.0 - beta
    if alpha <= 0:
        return ydata
    ydata_smooth = pd.Series(ydata).ewm(alpha=alpha).mean().values
    return ydata_smooth


def inlier_ylim(ydatas):
    """
    outlier removal used by tensorboard
    """
    low, high = None, None
    for ydata in ydatas:
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

        low_ = low_ - pad1
        high_ = high_ + pad2

        low = low_ if low is None else min(low_, low)
        high = high_ if high is None else max(high_, high)
    return (low, high)
