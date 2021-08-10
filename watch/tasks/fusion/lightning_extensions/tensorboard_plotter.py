"""
Derived from netharn/mixins.py for dumping tensorboard plots to disk
"""
# from distutils.version import LooseVersion
import ubelt as ub
import numpy as np
from os.path import join
import pandas as pd
import pytorch_lightning as pl
import netharn as nh
import torch


#
# TODO: expose as a toydata module
class LightningToyNet2d(pl.LightningModule):
    def __init__(self, num_train=100, num_val=10, batch_size=4):
        super().__init__()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size
        self.model = nh.models.ToyNet2d()

    def forward(self, x):
        return self.model(x)

    def forward_step(self, batch, batch_idx):
        if self.trainer is None:
            stage = 'disconnected'
        else:
            stage = self.trainer.state.stage.lower()
        inputs, targets = batch
        logits = self.forward(inputs)
        loss = torch.nn.functional.nll_loss(logits.log_softmax(dim=1), targets)
        # https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html
        self.log(f'{stage}_loss', loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx)

    def train_dataloader(self):
        dataset = nh.data.toydata.ToyData2d(n=self.num_train)
        loader = dataset.make_loader(batch_size=self.batch_size)
        return loader

    def val_dataloader(self):
        dataset = nh.data.toydata.ToyData2d(n=self.num_val)
        loader = dataset.make_loader(batch_size=self.batch_size)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


class TensorboardPlotter(pl.callbacks.Callback):
    """
    Asynchronously dumps PNGs to disk visualize tensorboard scalars.

    CommandLine:
        xdoctest -m watch.tasks.fusion.lightning_extensions.tensorboard_plotter TensorboardPlotter

    Example:
        >>> #
        >>> self = LightningToyNet2d(num_train=55)
        >>> default_root_dir = ub.ensure_app_cache_dir('lightning_ext/tests/TensorboardPlotter')
        >>> #
        >>> trainer = pl.Trainer(callbacks=[TensorboardPlotter()],
        >>>                      default_root_dir=default_root_dir,
        >>>                      max_epochs=10)
        >>> trainer.fit(self)
        >>> train_dpath = trainer.logger.log_dir
        >>> print('trainer.logger.log_dir = {!r}'.format(train_dpath))
        >>> data = read_tensorboard_scalars(train_dpath)
        >>> for key in data.keys():
        >>>     d = data[key]
        >>>     df = pd.DataFrame({key: d['ydata'], 'step': d['xdata'], 'wall': d['wall']})
        >>>     print(df)
    """

    def on_epoch_end(self, trainer, logs=None):
        # The following function draws the tensorboard result. This might take
        # a some non-trivial amount of time so we attempt to run in a separate
        # process.
        serial = False

        train_dpath = trainer.logger.log_dir

        func = _dump_measures
        args = (train_dpath,)

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
    cacher = ub.Cacher('tb_scalars', cfgstr=cfgstr, enabled=cache,
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
        for key, subdatas in datas.items():
            sortx = ub.argsort(subdatas['wall'])
            for d, vals in subdatas.items():
                subdatas[d] = list(ub.take(vals, sortx))
        cacher.save(datas)
    return datas


def _dump_measures(train_dpath, smoothing=0.0, ignore_outliers=True):
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
        meta = tb_data.get('meta', {})
        nice = meta.get('name', '?name?')
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
