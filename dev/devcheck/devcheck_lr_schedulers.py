"""
Exploratory script to gain understanding of excatly how things like batch size,
dataset length, accum grad batches, etc interact.
"""

import kwutil
import pytorch_lightning as pl
import rich
import scriptconfig
import torch
import torch.nn
import torch.utils.data
import ubelt as ub
import yaml
from torch.utils.data import Dataset


class RelevantConfig(scriptconfig.DataConfig):
    # MAX_STEPS               = 101
    # MAX_EPOCHS              = 197
    # BATCH_SIZE              = 5
    # ACCUMULATE_GRAD_BATCHES = 3
    # TRAIN_ITEMS_PER_EPOCH         = 17

    # Ideal divisibility variant
    # MAX_STEPS               = 400
    # MAX_EPOCHS              = 20
    # BATCH_SIZE              = 5
    # ACCUMULATE_GRAD_BATCHES = 3
    # TRAIN_ITEMS_PER_EPOCH   = 15 * 20

    if 1:
        # Prime number variant
        MAX_STEPS               = 907
        MAX_EPOCHS              = 107
        BATCH_SIZE              = 3
        ACCUMULATE_GRAD_BATCHES = 11
        TRAIN_ITEMS_PER_EPOCH   = 313

    # EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATE_GRAD_BATCHES
    # STEPS_PER_EPOCH = int(TRAIN_ITEMS_PER_EPOCH / EFFECTIVE_BATCH_SIZE)

# We will track the real number of steps used
MEASURED_COUNTS = dict(
    NUM_LR_STEPS=0,
    NUM_OPTIM_STEPS=0,
    NUM_TRAINING_STEPS=0,
    NUM_VALI_STEPS=0,
    NUM_GETITEM_CALLS=0,
    NUM_BATCHES=0,
    NUM_EPOCHS=0,
)


def jsonargparse_yaml_workarounds():
    from jsonargparse import set_loader, set_dumper
    # Not very safe, but needed to parse tuples
    # TODO: yaml.SafeLoader + tuple parsing

    def custom_yaml_load(stream):
        return yaml.load(stream, Loader=yaml.FullLoader)

    def custom_yaml_dump(data):
        return yaml.dump(data, Dumper=yaml.Dumper)

    set_loader('yaml_unsafe_for_tuples', custom_yaml_load)
    set_dumper('yaml_unsafe_for_tuples', custom_yaml_dump)


class CustomDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        MEASURED_COUNTS['NUM_EPOCHS'] += 1
        for batch in super().__iter__():
            MEASURED_COUNTS['NUM_BATCHES'] += 1
            yield batch


class CustomAdamW(torch.optim.AdamW):

    def step(self, *args, **kwargs):
        MEASURED_COUNTS['NUM_OPTIM_STEPS'] += 1
        return super().step(*args, **kwargs)


class CustomOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):

    def step(self, *args, **kwargs):
        MEASURED_COUNTS['NUM_LR_STEPS'] += 1
        return super().step(*args, **kwargs)


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleDict()
        self.layers['layer1'] = torch.nn.Conv2d(2, 3, 1, 1)
        # self.layers['layer2'] = torch.nn.Conv2d(3, 5, 1, 1)

    def forward(self, inputs):
        x = inputs
        x = self.layers['layer1'](x)
        # x = self.layers['layer2'](x)
        return x

    def forward_step(self, batch):
        """
        Generic forward step used for test / train / validation
        """
        batch = torch.stack(batch, dim=0)
        x = self.forward(batch)
        loss = x.sum()
        return loss

    def training_step(self, batch, batch_idx=None):
        MEASURED_COUNTS['NUM_TRAINING_STEPS'] += 1
        outputs = self.forward_step(batch)
        return outputs

    def validation_step(self, batch, batch_idx=None):
        MEASURED_COUNTS['NUM_VALI_STEPS'] += 1
        outputs = self.forward_step(batch)
        return outputs


class SimpleDataset(Dataset):
    def __init__(self, max_epoch_length=100):
        self.max_epoch_length = max_epoch_length

    def __len__(self):
        return self.max_epoch_length

    def __getitem__(self, index):
        MEASURED_COUNTS['NUM_GETITEM_CALLS'] += 1
        return torch.rand(2, 1, 1)


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=0, max_epoch_length=100):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        commonkw = ub.compatible(self.hparams, SimpleDataset.__init__)
        trainkw = commonkw.copy()
        valikw = commonkw.copy()
        self.train_dataset = SimpleDataset(**trainkw)
        self.vali_dataset = SimpleDataset(**valikw)

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.vali_dataset, shuffle=False)

    def _make_dataloader(self, dataset, shuffle=False):
        # loader = torch.utils.data.DataLoader(
        loader = CustomDataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle, pin_memory=True,
            collate_fn=lambda x: x
        )
        return loader


def inspect_relevant_interactions(relevant):
    import sympy
    import ubelt as ub
    symbolic_names = 'TRAIN_ITEMS_PER_EPOCH, BATCH_SIZE, ACCUMULATE_GRAD_BATCHES, MAX_EPOCHS, MAX_STEPS'.split(', ')
    # symbolic_vars = sympy.symbols(symbolic_names, integer=True, positive=True)
    symbolic_vars = sympy.symbols(symbolic_names)
    TRAIN_ITEMS_PER_EPOCH, BATCH_SIZE, ACCUMULATE_GRAD_BATCHES, MAX_EPOCHS, MAX_STEPS = symbolic_vars

    rich.print('[white]---------------------------')
    rich.print('[white]Check Relevant Interactions')
    rich.print('[white]---------------------------')
    rich.print(f'relevant = {ub.urepr(relevant, nl=1, align=":")}')

    # Build substitution dictionary for sympy
    subs = ub.dzip(symbolic_vars, ub.udict(relevant).take(symbolic_names))

    effective_batch_size = ACCUMULATE_GRAD_BATCHES * BATCH_SIZE
    steps_per_epoch = TRAIN_ITEMS_PER_EPOCH / effective_batch_size
    # This next line is more correct, but prevents the symbolic solver from
    # working. Can uncomment if we fixup the numeric solver to work better.
    # steps_per_epoch = sympy.floor(TRAIN_ITEMS_PER_EPOCH / effective_batch_size)
    total_steps = MAX_EPOCHS * steps_per_epoch
    total_steps.subs(subs)

    steps_per_epoch_ = steps_per_epoch.subs(subs).evalf()
    effective_batch_size_ = effective_batch_size.subs(subs).evalf()

    # The training progress iterator should show this number as the total number
    import math
    train_epoch_progbar_total_ = math.ceil((TRAIN_ITEMS_PER_EPOCH / BATCH_SIZE).subs(subs).evalf())

    print(f'steps_per_epoch_           = {steps_per_epoch_}')
    print(f'effective_batch_size_      = {effective_batch_size_}')
    print(f'train_epoch_progbar_total_ = {train_epoch_progbar_total_}')

    diff = MAX_STEPS - total_steps
    curr_diff = diff.subs(subs)
    print(f'curr_diff={curr_diff.evalf()}')

    if curr_diff == 0:
        print('Parameters are perfectly balanced')
    elif curr_diff > 0:
        print('Not enough total steps to fill MAX_STEPS')
    else:
        print('MAX STEPS will stop training short')

    def numeric_solve(to_zero, k):
        from scipy.optimize import minimize

        def func(x):
            v = float(x[0])
            result = to_zero.subs({k: v}).evalf() ** 2
            return float(result)

        guess = relevant[str(k)]
        results = minimize(func, guess)
        return int(results.x[0])

    print('--- Possible Adjustments ---')
    for k, v in subs.items():
        tmp_subs = (ub.udict(subs) - {k})
        to_zero = diff.subs(tmp_subs)
        initial = relevant[str(k)]
        try:
            solutions = sympy.solve(to_zero, k)
            solutions = [s.evalf() for s in solutions]
            if len(solutions) == 0:
                raise Exception
            suggestion = solutions
            method = 'symbolic'
        except Exception:
            numeric_solution = numeric_solve(to_zero, k)
            suggestion = numeric_solution
            method = 'numeric'
        print(f' * {k}: {initial} -> {suggestion} ({method})')


def main():
    dpath = ub.Path.appdir('geowatch/devcheck/lr_scheduler').delete().ensuredir()

    relevant = RelevantConfig()
    relevant_ = dict(relevant)
    relevant_['DEFAULT_ROOT_DIR'] = dpath
    relevant_['TARGET_LR'] = 1000
    relevant_['WEIGHT_DECAY'] = relevant_['TARGET_LR'] * 1e-2
    argstr = kwutil.partial_format.subtemplate(
        '''
        data:
            class_path: SimpleDataModule
            init_args:
                num_workers      : 0
                max_epoch_length : $TRAIN_ITEMS_PER_EPOCH
                batch_size       : $BATCH_SIZE

        model:
            class_path: SimpleModel

        lr_scheduler:
            class_path: CustomOneCycleLR
            # class_path: torch.optim.lr_scheduler.OneCycleLR
            init_args:
                max_lr           : $TARGET_LR

                # ---------
                # FIXME

                # total_steps      : $MAX_STEPS

                # LightningCLI Weirdness.
                # It seems to only step the scheduler every epoch,
                # https://github.com/Lightning-AI/pytorch-lightning/issues/15340
                total_steps      : $MAX_EPOCHS

                # epochs           : $MAX_STEPS
                # steps_per_epoch  : $STEPS_PER_EPOCH

                # ---------

                div_factor       : 25
                final_div_factor : 1000
                anneal_strategy  : cos
                pct_start        : 0.3

                # verbose          : False
                # verbose          : True

        optimizer:
            class_path: CustomAdamW
            #class_path: torch.optim.AdamW
            init_args:
                lr           : $TARGET_LR
                weight_decay : $WEIGHT_DECAY

        trainer:
            accumulate_grad_batches : $ACCUMULATE_GRAD_BATCHES
            default_root_dir        : $DEFAULT_ROOT_DIR
            accelerator             : cpu
            limit_train_batches     : $TRAIN_ITEMS_PER_EPOCH
            max_epochs              : $MAX_EPOCHS
            limit_val_batches       : 0
            log_every_n_steps       : 1
            check_val_every_n_epoch : 1
            enable_checkpointing    : true
            enable_model_summary    : true
            num_sanity_val_steps    : 0

            # NOT SURE WHY WE CANT SPECIFY A CALLBACK LIKE THIS!
            # callbacks:
            #     - class_path: pytorch_lightning.callbacks.ModelCheckpoint
            #       init_args:
            #           monitor    : val_loss
            #           mode       : min
            #           save_top_k : 5
            #           filename   : '{epoch:04d}-{step:06d}-{val_loss:.3f}.ckpt'
            #           save_last  : true
        ''', **relevant_)
    # assert '$' not in argstr

    # Oh LightningCLI, you are convinient, but also difficult.
    nested = kwutil.util_yaml.Yaml.coerce(argstr, backend='pyyaml')

    # Very annoying that we need to prefix with "fit", which breaks from normal
    # config usage. CLI and programatic use should be 1-to-1!
    nested = {'fit.' + k: v for k, v in nested.items()}

    def nested_to_jsonnest(nested):
        config = {}
        for p, v in ub.IndexableWalker(nested):
            if not isinstance(v, (dict, list)):
                k = '.'.join(list(map(str, p)))
                config[k] = v
        return config
    config = nested_to_jsonnest(nested)
    # config['subcommand'] = 'fit'
    rich.print(f'nested = {ub.urepr(nested, nl=True)}')
    rich.print(f'config = {ub.urepr(config, nl=True)}')
    rich.print('\n---\n')

    inspect_relevant_interactions(relevant)

    rich.print(f"\nTrainer log dpath:\n\n[link={dpath}]{dpath}[/link]\n")

    default_callbacks = [
        pl.callbacks.RichProgressBar(),
        pl.callbacks.LearningRateMonitor(logging_interval='step', log_momentum=True),
        # pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
    ]

    if 1:
        from geowatch.utils import lightning_ext as pl_ext
        default_callbacks.append(pl_ext.callbacks.TensorboardPlotter())

    jsonargparse_yaml_workarounds()

    from pytorch_lightning.cli import LightningCLI
    try:
        cli = LightningCLI(
            args=config,
            subclass_mode_model=True,
            save_config_kwargs={
                'overwrite': True,
            },
            parser_kwargs=dict(
                parser_mode='yaml_unsafe_for_tuples',
                error_handler=None,
            ),
            trainer_defaults=dict(
                callbacks=default_callbacks
            ),
            # Setting run to false has unintuitive behavior.
            # run=False,
        )
        print(f'cli={cli}')
        # cli.subcommand = 'fit'
        # cli._run_subcommand(cli.subcommand)
    finally:
        rich.print('[white]------------------')
        rich.print('[white] Finished Training')
        rich.print('[white]------------------')
        rich.print(f'relevant = {ub.urepr(relevant, nl=1, align=":")}')

        effective_num_batches = MEASURED_COUNTS['NUM_BATCHES'] // relevant.ACCUMULATE_GRAD_BATCHES

        MEASURED_COUNTS['NUM_OPTIM_STEPS']

        relevant.MAX_EPOCHS

        MEASURED_COUNTS['effective_num_batches'] = effective_num_batches
        MEASURED_COUNTS['train_batches_per_epoch'] = MEASURED_COUNTS['effective_num_batches'] / MEASURED_COUNTS['NUM_EPOCHS']

        MEASURED_COUNTS['batch_size'] = MEASURED_COUNTS['NUM_GETITEM_CALLS'] / MEASURED_COUNTS['NUM_BATCHES']
        MEASURED_COUNTS['train_items_per_epoch'] = MEASURED_COUNTS['train_batches_per_epoch'] * MEASURED_COUNTS['batch_size']

        if MEASURED_COUNTS['effective_num_batches'] != MEASURED_COUNTS['NUM_OPTIM_STEPS']:
            print('The effective number of batches should be the same as the number of optimization steps')

        rich.print(f'MEASURED_COUNTS = {ub.urepr(MEASURED_COUNTS, nl=1, align=":")}')
        rich.print(f"\nTrainer log dpath:\n\n[link={dpath}]{dpath}[/link]\n")

        scripts = list(dpath.glob('*/*/monitor/tensorboard/redraw.sh'))
        print(f'scripts={scripts}')
        script = scripts[0]
        ub.cmd(str(script), verbose=3, system=True)
    return cli

# Might be worth trying to do this manually, lightningcli has lots of overhead
# kwargs = dict(
#     max_lr=TARGET_LR,
#     div_factor=25,
#     final_div_factor=1000,
#     total_steps=MAX_STEPS,
#     anneal_strategy='cos',
#     pct_start=0.3,
#     verbose=True)


# optim_cls, optim_kw = nh.api.Optimizer.coerce(
#     optimizer='adam',
#     lr=TARGET_LR,
#     weight_decay=0)

# model = torch.nn.Linear(10, 10)
# params = list(model.parameters())
# optim_kw['params'] = params
# optimizer = optim_cls(**optim_kw)
# lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)

# rows = []

# batch_counter = 0
# accum_counter = 0

# # Insepct what the LR curve will look like
# for _ in range(max_epochs):

#     accum_counter = 0
#     for item_idx in range(ITEMS_PER_EPOCH):
#         accum_counter += 1

#         if accum_counter >= ACCUMULATE_GRAD_BATCHES:
#             optimizer.zero_grad()
#             optimizer.step()
#             lr_scheduler.step()
#             accum_counter = 0

#         lr = lr_scheduler.get_last_lr()[0]
#         rows.append({'lr': lr, 'last_epoch': lr_scheduler.last_epoch, 'step': lr_scheduler._step_count})

# # xdoctest: +REQUIRES(--show)
# data = pd.DataFrame(rows)
# sns = kwplot.autosns()
# sns.lineplot(data=data, y='lr', x='last_epoch')


if __name__ == '__main__':
    """
    CommandLine:
        cd ~/code/geowatch/dev/devcheck
        python ~/code/geowatch/dev/devcheck/devcheck_lr_schedulers.py
        python -m geowatch.utils.lightning_ext.callbacks.tensorboard_plotter "$HOME"/.cache/geowatch/devcheck/lr_scheduler/lightning_logs/version_0/
    """
    main()
