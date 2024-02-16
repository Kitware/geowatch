#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class RecommendSizeAdjustmentsCLI(scfg.DataConfig):
    """
    Helper to recommend adjustments to network size parameters
    """
    MAX_STEPS               = scfg.Value(None, help='The number of optimizer steps to be taken')
    MAX_EPOCHS              = scfg.Value(None, help='The maximum number of train epochs')
    BATCH_SIZE              = scfg.Value(None, help='The physical batch size')
    ACCUMULATE_GRAD_BATCHES = scfg.Value(1, help='Accumulate gradients for this many batches for stepping the optimizer. I.e. the multiplier for effective batch size')
    TRAIN_ITEMS_PER_EPOCH   = scfg.Value(None, help='The number of items the training dataloader can produce in one epoch')
    TRAIN_BATCHES_PER_EPOCH = scfg.Value(None, help='The number of items the training dataloader can produce in one epoch')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from geowatch.cli.experimental.recommend_size_adjustments import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = RecommendSizeAdjustmentsCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        import sympy
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + ub.urepr(config, nl=1, align=":"))

        assert config.BATCH_SIZE is not None

        if config.MAX_EPOCHS is None:
            ...

        if config.TRAIN_ITEMS_PER_EPOCH is None:
            config.TRAIN_ITEMS_PER_EPOCH = config.TRAIN_BATCHES_PER_EPOCH * config.BATCH_SIZE

        symbolic_names = 'TRAIN_ITEMS_PER_EPOCH, BATCH_SIZE, ACCUMULATE_GRAD_BATCHES, MAX_EPOCHS, MAX_STEPS'.split(', ')
        # symbolic_vars = sympy.symbols(symbolic_names, integer=True, positive=True)
        symbolic_vars = sympy.symbols(symbolic_names)
        TRAIN_ITEMS_PER_EPOCH, BATCH_SIZE, ACCUMULATE_GRAD_BATCHES, MAX_EPOCHS, MAX_STEPS = symbolic_vars

        # Build substitution dictionary for sympy
        subs = ub.dzip(symbolic_vars, ub.udict(config).take(symbolic_names))

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
        step_difference = diff.subs(subs)
        print(f'step_difference={step_difference.evalf()}')

        if step_difference == 0:
            print('Parameters are perfectly balanced')
        elif step_difference > 0:
            print('Not enough total steps to fill MAX_STEPS')
        else:
            print('MAX STEPS will stop training short')

        def numeric_solve(to_zero, k):
            from scipy.optimize import minimize

            def func(x):
                v = float(x[0])
                result = to_zero.subs({k: v}).evalf() ** 2
                return float(result)

            guess = config[str(k)]
            results = minimize(func, guess)
            return int(results.x[0])

        rich.print('[white]--- Possible Adjustments ---')
        for k, v in subs.items():
            tmp_subs = (ub.udict(subs) - {k})
            to_zero = diff.subs(tmp_subs)
            initial = config[str(k)]
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
            rich.print(f' * {k}: {initial} -> {suggestion} ({method})')


__cli__ = RecommendSizeAdjustmentsCLI
main = __cli__.main

if __name__ == '__main__':
    """

    CommandLine:
        python -m geowatch.cli.experimental.recommend_size_adjustments --help
    """
    main()
