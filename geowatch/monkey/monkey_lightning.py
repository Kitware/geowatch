def disable_lightning_hardware_warnings():
    """
    Lightning likes to warn us when we use a CPU or when we aren't using extr
    workers, even when we explicilty requested it. This lets us squash these
    warnings.
    """
    import warnings
    warnings.filterwarnings(
        action='ignore',
        message='.*Consider increasing the value of the `num_workers` argument.*',
        module=r'.*lightning.*'
    )
    warnings.filterwarnings(
        action='ignore',
        message='.*available but not used.*',
        module=r'.*lightning.*'
    )

    warnings.filterwarnings(
        action='ignore',
        message='.*The number of training batches .*is smaller than the logging interval.*',
        module=r'.*lightning.*'
    )

    warnings.filterwarnings(
        action='ignore',
        message='.*You defined a `validation_step` but have no `val_dataloader`.*',
        module=r'.*lightning.*'
    )

    warnings.filterwarnings(
        action='ignore',
        message='.*The `srun` command is available on your system but is not used.*',
        module=r'.*lightning.*'
    )
    # warnings.warn("FILTERING WARNINGS")
    # warnings.warn("FILTERING WARNINGS")
