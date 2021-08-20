# flake8: noqa
method_bases = {
    "mmvit-DotProdCD-stm-s12": dict(
        method="MultimodalTransformerDotProdCD",
        arch_name="smt_it_stm_t12",

        window_size=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=0.1,
    ),
}


channel_sets = {
    "all": "B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A",
    "no60": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
    "sample": "B01|B02|B03|B04|B08|B10|B12",
    "vnir": "B05|B06|B07|B08|B8A",
    "swir": "B09|B10|B11|B12",
    "bgr": "B02|B03|B04",
    "uv": "B01",
}


def main():
    """
    Example:
        # Set the path to your data
        DVC_DPATH=$HOME/Projects/smart_watch_dvc
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        ls $DVC_DPATH/extern/onera_2018
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # Invoke the training script
        python -m watch.tasks.fusion.experiments.subset_experiment \
            --train_kwcoco_path=$TRAIN_FPATH \
            --batch_size=1 \
            --num_workers=0 \
            --chip_size=32 \
            --workdir=$HOME/work/watch/fit/runs
    """
    import itertools as it
    from types import SimpleNamespace
    from watch.tasks.fusion import fit, predict, evaluate

    for (method_key, method_base), (channel_key, channel_subset) in it.product(method_bases.items(), channel_sets.items()):

        print("training...")
        defaults = dict(
            dataset="KWCocoDataModule",
            channels=channel_subset,

            # trainer params
            workdir="_subset_experiment/trained_models",
            terminate_on_nan=True,
        )
        config = {**defaults, **method_base}

        package_fpath = fit.fit_model(args=None, cmdline=True, **config)


if __name__ == "__main__":
    main()
