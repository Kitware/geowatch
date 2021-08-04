method_bases = {
    "mmvit-DotProdCD-stm-s12": dict(
        method="MultimodalTransformerDotProdCD",
        model_name="smt_it_stm_s12",

        window_size=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=0.1,
    ),
    "mmvit-DirectCD-stm-s12": dict(
        method="MultimodalTransformerDirectCD",
        model_name="smt_it_stm_s12",

        window_size=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=0.1,
    ),
    "unet": dict(
        method="UNetChangeDetector",

        feature_dim=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
    ),
}


channel_sets = {
    "all": "costal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A",
    "no60": "blue|green|red|B05|B06|B07|nir|swir16|swir22|B8A",
    "sample": "costal|blue|green|red|nir|cirrus|swir22",
    "swir": "B09|cirrus|swir16|swir22",
    "vnir": "B05|B06|B07|nir|B8A",
    "bgr": "blue|green|red",
    "uv": "costal",
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
    from typing import SimpleNamespace
    from watch.tasks.fusion import fit, predict, evaluate

    for (method_key, method_base), (channel_key, channel_subset) in it.product(method_bases.items(), channel_sets.items()):

        print("training...")
        defaults = dict(
            dataset="Drop0AlignMSI_S2",
            channels=channel_subset,

            # trainer params
            workdir="_subset_experiment/trained_models"
            terminate_on_nan=True,
        )
        config = {**defaults, **method_base}

        args = fit.make_fit_config(cmdline=True, **config)
        fit.fit_model(args=args)

        print("predicting...")
        predict_args = SimpleNamespace(
            tag=f"Drop0AlignMSI_{channel_key}_{method_key}"
            dataset="Drop0AlignMSI_S2",
            channels=channel_subset,
            checkpoint_path=args.default_root_dir / "package.pt",
            results_dir="_subset_experiment/results",
            results_path="_subset_experiment/results/Drop0AlignMSI_results.kwcoco.json",
            use_gpu=True,
        )
        predict.main(predict_args)

    print("evaluating...")
    evaluate_args = SimpleNamespace(
        result_kwcoco_path="_subset_experiment/results/Drop0AlignMSI_results.kwcoco.json",
        metrics_path="_subset_experiment/results/Drop0AlignMSI_metrics.kwcoco.json",
        figure_root="_subset_experiment/figures/Drop0AlignMSI",
    )
    evaluate.main(evaluate_args)


if __name__ == "__main__":
    main()
