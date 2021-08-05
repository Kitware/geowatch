model_names = [
    # "smt_it_stm_n12",
    # "smt_it_hwtm_n12",
    # "smt_it_stm_t12",
    # "smt_it_hwtm_t12",
    "smt_it_stm_s12",
    # "smt_it_hwtm_s12",
]

methods = [
    "MultimodalTransformerDotProdCD",
    "MultimodalTransformerDirectCD",
]


def main():
    """
    Example:
        # Set the path to your data
        DVC_DPATH=$HOME/Projects/smart_watch_dvc
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        ls $DVC_DPATH/extern/onera_2018
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # Invoke the training script
        python -m watch.tasks.fusion.onera_channelwisetransformer_train \
            --train_kwcoco_path=$TRAIN_FPATH \
            --batch_size=1 \
            --num_workers=0 \
            --chip_size=32 \
            --workdir=$HOME/work/watch/fit/runs
    """
    import itertools as it
    # from types import SimpleNamespace
    from . import fit

    for method, model_name in it.product(methods, model_names):

        defaults = dict(
            dataset="OneraCD_2018",
            method=method,
            model_name=model_name,

            # model params
            window_size=8,
            learning_rate=1e-3,
            weight_decay=1e-4,
            dropout=0.1,

            # trainer params
            terminate_on_nan=True,
        )
        fit.fit_model(cmdline=False, **defaults)


if __name__ == "__main__":
    main()
