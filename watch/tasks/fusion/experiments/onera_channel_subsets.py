import pathlib

model_names = [
    # "smt_it_stm_n12",
    # "smt_it_hwtm_n12",
    "smt_it_stm_t12",
    # "smt_it_hwtm_t12",
    # "smt_it_stm_s12",
    # "smt_it_hwtm_s12",
]

methods = [
    "MultimodalTransformerDotProdCD",
    # "MultimodalTransformerDirectCD",
]

datasets = {
    "onera": "OneraCD_2018",
    "drop0_s2": "Drop0AlignMSI_S2",
}

dataset_kwcocos = {
    "onera": pathlib.Path("~/Projects/smart_watch_dvc/extern/onera_2018/onera_all.kwcoco.json").expanduser(),
    "drop0_s2": pathlib.Path("~/Projects/smart_watch_dvc/drop0_aligned_context5/data.kwcoco.json").expanduser(),
}

dataset_channel_sets = {
    "all": {
        "onera": "B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A",
        "drop0_s2": "costal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A",
    },
    "no60": {
        "onera": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
        "drop0_s2": "blue|green|red|B05|B06|B07|nir|swir16|swir22|B8A",
    },
    "sample": {
        "onera": "B01|B02|B03|B04|B08|B10|B12",
        "drop0_s2": "costal|blue|green|red|nir|cirrus|swir22",
    },
    "vnir": {
        "onera": "B05|B06|B07|B08|B8A",
        "drop0_s2": "B05|B06|B07|nir|B8A",
    },
    "swir": {
        "onera": "B09|B10|B11|B12|B8A",
        "drop0_s2": "B09|cirrus|swir16|swir22|B8A",
    },
    "bgr": {
        "onera": "B02|B03|B04",
        "drop0_s2": "blue|green|red",
    },
    "uv": {
        "onera": "B01",
        "drop0_s2": "costal",
    },
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
        python -m watch.tasks.fusion.onera_channelwisetransformer_train \
            --train_kwcoco_path=$TRAIN_FPATH \
            --batch_size=1 \
            --num_workers=0 \
            --chip_size=32 \
            --workdir=$HOME/work/watch/fit/runs
    """
    import itertools as it
    from watch.tasks.fusion import fit, predict, evaluate  # NOQA

    for method, model_name, (channel_key, channel_sets) in it.product(methods, model_names, dataset_channel_sets.items()):

        workdir = f"_onera_channel_subsets/{method}/{model_name}/{channel_key}"

        defaults = dict(
            dataset="OneraCD_2018",
            method=method,
            model_name=model_name,

            # dataset params
            tfms_channel_subset=channel_sets["onera"],
            train_kwcoco_path=dataset_kwcocos["onera"],
            chip_size=128,

            # model params
            window_size=8,
            learning_rate=1e-3,
            weight_decay=1e-5,

            # trainer params
            terminate_on_nan=True,
            max_epochs=400,
            workdir=workdir,
        )
        fit.fit_model(cmdline=True, **defaults)


if __name__ == "__main__":
    main()
