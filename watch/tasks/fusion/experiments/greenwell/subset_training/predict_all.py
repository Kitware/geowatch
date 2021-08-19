import pathlib
from watch.tasks.fusion import utils, predict

channel_sets = {
    "drop1": {
        "all": "coastal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A",
        "no60": "blue|green|red|B05|B06|B07|nir|swir16|swir22|B8A",
        "sample": "coastal|blue|green|red|nir|cirrus|swir22",
        "swir": "B09|cirrus|swir16|swir22",
        "vnir": "B05|B06|B07|nir|B8A",
        "bgr": "blue|green|red",
        "uv": "coastal",
    },
    "onera": {
        "all": "B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A",
        "no60": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
        "sample": "B01|B02|B03|B04|B08|B10|B12",
        "vnir": "B05|B06|B07|B08|B8A",
        "swir": "B09|B10|B11|B12",
        "bgr": "B02|B03|B04",
        "uv": "B01",
    },
}
channel_keys = {
    ds: dict(zip(channel_set.values(), channel_set.keys()))
    for ds, channel_set in channel_sets.items()
}

model_roots = pathlib.Path("_subset_experiment/trained_models").glob("*")
for model_root in model_roots:
    model_path = model_root / "package.pt"
    model = utils.load_model_from_package(model_path)

    dataset = "drop1" if ("drop1" in str(model.datamodule_hparams.train_dataset)) else "onera"
    method_key = model_root.stem
    channel_subset = model.datamodule_hparams.channels
    channel_key = channel_keys[dataset][channel_subset]
    if dataset == "drop1":
        test_dataset = str(model.datamodule_hparams.vali_dataset)
    else:
        test_dataset = str(model.datamodule_hparams.train_dataset).replace("train", "test")

    print("predicting...")
    predict_args = dict(
        tag=f"{dataset}_{channel_key}_{method_key}",
        dataset="WatchDataModule",
        test_dataset=test_dataset,
        channels=channel_subset,
        checkpoint_path=model_path,
        results_dir=f"_subset_experiment/results_{dataset}",
        results_path=f"_subset_experiment/results/{dataset}_results.kwcoco.json",
        use_gpu=True,
        batch_size=1,
    )
    all_args = {**dict(model.datamodule_hparams), **predict_args}
    print(all_args)
    predict.predict(cmdline=False, **all_args)
