from types import SimpleNamespace
import pathlib
import predict

datasets = {
    "onera": "OneraCD_2018",
    "drop0_s2": "Drop0AlignMSI_S2",
}

dataset_kwcocos = {
    "onera": pathlib.Path("~/Projects/smart_watch_dvc/extern/onera_2018/onera_all.kwcoco.json").expanduser(),
    "drop0_s2": pathlib.Path("~/Projects/smart_watch_dvc/drop0_aligned_msi/data.kwcoco.json").expanduser(),
}

dataset_channel_sets = {
    "onera": {
        "all": None,
        "uv": "B01",
        "bgr": "B02|B03|B04",
        "vnir": "B05|B06|B07|B08|B8A",
        "swir": "B09|B10|B11|B12",
        "sample": "B01|B02|B03|B04|B08|B10|B12",
        "no60": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
    },
    "drop0_s2": {
        "all": None,
        "uv": "B01",
        "bgr": "B02|B03|B04",
        "vnir": "B05|B06|B07|B08|B8A",
        "swir": "B09|B10|B11|B12",
        "sample": "B01|B02|B03|B04|B08|B10|B12",
        "no60": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
    },
}

methods = {
    "Axial": "AxialTransformerChangeDetector",
    # "Joint": "JointTransformerChangeDetector",
    "SpaceTimeMode": "SpaceTimeModeTransformerChangeDetector",
    "SpaceMode": "SpaceModeTransformerChangeDetector",
    "SpaceTime": "SpaceTimeTransformerChangeDetector",
    "TimeMode": "TimeModeTransformerChangeDetector",
    "Space": "SpaceTransformerChangeDetector",
}

for ckpt_dir in pathlib.Path("_trained_models").glob("*/ctf/*/"):
    dataset_name = ckpt_dir.parts[-3]
    method_name = ckpt_dir.parts[-1]

    dataset = datasets[dataset_name]
    method = methods[method_name]
    test_kwcoco_path = dataset_kwcocos[dataset_name]

    ckpt_paths = ckpt_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    ckpt_paths = sorted(list(ckpt_paths))
    ckpt_path = ckpt_paths[-1]
    
    for channel_key, channel_subset in dataset_channel_sets[dataset_name].items():

        print(f"{method}_{dataset}_{channel_key}\n=========================")

        args = SimpleNamespace(
            dataset=dataset,
            method=method,
            tag=f"{method}_{channel_key}",
            checkpoint_path=ckpt_path,
            results_dir=pathlib.Path("_results") / dataset,
            results_path=pathlib.Path("_results") / f"{dataset}_results.kwcoco.json",
            test_kwcoco_path=test_kwcoco_path,
            tfms_channel_subset=channel_subset,
            # common args
            use_gpu=True,
            batch_size=1,
            time_steps=2,
            chip_size=128,
            time_overlap=0.5,
            chip_overlap=0.1,
            transform_key="channel_transformer",
            tfms_scale=2000.,
            tfms_window_size=8,
        )
        predict.main(args)

