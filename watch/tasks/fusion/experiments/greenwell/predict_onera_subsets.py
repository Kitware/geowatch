from types import SimpleNamespace
import pathlib
from watch.tasks.fusion import predict

dataset_channel_sets = {
    "all": "B01|B02|B03|B04|B05|B06|B07|B08|B09|B10|B11|B12|B8A",
    "no60": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
    "sample": "B01|B02|B03|B04|B08|B10|B12",
    "swir": "B09|B10|B11|B12",
    "vnir": "B05|B06|B07|B08|B8A",
    "bgr": "B02|B03|B04",
    "uv": "B01",
}

for ckpt_dir in pathlib.Path("_trained_models").glob("MultimodalTransformer*"):
    run_name = ckpt_dir.parts[-1]
    method, config_hash = run_name.split("-")

    ckpt_paths = ckpt_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    ckpt_paths = sorted(list(ckpt_paths))
    if len(ckpt_paths) < 1:
        continue
    ckpt_path = ckpt_paths[-1]

    for channel_key, channel_subset in dataset_channel_sets.items():

        print(f"{run_name}_Onera_{channel_key}\n=========================")

        args = SimpleNamespace(
            dataset="OneraCD_2018",
            method=method,
            tag=f"{run_name}_{channel_key}",
            checkpoint_path=ckpt_path,
            results_dir=pathlib.Path("_results") / "OneraCD_2018",
            results_path=pathlib.Path("_results") / "OneraCD_2018_results.kwcoco.json",
            test_kwcoco_path=pathlib.Path("~/Projects/smart_watch_dvc/extern/onera_2018/onera_all.kwcoco.json").expanduser(),
            tfms_channel_subset=channel_subset,
            # common args
            use_gpu=True,
            batch_size=1,
            time_steps=2,
            chip_size=128,
            time_overlap=0.5,
            chip_overlap=0.1,
        )
        predict.main(args)
