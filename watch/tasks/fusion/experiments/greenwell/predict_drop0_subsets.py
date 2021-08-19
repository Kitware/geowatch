from types import SimpleNamespace
import pathlib
from watch.tasks.fusion import predict

dataset_channel_sets = {
    "all": "costal|blue|green|red|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A",
    "no60": "blue|green|red|B05|B06|B07|nir|swir16|swir22|B8A",
    "sample": "costal|blue|green|red|nir|cirrus|swir22",
    "swir": "B09|cirrus|swir16|swir22",
    "vnir": "B05|B06|B07|nir|B8A",
    "bgr": "blue|green|red",
    "uv": "costal",
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
            dataset="Drop0AlignMSI_S2",
            method=method,
            tag=f"{run_name}_{channel_key}",
            checkpoint_path=ckpt_path,
            results_dir=pathlib.Path("_results") / "Drop0AlignMSI_S2",
            results_path=pathlib.Path("_results") / "Drop0AlignMSI_S2_results.kwcoco.json",
            test_kwcoco_path=pathlib.Path("~/Projects/smart_watch_dvc/drop0_aligned_context5/combo_data.kwcoco.json").expanduser(),
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
