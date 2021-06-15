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
    dataset = datasets[ckpt_dir.parts[-3]]
    method = methods[ckpt_dir.parts[-1]]
    test_kwcoco_path = dataset_kwcocos[ckpt_dir.parts[-3]]

    ckpt_paths = ckpt_dir.glob("lightning_logs/version_*/checkpoints/*.ckpt")
    ckpt_paths = sorted(list(ckpt_paths))
    ckpt_path = ckpt_paths[-1]

    args = SimpleNamespace(
        dataset=dataset,
        method=method,
        tag=f"{method}_all",
        checkpoint_path=ckpt_path,
        results_dir=pathlib.Path("_results") / dataset,
        results_path=pathlib.Path("_results") / f"{dataset}_results2.kwcoco.json",
        test_kwcoco_path=test_kwcoco_path,
        # common args
        batch_size=1,
        time_steps=2,
        chip_size=128,
        time_overlap=0,
        chip_overlap=0.1,
        transform_key="channel_transformer",
        tfms_scale=2000.,
        tfms_window_size=8,
    )
    predict.main(args)

