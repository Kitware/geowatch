import pathlib
from . import fit


channel_combos = {
    "all": "costal|B02|B03|B04|B05|B06|B07|nir|B09|cirrus|swir16|swir22|B8A",
    "uv": "costal",
    "bgr": "B02|B03|B04",
    "vnir": "B05|B06|B07|nir|B8A",
    "swir": "B09|cirrus|swir16|swir22",
    "sample": "costal|B02|B03|B04|nir|cirrus|swir22",
    "no60": "B02|B03|B04|B05|B06|B07|nir|swir16|swir22|B8A",
}

if __name__ == "__main__":

    from types import SimpleNamespace

    args = SimpleNamespace(
        dataset="Drop0AlignMSI_S2",
        method="UNetChangeDetector",

        # dataset params
        train_kwcoco_path=pathlib.Path("~/Projects/smart_watch_dvc/drop0_aligned_context5/train_data.kwcoco.json").expanduser(),
        batch_size=64,
        num_workers=8,

        # model params
        feature_dim=128,
        learning_rate=1e-3,
        weight_decay=1e-5,

        # trainer params
        gpus=1,
        max_epochs=400,
    )

    for key, channels in channel_combos.items():
        args.channels = channels
        args.default_root_dir = f"_trained_models/drop0/unet/{key}"
        fit.main(args)
