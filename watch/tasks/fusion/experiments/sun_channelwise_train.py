import pathlib
import itertools as it
from . import fit

model_names = [
    "sm_it_sm_t12",
    "sm_it_joint_t12",
    "sm_it_sm_s12",
    "sm_it_joint_s12",
]

methods = [
    "MultimodalTransformerSegmentation",
]

if __name__ == "__main__":

    from types import SimpleNamespace

    args = SimpleNamespace(
        dataset="SUN_RGBD",

        # dataset params
        data_root=pathlib.Path("~/Projects/smart_watch_dvc/extern/icl_sun_rgbd").expanduser(),
        batch_size=16,
        num_workers=8,
        chip_size=128,
        tfms_train_channel_size=2,

        # model params
        n_classes=13,
        window_size=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=0.1,

        # trainer params
        gpus=1,
        # accelerator="ddp",
        precision=16,
        max_epochs=400,
        accumulate_grad_batches=4,
        terminate_on_nan=True,
    )

    for method, model_name in it.product(methods, model_names):
        print(f"{method} / {model_name}\n====================")
        args.method = method
        args.model_name = model_name
        args.default_root_dir = f"_trained_models/sun_rgbd/ctf_drop2/{method}-{model_name}"
        fit.main(args)
