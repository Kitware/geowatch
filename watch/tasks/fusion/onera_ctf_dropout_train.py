import pathlib
import itertools as it
from . import fit
from .datasets import onera_2018

model_names = [
    "smt_it_t_t12",
    "smt_it_st_t12",
    "smt_it_stm_t12",
    "smt_it_t_s12",
    "smt_it_st_s12",
    "smt_it_stm_s12",
]

methods = [
    "MultimodalTransformerDotProdCD",
    "MultimodalTransformerDirectCD",
]

if __name__ == "__main__":

    from types import SimpleNamespace

    args = SimpleNamespace(
        dataset="OneraCD_2018",

        # dataset params
        train_kwcoco_path=pathlib.Path("~/Projects/smart_watch_dvc/extern/onera_2018/onera_train.kwcoco.json"),
        batch_size=16,
        num_workers=8,
        chip_size=128,
        tfms_train_channel_size=8,

        # model params
        window_size=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        dropout=0.1,

        # trainer params
        gpus=1,
        #accelerator="ddp",
        precision=16,
        max_epochs=400,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        terminate_on_nan=True,
    )
    
    for method, model_name in it.product(methods, model_names):
        print(f"{method} / {model_name}\n====================")
        args.method = method
        args.model_name = model_name
        args.default_root_dir = f"_trained_models/onera/ctf_drop{args.tfms_train_channel_size}/{method}-{model_name}"
        try:
            fit.main(args)
        except:
            continue
