import fit
import pathlib
import pytorch_lightning as pl
import itertools as it
import methods
from datasets import onera_2018
import utils

model_names = [
    "smt_it_joint_p8",
    "smt_it_stm_p8",
    "smt_it_hwtm_p8",
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
        batch_size=32,
        num_workers=8,
        chip_size=128,
        
        # model params
        window_size=8,
        learning_rate=1e-3,
        weight_decay=0,
        dropout=0,
        pos_weight=5.0,
        
        # trainer params
        gpus=1,
        #accelerator="ddp",
        precision=16,
        max_epochs=200,
        accumulate_grad_batches=2,
        terminate_on_nan=True,
    )
    
    for method, model_name in zip(methods, model_names):
        print(f"{method} / {model_name}\n====================")
        args.method = method
        args.model_name = model_name
        args.default_root_dir = f"_trained_models/onera/ctf/{method}-{model_name}"
        fit.main(args)
