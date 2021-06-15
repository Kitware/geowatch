import fit
import pathlib
import pytorch_lightning as pl

import methods
from datasets import onera_2018
import utils

ctf_methods = {
#    "Joint": "JointTransformerChangeDetector",
    "SpaceTimeMode": "SpaceTimeModeTransformerChangeDetector",
    "SpaceMode": "SpaceModeTransformerChangeDetector",
    "SpaceTime": "SpaceTimeTransformerChangeDetector",
    "TimeMode": "TimeModeTransformerChangeDetector",
    "Space": "SpaceTransformerChangeDetector",
    "Axial": "AxialTransformerChangeDetector",
}

if __name__ == "__main__":
    
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        dataset="OneraCD_2018",
        
        # dataset params
        train_kwcoco_path=pathlib.Path("~/Projects/smart_watch_dvc/extern/onera_2018/onera_train.kwcoco.json"),
        batch_size=16,
        num_workers=8,
        transform_key="channel_transformer",
        tfms_scale=2000.,
        tfms_window_size=8,
        
        # model params
        window_size=8,
        embedding_dim=256,
        n_layers=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        pos_weight=5.0,
        
        # trainer params
        gpus=1,
        precision=16,
        max_epochs=200,
        accumulate_grad_batches=2,
    )
    
    for key, method in ctf_methods.items():
        print(key, "\n====================")
        args.method = method
        args.default_root_dir = f"_trained_models/onera/ctf/{key}"
        fit.main(args)
