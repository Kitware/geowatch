import fit
import pathlib
import pytorch_lightning as pl

from methods import baseline
from datasets import onera_2018
import utils


channel_combos = {
    "all": "<all>",
    "uv": "B01",
    "bgr": "B02|B03|B04",
    "vnir": "B05|B06|B07|B08|B8A",
    "swir": "B09|B10|B11|B12",
    "sample": "B01|B02|B03|B04|B08|B10|B12",
    "no60": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
}

if __name__ == "__main__":
    
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        dataset="OneraCD_2018",
        method="UNetChangeDetector",
        
        # dataset params
        train_kwcoco_path=pathlib.Path("~/Projects/smart_watch_dvc/extern/onera_2018/onera_train.kwcoco.json"),
        batch_size=64,
        num_workers=8,
        
        # model params
        feature_dim=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
        pos_weight=5.0,
        
        # trainer params
        gpus=1,
        max_epochs=200,
    )
    
    for key, channels in channel_combos.items():
        args.channels = channels
        args.default_root_dir = f"_trained_models/onera/unet/{key}"
        fit.main(args)
