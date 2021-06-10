import fit
import pathlib
import pytorch_lightning as pl

from methods import baseline
from datasets import onera_2018
import utils


channel_combos = {
    "all": "<all>",
    "uv": "costal",
    "bgr": "blue|green|red",
    "vnir": "B05|B06|B07|nir|B8A",
    "swir": "B09|cirrus|swir16|swir22",
}

if __name__ == "__main__":
    
    from types import SimpleNamespace
    
    args = SimpleNamespace(
        dataset="Drop0AlignMSI_S2",
        method="TransformerChangeDetector",
        
        # dataset params
        train_kwcoco_path=pathlib.Path("~/Projects/smart_watch_dvc/drop0_aligned_msi/data.kwcoco.json"),
        batch_size=64,
        num_workers=8,
        
        # model params
        window_size=8,
        embedding_dim=256,
        n_layers=8,
        learning_rate=1e-3,
        weight_decay=1e-5,
        pos_weight=5.0,
        
        # trainer params
        gpus=1,
        max_epochs=200,
    )
    
    for key, channels in channel_combos.items():
        args.channels = channels
        args.default_root_dir = f"_trained_models/drop0_s2/tf/{key}"
        fit.main(args)
