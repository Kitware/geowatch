import pathlib
import scriptconfig as scfg
import ubelt as ub
from watch.tasks.fusion import fit


channel_combos = {
    "all": "<all>",
    "uv": "B01",
    "bgr": "B02|B03|B04",
    "vnir": "B05|B06|B07|B08|B8A",
    "swir": "B09|B10|B11|B12",
    "sample": "B01|B02|B03|B04|B08|B10|B12",
    "no60": "B02|B03|B04|B05|B06|B07|B08|B11|B12|B8A",
}


class OneraUnetTrainConfig(scfg.Config):
    default = {
        'train_dataset': scfg.Path(None, help='path to train kwcoco file'),
        'workdir': scfg.Path('_trained_models/onera/unet/', help=ub.paragraph(
            '''
            Directory where training data can be written
            '''))
    }


def main():
    r"""
    Example:

        # Set the path to your data
        DVC_DPATH=$HOME/Projects/smart_watch_dvc
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # Invoke the training script
        python -m watch.tasks.fusion.onera_unet_train \
            --train_dataset=$TRAIN_FPATH \
            --workdir=$HOME/work/watch/onera/unet/

    """
    from types import SimpleNamespace
    config = OneraUnetTrainConfig(cmdline=True)

    args = SimpleNamespace(
        dataset="OneraCD_2018",
        method="UNetChangeDetector",

        # dataset params
        train_kwcoco_path=pathlib.Path(config['train_dataset']),
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
        args.default_root_dir = pathlib.Path(config['workdir']) / key
        fit.main(args)

if __name__ == "__main__":
    main()
