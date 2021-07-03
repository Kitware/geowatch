import pytorch_lightning as pl

from watch.tasks.fusion import datasets
from watch.tasks.fusion import methods
from watch.tasks.fusion import utils

import scriptconfig as scfg
import ubelt as ub
import pathlib


# available_methods = dir(methods)
available_methods = [
    'End2EndVotingModel',
    'MultimodalTransformerDirectCD',
    'MultimodalTransformerDotProdCD',
    'MultimodalTransformerSegmentation',
    'TransformerChangeDetector',
    'UNetChangeDetector',
    'VotingModel',
]

available_models = [
    "smt_it_joint_p8",
    "smt_it_stm_p8",
    "smt_it_hwtm_p8",
]

# dir(datasets)
available_datasets = [
    'Drop0AlignMSI_S2',
    'Drop0Raw_S2',
    'WatchDataModule',
    # # 'common',
    # 'onera_2018',
    # 'project_data'
]


class TrainFusionConfig(scfg.Config):
    default = {

        # Basic parameters
        'method': scfg.Value('MultimodalTransformerDirectCD', choices=available_methods),
        'model_name': scfg.Value("smt_it_stm_p8", choices=available_models),

        # TODO: Is a lightning Dataset Module more like a task than a dataset?
        # special:vidshapes8-multispectral and special:vidshapes8 and sometimes
        # on special:shapes8
        'dataset': scfg.Value("WatchDataModule", choices=available_datasets),
        'train_dataset': scfg.Path('vidshapes8:multispectral', help='path to train kwcoco file'),
        'vali_dataset': scfg.Path(None, help='path to vali kwcoco file'),
        'test_dataset': scfg.Path(None, help='path to test kwcoco file'),


        # Sensible defaults
        'batch_size': scfg.Value(32, help='numer of samples per batch'),
        'num_workers': scfg.Value(8, help='number of dataloader workers'),
        'chip_size': scfg.Value(128, help='width and height of patches'),
        'workdir': scfg.Path('_trained_models/onera/ctf/', help=ub.paragraph(
            '''
            Directory where training data can be written
            ''')),
        # model params
        'window_size': 8,
        'learning_rate': 1e-3,
        'weight_decay': 0,
        'dropout': 0,
        'pos_weight': 5.0,

        # trainer params
        'gpus': 1,
        #accelerator="ddp",
        'precision': 16,
        'max_epochs': 200,
        'accumulate_grad_batches': 2,
        'terminate_on_nan': True,
    }


def main(**kwargs):
    """
    Example:
        pass

    CommandLine:
        # Set the path to your data
        DVC_DPATH=$HOME/Projects/smart_watch_dvc
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # View the help docs
        python -m watch.tasks.fusion.fit --help

        # Invoke the training script

        # This task required 5.12GB on a 3090
        python -m watch.tasks.fusion.fit \
            --model_name=smt_it_stm_p8 \
            --method=MultimodalTransformerDotProdCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=2 \
            --num_workers=4 \
            --chip_size=96 \
            --workdir=$HOME/work/watch/fit

        # This task required 17GB on a 3090
        python -m watch.tasks.fusion.fit \
            --model_name=smt_it_stm_p8 \
            --method=MultimodalTransformerDotProdCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=16 \
            --num_workers=4 \
            --chip_size=128 \
            --workdir=$HOME/work/watch/fit

        # This task required 20GB on a 3090
        python -m watch.tasks.fusion.onera_channelwisetransformer_train \
            --model_name=smt_it_joint_p8 \
            --method=MultimodalTransformerDirectCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=4 \
            --num_workers=4 \
            --chip_size=96 \
            --workdir=$HOME/work/watch/fit
    """
    config = dict(TrainFusionConfig(default=kwargs, cmdline=True))

    learning_irrelevant = {
        'num_workers',
        'workdir',
        'model_name',
        'gpus',
    }

    learning_config = ub.dict_diff(config, learning_irrelevant)
    train_hashid = ub.hash_data(ub.map_vals(str, learning_config))[0:16]

    method = config['method']
    model_name = config['model_name']
    key = f"{method}-{model_name}-{train_hashid}"
    print(f"{key}\n====================")

    # init method from args
    method_class = getattr(methods, config['method'])
    method_var_dict = utils.filter_args(config, method_class.__init__)
    # Note: Changed name from method to model
    model = method_class(**method_var_dict)

    # init dataset from args
    dataset_class = getattr(datasets, config['dataset'], None)

    dataset_var_dict = utils.filter_args(config, dataset_class.__init__)
    dataset_var_dict["train_kwcoco_path"] = pathlib.Path(config['train_dataset'])
    dataset_var_dict["preprocessing_step"] = model.preprocessing_step
    dataset = dataset_class(**dataset_var_dict)
    dataset.setup("fit")

    # init trainer from args
    from types import SimpleNamespace
    args = SimpleNamespace(**dict(config))
    args.default_root_dir = pathlib.Path(config['workdir']) / key
    trainer = pl.Trainer.from_argparse_args(args)

    # TODO: perhaps netharn or some other config module could be executed here
    # to give pytorch lightning, netharn like directory structures

    # prime the model, incase it has a lazy layer
    batch = next(iter(dataset.train_dataloader()))
    result = model(batch["images"][[0], ...])

    # fit the model
    trainer.fit(model, dataset)


if __name__ == "__main__":
    main()

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("dataset")
    # parser.add_argument("method")

    # # parse the dataset and method strings
    # temp_args, _ = parser.parse_known_args()

    # # get the dataset and method classes
    # dataset_class = getattr(datasets, temp_args.dataset)
    # method_class = getattr(methods, temp_args.method)

    # # add the appropriate args to the parse
    # # for dataset, method, and trainer
    # parser = dataset_class.add_data_specific_args(parser)
    # parser = method_class.add_model_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)

    # # parse and pass to main
    # args = parser.parse_args()
    # main(args)
