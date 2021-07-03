model_names = [
    "smt_it_joint_p8",
    "smt_it_stm_p8",
    "smt_it_hwtm_p8",
]

methods = [
    "MultimodalTransformerDotProdCD",
    "MultimodalTransformerDirectCD",
    'voting',
]


def main():
    """
    Example:
        # Set the path to your data
        DVC_DPATH=$HOME/Projects/smart_watch_dvc
        DVC_DPATH=$HOME/data/dvc-repos/smart_watch_dvc
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # Invoke the training script

        # This task required 17GB on a 3090
        python -m watch.tasks.fusion.fit \
            --model_name=smt_it_stm_p8 \
            --method=MultimodalTransformerDotProdCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=16 \
            --num_workers=4 \
            --chip_size=128 \
            --workdir=$HOME/work/watch/fit/runs

        # This task required 20GB on a 3090
        python -m watch.tasks.fusion.fit \
            --model_name=smt_it_joint_p8 \
            --method=MultimodalTransformerDirectCD \
            --train_dataset=$TRAIN_FPATH \
            --batch_size=4 \
            --num_workers=4 \
            --chip_size=96 \
            --workdir=$HOME/work/watch/fit/runs
    """
    from watch.tasks.fusion import fit
    fit.main(dataset='OneraCD_2018')

    # args = SimpleNamespace(**dict(config))
    #     dataset="OneraCD_2018",

    #     # dataset params
    #     train_kwcoco_path=pathlib.Path(config['train_dataset']),
    #     batch_size=config['batch_size'],
    #     num_workers=config['num_workers'],
    #     chip_size=config['chip_size'],

    #     # model params
    #     window_size=8,
    #     learning_rate=1e-3,
    #     weight_decay=0,
    #     dropout=0,
    #     pos_weight=5.0,

    #     # trainer params
    #     gpus=1,
    #     #accelerator="ddp",
    #     precision=16,
    #     max_epochs=200,
    #     accumulate_grad_batches=2,
    #     terminate_on_nan=True,
    # )
    # train_hashid = ub.hash_data(ub.map_vals(str, args.__dict__))[0:16]

    # method = config['method']
    # model_name = config['model_name']
    # print(f"{method} / {model_name}\n====================")
    # args.method = method
    # args.model_name = model_name
    # key = f"{method}-{model_name}-{train_hashid}"
    # args.default_root_dir = pathlib.Path(config['workdir']) / key


if __name__ == "__main__":
    main()
