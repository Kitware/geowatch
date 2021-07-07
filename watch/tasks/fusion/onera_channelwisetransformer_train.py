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
        ls $DVC_DPATH/extern/onera_2018
        TRAIN_FPATH=$DVC_DPATH/extern/onera_2018/onera_train.kwcoco.json

        # Invoke the training script
        python -m watch.tasks.fusion.onera_channelwisetransformer_train \
            --method=MultimodalTransformerDotProdCD \
            --model_name=smt_it_joint_p8 \
            --train_kwcoco_path=$TRAIN_FPATH \
            --batch_size=1 \
            --num_workers=0 \
            --chip_size=32 \
            --workdir=$HOME/work/watch/fit/runs
    """
    from watch.tasks.fusion import fit
    defaults = dict(
        dataset="OneraCD_2018",
        method='MultimodalTransformerDotProdCD',

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
    fit.main(**defaults)


if __name__ == "__main__":
    main()
