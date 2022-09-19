from watch.tasks.fusion.datamodules.kwcoco_video_data import KWCocoVideoDataModule  # NOQA
from watch.tasks.fusion.methods import SequenceAwareModel

import pathlib

"""
The two Wrapped classes below are examples of why we should eventually factor out the current configuraiton system. LightningCLI interogates the __init__ methods belonging to LightningModule and LightningDataModule to decide which parameters can be configured.
"""


class WrappedKWCocoDataModule(KWCocoVideoDataModule):
    def __init__(
        self,
        train_dataset=None,
        vali_dataset=None,
        test_dataset=None,
        channels=None,
        batch_size=4,
        space_scale="native",
        num_workers="avail/2",
        time_steps=2,
        chip_size=128,
        neg_to_pos_ratio=0,
        chip_overlap=0,
    ):

        super().__init__(
            train_dataset=pathlib.Path(train_dataset) if (train_dataset is not None) else None,
            vali_dataset=pathlib.Path(vali_dataset) if (vali_dataset is not None) else None,
            test_dataset=pathlib.Path(test_dataset) if (test_dataset is not None) else None,
            batch_size=batch_size,
            channels=channels,
            space_scale=space_scale,
            num_workers=num_workers,
            time_steps=time_steps,
            chip_size=chip_size,
            neg_to_pos_ratio=neg_to_pos_ratio,
            chip_overlap=chip_overlap,
        )

        self.setup("fit")


class WrappedSequenceAwareModel(SequenceAwareModel):
    def __init__(
        self,
        dataset_stats,
        optimizer="RAdam",
        learning_rate=0.001,
        weight_decay=0.0,
        positive_change_weight=1.0,
        negative_change_weight=1.0,
        class_weights="auto",
        saliency_weights="auto",
        tokenizer="linconv",
        token_norm="none",
        decoder="mlp",
        global_class_weight=1.0,
        global_change_weight=1.0,
        global_saliency_weight=1.0,
        change_loss="cce",
        class_loss="focal",
        saliency_loss="focal",
        window_size=8,
        perceiver_depth=4,
        perceiver_latents=512,
        training_limit_queries=1024,
    ):

        super().__init__(
            dataset_stats=dataset_stats,
            optimizer=optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            positive_change_weight=positive_change_weight,
            negative_change_weight=negative_change_weight,
            class_weights=class_weights,
            saliency_weights=saliency_weights,
            tokenizer=tokenizer,
            token_norm=token_norm,
            decoder=decoder,
            global_class_weight=global_class_weight,
            global_change_weight=global_change_weight,
            global_saliency_weight=global_saliency_weight,
            change_loss=change_loss,
            class_loss=class_loss,
            saliency_loss=saliency_loss,
            window_size=window_size,
            perceiver_depth=perceiver_depth,
            perceiver_latents=perceiver_latents,
            training_limit_queries=training_limit_queries,
        )


def main():
    from pytorch_lightning.cli import LightningCLI

    import yaml
    from jsonargparse import set_loader, set_dumper

    # Not very safe, but needed to parse tuples e.g. datamodule.dataset_stats
    # TODO: yaml.SafeLoader + tuple parsing
    def custom_yaml_load(stream):
        return yaml.load(stream, Loader=yaml.FullLoader)
    set_loader('yaml_unsafe_for_tuples', custom_yaml_load)

    def custom_yaml_dump(data):
        return yaml.dump(data, Dumper=yaml.Dumper)
    set_dumper('yaml_unsafe_for_tuples', custom_yaml_dump)

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments(
                "data.dataset_stats",
                "model.dataset_stats",
                apply_on="instantiate")

    cli = MyLightningCLI(
        WrappedSequenceAwareModel,
        WrappedKWCocoDataModule,
        parser_kwargs=dict(parser_mode='yaml_unsafe_for_tuples'),
    )
    cli


if __name__ == '__main__':
    """
    Example invocation: python fit_lightning.py fit --data.train_dataset=$SMART_DVC/extern/onera_2018/onera_train.kwcoco.json --trainer.accelerator="gpu" --trainer.devices=0, --trainer.precision=16 --trainer.fast_dev_run=5
    """
    main()
