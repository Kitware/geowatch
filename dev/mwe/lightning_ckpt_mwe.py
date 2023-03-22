import torch
import torch.nn
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pytorch_lightning.cli import LightningCLI


class SimpleModel(pl.LightningModule):
    def __init__(self, ascending=False):
        super().__init__()
        self.layers = torch.nn.ModuleDict()
        if ascending:
            self.layers['layer1'] = torch.nn.Conv2d(3, 5, 1, 1)
            self.layers['layer2'] = torch.nn.Conv2d(5, 7, 1, 1)
        else:
            self.layers['layer2'] = torch.nn.Conv2d(5, 7, 1, 1)
            self.layers['layer1'] = torch.nn.Conv2d(3, 5, 1, 1)

    def forward(self, inputs):
        x = inputs
        x = self.layers['layer1'](x)
        x = self.layers['layer2'](x)
        return x

    def forward_step(self, batch):
        """
        Generic forward step used for test / train / validation
        """
        batch = torch.stack(batch, dim=0)
        x = self.forward(batch)
        loss = x.sum()
        return loss

    def training_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch)
        return outputs

    def validation_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch)
        return outputs


class SimpleDataset(Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index):
        return torch.rand(3, 10, 10)


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=0):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage):
        self.train_dataset = SimpleDataset()
        self.vali_dataset = SimpleDataset()

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.vali_dataset, shuffle=False)

    def _make_dataloader(self, dataset, shuffle=False):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle, pin_memory=True,
            collate_fn=lambda x: x
        )
        return loader


def main():
    LightningCLI(
        model_class=SimpleModel,
        datamodule_class=SimpleDataModule,
    )


if __name__ == '__main__':
    """
    CommandLine:
        cd ~/code/watch/dev/mwe/

        DEFAULT_ROOT_DIR=./mwe_train_dir

        python lightning_ckpt_mwe.py fit --config "
            model:
                ascending: True
            data:
                num_workers: 8
                batch_size: 2
            optimizer:
              class_path: torch.optim.Adam
              init_args:
                lr: 1e-7
            trainer:
              default_root_dir     : $DEFAULT_ROOT_DIR
              accelerator          : gpu
              devices              : 0,
              max_epochs: 100
        "

        CKPT_FPATH=$(python -c "import pathlib; print(sorted(pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs').glob('*/checkpoints/*.ckpt'))[-1])")
        echo "CKPT_FPATH = $CKPT_FPATH"

        # Even though the model is "the same", the ordering of layers is different and
        # and that causes an error
        python lightning_ckpt_mwe.py fit --config "
            model:
                ascending: False
            data:
                num_workers: 8
                batch_size: 2
            optimizer:
              class_path: torch.optim.Adam
              init_args:
                lr: 1e-7
            trainer:
              default_root_dir     : $DEFAULT_ROOT_DIR
              accelerator          : gpu
              devices              : 0,
              max_epochs: 100
        " --ckpt_path="$CKPT_FPATH"
    """
    main()
