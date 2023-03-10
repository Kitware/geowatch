import torch
import torch.nn
from torch import nn
from torch.utils.data import Dataset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from typing import List, Dict


class MWE_HeterogeneousModel(pl.LightningModule):
    """
    Example:
        >>> from lightning_cli_ckpt_path_error import *  # NOQA
        >>> dataset = MWE_HeterogeneousDataset()
        >>> self = MWE_HeterogeneousModel()
        >>> batch = [dataset[i] for i in range(2)]
        >>> self.forward(batch)
    """
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

        if dataset_stats is None:
            raise ValueError('must be given dataset stats')

        self.d_model = 16
        self.num_classes = 5

        self.stems = torch.nn.ModuleDict()
        self.dataset_stats = dataset_stats

        # THIS IS THE ISSUE
        # USING A SET HERE CAN RESULT IN INCONSISTENT ORDER AND
        # TORCH OR LIGHTNING DOESNT LIKE THAT
        self.known_sensorchan = {
            (mode['sensor'], mode['channels'], mode['num_bands'])
            for mode in self.dataset_stats['known_modalities']
        }
        for sensor, channels, num_bands in self.known_sensorchan:
            if sensor not in self.stems:
                self.stems[sensor] = torch.nn.ModuleDict()
            self.stems[sensor][channels] = torch.nn.Conv2d(num_bands, self.d_model, kernel_size=1)

        self.backbone = torch.nn.Transformer(
            d_model=self.d_model,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=32,
            batch_first=True
        )

        self.heads = torch.nn.ModuleDict()
        self.heads['class'] = torch.nn.Conv2d(self.d_model, self.num_classes, kernel_size=1)

    @property
    def main_device(self):
        for key, item in self.state_dict().items():
            return item.device

    def tokenize_inputs(self, item: Dict):
        device = self.device

        input_sequence = []
        for input_item in item['inputs']:
            stem = self.stems[input_item['sensor_code']][input_item['channel_code']]
            out = stem(input_item['data'])
            tokens = out.view(self.d_model, -1).T
            input_sequence.append(tokens)

        output_sequence = []
        for output_item in item['outputs']:
            shape = tuple(output_item['dims']) + (self.d_model,)
            tokens = torch.rand(shape, device=device).view(-1, self.d_model)
            output_sequence.append(tokens)
        if len(input_sequence) == 0 or len(output_sequence) == 0:
            return None, None
        in_tokens = torch.concat(input_sequence, dim=0)
        out_tokens = torch.concat(output_sequence, dim=0)
        return in_tokens, out_tokens

    def forward(self, batch: List[Dict]) -> List[Dict]:
        batch_in_tokens = []
        batch_out_tokens = []

        given_batch_size = len(batch)
        valid_batch_indexes = []

        # Prepopulate an output for each input
        batch_logits = [{} for _ in range(given_batch_size)]

        # Handle heterogeneous style inputs on a per-item level
        for batch_idx, item in enumerate(batch):
            in_tokens, out_tokens = self.tokenize_inputs(item)
            if in_tokens is not None:
                valid_batch_indexes.append(batch_idx)
                batch_in_tokens.append(in_tokens)
                batch_out_tokens.append(out_tokens)

        # Some batch items might not be valid
        valid_batch_size = len(valid_batch_indexes)
        if not valid_batch_size:
            # No inputs were valid
            return batch_logits

        # Pad everything into a batch to be more efficient
        padding_value = -9999.0
        input_seqs = nn.utils.rnn.pad_sequence(
            batch_in_tokens,
            batch_first=True,
            padding_value=padding_value,
        )
        output_seqs = nn.utils.rnn.pad_sequence(
            batch_out_tokens,
            batch_first=True,
            padding_value=padding_value,
        )

        input_masks = input_seqs[..., 0] > padding_value
        output_masks = output_seqs[..., 0] > padding_value
        input_seqs[~input_masks] = 0.
        output_seqs[~output_masks] = 0.

        decoded = self.backbone(
            src=input_seqs,
            tgt=output_seqs,
            src_key_padding_mask=~input_masks,
            tgt_key_padding_mask=~output_masks,
        )
        B = valid_batch_size
        decoded_features = decoded.view(B, -1, 3, 3, self.d_model)
        decoded_masks = output_masks.view(B, -1, 3, 3)

        # Reconstruct outputs corresponding to the inputs
        for batch_idx, feat, mask in zip(valid_batch_indexes, decoded_features, decoded_masks):
            item_feat = feat[mask].view(-1, 3, 3, 16).permute(0, 3, 1, 2)
            item_logits = batch_logits[batch_idx]
            for head_name, head_layer in self.heads.items():
                head_logits = head_layer(item_feat)
                item_logits[head_name] = head_logits
            batch_logits.append(item_logits)
        return batch_logits

    def forward_step(self, batch: List[Dict], with_loss=False, stage='unspecified'):
        """
        Generic forward step used for test / train / validation
        """
        batch_logits : List[Dict] = self.forward(batch)
        outputs = {}
        outputs['logits'] = batch_logits

        if with_loss:
            losses = []
            valid_batch_size = 0
            for item, item_logits in zip(batch, batch_logits):
                if len(item_logits):
                    valid_batch_size += 1
                for head_name, head_logits in item_logits.items():
                    head_target = torch.stack([label['data'] for label in item['labels'] if label['head'] == head_name], dim=0)
                    # dummy loss function
                    head_loss = torch.nn.functional.mse_loss(head_logits, head_target)
                    losses.append(head_loss)
            total_loss = sum(losses) if len(losses) > 0 else None
            if total_loss is not None:
                self.log(f'{stage}_loss', total_loss, prog_bar=True, batch_size=valid_batch_size)
            outputs['loss'] = total_loss

        return outputs

    def training_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch, with_loss=True, stage='train')
        if outputs['loss'] is None:
            return None
        return outputs

    def validation_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch, with_loss=True, stage='val')
        return outputs

    def test_step(self, batch, batch_idx=None):
        outputs = self.forward_step(batch, with_loss=True, stage='test')
        return outputs


class MWE_HeterogeneousDatamodule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=0):
        super().__init__()
        self.save_hyperparameters()
        self.torch_datasets = {}
        self.dataset_stats = None
        self._did_setup = False

    def setup(self, stage):
        if self._did_setup:
            return
        self.torch_datasets['train'] = MWE_HeterogeneousDataset()
        self.torch_datasets['test'] = MWE_HeterogeneousDataset()
        self.torch_datasets['vali'] = MWE_HeterogeneousDataset()
        self.dataset_stats = self.torch_datasets['train']
        self._did_setup = True

    def train_dataloader(self):
        return self._make_dataloader('train', shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader('vali', shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader('test', shuffle=False)

    @property
    def train_dataset(self):
        return self.torch_datasets.get('train', None)

    @property
    def test_dataset(self):
        return self.torch_datasets.get('test', None)

    @property
    def vali_dataset(self):
        return self.torch_datasets.get('vali', None)

    def _make_dataloader(self, stage, shuffle=False):
        loader = self.torch_datasets[stage].make_loader(
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )
        return loader


# Global hack because having a hard time linking the args
dataset_stats =  {
    'known_modalities': [
        {'sensor': 'sensor1', 'channels': 'rgb', 'num_bands': 3},
        {'sensor': 'sensor2', 'channels': 'rgb', 'num_bands': 3},
        {'sensor': 'sensor3', 'channels': 'rgb', 'num_bands': 3},
        {'sensor': 'sensor4', 'channels': 'rgb', 'num_bands': 3},
        {'sensor': 'sensor1', 'channels': 'rgb', 'num_bands': 3},
        {'sensor': 'sensor2', 'channels': 'ir', 'num_bands': 3},
        {'sensor': 'sensor2', 'channels': 'depth', 'num_bands': 3},
        {'sensor': 'sensor4', 'channels': 'flowxy', 'num_bands': 2},
    ],
    'known_tasks': [
        {'name': 'class'},
    ]
}


class MWE_HeterogeneousDataset(Dataset):
    """
    A dataset that produces heterogeneous outputs

    Example:
        >>> from lightning_cli_ckpt_path_error import *  # NOQA
        >>> self = MWE_HeterogeneousDataset()
        >>> self[0]
    """
    def __init__(self):
        super().__init__()
        self.rng = np.random
        # In practice the dataset computes stats about itself.
        # In this example we just hard code it.
        self.dataset_stats = dataset_stats

    def __len__(self):
        return 100

    def __getitem__(self, index):
        """
        Constructs a sequence of:
            * inputs - a list of observations
            * outputs - a list of what we want to predict
            * labels - ground truth if we have it
        """
        inputs = []
        outputs = []
        labels = []
        num_frames = self.rng.randint(1, 10)
        for frame_index in range(num_frames):
            had_input = 0
            # In general we may have any number of observations per frame
            for modality in self.dataset_stats['known_modalities']:
                sensor = modality['sensor']
                channels = modality['channels']
                num_bands = modality['num_bands']

                # Randomly include each sensorchan on each frame
                if self.rng.rand() > 0.5:
                    had_input = 1
                    c = num_bands
                    if channels == 'rgb':
                        h, w = 10, 10
                        if sensor == 'sensor3':
                            h, w = 17, 17
                    elif channels == 'ir':
                        h, w = 5, 5
                    elif channels == 'depth':
                        h, w = 7, 7
                    inputs.append({
                        'type': 'input',
                        'channel_code': channels,
                        'sensor_code': sensor,
                        'frame_index': frame_index,
                        'data': torch.rand(c, h, w),
                    })
            if had_input:
                task = 'class'
                oh, ow = 3, 3
                outputs.append({
                    'type': 'output',
                    'head': task,
                    'frame_index': frame_index,
                    'dims': (oh, ow),
                })
                labels.append({
                    'type': 'label',
                    'head': task,
                    'frame_index': frame_index,
                    'data': torch.rand(5, 3, 3),
                })
        item = {
            'inputs': inputs,
            'outputs': outputs,
            'labels': labels,
        }
        return item

    def make_loader(self, subset=None, batch_size=1, num_workers=0, shuffle=False,
                    pin_memory=False):
        """
        Use this to make the dataloader so we ensure that we have the right
        worker init function.
        """
        if subset is None:
            dataset = self
        else:
            dataset = subset
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            shuffle=shuffle, pin_memory=pin_memory,
            collate_fn=lambda x: x  # disable collation
        )
        return loader


class MWE_LightningCLI(LightningCLI):
    ...

    # Having trouble linking the dataset stats to the model
    # def add_arguments_to_parser(self, parser):
    #     def data_value_getter(key):
    #         # Hack to call setup on the datamodule before linking args
    #         def get_value(data):
    #             if not data._did_setup:
    #                 data.setup('fit')
    #             return getattr(data, key)
    #         return get_value

    #     # pass dataset stats to model after initialization datamodule
    #     parser.link_arguments(
    #         "data",
    #         "model.init_args.dataset_stats",
    #         compute_fn=data_value_getter('dataset_stats'),
    #         apply_on="instantiate")
    #     # parser.link_arguments(
    #     #     "data",
    #     #     "model.init_args.classes",
    #     #     compute_fn=data_value_getter('classes'),
    #     #     apply_on="instantiate")
    #     super().add_arguments_to_parser(parser)


def main():
    MWE_LightningCLI(
        model_class=MWE_HeterogeneousModel,
        datamodule_class=MWE_HeterogeneousDatamodule,
    )


if __name__ == '__main__':
    """
    CommandLine:
        cd ~/code/watch/dev/mwe/

        DEFAULT_ROOT_DIR=./mwe_train_dir

        python lightning_cli_ckpt_path_error.py fit --config "
            data:
                num_workers: 2
            optimizer:
              class_path: torch.optim.Adam
              init_args:
                lr: 1e-7
            lr_scheduler:
              class_path: torch.optim.lr_scheduler.ExponentialLR
              init_args:
                gamma: 0.1
            trainer:
              accumulate_grad_batches: 16
              callbacks:
                - class_path: pytorch_lightning.callbacks.ModelCheckpoint
                  init_args:
                    monitor: val_loss
                    mode: min
                    save_top_k: 5
                    auto_insert_metric_name: true
              default_root_dir     : $DEFAULT_ROOT_DIR
              accelerator          : gpu
              devices              : 0,
              #devices              : 0,1
              #strategy             : ddp
              check_val_every_n_epoch: 1
              enable_checkpointing: true
              enable_model_summary: true
              log_every_n_steps: 5
              logger: true
              max_steps: 10000
              num_sanity_val_steps: 0
              replace_sampler_ddp: true
              track_grad_norm: -1
              limit_val_batches: 10
        "

        CKPT_FPATH=$(python -c "import pathlib; print(list(pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs').glob('*/checkpoints/*.ckpt'))[0])")
        CONFIG_FPATH=$(python -c "import pathlib; print(sorted(pathlib.Path('$DEFAULT_ROOT_DIR/lightning_logs').glob('*/config.yaml'))[-1])")
        echo "CONFIG_FPATH = $CONFIG_FPATH"
        echo "CKPT_FPATH = $CKPT_FPATH"

        python lightning_cli_ckpt_path_error.py fit --config "$CONFIG_FPATH" --ckpt_path="$CKPT_FPATH"
    """
    main()
