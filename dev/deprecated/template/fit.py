#!/usr/bin/env python
"""
This is a Template for writing training logic.

NOTE: This template has been deprecated in favor of a pytorch-lightning variant
"""
import kwcoco
import ndsampler
import scriptconfig as scfg
import ubelt as ub
import numpy as np
import torch
import kwimage


class TemplateFitConfig(scfg.Config):
    """
    Name this config object based on your machine learning method.

    Use this docstring to write a small blurb about it. It will be printed
    when someone runs ``python -m watch.tasks.template.fit --help``
    """

    default = {
        'name': scfg.Value('untitled', help='an experiment name'),

        'workdir': scfg.Path(
            '~/work/smart/watch/tasks/template',
            help='place where the experiment can write things'),

        'train_dataset': scfg.Value(None, help=ub.paragraph(
            ''' A path to a kwcoco dataset used to learn model parameters ''')),
        'vali_dataset': scfg.Value(None, help=ub.paragraph(
            ''' A optional path to a kwcoco dataset used to adjust hyper parameters. ''')),
        'test_dataset': scfg.Value(None, help=ub.paragraph(
            ''' A optional path to a kwcoco dataset used to evaluate performance. ''')),
    }


# A Netharn implementation of the training logic is optional.
import netharn as nh  # NOQA


# But if you do use it, you will find out that its outputs are fairly nice.
class TemplateFitHarn(nh.FitHarn):
    def run_batch(harn, batch):
        """
        See netharn examples and docs for how to write this

        Example:
            >>> from watch.tasks.template.fit import *  # NOQA
            >>> harn = setup_datasets_and_training_harness(
            ...     train_dataset='special:vidshapes8-multispectral')
            >>> harn.initialize()
            >>> batch = harn._demo_batch()
            >>> outputs, loss = harn.run_batch(batch)
            >>> print('outputs = {}'.format(ub.repr2(outputs, nl=1)))
            >>> print('loss = {}'.format(ub.repr2(loss, nl=1)))
        """

        late_fused_inputs = batch['inputs']
        cidxs = batch['labels']['cidxs']

        an_input = ub.peek(late_fused_inputs.values())

        # Hacked one channel inputs
        im = an_input.sum(dim=1, keepdims=True)
        output = harn.model(im)

        # This is not a real loss function
        target = cidxs.sum()
        predict = output.sum()
        weird_loss = ((target - predict) ** 2).sum()

        outputs = {
            'model_output': output,
        }

        loss = {
            'my_made_up_loss': weird_loss,
        }
        return outputs, loss


class TemplateDataset(torch.utils.data.Dataset):
    """
    The ndsampler and kwcoco packages can help you write this.

    Example:
        >>> # xdoctest: +SKIP
        >>> from watch.tasks.template.fit import *  # NOQA
        >>> coco_dset = kwcoco.CocoDataset.demo('vidshapes2-multispectral')
        >>> input_dims = (128, 128)
        >>> window_dims = 'full'
        >>> self = TemplateDataset(coco_dset, input_dims, window_dims)
        >>> index = len(self) // 2
        >>> item = self[index]
    """

    def __init__(self, coco_dset, input_dims, window_dims):
        self.coco_dset = coco_dset
        self.input_dims = input_dims
        self.window_dims = window_dims
        self.sampler = ndsampler.CocoSampler(self.coco_dset)

        self.task_grid = ndsampler.coco_regions.new_image_sample_grid(
            self.sampler.dset,
            # self.task_grid = self.sampler.regions.new_sample_grid(
            #     task='image_detection',
            window_dims=window_dims,
            # window_dims='full',  # full image
            # window_dims=(128, 128),  # sub image
            window_overlap=0,
        )
        self.grid = self.task_grid['targets']
        # .values()))

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, index):
        tr = self.grid[index]
        tr['channels'] = '<all>'
        # tr['channels'] = 'B1|B8|B11|B8a' # fixme on images
        sample = self.sampler.load_sample(tr)
        hwc = sample['im'].astype(np.float32)
        dets = sample['annots']['frame_dets'][0]

        cidxs = [dets.classes.id_to_idx[cid] for cid in dets.data['cids']]

        # dims are row-major (e.g. h, w), dsize is always (w, h)
        input_dsize = self.input_dims[-2:][::-1]
        hwc, resize_info = kwimage.imresize(
            hwc, dsize=input_dsize, antialias=True, interpolation='linear',
            return_info=True)

        resize_transform = kwimage.Affine.coerce(**resize_info).matrix
        dets = dets.warp(resize_transform)

        chw = torch.from_numpy(hwc.transpose(2, 0, 1)).float()
        tlbr = torch.from_numpy(dets.boxes.to_tlbr().data).float()
        chan_spec = 'B1|B8|B11|B8a'

        item = {
            'inputs': {
                chan_spec: chw,
                # Unfused inputs should be key-labeled by channel specs.

            },
            'labels': {
                'cidxs': torch.from_numpy(np.array(cidxs)).long().view(-1)[0:1],
                'tlbr': tlbr.view(-1)[0:1],
            },
        }
        return item


def setup_datasets_and_training_harness(cmdline=False, **kwargs):
    """
    Kwargs:
        see TemplateFitConfig

    Example:
        >>> # xdoctest: +SKIP
        >>> from watch.tasks.template.fit import *  # NOQA
        >>> kwargs = {
        ...     'train_dataset': 'special:vidshapes32-multispectral',
        ...     'vali_dataset': 'special:vidshapes16-multispectral',
        ... }
        >>> cmdline = False
        >>> harn = setup_datasets_and_training_harness(**kwargs)
        >>> harn.initialize()
        >>> #harn.run()

    """
    config = TemplateFitConfig(data=kwargs, cmdline=cmdline)
    coco_datasets = {}
    for key in {'train_dataset', 'vali_dataset', 'test_dataset'}:
        if config[key] is not None:
            tag = key.split('_')[0]  # tag is train, vali, or test
            coco_datasets[tag] = kwcoco.CocoDataset.coerce(config[key])

    datasets = {}
    datasets['train'] =  TemplateDataset(
        coco_datasets['train'],
        input_dims=(128, 128),
        window_dims='full',
    )
    for key in ['vali', 'test']:
        coco_dset = coco_datasets.get(key, None)
        if coco_dset is not None:
            datasets[key] = TemplateDataset(
                coco_dset,
                input_dims=(128, 128),
                window_dims='full'
            )

    print('datasets = {!r}'.format(datasets))
    loaders = nh.api.Loaders.coerce(datasets, ub.dict_union(config, {
        'batch_size': 1,
        'workers': 0,
    }))
    print('loaders = {!r}'.format(loaders))

    hyper = nh.HyperParams(**{
        'name': config['name'],
        'workdir': config['workdir'],
        'xpu': nh.XPU('cpu'),

        'datasets': datasets,
        'loaders': loaders,

        # 'model': nh.models.ToyNet2d(),
        'model': (nh.models.ToyNet2d, {}),

        # 'optimizer': nh.api.Optimizer.coerce(config),
        # 'initializer': nh.api.Initializer.coerce(config),
        # 'scheduler': nh.api.Scheduler.coerce(config),
        # 'dynamics': nh.api.Dynamics.coerce(config),
        'monitor': (nh.Monitor, dict(
            minimize=['loss'], patience=5, max_epoch=10
        )),

        # 'augment'
        # 'extra'
        # 'other'
    })
    print('hyper = {!r}'.format(hyper))

    harn = TemplateFitHarn(hyper)
    harn.preferences.update({
        'num_keep': 10,
        'keep_freq': 5,
        'export_modules': ['watch'],  # TODO
        'prog_backend': 'progiter',  # alternative: 'tqdm'
        'keyboard_debug': False,
        'deploy_after_error': True,
        # 'timeout': config['timeout'],

        # 'allow_unicode': config['allow_unicode'],
        # 'eager_dump_tensorboard': config['eager_dump_tensorboard'],
        # 'dump_tensorboard': config['dump_tensorboard'],
    })
    harn.intervals.update({
        'log_iter_train': 100,
        'test': 0,
        # 'vali': config['vali_intervals'],
        # 'vali': 100,
        # config['vali_intervals'],
    })
    harn.script_config = config

    return harn


def main(**kwargs):
    # Parse the config for this script and allow the command line (i.e. the
    # current value of sys.argv) to overwrite
    harn = setup_datasets_and_training_harness(cmdline=True, **kwargs)
    harn.initialize()
    harn.run()


if __name__ == '__main__':
    """
    CommandLine:
        python -m watch.tasks.template.fit --help

        python -m watch.tasks.template.fit \
            --train_dataset=special:vidshapes32-multispectral \
            --vali_dataset=special:vidshapes16-multispectral
    """
    main()
