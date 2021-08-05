#!/usr/bin/env python
## flake8: noqa
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
# import ubelt as ub
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torchvision.models as models
import itertools
import numpy as np

# import moco.loader
# import watch.tasks.rutgers_material_seg.utils.utils as utils
# from watch.tasks.rutgers_material_seg.datasets.iarpa_dataset import SequenceDataset
from watch.tasks.rutgers_material_seg.datasets import build_dataset
import watch.tasks.rutgers_material_seg.models.moco as moco

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# mask_mapping = {0: 0,    # 0, unknown
#                 179: 1,  # 179, urban land
#                 226: 2,  # 226, agriculture land
#                 105: 3,  # 105, rangeland, non-forest, non farm, green land
#                 150: 4,  # 150, forest land
#                 29: 5,   # 29, water
#                 255: 6}  # 255, barren land, mountain, rock, dessert

mask_mapping = {0: "unknown",    # 0, unknown
                1: "urban" ,  # 179, urban land
                2: "agriculture",  # 226, agriculture land
                3: "rangeland",  # 105, rangeland, non-forest, non farm, green land
                4: "forest",  # 150, forest land
                5: "water",   # 29, water
                6: "barren"}  # 255, barren land, mountain, rock, dessert

possible_combinations = [list(i) for i in itertools.product([0, 1], repeat=len(mask_mapping.keys()))]
for index, item in enumerate(possible_combinations):
    num_labels = len(np.argwhere(np.array(item) == 1))
    if num_labels == 0:
        continue
    possible_combinations[index] = np.divide(possible_combinations[index], num_labels)

# print(np.unique(possible_combinations, return_counts=True, axis=0))
# possible_combinations = np.divide(possible_combinations, len(mask_mapping.keys()))
# print(possible_combinations)
verbose_labels = {}
for index, item in enumerate(possible_combinations):
    verbose_label = ""
    for label_index, label in enumerate(item):
        if label != 0:
            verbose_label += f"{mask_mapping[label_index]}: {label}, "
    verbose_labels[index] = verbose_label

# print(verbose_labels)

# project_root = "/home/native/projects/watch/watch/tasks/rutgers_material_seg/"
# main_config_path = f"{project_root}/configs/main.yaml"

# initial_config = utils.load_yaml_as_dict(main_config_path)
# experiment_config_path = f"{project_root}/configs/{initial_config['dataset']}.yaml"

# experiment_config = utils.config_parser(experiment_config_path,experiment_type="training")
# config = {**initial_config, **experiment_config}
# config['start_time'] = datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

# project_name = "MoCo Experiments"#_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M')}"
# experiment_name = f"MoCo_{datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}"
# cometml_experiment = comet_ml.Experiment(api_key=config['cometml']['api_key'],
#                                     project_name=project_name,
#                                     workspace=config['cometml']['workspace'],
#                                     display_summary_level=0)
# cometml_experiment.set_name(experiment_name)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # print(model)
    # print(args.gpu)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        print("hello")
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [  # NOQA
            # transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]

    # train_dataset = datasets.ImageFolder(
        # traindir,
        # moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    # coco_json = "/media/native/data/data/smart_watch_dvc/drop0_aligned_msi/material_labels2.kwcoco.json"
    # coco_fpath = ub.expandpath(coco_json)
    # dset = kwcoco.CocoDataset(coco_fpath)
    # sampler = ndsampler.CocoSampler(dset)
    # number_of_timestamps, h, w = 1, 64, 64
    # window_dims = (number_of_timestamps, h, w) #[t,h,w]
    # input_dims = (h, w)
    # # channels = 'red|green|blue|nir|swir16|swir22|cirrus'
    # channels = 'red|green|blue'
    # train_dataset = SequenceDataset(sampler, window_dims, input_dims, channels)

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None

    # train_loader = train_dataset.make_loader(batch_size=args.batch_size,
    #                                          pin_memory=True,
    #                                          drop_last=True
    #                                          )

    train_loader = build_dataset(dataset_name="deepglobe",
                                 root="/media/native/data/data/DeepGlobe/crops/",
                                 batch_size=args.batch_size,
                                 num_workers=1,
                                 split="train",
                                 image_size="300x300")

    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', '6.3')
    data_time = AverageMeter('Data', '6.3')
    losses = AverageMeter('Loss', '.4')
    top1 = AverageMeter('Acc@1', '6.2')
    top5 = AverageMeter('Acc@5', '6.2')  # NOQA
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    total_loss = 0
    end = time.time()

    for i, batch in enumerate(train_loader):
        # measure data loading time
        outputs = batch
        # image1, mask = outputs['inputs']['im'].data[0], batch['label']['class_masks'].data[0]
        # mask = torch.stack(mask)
        # mask[mask==-1]=0

        image1 = outputs['inputs']['image']
        mask = batch['inputs']['mask']
        labels = batch['inputs']['labels']
        # print(labels)
        bs, c, h, w = image1.shape

        mask = mask.long().squeeze(1)
        # labels = labels.long().squeeze(2)

        image1 = image1.squeeze(2)
        image2 = image1.clone()
        images = [image1, image2]
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        # print(target.shape)
        # print(labels.shape)
        loss = criterion(output, labels)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output, target, topk=(1,))
        acc1, preds = accuracy(output, target, topk=(1,))
        # print(preds[0])
        verbose_preds = [verbose_labels[pred.data] for pred in preds]
        # verbose_preds = [print(pred) for pred in preds]
        print(verbose_preds)
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        # top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        total_loss += loss.item()

        # cometml_experiment.log_metric("Training Accuracy", acc1, epoch=epoch+1)

        # if i % args.print_freq == 0:
        #     print(f"accuracy: {acc1}")
        #     progress.display(i)

    # cometml_experiment.log_metric("Training Loss", total_loss, epoch=epoch+1)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        stats = fmtstr.format(**self.__dict__)
        # print(self.fmt)
        # stats = f"{self.__dict__['name']} {self.__dict__['val']:{self.fmt}f} {self.__dict__['avg']:{self.fmt}f})"
        # stats = fmtstr.format(**self.__dict__)
        return stats


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        # print(meters)
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred[0]


if __name__ == '__main__':
    main()
