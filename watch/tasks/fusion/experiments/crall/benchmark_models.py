import ubelt as ub

try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


def benchmark_models():
    # https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
    # TODO: profile attention_impl
    import netharn as nh
    # from watch.tasks.fusion import datamodules
    import torch.profiler
    from watch.tasks.fusion.architectures import transformer
    from watch.tasks.fusion.methods.channelwise_transformer import MultimodalTransformer
    # from torch.profiler import profile, ProfilerActivity, record_function
    # datamodule = datamodules.KWCocoVideoDataModule(
    #     train_dataset='special:vidshapes8', num_workers=0)
    # datamodule.setup('fit')
    # loader = datamodule.train_dataloader()
    # batch = next(iter(loader))
    #self = MultimodalTransformer(arch_name='smt_it_joint_p8')
    # frames = batch[0]['frames']
    # collate_images = torch.cat([frame['modes']['r|g|b'][None, :].float() for frame in frames], dim=0)
    device = nh.XPU.coerce('cpu').main_device
    device = nh.XPU.coerce('gpu').main_device
    #device = nh.XPU.coerce('auto').main_device
    # images = collate_images[None, :].to(device)

    input_grid = list(ub.named_product({
        'S': [32, 64, 96, 128],
        # 'T': [2, 3, 5, 9],
        # 'T': [2, 5, 9],
        # 'T': [2, 5, 9],
        # 'T': [2],
        'T': [2, 9, 11, 17],
        # 'M': [3, 5, 7, 11, 13, 32, 64, 128, 256],
        'M': [3, 5, 7, 11, 13, 32, 64],
    }))

    # coco_fpath = ub.expandpath('$HOME/data/work/toy_change/vidshapes_msi_train/data.kwcoco.json')
    import watch
    from watch.tasks.fusion import datamodules
    if 1:
        coco_dset = watch.demo.coerce_kwcoco('watch-msi')
        channels = "B11,r|g|b,B1|B8|B11"
        datamodule = datamodules.KWCocoVideoDataModule(
            train_dataset=coco_dset,
            chip_size=224, batch_size=1, time_steps=3,
            channels=channels,
            normalize_inputs=True, neg_to_pos_ratio=0, num_workers='avail/2', true_multimodal=True,
        )
        datamodule.setup('fit')
        batch = next(iter(datamodule.train_dataloader()))

    encoder_info = transformer.encoder_configs.copy()

    import pandas as pd
    df = pd.DataFrame(encoder_info).T
    df['num_atten_mechs'] = df['axes'].map(len)
    df['axes'] = df['axes'].map(lambda x: '.'.join([
        ''.join([p[0] for p in part])
        for part in x
    ]))
    df['default_shape'] = df['default_shape'].map(lambda x: ''.join([c[0] for c in x]))
    df.drop('axes', axis=1)
    df.index.name = 'arch_name'
    # df = df.drop('axes', axis=1)
    # df = df.drop('default_shape', axis=1)
    flags = (
        (df['n_layers'] >= 12)
        & (df['n_layers'] < 24)
        & (df['num_atten_mechs'] == 1)
    )
    df_subset = df[flags]

    print(df_subset.to_string())

    df = df[df['n_heads'] >= 8]

    chosen_arch = [
        # 'smt_it_stm_p8',
        'smt_it_joint_p8',
        # 'smt_it_hwtm_p8',
        # 'smt_it_joint_n12',
    ]
    # chosen_arch = df_subset.index.values.tolist()

    # all_arch_names = list(transformer.encoder_configs.keys())
    model_basis = {
        'arch_name': chosen_arch,
        'squash_modes': [True, False],
        'window_size': [8, 4],
        'attention_impl': [
            'exact',
            'performer'
            # 'reformer'
        ],
    }
    model_grid = list(ub.named_product(model_basis))
    import itertools as it
    bench_grid = list(it.product(model_grid, input_grid))

    rows = []
    nicerows = []
    self = None
    images = None
    # train_prof = None
    output = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()

    # Pure memory benchmarks
    for modelkw, inputkw in ub.ProgIter(bench_grid, verbose=3):

        # for arch_name in ['smt_it_stm_p8']:
        M = inputkw['M']
        T = inputkw['T']
        S = inputkw['S']
        row = {}
        row.update(inputkw)
        row.update(modelkw)
        row.update(ub.dict_isect(transformer.encoder_configs[modelkw['arch_name']], {'n_layers', 'embedding_size', 'n_heads'}))

        # Get dummy input

        images = torch.rand(1, T, M, S, S).to(device)

        errored = False

        try:
            self = MultimodalTransformer(input_channels=M, **modelkw)
            num_params = nh.util.number_of_parameters(self)
            self = self.to(device)
            optim = torch.optim.SGD(self.parameters(), lr=1e-9)
            # with torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as train_prof:
            # with torch.profiler.profile(activities=[ProfilerActivity.CUDA], record_shapes=False, profile_memory=True) as train_prof:
            #     with record_function(f"train_{arch_name}"):
            optim.zero_grad()
            output = self(images)['change']
            output.sum().backward()
            optim.step()

            # total_memory = sum(event.cuda_memory_usage for event in train_prof.events())
            # total_mem_str = xdev.byte_str(total_memory)
            # print(total_mem_str)

            row.update({
                'num_params': num_params,
            })
            mem_stats = ({
                'max_mem_alloc': torch.cuda.max_memory_allocated(),
                'mem_alloc': torch.cuda.memory_allocated(),
                'mem_reserve': torch.cuda.memory_reserved(),
                'max_mem_reserve': torch.cuda.max_memory_reserved(),
            })
            row.update(mem_stats)
        except RuntimeError:
            errored = True
            pass
        except AssertionError:
            errored = True
            pass

        self = None
        images = None
        # train_prof = None
        output = None
        optim = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_max_memory_allocated()

        if not errored:
            rows.append(row)
            nicerow = row.copy()
            nicestats = {k + '_str': xdev.byte_str(v, unit='GB') if isinstance(v, int) else v for k, v in mem_stats.items()}
            nicerow.update(nicestats)
            nicerows.append(nicerow)
            print(nicerow)

    import pandas as pd
    df = (pd.DataFrame(nicerows))
    df = df.sort_values('max_mem_alloc')
    df['max_mem_alloc_str'] = df['max_mem_alloc'].map(lambda x: xdev.byte_str(x, 'GB'))
    print(df.to_string())

    df_subset = df[df.arch_name.apply(lambda x: 'joint' in x)]

    import kwplot
    kwplot.autompl()
    sns = kwplot.autosns()
    from matplotlib.colors import LogNorm

    fnum = 0

    grouper = list(model_basis.keys())
    for k, subdf in df.groupby(grouper):
        fnum += 1
        print('')
        print('k = {!r}'.format(k))
        piv = subdf.pivot(['S', 'T'], ['M', 'num_params'], ['max_mem_alloc_str'])
        print(piv)

        fig = kwplot.figure(fnum=fnum)
        fig.set_size_inches(8.69, 4.93)
        ax = fig.gca()

        piv = piv.droplevel((0, 2), axis=1)
        d = piv.applymap(lambda x: float(x.split(' ')[0]) if isinstance(x, str) else x)

        arch_cfg = transformer.encoder_configs[k[0]]
        cfg = ub.dzip(grouper, k)
        cfg.update(ub.dict_isect(arch_cfg, {'n_layers', 'n_heats', 'embedding_size'}))
        title = ub.repr2(cfg, compact=1, sort=0)

        sns.heatmap(d,
                    annot=piv,
                    ax=ax, fmt='s',
                    norm=LogNorm(vmin=1, vmax=24),
                    annot_kws={'size': 8},
                    cbar_kws={'label': 'memory', 'pad': 0.001})
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.figure.subplots_adjust(bottom=0.2)
        ax.set_title(title)
        ax.set_xlabel('Number of Modes (M)')
        ax.set_ylabel('Space (S) Time (S) dims')
    # ax.figure

    # import timerit
    # ti = timerit.Timerit(3, bestof=1, verbose=2)
    # #
    # for arch_name in ['smt_it_stm_p8', 'smt_it_joint_p8', 'smt_it_hwtm_p8']:
    #     print('====')
    #     # self = MultimodalTransformer(arch_name=arch_name, input_channels=datamodule.input_channels)
    #     num_params = nh.util.number_of_parameters(self)
    #     print('arch_name = {!r}'.format(arch_name))
    #     print('num_params = {!r}'.format(num_params))
    #     print('running')
    #     self = self.to(device)
    #     output = self(images)
    #     for timer in ti.reset(f'inference-{arch_name}'):
    #         torch.cuda.synchronize()
    #         with timer:
    #             output = self(images)['change']
    #             torch.cuda.synchronize()
    #     for timer in ti.reset(f'train-{arch_name}'):
    #         torch.cuda.synchronize()
    #         with timer:
    #             output = self(images)['change']
    #             output.sum().backward()
    #             torch.cuda.synchronize()
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as pred_prof:
    #         with record_function(f"pred_{arch_name}"):
    #             output = self(images)['change']
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as train_prof:
    #         with record_function(f"train_{arch_name}"):
    #             output = self(images)['change']
    #             output.sum().backward()
    #     print('arch_name = {!r}'.format(arch_name))
    #     print('num_params = {!r}'.format(num_params))
    #     print(pred_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #     print(train_prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    #     total_memory = sum(event.cuda_memory_usage for event in train_prof.events())
    #     total_mem_str = xdev.byte_str(total_memory)
    #     print(total_mem_str)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/watch/watch/tasks/fusion/experiments/crall/benchmark_models.py
    """
    benchmark_models()
