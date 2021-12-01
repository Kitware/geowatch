"""

        Ignore:
            python -m watch.cli.gifify \
                    -i /home/local/KHQ/jon.crall/data/work/toy_change/_overfit_viz7/ \
                    -o /home/local/KHQ/jon.crall/data/work/toy_change/_overfit_viz7.gif

            nh.initializers.functional.apply_initializer(self, torch.nn.init.kaiming_normal, {})


            # How to get data we need to step back into the dataloader
            # to debug the batch
            item = batch[0]

            item['frames'][0]['class_idxs'].unique()
            item['frames'][1]['class_idxs'].unique()
            item['frames'][2]['class_idxs'].unique()

            # print(item['frames'][0]['change'].unique())
            print(item['frames'][1]['change'].unique())
            print(item['frames'][2]['change'].unique())

            tr = item['tr']
            self = torch_dset
            kwplot.imshow(self.draw_item(item), fnum=3)

            kwplot.imshow(item['frames'][1]['change'].cpu().numpy(), fnum=4)

        Ignore:
            model = self
            model = self.to(0)

            for item in batch:
                for frame in item['frames']:
                    modes = frame['modes']
                    for key in modes.keys():
                        modes[key] = modes[key].to(0)
            out = model.forward_step(batch)

            batch2 = [ub.dict_diff(item, {'tr', 'index', 'video_name', 'video_id'})  for item in batch[0:1]]
            for item in batch2:
                item['frames'] = [
                    ub.dict_diff(frame, {
                        'gid', 'date_captured', 'sensor_coarse',
                        'change', 'ignore', 'class_idxs',
                    })
                    for frame in item['frames']
                ]

            traced = torch.jit.trace_module(model, {'forward_step': (batch2,)}, strict=False)

            traced = torch.jit.trace_module(model, {'forward': (images,)}, strict=False)

            import timerit
            ti = timerit.Timerit(5, bestof=1, verbose=2)
            for timer in ti.reset('time'):
                model.forward(images)
            for timer in ti.reset('time'):
                traced.forward(images)

            # traced = torch.jit.trace(model.forward, batch)
            traced = torch.jit.trace_module(model, {'forward_step': batch2})

"""
