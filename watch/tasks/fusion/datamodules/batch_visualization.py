import kwcoco
import kwimage
import numpy as np
import ubelt as ub
import einops


class BatchVisualizationBuilder:
    """
    Helper object to build a batch visualization.

    The basic logic is that we will build a column for each timestep and then
    arrange them from left to right to show how the scene changes over time.
    Each column will be made of "cells" which could show either the truth, a
    prediction, loss weights, or raw input channels.

    CommandLine:
        xdoctest -m watch.tasks.fusion.datamodules.batch_visualization BatchVisualizationBuilder

    Example:
        >>> from watch.tasks.fusion.datamodules.kwcoco_video_data import *  # NOQA
        >>> from watch.tasks.fusion.datamodules.batch_visualization import *  # NOQA
        >>> import ndsampler
        >>> import watch
        >>> coco_dset = watch.coerce_kwcoco('vidshapes2-watch', num_frames=5)
        >>> sampler = ndsampler.CocoSampler(coco_dset)
        >>> channels = 'r|g|b,B10|B8a|B1|B8|B11,X.2|Y.2'
        >>> combinable_extra = [['B10', 'B8', 'B8a']]  # special behavior
        >>> # combinable_extra = None  # uncomment for raw behavior
        >>> self = KWCocoVideoDataset(
        >>>     sampler, sample_shape=(5, 530, 610), channels=channels,
        >>>     use_centered_positives=True, neg_to_pos_ratio=0)
        >>> index = len(self) // 4
        >>> item = self[index]
        >>> # Calculate the probability of change for each frame
        >>> item_output = {}
        >>> change_prob_list = []
        >>> fliprot_params = item['tr'].get('fliprot_params', None)
        >>> for frame in item['frames'][1:]:
        >>>     change_prob = kwimage.Heatmap.random(
        >>>         dims=frame['target_dims'], classes=1).data['class_probs'][0]
        >>>     if fliprot_params:
        >>>         change_prob = fliprot(change_prob, **fliprot_params)
        >>>     change_prob_list += [change_prob]
        >>> change_probs = np.stack(change_prob_list)
        >>> item_output['change_probs'] = change_probs  # first frame does not have change
        >>> #
        >>> # Probability of each class for each frame
        >>> class_prob_list = []
        >>> for frame in item['frames']:
        >>>     class_prob = kwimage.Heatmap.random(
        >>>         dims=frame['target_dims'], classes=list(sampler.classes)).data['class_probs']
        >>>     class_prob = einops.rearrange(class_prob, 'c h w -> h w c')
        >>>     if fliprot_params:
        >>>         class_prob = fliprot(class_prob, **fliprot_params)
        >>>     class_prob_list += [class_prob]
        >>> class_probs = np.stack(class_prob_list)
        >>> item_output['class_probs'] = class_probs  # first frame does not have change
        >>> #
        >>> # Probability of "saliency" (i.e. non-background) for each frame
        >>> saliency_prob_list = []
        >>> for frame in item['frames']:
        >>>     saliency_prob = kwimage.Heatmap.random(
        >>>         dims=frame['target_dims'], classes=1).data['class_probs']
        >>>     saliency_prob = einops.rearrange(saliency_prob, 'c h w -> h w c')
        >>>     if fliprot_params:
        >>>         saliency_prob = fliprot(saliency_prob, **fliprot_params)
        >>>     saliency_prob_list += [saliency_prob]
        >>> saliency_probs = np.stack(saliency_prob_list)
        >>> item_output['saliency_probs'] = saliency_probs
        >>> #binprobs[0][:] = 0  # first change prob should be all zeros
        >>> builder = BatchVisualizationBuilder(
        >>>     item, item_output, classes=self.classes, requested_tasks=self.requested_tasks,
        >>>     default_combinable_channels=self.default_combinable_channels, combinable_extra=combinable_extra)
        >>> #builder.overlay_on_image = 1
        >>> #canvas = builder.build()
        >>> builder.max_channels = 3
        >>> builder.overlay_on_image = 0
        >>> canvas2 = builder.build()
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> #kwplot.imshow(canvas, fnum=1, pnum=(1, 2, 1))
        >>> #kwplot.imshow(canvas2, fnum=1, pnum=(1, 2, 2))
        >>> kwplot.imshow(canvas2, fnum=1, doclf=True)
        >>> kwplot.show_if_requested()
    """

    def __init__(builder, item, item_output=None, combinable_extra=None,
                 max_channels=5, max_dim=224, norm_over_time=0,
                 overlay_on_image=False, draw_weights=True, classes=None,
                 default_combinable_channels=None,
                 requested_tasks=None):
        builder.max_channels = max_channels
        builder.max_dim = max_dim
        builder.norm_over_time = norm_over_time
        builder.combinable_extra = combinable_extra
        builder.item_output = item_output
        builder.item = item
        builder.overlay_on_image = overlay_on_image
        builder.draw_weights = draw_weights
        builder.requested_tasks = requested_tasks

        builder.classes = classes
        builder.default_combinable_channels = default_combinable_channels

        combinable_channels = default_combinable_channels
        if combinable_extra is not None:
            combinable_channels = combinable_channels.copy()
            combinable_channels += list(map(ub.oset, combinable_extra))
        builder.combinable_channels = combinable_channels
        # print('builder.combinable_channels = {}'.format(ub.repr2(builder.combinable_channels, nl=1)))

    def build(builder):
        frame_metas = builder._prepare_frame_metadata()
        if 0:
            for idx, frame_meta in enumerate(frame_metas):
                print('---')
                print('idx = {!r}'.format(idx))
                frame_weight_shape = ub.map_vals(lambda x: x.shape, frame_meta['frame_weight'])
                print('frame_weight_shape = {}'.format(ub.repr2(frame_weight_shape, nl=1)))
                frame_meta['frame_weight']
        canvas = builder._build_canvas(frame_metas)
        return canvas

    def _prepare_frame_metadata(builder):
        import more_itertools
        item = builder.item
        combinable_channels = builder.combinable_channels

        truth_keys = []
        weight_keys = []
        if builder.requested_tasks['class']:
            truth_keys.append('class_idxs')
            weight_keys.append('class_weights')
        if builder.requested_tasks['saliency']:
            truth_keys.append('saliency')
            weight_keys.append('saliency_weights')
        if builder.requested_tasks['change']:
            truth_keys.append('change')
            weight_keys.append('change_weights')

        # Prepare metadata on each frame
        frame_metas = []
        for frame_idx, frame_item in enumerate(item['frames']):
            # Gather ground truth rasters
            frame_truth = {}
            for truth_key in truth_keys:
                truth_data = frame_item[truth_key]
                if truth_data is not None:
                    truth_data = truth_data.data.cpu().numpy()
                    frame_truth[truth_key] = truth_data

            frame_weight = {}
            for weight_key in weight_keys:
                weight_data = frame_item[weight_key]
                if weight_data is not None:
                    weight_data = weight_data.data.cpu().numpy()
                    frame_weight[weight_key] = weight_data
                else:
                    # HACK so saliency weights align correctly
                    frame_weight[weight_key] = None
                    # np.full((2, 2), fill_value=np.nan)

            # Breakup all of the modes into 1-channel per array
            frame_chan_names = []
            frame_chan_datas = []
            frame_modes = frame_item['modes']
            for mode_code, mode_data in frame_modes.items():
                mode_data = mode_data.data.cpu().numpy()
                code_list = kwcoco.FusedChannelSpec.coerce(mode_code).normalize().as_list()
                for chan_data, chan_name in zip(mode_data, code_list):
                    frame_chan_names.append(chan_name)
                    frame_chan_datas.append(chan_data)
            full_mode_code = ','.join(list(frame_item['modes'].keys()))

            # Determine what single and combinable channels exist per stream
            perstream_available = []
            for mode_code in frame_modes.keys():
                code_list = kwcoco.FusedChannelSpec.coerce(mode_code).normalize().as_list()
                code_set = ub.oset(code_list)
                stream_combinables = []
                for combinable in combinable_channels:
                    if combinable.issubset(code_set):
                        stream_combinables.append(combinable)
                remain = code_set - set(ub.flatten(stream_combinables))
                stream_singletons = [(c,) for c in remain]
                # Prioritize combinable channels in each stream first
                stream_available = list(map(tuple, stream_combinables)) + stream_singletons
                perstream_available.append(stream_available)

            # Prioritize choosing a balance of channels from each stream
            frame_available_chans = list(more_itertools.roundrobin(*perstream_available))

            frame_meta = {
                'full_mode_code': full_mode_code,
                'frame_idx': frame_idx,
                'frame_item': frame_item,
                'frame_chan_names': frame_chan_names,
                'frame_chan_datas': frame_chan_datas,
                'frame_available_chans': frame_available_chans,
                'frame_truth': frame_truth,
                'frame_weight': frame_weight,
                'sensor': frame_item.get('sensor', '*'),
            }
            frame_metas.append(frame_meta)

        # Determine which frames to visualize For each frame choose N channels
        # such that common channels are aligned, visualize common channels in
        # the first rows and then fill with whatever is left
        # chan_freq = ub.dict_hist(ub.flatten(frame_meta['frame_available_chans']
        #                                     for frame_meta in frame_metas))
        # chan_priority = {k: (v, len(k), -idx) for idx, (k, v)
        #                  in enumerate(chan_freq.items())}
        for frame_meta in frame_metas:
            chan_keys = frame_meta['frame_available_chans']
            # print('chan_keys = {!r}'.format(chan_keys))
            # frame_priority = ub.dict_isect(chan_priority, chan_keys)
            # chosen = ub.argsort(frame_priority, reverse=True)[0:builder.max_channels]
            # print('chosen = {!r}'.format(chosen))
            chosen = chan_keys[0:builder.max_channels]
            frame_meta['chans_to_use'] = chosen

        # Gather channels to visualize
        for frame_meta in frame_metas:
            chans_to_use = frame_meta['chans_to_use']
            frame_chan_names = frame_meta['frame_chan_names']
            frame_chan_datas = frame_meta['frame_chan_datas']
            chan_idx_lut = {name: idx for idx, name in enumerate(frame_chan_names)}
            # Prepare and normalize the channels for visualization
            chan_rows = []
            for chan_names in chans_to_use:
                chan_code = '|'.join(chan_names)
                chanxs = list(ub.take(chan_idx_lut, chan_names))
                parts = list(ub.take(frame_chan_datas, chanxs))
                raw_signal = np.stack(parts, axis=2)
                row = {
                    'raw_signal': raw_signal,
                    'chan_code': chan_code,
                    'signal_text': f'{chan_code}',
                    'sensor': frame_meta['sensor'],
                }
                chan_rows.append(row)
            frame_meta['chan_rows'] = chan_rows
            assert len(chan_rows) > 0, 'no channels to draw on'

        if builder.draw_weights:
            # Normalize weights for visualization
            all_weight_overlays = []
            for frame_meta in frame_metas:
                frame_meta['weight_overlays'] = {}
                for weight_key, weight_data in frame_meta['frame_weight'].items():
                    overlay_row = {
                        'weight_key': weight_key,
                        'raw': weight_data,
                    }
                    frame_meta['weight_overlays'][weight_key] = overlay_row
                    all_weight_overlays.append(overlay_row)

            for weight_key, group in ub.group_items(all_weight_overlays, lambda x: x['weight_key']).items():
                # print('weight_key = {!r}'.format(weight_key))
                # maxval = -float('inf')
                # minval = float('inf')
                # for cell in group:
                #     maxval = max(maxval, cell['raw'].max())
                #     minval = min(minval, cell['raw'].min())
                # print('maxval = {!r}'.format(maxval))
                # print('minval = {!r}'.format(minval))
                for cell in group:
                    weight_data = cell['raw']
                    if weight_data is None:
                        h = w = builder.max_dim
                        weight_overlay = kwimage.draw_text_on_image(
                            {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                            valign='center', halign='center', fontScale=10,
                            color='red')
                        weight_overlay = kwimage.ensure_float01(weight_overlay)
                    else:
                        weight_overlay = kwimage.atleast_3channels(weight_data)
                    # weight_overlay = kwimage.ensure_alpha_channel(weight_overlay)
                    # weight_overlay[:, 3] = 0.5
                    cell['overlay'] = weight_overlay

        # Normalize raw signal into visualizable range
        if builder.norm_over_time:
            # Normalize all cells with the same channel code across time
            channel_cells = [cell for frame_meta in frame_metas for cell in frame_meta['chan_rows']]
            # chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'])
            chan_to_cells = ub.group_items(channel_cells, lambda c: (c['chan_code'], c['sensor']))
            for chan_code, cells in chan_to_cells.items():
                flat = [c['raw_signal'].ravel() for c in cells]
                cums = np.cumsum(list(map(len, flat)))
                combo = np.hstack(flat)
                mask = (combo != 0) & np.isfinite(combo)
                # try:
                combo_normed = kwimage.normalize_intensity(combo, mask=mask).copy()
                # except Exception:
                #     combo_normed = combo.copy()
                flat_normed = np.split(combo_normed, cums)
                for cell, flat_item in zip(cells, flat_normed):
                    norm_signal = flat_item.reshape(*cell['raw_signal'].shape)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    # norm_signal = np.nan_to_num(norm_signal)
                    norm_signal = kwimage.fill_nans_with_checkers(norm_signal)
                    cell['norm_signal'] = norm_signal
        else:
            # Normalize each timestep by itself
            for frame_meta in frame_metas:
                for row in frame_meta['chan_rows']:
                    raw_signal = row['raw_signal']
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='All-NaN slice')
                        needs_norm = np.nanmin(raw_signal) < 0 or np.nanmax(raw_signal) > 1
                    if needs_norm:
                        mask = (raw_signal != 0) & np.isfinite(raw_signal)
                        norm_signal = kwimage.normalize_intensity(raw_signal, mask=mask).copy()
                        # try:
                        # except Exception:
                        #     norm_signal = raw_signal.copy()
                    else:
                        norm_signal = raw_signal.copy()
                    norm_signal = kwimage.fill_nans_with_checkers(norm_signal)
                    # norm_signal = np.nan_to_num(norm_signal)
                    from watch.utils import util_kwimage
                    norm_signal = util_kwimage.ensure_false_color(norm_signal)
                    norm_signal = kwimage.atleast_3channels(norm_signal)
                    row['norm_signal'] = norm_signal

        return frame_metas

    def _build_canvas(builder, frame_metas):

        # Given prepared frame metadata, build a vertical stack of per-chanel
        # information, and then horizontally stack the timesteps.
        horizontal_stack = []

        truth_overlay_keys = set(ub.flatten([m['frame_truth'] for m in frame_metas]))
        weight_overlay_keys = set(ub.flatten([m['frame_weight'] for m in frame_metas]))

        for frame_meta in frame_metas:
            frame_canvas = builder._build_frame(
                frame_meta, truth_overlay_keys, weight_overlay_keys)
            horizontal_stack.append(frame_canvas)

        body_canvas = kwimage.stack_images(horizontal_stack, axis=1, pad=5)
        body_canvas = body_canvas[..., 0:3]  # drop alpha
        body_canvas = kwimage.ensure_uint255(body_canvas)  # convert to uint8

        width = body_canvas.shape[1]

        vid_text = f'video: {builder.item["video_id"]} - {builder.item["video_name"]}'

        sample_gsd = builder.item.get('sample_gsd', None)
        if sample_gsd is not None:
            if isinstance(sample_gsd, float):
                vid_text = vid_text + ' @ {:0.2f} GSD'.format(sample_gsd)
            else:
                vid_text = vid_text + ' @ {} GSD'.format(sample_gsd)

        vid_header = kwimage.draw_text_on_image(
            {'width': width}, vid_text, org=(width // 2, 3), valign='top',
            halign='center', color='pink')

        canvas = kwimage.stack_images([vid_header, body_canvas], axis=0, pad=3)
        return canvas

    def _build_frame_header(builder, frame_meta):
        """
        Make the text header for each timestep (frame)
        """
        header_stack = []

        frame_item = frame_meta['frame_item']
        frame_idx = frame_meta['frame_idx']
        gid = frame_item['gid']

        # Build column headers
        header_dims = {'width': builder.max_dim}
        header_part = kwimage.draw_header_text(
            image=header_dims, fit=False,
            text=f't={frame_idx} gid={gid}', color='salmon')
        header_stack.append(header_part)

        sensor = frame_item.get('sensor', '*')
        if sensor != '*':
            header_part = kwimage.draw_header_text(
                image=header_dims, fit=False, text=f'{sensor}',
                color='salmon')
            header_stack.append(header_part)

        date_captured = frame_item.get('date_captured', '')
        if date_captured:
            header_part = kwimage.draw_header_text(
                header_dims, fit='shrink', text=f'{date_captured}',
                color='salmon')
            header_stack.append(header_part)
        return header_stack

    def _build_frame(builder, frame_meta, truth_overlay_keys, weight_overlay_keys):
        """
        Build a vertical stack for a single frame
        """
        classes = builder.classes
        item_output = builder.item_output

        vertical_stack = []

        frame_idx = frame_meta['frame_idx']
        chan_rows = frame_meta['chan_rows']

        frame_truth = frame_meta['frame_truth']
        # frame_weight = frame_meta['frame_weight']

        # Build column headers
        header_stack = builder._build_frame_header(frame_meta)
        vertical_stack.extend(header_stack)

        # Build truth / metadata overlays
        if len(frame_truth):
            overlay_shape = ub.peek(frame_truth.values()).shape[0:2]
        else:
            overlay_shape = None

        # Create overlays for training objective targets
        overlay_items = []

        # Create the the true class label overlay
        overlay_key = 'class_idxs'
        if overlay_key in truth_overlay_keys and builder.requested_tasks['class']:
            class_idxs = frame_truth.get(overlay_key, None)
            true_heatmap = kwimage.Heatmap(class_idx=class_idxs, classes=classes)
            class_overlay = true_heatmap.colorize('class_idx')
            class_overlay[..., 3] = 0.5
            overlay_items.append({
                'overlay': class_overlay,
                'label_text': 'true class',
            })

        # Create the the true saliency label overlay
        overlay_key = 'saliency'
        if overlay_key in truth_overlay_keys and builder.requested_tasks['saliency']:
            saliency = frame_truth.get(overlay_key, None)
            if saliency is not None:
                if 1:
                    saliency_overlay = kwimage.make_heatmask(saliency.astype(np.float32), cmap='plasma').clip(0, 1)
                    saliency_overlay[..., 3] *= 0.5
                else:
                    saliency_overlay = np.zeros(saliency.shape + (4,), dtype=np.float32)
                    saliency_overlay = kwimage.Mask(saliency, format='c_mask').draw_on(saliency_overlay, color='dodgerblue')
                    saliency_overlay = kwimage.ensure_alpha_channel(saliency_overlay)
                    saliency_overlay[..., 3] = (saliency > 0).astype(np.float32) * 0.5
                overlay_items.append({
                    'overlay': saliency_overlay,
                    'label_text': 'true saliency',
                })

        # Create the true change label overlay
        overlay_key = 'change'
        if overlay_key in truth_overlay_keys and builder.requested_tasks['change']:
            change_overlay = np.zeros(overlay_shape + (4,), dtype=np.float32)
            changes = frame_truth.get(overlay_key, None)
            if changes is not None:
                if 1:
                    change_overlay = kwimage.make_heatmask(changes.astype(np.float32), cmap='viridis').clip(0, 1)
                    change_overlay[..., 3] *= 0.5
                else:
                    change_overlay = kwimage.Mask(changes, format='c_mask').draw_on(change_overlay, color='lime')
                    change_overlay = kwimage.ensure_alpha_channel(change_overlay)
                    change_overlay[..., 3] = (changes > 0).astype(np.float32) * 0.5
            overlay_items.append({
                'overlay': change_overlay,
                'label_text': 'true change',
            })

        weight_items = []
        if builder.draw_weights:
            weight_overlays = frame_meta['weight_overlays']
            for overlay_key in weight_overlay_keys:
                weight_overlay_info = weight_overlays.get(overlay_key, None)
                if weight_overlay_info is not None:
                    weight_items.append({
                        'overlay': weight_overlay_info['overlay'],
                        'label_text': overlay_key,
                    })

        resizekw = {
            'dsize': (builder.max_dim, builder.max_dim),
            # 'max_dim': builder.max_dim,
            # 'letterbox': False,
            'letterbox': True,
            'interpolation': 'nearest',
            # 'interpolation': 'linear',
        }

        # TODO: clean up logic
        key = 'class_probs'
        overlay_index = 0
        if item_output and key in item_output and builder.requested_tasks['class']:
            if builder.overlay_on_image:
                norm_signal = chan_rows[overlay_index]['norm_signal']
            else:
                norm_signal = np.zeros_like(chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal'])
            x = item_output[key][frame_idx]
            class_probs = einops.rearrange(x, 'h w c -> c h w')
            class_heatmap = kwimage.Heatmap(class_probs=class_probs, classes=classes)
            pred_part = class_heatmap.draw_on(norm_signal, with_alpha=0.7)
            # TODO: we might want to overlay the prediction on one or
            # all of the channels
            pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)
            pred_text = f'pred class t={frame_idx}'
            pred_part = kwimage.draw_text_on_image(
                pred_part, pred_text, (1, 1), valign='top',
                color='dodgerblue', border=3)
            vertical_stack.append(pred_part)

        key = 'saliency_probs'
        if item_output and key in item_output and builder.requested_tasks['saliency']:
            if builder.overlay_on_image:
                norm_signal = chan_rows[0]['norm_signal']
            else:
                norm_signal = np.zeros_like(chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal'])
            x = item_output[key][frame_idx]
            saliency_probs = einops.rearrange(x, 'h w c -> c h w')
            # Hard coded index, dont like
            is_salient_probs = saliency_probs[1]
            # saliency_heatmap = kwimage.Heatmap(class_probs=saliency_probs)
            # pred_part = saliency_heatmap.draw_on(norm_signal, with_alpha=0.7)
            pred_part = kwimage.make_heatmask(is_salient_probs, cmap='plasma')
            pred_part[..., 3] = 0.7
            # TODO: we might want to overlay the prediction on one or
            # all of the channels
            pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)
            pred_text = f'pred saliency t={frame_idx}'
            pred_part = kwimage.draw_text_on_image(
                pred_part, pred_text, (1, 1), valign='top',
                color='dodgerblue', border=3)
            vertical_stack.append(pred_part)

        key = 'change_probs'
        overlay_index = 1
        if item_output and key in item_output and builder.requested_tasks['change']:
            # Make a probability heatmap we can either display
            # independently or overlay on a rendered channel
            if frame_idx == 0:
                # BIG RED X
                # h, w = vertical_stack[-1].shape[0:2]
                h = w = builder.max_dim
                pred_mask = kwimage.draw_text_on_image(
                    {'width': w, 'height': h}, 'X', org=(w // 2, h // 2),
                    valign='center', halign='center', fontScale=10,
                    color='red')
                pred_part = pred_mask
            else:
                pred_raw = item_output[key][frame_idx - 1]
                # Draw predictions on the first item
                pred_mask = kwimage.make_heatmask(pred_raw, cmap='viridis')
                norm_signal = chan_rows[min(overlay_index, len(chan_rows) - 1)]['norm_signal']
                if builder.overlay_on_image:
                    norm_signal = norm_signal
                else:
                    norm_signal = np.zeros_like(norm_signal)
                pred_layers = [pred_mask, norm_signal]
                pred_part = kwimage.overlay_alpha_layers(pred_layers)
                # TODO: we might want to overlay the prediction on one or
                # all of the channels
                pred_part = kwimage.imresize(pred_part, **resizekw).clip(0, 1)
                pred_text = f'pred change t={frame_idx}'
                pred_part = kwimage.draw_text_on_image(
                    pred_part, pred_text, (1, 1), valign='top',
                    color='dodgerblue', border=3)
            vertical_stack.append(pred_part)

        if not builder.overlay_on_image:
            # FIXME: might be broken
            # Draw the overlays by themselves
            for overlay_info in overlay_items:
                label_text = overlay_info['label_text']
                row_canvas = overlay_info['overlay'][..., 0:3]
                row_canvas = kwimage.imresize(row_canvas, **resizekw).clip(0, 1)
                signal_bottom_y = 1  # hack: hardcoded
                row_canvas = kwimage.ensure_uint255(row_canvas)
                row_canvas = kwimage.draw_text_on_image(
                    row_canvas, label_text, (1, signal_bottom_y + 1),
                    valign='top', color='lime', border=3)
                vertical_stack.append(row_canvas)

        for overlay_info in weight_items:
            label_text = overlay_info['label_text']
            row_canvas = overlay_info['overlay'][..., 0:3]
            row_canvas = row_canvas.copy()
            row_canvas = kwimage.imresize(row_canvas, **resizekw).clip(0, 1)
            signal_bottom_y = 1  # hack: hardcoded
            row_canvas = kwimage.ensure_uint255(row_canvas)
            row_canvas = kwimage.draw_text_on_image(
                row_canvas, label_text, (1, signal_bottom_y + 1),
                valign='top', color='lime', border=3)
            vertical_stack.append(row_canvas)

        for iterx, row in enumerate(chan_rows):
            layers = []
            label_text = None
            if builder.overlay_on_image:
                # Draw truth on the image itself
                if iterx < len(overlay_items):
                    overlay_info = overlay_items[iterx]
                    layers.append(overlay_info['overlay'])
                    label_text = overlay_info['label_text']

            layers.append(row['norm_signal'])
            row_canvas = kwimage.overlay_alpha_layers(layers)[..., 0:3]

            # row_canvas = kwimage.imresize(row_canvas, **resizekw).clip(0, 1)
            row_canvas = kwimage.ensure_uint255(row_canvas)
            row_canvas = kwimage.draw_text_on_image(
                row_canvas, row['signal_text'], (1, 1), valign='top',
                color='white', border=3)

            if label_text:
                # TODO: make draw_text_on_image able to return the
                # geometry of what it drew and use that.
                signal_bottom_y = 31  # hack: hardcoded
                row_canvas = kwimage.draw_text_on_image(
                    row_canvas, label_text, (1, signal_bottom_y + 1),
                    valign='top', color='lime', border=3)
            vertical_stack.append(row_canvas)

        vertical_stack = [kwimage.ensure_uint255(p) for p in vertical_stack]
        frame_canvas = kwimage.stack_images(vertical_stack, overlap=-3)
        return frame_canvas
